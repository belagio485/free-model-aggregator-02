#!/usr/bin/env python3
"""Free-model aggregator: one server, two wires, live free models only.

Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages
(Anthropic) in JSON and streaming, plus GET /v1/models in the wire its
auth header names. Every completion is a real call to a free model
listed in the trial brief, dialled through the grader gateway on the
Chat Completions route. Model choice, ids and timestamps derive only
from the brief, the request, the trial seed / trial now, and the
gateway answers: no wall clock, no unseeded randomness, so a replay
leaves the same answers.
"""

import hashlib
import http.client
import http.server
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone

MAX_ATTEMPTS_PER_PROMPT = 8
MAX_CALLS_PER_TRIAL = 60
GATEWAY_TIMEOUT_SEC = 10
REGISTER_TIMEOUT_SEC = 5
WORK_TIMEOUT_SEC = 5
REGISTER_RETRIES = 30
RETRY_PAUSE_SEC = 0.5
POLL_INTERVAL_SEC = 0.5
DEFAULT_MAX_TOKENS = 1024
MAX_OUTPUT_TOKENS = 8192
STREAM_CHUNK_CHARS = 32
MAX_BODY_BYTES = 8 * 1024 * 1024
CATALOG_BASE_URL = "https://openrouter.ai/api/v1"
CATALOG_KEY_ENV = "OPENROUTER_API_KEY"
FALLBACK_EPOCH = 0
FALLBACK_ISO = "1970-01-01T00:00:00Z"

# Trial state, filled once in main() before the server starts.
BRIEF_MODELS = []
CATALOG = {}
CATALOG_TEXT = ""
TRIAL_EPOCH = FALLBACK_EPOCH
TRIAL_ISO = FALLBACK_ISO
TRIAL_SEED = ""
OPENAI_BASE_URL = ""
OPENAI_API_KEY = ""
_CALL_COUNT = 0
_CALL_LOCK = threading.Lock()
_IN_FLIGHT = 0
_IN_FLIGHT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Trial time: only MAKEFUN_TRIAL_NOW is ever used as "now".
# ---------------------------------------------------------------------------

def parse_trial_now(raw):
    """Return (epoch_seconds, iso_string) for a MAKEFUN_TRIAL_NOW value."""
    if raw is None:
        return FALLBACK_EPOCH, FALLBACK_ISO
    text = str(raw).strip()
    if not text:
        return FALLBACK_EPOCH, FALLBACK_ISO
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            epoch = int(text)
        except ValueError:
            return FALLBACK_EPOCH, FALLBACK_ISO
        try:
            iso = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except (OverflowError, OSError, ValueError):
            iso = FALLBACK_ISO
        return epoch, iso
    candidate = text
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return FALLBACK_EPOCH, FALLBACK_ISO
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        epoch = int(parsed.timestamp())
    except (OverflowError, OSError, ValueError):
        return FALLBACK_EPOCH, FALLBACK_ISO
    iso = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return epoch, iso


# ---------------------------------------------------------------------------
# Brief parsing: the models document embedded in the brief text.
# ---------------------------------------------------------------------------

def _decode_json_at(text, start):
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return value


def _normalise_model(row):
    if not isinstance(row, dict):
        return None
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    rating = row.get("rating_milli")
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        rating = 0
    context_length = row.get("context_length")
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        context_length = None
    display = row.get("display_name")
    if not isinstance(display, str) or not display:
        display = model_id
    return {
        "id": model_id,
        "display_name": display,
        "rating": int(rating),
        "context_length": context_length,
        "tools": bool(row.get("tools")),
        "vision": bool(row.get("vision")),
        "json": bool(row.get("json")),
    }


def _fallback_models(text):
    """Last-resort extraction of dialable ids when the JSON doc is absent."""
    found = []
    seen = set()
    for match in re.finditer(r'"id"\s*:\s*"(openrouter/[^"]+:free)"', text):
        model_id = match.group(1)
        if model_id in seen:
            continue
        seen.add(model_id)
        found.append(
            {
                "id": model_id,
                "display_name": model_id,
                "rating": 0,
                "context_length": None,
                "tools": True,
                "vision": True,
                "json": True,
            }
        )
    return found


def load_brief_models(path):
    """Read the brief file and return the ordered list of free models."""
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return []
    match = re.search(
        r'\{\s*"schema"\s*:\s*"free-model-aggregator-02/models/v1"', text
    )
    if match is not None:
        doc = _decode_json_at(text, match.start())
        if isinstance(doc, dict) and isinstance(doc.get("models"), list):
            models = []
            seen = set()
            for row in doc["models"]:
                item = _normalise_model(row)
                if item is not None and item["id"] not in seen:
                    seen.add(item["id"])
                    models.append(item)
            if models:
                return models
    return _fallback_models(text)


def build_catalog(models):
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": CATALOG_BASE_URL,
                "key_env": CATALOG_KEY_ENV,
                "wire": "openai",
                "models": [
                    {
                        "id": item["id"],
                        "free": True,
                        "context_length": item["context_length"],
                    }
                    for item in models
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Small HTTP client used for the gateway, register and /work.
# ---------------------------------------------------------------------------

def http_request(method, url, body_bytes, headers, timeout):
    """Return (status, raw_bytes, parsed_json_or_None). Raises on transport error."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname
    if not host:
        raise ValueError("URL without a host")
    port = parts.port
    if scheme == "https":
        connection = http.client.HTTPSConnection(
            host, port or 443, timeout=timeout
        )
    else:
        connection = http.client.HTTPConnection(host, port or 80, timeout=timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        connection.request(method, path, body=body_bytes, headers=headers)
        response = connection.getresponse()
        raw = response.read(MAX_BODY_BYTES + 1)
        status = response.status
    finally:
        connection.close()
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            parsed = None
    return status, raw, parsed


def error_code_from(status, parsed):
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if isinstance(value, str) and value:
                    return value[:128]
        value = parsed.get("code")
        if isinstance(value, str) and value:
            return value[:128]
    if status == 429:
        return "rate_limit_error"
    if status == 401:
        return "authentication_error"
    if status == 400:
        return "invalid_request_error"
    if status and status >= 500:
        return "api_error"
    return "upstream_error"


def error_message_from(parsed, fallback):
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message[:500]
        message = parsed.get("message")
        if isinstance(message, str) and message:
            return message[:500]
    return fallback


# ---------------------------------------------------------------------------
# Model selection: highest brief rating meeting the request's requirements.
# ---------------------------------------------------------------------------

def _openai_has_images(body):
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "image",
                    "input_image",
                ):
                    return True
    return False


def _anthropic_has_images(body):
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def requirements_from_openai(body):
    response_format = body.get("response_format")
    needs_json = isinstance(response_format, dict) and response_format.get(
        "type"
    ) in ("json_object", "json_schema")
    tools = body.get("tools")
    return {
        "tools": isinstance(tools, list) and len(tools) > 0,
        "vision": _openai_has_images(body),
        "json": needs_json,
    }


def requirements_from_anthropic(body):
    tools = body.get("tools")
    return {
        "tools": isinstance(tools, list) and len(tools) > 0,
        "vision": _anthropic_has_images(body),
        "json": False,
    }


def rank_models(models):
    return sorted(models, key=lambda item: (-item["rating"], item["id"]))


def choose_models(requirements):
    ranked = rank_models(BRIEF_MODELS)
    eligible = [
        item
        for item in ranked
        if (not requirements["tools"] or item["tools"])
        and (not requirements["vision"] or item["vision"])
        and (not requirements["json"] or item["json"])
    ]
    return eligible if eligible else ranked


def attempt_list(ranked_eligible, preferred_id=None):
    """Best model twice (one retry), then the ranking, capped per prompt."""
    ordered = list(ranked_eligible)
    if preferred_id:
        preferred = [item for item in ordered if item["id"] == preferred_id]
        rest = [item for item in ordered if item["id"] != preferred_id]
        ordered = preferred + rest
    if not ordered:
        return []
    sequence = [ordered[0]] + ordered
    result = []
    for item in sequence:
        model_id = item["id"]
        if model_id in result and model_id != ordered[0]["id"]:
            continue
        if result.count(model_id) >= 2:
            continue
        result.append(model_id)
        if len(result) >= MAX_ATTEMPTS_PER_PROMPT:
            break
    return result


# ---------------------------------------------------------------------------
# Upstream bodies: OpenAI passthrough, Anthropic translation.
# ---------------------------------------------------------------------------

def ensure_output_cap(body):
    """Guarantee a gateway-accepted output cap, preserving the client's field."""
    if isinstance(body.get("max_tokens"), (int, float)) and not isinstance(
        body.get("max_tokens"), bool
    ):
        body["max_tokens"] = max(1, min(int(body["max_tokens"]), MAX_OUTPUT_TOKENS))
        body.pop("max_completion_tokens", None)
        return body
    if isinstance(body.get("max_completion_tokens"), (int, float)) and not isinstance(
        body.get("max_completion_tokens"), bool
    ):
        body["max_completion_tokens"] = max(
            1, min(int(body["max_completion_tokens"]), MAX_OUTPUT_TOKENS)
        )
        return body
    body["max_tokens"] = DEFAULT_MAX_TOKENS
    return body


def upstream_openai_body(body):
    upstream = dict(body)
    upstream.pop("stream", None)
    upstream.pop("stream_options", None)
    upstream.pop("store", None)
    return ensure_output_cap(upstream)


def _anthropic_text_of(blocks):
    texts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _anthropic_system_messages(system):
    if isinstance(system, str):
        if system:
            return [{"role": "system", "content": system}]
        return []
    if isinstance(system, list):
        text = _anthropic_text_of(system)
        if text:
            return [{"role": "system", "content": text}]
        return []
    return []


def _anthropic_content_parts(blocks):
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif kind == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "base64":
                media = source.get("media_type") or "image/jpeg"
                data = source.get("data") or ""
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s" % (media, data)
                        },
                    }
                )
            elif source.get("type") == "url":
                url = source.get("url")
                if isinstance(url, str) and url:
                    parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
    return parts


def _tool_result_text(block):
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _anthropic_text_of(content)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), sort_keys=True)


def translate_anthropic_messages(body):
    messages = []
    messages.extend(_anthropic_system_messages(body.get("system")))
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return messages
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        tool_uses = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        tool_results = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if tool_results and not tool_uses:
            for block in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _tool_result_text(block),
                    }
                )
            continue
        if tool_uses:
            text = _anthropic_text_of(content)
            tool_calls = []
            for block in tool_uses:
                arguments = block.get("input")
                if isinstance(arguments, str):
                    arguments_text = arguments
                else:
                    arguments_text = json.dumps(
                        arguments if arguments is not None else {},
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": arguments_text,
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": text if text else None,
                    "tool_calls": tool_calls,
                }
            )
            continue
        parts = _anthropic_content_parts(content)
        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": role, "content": parts[0]["text"]})
        elif parts:
            messages.append({"role": role, "content": parts})
        else:
            messages.append({"role": role, "content": ""})
    return messages


def translate_anthropic_tools(body):
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    translated = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            schema = {}
        translated.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": schema,
                },
            }
        )
    return translated if translated else None


def translate_anthropic_tool_choice(body):
    choice = body.get("tool_choice")
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        name = choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
        return "required"
    return None


def upstream_anthropic_body(body):
    upstream = {"messages": translate_anthropic_messages(body)}
    tools = translate_anthropic_tools(body)
    if tools is not None:
        upstream["tools"] = tools
    tool_choice = translate_anthropic_tool_choice(body)
    if tool_choice is not None:
        upstream["tool_choice"] = tool_choice
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            upstream[key] = body[key]
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        upstream["stop"] = stop
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool):
        upstream["max_tokens"] = max(
            1, min(int(max_tokens), MAX_OUTPUT_TOKENS)
        )
    else:
        upstream["max_tokens"] = DEFAULT_MAX_TOKENS
    return upstream


# ---------------------------------------------------------------------------
# Gateway dialling with retry + failover.
# ---------------------------------------------------------------------------

def gateway_chat(model_id, upstream_body):
    """One Chat Completions call. Returns (status, raw, parsed)."""
    global _CALL_COUNT
    with _CALL_LOCK:
        _CALL_COUNT += 1
    base = OPENAI_BASE_URL.rstrip("/")
    url = base + "/chat/completions"
    payload = json.dumps(
        upstream_body, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    headers = {
        "Authorization": "Bearer %s" % OPENAI_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    status, raw, parsed = http_request(
        "POST", url, payload, headers, GATEWAY_TIMEOUT_SEC
    )
    return status, raw, parsed


def calls_made():
    with _CALL_LOCK:
        return _CALL_COUNT


def valid_gateway_answer(parsed):
    if not isinstance(parsed, dict):
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    message = first.get("message")
    return isinstance(message, dict)


def dial_with_failover(upstream_template, ranked, preferred_id):
    """Try models in order. Returns (model_id, parsed, raw, failures, last).

    failures is the ordered refusal trail; last is (status, code, message)
    describing the final refusal, for the give-up error response.
    """
    failures = []
    last = (502, "upstream_error", "Upstream model call failed")
    for model_id in attempt_list(ranked, preferred_id):
        if calls_made() >= MAX_CALLS_PER_TRIAL:
            last = (429, "request_cap", "Trial model-call cap reached")
            break
        upstream = dict(upstream_template)
        upstream["model"] = model_id
        try:
            status, raw, parsed = gateway_chat(model_id, upstream)
        except Exception:
            status, raw, parsed = None, b"", None
        if status == 200 and valid_gateway_answer(parsed):
            return model_id, parsed, raw, failures, last
        if status == 200:
            record_status, code = 502, "bad_gateway_response"
            message = "Upstream answer was not a usable completion"
        elif status is None:
            record_status, code = 502, "gateway_error"
            message = "Upstream model call failed"
        else:
            record_status = status
            code = error_code_from(status, parsed)
            message = error_message_from(parsed, "Upstream model call failed")
        failures.append(
            {"status": record_status, "code": code, "model": model_id}
        )
        last = (record_status, code, message)
    return None, None, b"", failures, last


def failures_header_value(failures):
    rows = failures[:8]
    text = json.dumps(rows, separators=(",", ":"))
    encoded = text.encode("utf-8")
    while len(encoded) > 8192 and rows:
        rows = rows[:-1]
        text = json.dumps(rows, separators=(",", ":"))
        encoded = text.encode("utf-8")
    return text


# ---------------------------------------------------------------------------
# Rendering: gateway answers back into the wire the client used.
# ---------------------------------------------------------------------------

def deterministic_id(prefix, request_raw, gateway_raw):
    digest = hashlib.sha256()
    digest.update(TRIAL_SEED.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(request_raw or b"")
    digest.update(b"\x00")
    digest.update(gateway_raw or b"")
    return prefix + digest.hexdigest()[:24]


def split_text(text, size=STREAM_CHUNK_CHARS):
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def usage_counts(parsed, request_chars, response_chars):
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(
            prompt_tokens, bool
        ):
            prompt_tokens = max(0, prompt_tokens)
        else:
            prompt_tokens = max(0, request_chars // 4)
        if isinstance(completion_tokens, int) and not isinstance(
            completion_tokens, bool
        ):
            completion_tokens = max(0, completion_tokens)
        else:
            completion_tokens = max(1, response_chars // 4)
        if response_chars > 0 and completion_tokens == 0:
            completion_tokens = 1
        return prompt_tokens, completion_tokens
    return max(0, request_chars // 4), max(1, response_chars // 4)


def message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)
    return ""


def normalise_tool_call(call, basis):
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        function = {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments_text = arguments
    elif arguments is None:
        arguments_text = ""
    else:
        arguments_text = json.dumps(
            arguments, separators=(",", ":"), sort_keys=True
        )
    call_id = call.get("id") if isinstance(call, dict) else None
    if not isinstance(call_id, str) or not call_id:
        digest = hashlib.sha256()
        digest.update(basis)
        digest.update(b"\x00")
        digest.update(str(function.get("name")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(arguments_text.encode("utf-8"))
        call_id = "call_" + digest.hexdigest()[:24]
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": function.get("name"),
            "arguments": arguments_text,
        },
    }


def filter_tool_calls(parsed, declared_names):
    """Drop tool calls naming undeclared tools, only when some remain."""
    if not declared_names:
        return parsed
    choices = parsed.get("choices")
    if not isinstance(choices, list):
        return parsed
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        kept = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if name in declared_names:
                kept.append(call)
        if kept and len(kept) != len(calls):
            message["tool_calls"] = kept
    return parsed


def declared_tool_names(body):
    tools = body.get("tools")
    names = set()
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            else:
                name = tool.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def openai_response_body(parsed, model_id, response_id, request_chars):
    choices_out = []
    response_chars = 0
    for index, choice in enumerate(parsed.get("choices", [])):
        message = choice.get("message") or {}
        text = message_text(message)
        response_chars += len(text)
        raw_content = message.get("content")
        if isinstance(raw_content, list):
            content_value = text
        elif raw_content is None:
            content_value = text if text else None
        else:
            content_value = raw_content
        out_message = {
            "role": message.get("role") or "assistant",
            "content": content_value,
        }
        if message.get("refusal") is not None:
            out_message["refusal"] = message.get("refusal")
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            basis = (response_id + str(index)).encode("utf-8")
            normalised = [normalise_tool_call(c, basis) for c in calls]
            out_message["tool_calls"] = normalised
            for item in normalised:
                response_chars += len(item["function"]["arguments"])
        finish = choice.get("finish_reason") or "stop"
        choices_out.append(
            {
                "index": choice.get("index", index),
                "message": out_message,
                "finish_reason": finish,
                "logprobs": choice.get("logprobs"),
            }
        )
    if not choices_out:
        choices_out = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ]
    prompt_tokens, completion_tokens = usage_counts(
        parsed, request_chars, response_chars
    )
    body = {
        "id": response_id,
        "object": "chat.completion",
        "created": TRIAL_EPOCH,
        "model": model_id,
        "choices": choices_out,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if parsed.get("system_fingerprint") is not None:
        body["system_fingerprint"] = parsed.get("system_fingerprint")
    return body


def openai_stream_body(parsed, model_id, response_id, request_chars):
    rendered = openai_response_body(parsed, model_id, response_id, request_chars)
    chunks = []
    for choice in rendered["choices"]:
        index = choice["index"]
        message = choice["message"]
        base = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": TRIAL_EPOCH,
            "model": model_id,
        }
        role_chunk = dict(base)
        role_chunk["choices"] = [
            {
                "index": index,
                "delta": {"role": "assistant"},
                "finish_reason": None,
                "logprobs": None,
            }
        ]
        chunks.append(role_chunk)
        content = message.get("content")
        if isinstance(content, str):
            for piece in split_text(content):
                chunk = dict(base)
                chunk["choices"] = [
                    {
                        "index": index,
                        "delta": {"content": piece},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ]
                chunks.append(chunk)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call_index, call in enumerate(calls):
                first = dict(base)
                first["choices"] = [
                    {
                        "index": index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": call_index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ]
                chunks.append(first)
                for piece in split_text(call["function"]["arguments"]):
                    chunk = dict(base)
                    chunk["choices"] = [
                        {
                            "index": index,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": call_index,
                                        "function": {"arguments": piece},
                                    }
                                ]
                            },
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ]
                    chunks.append(chunk)
        finish_chunk = dict(base)
        finish_chunk["choices"] = [
            {
                "index": index,
                "delta": {},
                "finish_reason": choice["finish_reason"],
                "logprobs": None,
            }
        ]
        chunks.append(finish_chunk)
    usage_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": TRIAL_EPOCH,
        "model": model_id,
        "choices": [],
        "usage": rendered["usage"],
    }
    chunks.append(usage_chunk)
    text = "".join(
        "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
        for chunk in chunks
    )
    text += "data: [DONE]\n\n"
    return text.encode("utf-8")


def tool_input_from_arguments(arguments_text):
    try:
        value = json.loads(arguments_text)
    except (json.JSONDecodeError, ValueError):
        return {"arguments": arguments_text}
    if isinstance(value, dict):
        return value
    return {"arguments": value}


def anthropic_stop_reason(finish_reason):
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }
    return mapping.get(finish_reason, "end_turn")


def anthropic_blocks_and_usage(parsed, response_id, request_chars):
    choices = parsed.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = message_text(message)
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    response_chars = len(text)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        basis = response_id.encode("utf-8")
        for call in calls:
            normalised = normalise_tool_call(call, basis)
            arguments_text = normalised["function"]["arguments"]
            response_chars += len(arguments_text)
            blocks.append(
                {
                    "type": "tool_use",
                    "id": normalised["id"],
                    "name": normalised["function"]["name"],
                    "input": tool_input_from_arguments(arguments_text),
                }
            )
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    prompt_tokens, completion_tokens = usage_counts(
        parsed, request_chars, response_chars
    )
    finish = choice.get("finish_reason") or "stop"
    return blocks, anthropic_stop_reason(finish), prompt_tokens, completion_tokens


def anthropic_response_body(parsed, model_id, response_id, request_chars):
    blocks, stop_reason, prompt_tokens, completion_tokens = (
        anthropic_blocks_and_usage(parsed, response_id, request_chars)
    )
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model_id,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def anthropic_stream_body(parsed, model_id, response_id, request_chars):
    blocks, stop_reason, prompt_tokens, completion_tokens = (
        anthropic_blocks_and_usage(parsed, response_id, request_chars)
    )
    events = []
    events.append(
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": response_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_id,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": 1 if completion_tokens >= 1 else 0,
                    },
                },
            },
        )
    )
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            for piece in split_text(block["text"]):
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": piece},
                        },
                    )
                )
            events.append(
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index},
                )
            )
        else:
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    },
                )
            )
            partial = json.dumps(
                block["input"], separators=(",", ":"), sort_keys=True
            )
            for piece in split_text(partial):
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": piece,
                            },
                        },
                    )
                )
            events.append(
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index},
                )
            )
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                },
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    text = "".join(
        "event: %s\ndata: %s\n\n"
        % (name, json.dumps(payload, separators=(",", ":")))
        for name, payload in events
    )
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------

def _in_flight_add(delta):
    global _IN_FLIGHT
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT += delta
        return _IN_FLIGHT


def in_flight_count():
    with _IN_FLIGHT_LOCK:
        return _IN_FLIGHT


class AggregatorHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *args):
        sys.stderr.write(
            "%s %s\n" % (self.command, self.path.split("?", 1)[0])
        )

    def _send_bytes(self, status, payload, content_type, failures=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if failures:
                self.send_header(
                    "x-makefun-upstream-failures",
                    failures_header_value(failures),
                )
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_json(self, status, body, failures=None):
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, payload, "application/json", failures)

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size <= 0:
            return b""
        if size > MAX_BODY_BYTES:
            return None
        try:
            return self.rfile.read(size)
        except OSError:
            return b""

    def _path(self):
        path = urllib.parse.urlsplit(self.path).path
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        return path

    def do_GET(self):
        if self._path() == "/v1/models":
            self._handle_models()
        else:
            self._send_json(
                404,
                {
                    "error": {
                        "message": "Not found",
                        "type": "invalid_request_error",
                        "code": "not_found",
                    }
                },
            )

    def do_POST(self):
        path = self._path()
        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path == "/v1/messages":
            self._handle_messages()
        else:
            self._read_body()
            self._send_json(
                404,
                {
                    "error": {
                        "message": "Not found",
                        "type": "invalid_request_error",
                        "code": "not_found",
                    }
                },
            )

    def do_PUT(self):
        self._read_body()
        self._send_json(
            404,
            {
                "error": {
                    "message": "Not found",
                    "type": "invalid_request_error",
                    "code": "not_found",
                }
            },
        )

    do_DELETE = do_PUT
    do_PATCH = do_PUT

    def _handle_models(self):
        authorisation = self.headers.get("Authorization") or ""
        api_key = self.headers.get("x-api-key")
        if authorisation.lower().startswith("bearer"):
            self._send_json(200, self._openai_models())
        elif api_key:
            self._send_json(200, self._anthropic_models())
        else:
            self._send_json(200, self._openai_models())

    def _openai_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": item["id"],
                    "object": "model",
                    "created": TRIAL_EPOCH,
                    "owned_by": "openrouter",
                }
                for item in BRIEF_MODELS
            ],
        }

    def _anthropic_models(self):
        data = [
            {
                "id": item["id"],
                "type": "model",
                "display_name": item["display_name"],
                "created_at": TRIAL_ISO,
            }
            for item in BRIEF_MODELS
        ]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def _parse_json_body(self):
        raw = self._read_body()
        if raw is None:
            return None, b"", "body_too_large"
        if not raw:
            return None, raw, "empty_body"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return None, raw, "invalid_json"
        if not isinstance(parsed, dict):
            return None, raw, "invalid_json"
        return parsed, raw, None

    def _preferred_model(self, body):
        named = body.get("model")
        if isinstance(named, str) and named != "auto":
            for item in BRIEF_MODELS:
                if item["id"] == named:
                    return named
        return None

    def _handle_chat(self):
        _in_flight_add(1)
        try:
            self._serve_chat()
        finally:
            _in_flight_add(-1)

    def _serve_chat(self):
        body, raw, problem = self._parse_json_body()
        if problem is not None:
            self._send_json(
                400,
                {
                    "error": {
                        "message": "Request body is not valid JSON",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(
                400,
                {
                    "error": {
                        "message": "Request must carry a non-empty messages list",
                        "type": "invalid_request_error",
                        "code": "invalid_request_error",
                    }
                },
            )
            return
        ranked = choose_models(requirements_from_openai(body))
        upstream = upstream_openai_body(body)
        model_id, parsed, gateway_raw, failures, last = dial_with_failover(
            upstream, ranked, self._preferred_model(body)
        )
        if model_id is None:
            status, code, message = last
            self._send_json(
                status,
                {
                    "error": {
                        "message": message,
                        "type": "api_error",
                        "code": code,
                    }
                },
                failures,
            )
            return
        parsed = filter_tool_calls(parsed, declared_tool_names(body))
        response_id = deterministic_id("chatcmpl-", raw, gateway_raw)
        request_chars = len(raw.decode("utf-8", errors="replace"))
        if body.get("stream") is True:
            payload = openai_stream_body(
                parsed, model_id, response_id, request_chars
            )
            self._send_bytes(
                200, payload, "text/event-stream; charset=utf-8", failures
            )
        else:
            rendered = openai_response_body(
                parsed, model_id, response_id, request_chars
            )
            self._send_json(200, rendered, failures)

    def _handle_messages(self):
        _in_flight_add(1)
        try:
            self._serve_messages()
        finally:
            _in_flight_add(-1)

    def _serve_messages(self):
        body, raw, problem = self._parse_json_body()
        if problem is not None:
            self._send_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Request body is not valid JSON",
                    },
                },
            )
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Request must carry a non-empty messages list",
                    },
                },
            )
            return
        ranked = choose_models(requirements_from_anthropic(body))
        upstream = upstream_anthropic_body(body)
        model_id, parsed, gateway_raw, failures, last = dial_with_failover(
            upstream, ranked, self._preferred_model(body)
        )
        if model_id is None:
            status, code, message = last
            if status == 429:
                error_type = "rate_limit_error"
            elif status == 401:
                error_type = "authentication_error"
            elif status == 400:
                error_type = "invalid_request_error"
            else:
                error_type = "api_error"
            self._send_json(
                status,
                {
                    "type": "error",
                    "error": {"type": error_type, "message": message},
                },
                failures,
            )
            return
        parsed = filter_tool_calls(parsed, declared_tool_names(body))
        response_id = deterministic_id("msg_", raw, gateway_raw)
        request_chars = len(raw.decode("utf-8", errors="replace"))
        if body.get("stream") is True:
            payload = anthropic_stream_body(
                parsed, model_id, response_id, request_chars
            )
            self._send_bytes(
                200, payload, "text/event-stream; charset=utf-8", failures
            )
        else:
            rendered = anthropic_response_body(
                parsed, model_id, response_id, request_chars
            )
            self._send_json(200, rendered, failures)


class AggregatorServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Env registration and work polling.
# ---------------------------------------------------------------------------

def register_with_env(env_url, port):
    body = {"protocol_version": 1, "port": port, "catalog": CATALOG}
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    url = env_url.rstrip("/") + "/register"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for _attempt in range(REGISTER_RETRIES):
        try:
            status, _raw, parsed = http_request(
                "POST", url, payload, headers, REGISTER_TIMEOUT_SEC
            )
        except Exception:
            status, parsed = None, None
        if status == 200 and isinstance(parsed, dict) and parsed.get("ok") is True:
            return True
        time.sleep(RETRY_PAUSE_SEC)
    return False


def work_is_done(env_url):
    url = env_url.rstrip("/") + "/work"
    _status, _raw, parsed = http_request(
        "GET", url, None, {"Accept": "application/json"}, WORK_TIMEOUT_SEC
    )
    return isinstance(parsed, dict) and parsed.get("done") is True


def serve_forever_standalone(server):
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()


def main(argv):
    global BRIEF_MODELS, CATALOG, CATALOG_TEXT, TRIAL_EPOCH, TRIAL_ISO
    global TRIAL_SEED, OPENAI_BASE_URL, OPENAI_API_KEY

    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    BRIEF_MODELS = load_brief_models(prompt_file)
    CATALOG = build_catalog(BRIEF_MODELS)
    CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"))
    TRIAL_EPOCH, TRIAL_ISO = parse_trial_now(
        os.environ.get("MAKEFUN_TRIAL_NOW", "")
    )
    TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    if "--list" in argv:
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return 0

    if not OPENAI_BASE_URL:
        sys.stderr.write("OPENAI_BASE_URL is not set\n")
        return 1

    server = AggregatorServer(("0.0.0.0", 0), AggregatorHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sys.stderr.write("listening on port %d\n" % port)

    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        serve_forever_standalone(server)
        return 0

    if not register_with_env(env_url, port):
        sys.stderr.write("registration with the env failed\n")
        server.shutdown()
        server.server_close()
        return 1

    try:
        while True:
            try:
                done = work_is_done(env_url)
            except Exception:
                done = False
            if done and in_flight_count() == 0:
                break
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
