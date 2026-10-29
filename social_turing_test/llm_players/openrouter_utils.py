"""Shared helpers for OpenRouter API calls.

Some OpenRouter models (notably OpenAI's GPT-5.x family and the o-series
reasoning models) reject the `temperature` parameter and return HTTP 400.
Use these helpers to:

* skip `temperature` automatically for known-incompatible models
* detect 400 responses caused by unsupported parameters and retry without
  the offending field
"""

from __future__ import annotations

import re
import json
import math
import time

import requests


class OpenRouterRequestError(RuntimeError):
    """A failed request, not a model WAIT decision. Contains metadata only."""

    def __init__(self, model, component, category, attempts):
        self.category = category
        self.attempts = attempts
        last = attempts[-1] if attempts else {}
        super().__init__(
            f"OpenRouter {component} failed: {category}; model={model}; "
            f"attempts={len(attempts)}; status={last.get('status')}; "
            f"finish_reason={last.get('finish_reason')}; "
            f"max_tokens={last.get('max_tokens')}; output_tokens={last.get('output_tokens')}; "
            f"reasoning_tokens={last.get('reasoning_tokens')}; "
            f"reasoning_present={last.get('reasoning_present')}; "
            f"request_id={last.get('request_id')}"
        )

# Model name prefixes (case-insensitive) that do NOT accept `temperature`/`top_p`
# on OpenRouter. Extend this list as new models are released.
_NO_TEMPERATURE_PATTERNS = (
    re.compile(r"^openai/gpt-5(\b|[.\-])", re.IGNORECASE),  # gpt-5, gpt-5.1, gpt-5-mini, gpt-5.4-pro, etc.
    re.compile(r"^openai/o[1-9](\b|[.\-])",  re.IGNORECASE),  # o1, o3, o4-mini, o4-mini-high, etc.
)

# Models that enforce a minimum `max_tokens` (e.g. Azure-hosted GPT-5.x requires >= 16).
# Use the largest known floor so all variants succeed.
_MIN_MAX_TOKENS_PATTERNS = (
    (re.compile(r"^openai/gpt-5(\b|[.\-])", re.IGNORECASE), 16),
    (re.compile(r"^openai/o[1-9](\b|[.\-])",  re.IGNORECASE), 16),
)

# Reasoning ("thinking") models that, on OpenRouter, default to a high reasoning
# effort and so are slow and burn the tiny chat-message token budget on internal
# reasoning. Send an explicit LOW effort to keep latency down for short messages.
#   * OpenAI GPT-5.x / o-series   → reasoning models
#   * Google Gemini Pro (2.5 / 3) → thinking models routed via OpenRouter
# NOTE: Anthropic Claude is intentionally excluded - extended thinking is OFF by
# default on the OpenRouter chat API, so sending `reasoning` would only slow it down.
_LOW_REASONING_PATTERNS = (
    re.compile(r"^openai/gpt-5(\b|[.\-])",        re.IGNORECASE),
    re.compile(r"^openai/o[1-9](\b|[.\-])",       re.IGNORECASE),
    re.compile(r"^google/gemini-3.*pro",          re.IGNORECASE),
    re.compile(r"^google/gemini-2\.5-pro",        re.IGNORECASE),
    # Grok 4.5 rejects reasoning:{enabled:false} with HTTP 400, so use low
    # effort rather than the full-disable setting supported by other variants.
    re.compile(r"^x-ai/grok-4\.5(\b|[.\-]|$)",    re.IGNORECASE),
)

# Models where reasoning can be fully DISABLED - the right mode for 2-6 word
# chat messages, avoiding internal reasoning overhead on tiny replies.
_DISABLE_REASONING_PATTERNS = (
    re.compile(r"^x-ai/grok-(?!4\.5)", re.IGNORECASE),   # grok-4.3, 4.20, build…
)

# Chat-tuned variants that already skip reasoning (e.g. "gpt-5.2-chat"). Sending a
# `reasoning` payload to these is unnecessary and may be rejected - skip them.
_NO_REASONING_PATTERNS = (
    re.compile(r"-chat(\b|[.\-]|$)", re.IGNORECASE),
)


def min_max_tokens_for(model_name: str) -> int:
    """Return the minimum `max_tokens` value the provider requires for this model (0 = no minimum)."""
    if not model_name:
        return 0
    for pattern, floor in _MIN_MAX_TOKENS_PATTERNS:
        if pattern.match(model_name):
            return floor
    return 0


def is_max_tokens_too_low_error(status_code: int, body_text: str) -> bool:
    """Detect 400 errors complaining that max_tokens / max_output_tokens is below the minimum."""
    if status_code != 400 or not body_text:
        return False
    lower = body_text.lower()
    return ("max_tokens" in lower or "max_output_tokens" in lower) and (
        "below minimum" in lower or "minimum value" in lower or "too small" in lower
        or "integer_below_min_value" in lower or ">= " in lower
    )


def model_supports_temperature(model_name: str) -> bool:
    """Return False for OpenRouter models known to reject `temperature`."""
    if not model_name:
        return True
    return not any(p.match(model_name) for p in _NO_TEMPERATURE_PATTERNS)


def reasoning_payload_for(model_name: str):
    """Return an OpenRouter `reasoning` payload to minimize latency for reasoning
    models, or None if the model should be left untouched.

    Reasoning models default to a high thinking effort on OpenRouter, which is slow
    and consumes the small chat token budget on internal reasoning. We force LOW
    effort for short Turing-test messages. Chat-tuned variants and Claude (thinking
    off by default) are deliberately left alone.
    """
    if not model_name:
        return None
    if any(p.search(model_name) for p in _NO_REASONING_PATTERNS):
        return None
    if any(p.match(model_name) for p in _DISABLE_REASONING_PATTERNS):
        return {"enabled": False}
    if any(p.match(model_name) for p in _LOW_REASONING_PATTERNS):
        return {"effort": "low"}
    return None


def model_uses_reasoning_tokens(model_name: str) -> bool:
    """Return whether a model may consume its completion budget on reasoning tokens."""
    if not model_name or any(p.search(model_name) for p in _NO_REASONING_PATTERNS):
        return False
    return any(p.match(model_name) for p in _LOW_REASONING_PATTERNS)


def completion_budget(model_name: str, requested: int, component: str) -> int:
    """OpenRouter-only total completion budget, including hidden reasoning."""
    budget = max(requested, min_max_tokens_for(model_name))
    if model_uses_reasoning_tokens(model_name):
        budget = max(budget, 1024 if component == "scheduler" else 2048)
    return budget


def is_unsupported_param_error(status_code: int, body_text: str) -> bool:
    """Heuristic: detect OpenRouter 400 errors caused by unsupported sampling params.

    OpenRouter / underlying providers return 400 with messages like
    "Unsupported value: 'temperature'" or "temperature is not supported".
    """
    if status_code != 400 or not body_text:
        return False
    lower = body_text.lower()
    return ("temperature" in lower or "top_p" in lower or "top-p" in lower) and (
        "unsupported" in lower or "not supported" in lower or "does not support" in lower
        or "invalid" in lower
    )


def request_completion(model_name, api_key, messages, generation_parameters, *,
                       component, log, record_usage, estimate_cost=None, timeout_s=30.0):
    """One bounded OpenRouter transport for generation, scheduling and review."""
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("OpenRouter timeout must be a positive finite duration")
    payload = {"model": model_name, "messages": messages}
    if "max_tokens" in generation_parameters:
        payload["max_tokens"] = completion_budget(
            model_name, int(generation_parameters["max_tokens"]), component
        )
    if "temperature" in generation_parameters and model_supports_temperature(model_name):
        payload["temperature"] = generation_parameters["temperature"]
    reasoning = reasoning_payload_for(model_name)
    if reasoning:
        payload["reasoning"] = reasoning
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    max_attempts = 4
    deadline = time.monotonic() + min(timeout_s, 30.0)
    attempts = []
    token_bumped = False
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    if estimate_cost is not None:
        totals["estimated_cost_usd"] = 0.0

    def token_count(value):
        return value if type(value) is int and value >= 0 else None

    def safe_id(value):
        return value if (
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", value)
            and api_key not in value
        ) else None

    def record(meta, usage):
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        it, ot = token_count(usage.get("prompt_tokens")), token_count(usage.get("completion_tokens"))
        rt = token_count(details.get("reasoning_tokens"))
        meta.update(input_tokens=it, output_tokens=ot, reasoning_tokens=rt)
        totals["input_tokens"] += it or 0
        totals["output_tokens"] += ot or 0
        totals["reasoning_tokens"] += rt or 0
        if estimate_cost is not None:
            totals["estimated_cost_usd"] += estimate_cost(it or 0, ot or 0)
        attempts.append(meta)
        record_usage(dict(totals, attempts=list(attempts)))
        log("openrouter_attempt", json.dumps(meta, allow_nan=False))

    def fail(category):
        error = OpenRouterRequestError(model_name, component, category, list(attempts))
        log("AGENT_ERROR", str(error))
        raise error

    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail("deadline_exceeded")
        started = time.monotonic()
        meta = {"attempt": attempt, "component": component, "status": None,
                "request_id": None, "finish_reason": None,
                "max_tokens": payload.get("max_tokens"), "reasoning_present": False}
        usage = {}
        retry = False
        retry_after = 0.0
        output = None
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload,
                timeout=(min(5.0, remaining / 2), min(20.0, remaining / 2)),
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            category, retry = "transport_error", True
        except requests.exceptions.RequestException:
            category = "request_error"
        else:
            try:
                status = response.status_code
                meta["status"] = status
                meta["request_id"] = safe_id(response.headers.get("x-request-id"))
                try:
                    body = response.json()
                except ValueError:
                    body = None
                body = body if isinstance(body, dict) else {}
                usage = body.get("usage", {})
                meta["request_id"] = meta["request_id"] or safe_id(body.get("id"))
                embedded = body.get("error")
                if 200 <= status < 300 and isinstance(embedded, dict):
                    code = embedded.get("code")
                    if isinstance(code, str) and re.fullmatch(r"[45]\d\d", code):
                        code = int(code)
                    status = code if type(code) is int and 400 <= code <= 599 else 502
                    meta["status"] = status
                if not 200 <= status < 300:
                    category = {401: "authentication", 402: "billing", 403: "permission",
                                404: "model_unavailable", 429: "rate_limit"}.get(status, "http_error")
                    retry = status in (408, 429, 500, 502, 503, 504)
                    if status == 400:
                        # Inspect error text for compatibility only; never log it.
                        error_text = response.text.lower()
                        if "temperature" in payload and is_unsupported_param_error(status, error_text):
                            payload.pop("temperature")
                            category, retry = "unsupported_temperature", True
                        elif "reasoning" in payload and "reasoning" in error_text and any(
                            word in error_text for word in ("unsupported", "not supported", "invalid")
                        ):
                            payload.pop("reasoning")
                            category, retry = "unsupported_reasoning", True
                        elif not token_bumped and is_max_tokens_too_low_error(status, error_text):
                            budget = payload.get("max_tokens", 0)
                            if budget < 8192:
                                payload["max_tokens"] = min(8192, max(1024, budget * 2))
                                token_bumped = True
                                category, retry = "minimum_budget", True
                    value = response.headers.get("Retry-After")
                    if value:
                        try:
                            retry_after = float(value)
                        except ValueError:
                            retry_after = 0.0
                        if not math.isfinite(retry_after) or retry_after < 0:
                            retry_after = 0.0
                else:
                    choices = body.get("choices")
                    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                    message = choice.get("message")
                    message = message if isinstance(message, dict) else {}
                    finish, content = choice.get("finish_reason"), message.get("content")
                    valid_shape = (
                        bool(choice) and bool(message) and embedded is None
                        and (finish is None or isinstance(finish, str))
                        and (content is None or isinstance(content, str))
                    )
                    if not valid_shape:
                        category = "invalid_response"
                    else:
                        meta["finish_reason"] = finish if finish in {
                            "stop", "length", "content_filter", "tool_calls", "error"
                        } else "unknown"
                        meta["reasoning_present"] = bool(message.get("reasoning") or message.get("reasoning_details"))
                        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
                        rt = token_count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
                        reasoning_exhausted = (
                            not content and rt is not None
                            and rt >= payload.get("max_tokens", float("inf"))
                        )
                        if finish == "content_filter" or message.get("refusal"):
                            category = "refused"
                        elif finish == "length" or reasoning_exhausted:
                            category = "output_budget_exhausted"
                            budget = payload.get("max_tokens", 0)
                            if not token_bumped and budget < 8192:
                                payload["max_tokens"] = min(8192, max(1024, budget * 2))
                                token_bumped, retry = True, True
                        elif finish in ("tool_calls", "error"):
                            category = "unexpected_completion"
                        elif isinstance(content, str) and content.strip():
                            category, output = "success", content
                        else:
                            category, retry = "empty_content", True
            finally:
                response.close()
        meta.update(outcome=category, latency_s=round(time.monotonic() - started, 3))
        record(meta, usage)
        if output is not None:
            return output
        if not retry or attempt == max_attempts:
            fail(category)
        delay = max(min(2 ** (attempt - 1), 4), retry_after)
        if delay >= deadline - time.monotonic():
            fail("deadline_exceeded")
        time.sleep(delay)
    fail("attempt_limit")
