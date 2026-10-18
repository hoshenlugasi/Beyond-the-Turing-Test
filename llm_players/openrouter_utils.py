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
# NOTE: Anthropic Claude is intentionally excluded — extended thinking is OFF by
# default on the OpenRouter chat API, so sending `reasoning` would only slow it down.
_LOW_REASONING_PATTERNS = (
    re.compile(r"^openai/gpt-5(\b|[.\-])",        re.IGNORECASE),
    re.compile(r"^openai/o[1-9](\b|[.\-])",       re.IGNORECASE),
    re.compile(r"^google/gemini-3.*pro",          re.IGNORECASE),
    re.compile(r"^google/gemini-2\.5-pro",        re.IGNORECASE),
    # Grok 4.5 REJECTS reasoning:{enabled:false} with HTTP 400 (measured
    # 2026-07-09), so the best available is low effort (~2-3s for a 5-word
    # reply, 200+ reasoning tokens). Prefer grok-4.3 for live games — see
    # _DISABLE_REASONING_PATTERNS below.
    re.compile(r"^x-ai/grok-4\.5(\b|[.\-]|$)",    re.IGNORECASE),
)

# Models where reasoning can be fully DISABLED — the right mode for 2-6 word
# chat messages. Measured on grok-4.3 (2026-07-09): default = 3-4s with ~300
# wasted reasoning tokens per tiny reply; {"enabled": false} = 0.7s, clean.
_DISABLE_REASONING_PATTERNS = (
    re.compile(r"^x-ai/grok-(?!4\.5)", re.IGNORECASE),   # grok-4.3, 4.20, build…
)

# Chat-tuned variants that already skip reasoning (e.g. "gpt-5.2-chat"). Sending a
# `reasoning` payload to these is unnecessary and may be rejected — skip them.
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
