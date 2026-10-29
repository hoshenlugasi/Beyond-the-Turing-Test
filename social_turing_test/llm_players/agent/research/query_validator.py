"""Deterministic post-classification query validator.

Runs AFTER the model classifier (never decides WHETHER to search — the classifier
already did). It normalizes/validates a proposed search query and may reject or
rewrite it: computable-locally time/date queries, malformed/empty, dedup against
in-flight, fresh exact-topic cache hits, and unsupported/absurd years. It must
NOT reject a topic merely for not matching a pattern.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from social_turing_test.llm_players.agent.conversation_analysis import is_trivial_test_question

DECISION_ACCEPTED = "accepted"
DECISION_REWRITTEN = "rewritten"
DECISION_REJECTED = "rejected"
DECISION_CACHE_HIT = "cache_hit"

# Time/date that the agent can compute locally (zoneinfo) - never search these.
_COMPUTABLE_RE = re.compile(
    r"\b(what('?s| is) the (time|date)|what time is it|current (time|date)|"
    r"today'?s date|time in \w+ now|day of the week)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20|21)\d{2}\b")
_WORD_MATH_RE = re.compile(
    r"^\s*(?:what(?:'s| is)\s+)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
    r"\s+(?:plus|times|multiplied by)\s+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
    r"\s*\??\s*$",
    re.IGNORECASE,
)
_META_CHAT_RE = re.compile(
    r"\b(how (?:should|do) (?:i|you) (?:answer|reply|respond)|"
    r"what should (?:i|you) say|chat game|social turing test|"
    r"prompt injection|system prompt|chain of thought|"
    r"who is the (?:ai|bot)|which player is the (?:ai|bot)|"
    # Queries about hiding the AI or detecting players concern game strategy,
    # not external facts needed for the conversation.
    r"signs? of (?:an? )?(?:ai|bot|chatbot)|"
    r"(?:appear|sound|seem|look) (?:more )?human|"
    r"(?:chatbot|ai|bot) detection|detect (?:an? )?(?:ai|bot|chatbot)|"
    r"pass (?:as|for) (?:a )?human|meaning of scripted)\b",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip()).lower()


def non_web_query_reason(query: str) -> str:
    """Return a stable reason when a proposed query should never hit the web."""
    text = (query or "").strip()
    if _COMPUTABLE_RE.search(text):
        return "computable_locally"
    if is_trivial_test_question(text) or _WORD_MATH_RE.match(text):
        return "trivial_test"
    if _META_CHAT_RE.search(text):
        return "game_meta_query"
    return ""


def validate_query(query: str, *, cache_norm_queries: Iterable[str] = (),
                   current_year: Optional[int] = None) -> Dict[str, Any]:
    """Return a QueryValidation dict
    ``{original_query, normalized_query, decision, reason_code}``."""
    current_year = current_year or datetime.now().year
    original = query or ""
    norm = normalize_query(original)

    if not norm:
        return _qv(original, norm, DECISION_REJECTED, "empty_or_malformed")
    non_web_reason = non_web_query_reason(original)
    if non_web_reason:
        return _qv(original, norm, DECISION_REJECTED, non_web_reason)

    # Unsupported / absurd year (e.g. far future) → reject; a plausible future
    # year (current_year+1, e.g. an upcoming event) is allowed.
    for m in _YEAR_RE.finditer(original):
        yr = int(m.group())
        if yr > current_year + 1:
            return _qv(original, norm, DECISION_REJECTED, "unsupported_year")

    cache = {normalize_query(c) for c in cache_norm_queries}
    if norm in cache:
        return _qv(original, norm, DECISION_CACHE_HIT, "fresh_cache_hit")

    if norm != original.strip():
        return _qv(original, norm, DECISION_REWRITTEN, "normalized")
    return _qv(original, norm, DECISION_ACCEPTED, "ok")


def _qv(original: str, norm: str, decision: str, reason: str) -> Dict[str, Any]:
    return {"original_query": original, "normalized_query": norm,
            "decision": decision, "reason_code": reason}
