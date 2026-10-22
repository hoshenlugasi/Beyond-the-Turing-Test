"""Structured event payloads for the Social Turing Test agent.

Builders report only values actually evaluated by callers. Turn correlation is
stamped by SocialTuringTestAgent.write_game_event; no delivery is inferred from
reviewer selection. All builders are pure and preserve nullable observations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Canonical component-log label → produces logs/{name}_reviewer.txt etc.
REVIEWER_COMPONENT = "reviewer"


def reviewer_selected_event(
    *,
    selected: str,
    num_candidates: int,
    source: str,
    review_status: str = "ok",
    review_risk: Optional[str] = None,
    reason_codes: Optional[List[str]] = None,
    selected_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Payload for ``reviewer_selected``.

    ``source``        — selection provenance: "model" (model picked the index),
                        "fallback" (deterministic pick — model output could not
                        be parsed / errored), or "rewrite" (single-draft path).
    ``review_status`` — "ok" | "fallback" | "error". Honest about provenance.
    ``review_risk``   — only set when an actual risk judgement was made; ``None``
                        when no semantic risk was evaluated. The ranker path does
                        NOT evaluate a risk score, so it passes ``None`` and the
                        field is OMITTED — never auto-defaulted to "low".
    """
    payload: Dict[str, Any] = {
        "selected_text": selected,
        "selected_word_count": len((selected or "").split()),
        "num_candidates": int(num_candidates),
        "source": source,
        "review_status": review_status,
        "review_reason_codes": list(reason_codes or []),
    }
    if review_risk is not None:
        payload["review_risk"] = review_risk
    if selected_index is not None:
        payload["selected_index"] = int(selected_index)
    return payload


def reviewer_drop_event(
    *,
    num_candidates: int,
    source: str,
    review_status: str = "drop",
    review_risk: Optional[str] = None,
    reason_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Payload for ``reviewer_drop`` — the reviewer selected nothing this turn.

    ``review_status`` distinguishes an explicit model DROP ("drop") from a
    fallback that found no usable candidate ("fallback"). ``review_risk`` is
    NEVER fabricated: included only when actually evaluated — a model
    DROP carries the model risk (low|med|high), a fallback carries None (omitted)."""
    payload = {
        "num_candidates": int(num_candidates),
        "source": source,
        "review_status": review_status,
        "review_reason_codes": list(reason_codes or []),
    }
    if review_risk is not None:
        payload["review_risk"] = review_risk
    return payload


def reviewer_analysis_event(*, verdicts: Optional[List[Dict[str, Any]]] = None,
                            selected_index: Optional[int] = None,
                            num_candidates: int = 0) -> Dict[str, Any]:
    """Payload for ``reviewer_analysis`` — the per-candidate assessment the
    reviewer produced while ranking. Each verdict is the reviewer's OWN judgement
    of one candidate: ``{index, text, type, pros, cons, scores, score, red_flags,
    verdict, selected}``.

    This is honest by construction: it carries only what the model actually
    returned. When the model gave no structured reasoning (fallback/error paths),
    ``verdicts`` is empty — we never fabricate pros/cons. Used by the dashboard to
    show, on click, why each candidate was kept or rejected.
    """
    clean: List[Dict[str, Any]] = []
    for v in (verdicts or []):
        if not isinstance(v, dict):
            continue
        raw_scores = v.get("scores") or {}
        scores: Dict[str, int] = {}
        if isinstance(raw_scores, dict):
            for k, val in raw_scores.items():
                try:
                    scores[str(k)] = int(val)
                except (TypeError, ValueError):
                    scores[str(k)] = 0
        try:
            score = int(v.get("score"))
        except (TypeError, ValueError):
            score = sum(scores.values()) if scores else 0
        raw_index = v.get("index")
        if raw_index is None:
            raw_index = v.get("i")
        try:
            clean_index = int(raw_index) if raw_index is not None else -1
        except (TypeError, ValueError):
            clean_index = -1
        clean_verdict = {
            # Zero is a valid first-candidate index. Never use ``or -1`` here:
            # truthiness would misreport candidate 0 as a missing selection.
            "index": clean_index,
            "text": str(v.get("text", "")),
            "type": str(v.get("type", "")),
            "pros": str(v.get("pros", "")),
            "cons": str(v.get("cons", "")),
            "scores": scores,
            "score": score,
            "red_flags": [str(f) for f in (v.get("red_flags") or []) if str(f).strip()],
            "verdict": str(v.get("verdict", "")),
            "selected": bool(v.get("selected", False)),
        }
        if v.get("pool_index") is not None:
            try:
                clean_verdict["pool_index"] = int(v["pool_index"])
            except (TypeError, ValueError):
                pass
        clean.append(clean_verdict)
    payload: Dict[str, Any] = {
        "verdicts": clean,
        "num_candidates": int(num_candidates),
    }
    if selected_index is not None:
        payload["selected_index"] = int(selected_index)
    return payload


# ── Fact usage, turn/UI semantics and poll reduction ─────────────────────────

def fact_usage_event(*, available_fact_ids: Optional[List[str]] = None,
                     declared_fact_ids: Optional[List[str]] = None,
                     verified_fact_ids_used: Optional[List[str]] = None) -> Dict[str, Any]:
    """Honest fact-usage payload. A fact is "used" ONLY when the sent answer
    actually relied on it (``verified_fact_ids_used``) — never just because it was
    available. ``fact_usage_status``: verified (a verified fact was used) |
    unverified (facts were declared but none verified-used) | none."""
    available = list(available_fact_ids or [])
    declared = list(declared_fact_ids or [])
    used = list(verified_fact_ids_used or [])
    if used:
        status = "verified"
    elif declared:
        status = "unverified"
    else:
        status = "none"
    return {
        "available_fact_ids": available,
        "declared_fact_ids": declared,
        "verified_fact_ids_used": used,
        "fact_usage_status": status,
    }


def turn_pipeline_complete_event(*, turn_id: str, snapshot_revision: Any,
                                 target_message_id: Optional[str],
                                 final_response_act: Optional[str],
                                 review_status: str, review_risk: Optional[str],
                                 review_reason_codes: Optional[List[str]] = None,
                                 research_dependencies: Optional[List[Dict[str, Any]]] = None,
                                 artificial_delay_requested_s: float = 0.0,
                                 chat_append_at: Any = None,
                                 fact_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-turn UI/analyzer row. Pipeline completion is the PUBLIC CHAT APPEND
    (``chat_append_at``), not reviewer completion. ``review_risk`` is included
    only when actually evaluated (None → omitted)."""
    payload: Dict[str, Any] = {
        "turn_id": turn_id,
        "snapshot_revision": snapshot_revision,
        "target_message_id": target_message_id,
        "final_response_act": final_response_act,
        "review_status": review_status,
        "review_reason_codes": list(review_reason_codes or []),
        "research_dependencies": research_dependencies or [],
        "artificial_delay_requested_s": round(float(artificial_delay_requested_s), 4),
        "chat_append_at": chat_append_at,
        "pipeline_complete_at": chat_append_at,   # completion == chat append
    }
    if review_risk is not None:
        payload["review_risk"] = review_risk
    if fact_usage is not None:
        payload.update(fact_usage)
    return payload


def turn_pipeline_ready_event(*, turn_id: str, snapshot_revision: Any,
                              target_message_id: Optional[str],
                              final_response_act: Optional[str],
                              review_status: str, review_risk: Optional[str],
                              review_reason_codes: Optional[List[str]] = None,
                              research_dependencies: Optional[List[Dict[str, Any]]] = None,
                              artificial_delay_requested_s: float = 0.0,
                              ready_to_append_at: Any = None,
                              fact_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A turn whose message is APPROVED and READY to be
    published, but whose public append is performed by the outer game loop. This
    carries ``ready_to_append_at`` and explicitly NO ``pipeline_complete_at`` /
    ``chat_append_at`` — the completion is emitted later by the real append layer
    (``record_public_append``), so the analyzer never counts a ready message as a
    completed/visible send."""
    payload: Dict[str, Any] = {
        "turn_id": turn_id,
        "snapshot_revision": snapshot_revision,
        "target_message_id": target_message_id,
        "final_response_act": final_response_act,
        "review_status": review_status,
        "review_reason_codes": list(review_reason_codes or []),
        "research_dependencies": research_dependencies or [],
        "artificial_delay_requested_s": round(float(artificial_delay_requested_s), 4),
        "ready_to_append_at": ready_to_append_at,
        "pipeline_complete_at": None,        # NOT complete until the real append
        "chat_append_at": None,
    }
    if review_risk is not None:
        payload["review_risk"] = review_risk
    if fact_usage is not None:
        payload.update(fact_usage)
    return payload


TERMINAL_EVENTS = frozenset({
    "turn_sent", "turn_ready", "turn_scheduler_wait", "turn_generator_failure",
    "turn_reviewer_drop", "turn_pre_send_cancel", "turn_active_exception",
})


def turn_terminal_event(*, turn_id: str, stage: str, event_type: str,
                        target_message_id: Optional[str] = None,
                        original_snapshot_revision: Any = None,
                        fresh_snapshot_revision: Any = None,
                        pipeline_complete_at: Any = None,
                        **fields: Any) -> Dict[str, Any]:
    """One DISTINCT terminal event per pipeline outcome — a real send
    (``turn_sent``), a scheduler wait, a generator failure, a reviewer drop, a
    pre-send cancellation, or an active-path exception. ONLY a real append carries
    ``pipeline_complete_at`` — every non-send outcome leaves it None so the
    analyzer never mistakes a drop/cancel for a completed turn. Carries the
    original AND post-delay fresh snapshot revisions, gate reason, delays and
    validated fact ids when the caller has them."""
    assert event_type in TERMINAL_EVENTS, f"unknown terminal event {event_type!r}"
    payload: Dict[str, Any] = {
        "event": event_type,
        "turn_id": turn_id,
        "stage": stage,
        "target_message_id": target_message_id,
        "original_snapshot_revision": original_snapshot_revision,
        "fresh_snapshot_revision": fresh_snapshot_revision,
        # Completion == real public chat append; non-send terminals stay None.
        "pipeline_complete_at": pipeline_complete_at if event_type == "turn_sent" else None,
        "is_send": event_type == "turn_sent",
    }
    for k, v in fields.items():
        if v is not None:
            payload[k] = v
    return payload


class PollAggregator:
    """Log-noise reduction: emit a structured poll event only when the BLOCK
    REASON changes; otherwise aggregate counts to flush periodically.

    Tracks BOTH the lifetime total per reason AND the CURRENT
    consecutive run length. ``previous_run_count`` is the length of the
    immediately-preceding consecutive streak — not the lifetime total."""

    def __init__(self) -> None:
        self._last_reason: Optional[str] = None
        self._current_run: int = 0
        self.counts: Dict[str, int] = {}

    def observe(self, reason: str) -> Optional[Dict[str, Any]]:
        """Returns a payload to emit only on a reason CHANGE (carrying the run
        length of the previous reason's consecutive streak); otherwise None."""
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if reason == self._last_reason:
            self._current_run += 1
            return None
        prev = self._last_reason
        prev_run = self._current_run
        self._last_reason = reason
        self._current_run = 1
        return {"event": "poll_reason_change", "reason": reason,
                "previous_reason": prev, "previous_run_count": prev_run}

    def summary(self) -> Dict[str, int]:
        return dict(self.counts)
