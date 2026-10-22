"""Pure, offline projection of neutral runtime records into typed timelines.

The caller owns name migration, pseudonym/turn mapping and privacy review. Text
redaction is mandatory but is not privacy clearance. No raw record is copied.
Unknown records return None; classify_event supplies the manifest disposition.
Unlisted input fields (including prompts, history, provider results, titles,
errors and full snapshots) are discarded before type checking. Retained invalid
values fail closed. Missing/null is unknown; false, zero and empty are evidence.

SCHEMAS is the executable field dictionary: str/int/float/bool are nullable
scalars, [schema] is a nullable list, and dicts are fixed nullable structs. All
keys of every present struct are emitted, even when their values are null.
Float fields accept finite ints on input and are always floats on output.
EVENT_SPECS lists every supported neutral event and its allowed input fields.
Historical naming conversion belongs exclusively to the private adapter.

Ordering is supplied write order, not inferred causal order. Timing is supplied
verified relative timing, never derived here from raw clocks. Internet returned
item counts are NOT independent cited-source counts. Grounding is never inferred.
Persona/social outputs are model claims/interpretations, not verified identities.
Selection, pipeline completion and observed context are NOT delivery evidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Mapping


COMPONENTS = (
    "Scheduler", "Generator", "Reviewer", "Internet Context", "Round Summarizer",
    "Persona Monitor", "Social State Tracker", "Behavior Adapter",
    "Repetition Monitor", "System",
)
TIME_BASES = frozenset({"unknown", "since_game_start", "since_first_event",
                        "verified_relative", "monotonic_segment"})
OMISSION_REASONS = frozenset({"unknown_event", "retired_diagnostic",
                              "privacy_withheld", "malformed_input",
                              "unavailable_input"})

VERDICT = dict(index="int", pool_index="int", text="str", type="str",
               pros="str", cons="str", score="float", red_flags=["str"],
               verdict="str", selected="bool")
REMOVED = dict(text="str", reason="str")
PLAN = dict(query="str", volatility="str", confidence="str",
            reason="str", needs_search="bool")
STORED = dict(topic="str", volatility="str", confidence="str",
              returned_item_count="int", origin="str")
PLAYER = dict(pseudonym="str", observation="str")
FACTS = dict(location="str", city="str", country="str", timezone="str",
             nationality="str", age="str", occupation="str", name="str")
CADENCE = dict(level="str", reason="str", talk_share="float")

SCHEMAS = {
    "scheduler": dict(decision="str", send_reason="str", reason="str",
                      latency_ms="int", last_send_ago_s="float", nudges="str"),
    "candidates": dict(count="int", candidates=["str"], reason="str",
                       history_lines="int", current_info_chars="int"),
    "review": dict(num_candidates="int", selected_index="int", verdicts=[VERDICT],
                   selected_text="str", selected_word_count="int", source="str",
                   review_status="str", review_risk="str", review_reason_codes=["str"]),
    "filter": dict(removed=[REMOVED], removed_count="int", kept="int", demoted=["str"], tag="str",
                   pool_size="int", mode="str", dodge_streak="int", streak="int"),
    "delivery": dict(message="str", outcome="str", reason="str", stage="str",
                     new_messages="int", suppressed_message="str",
                     pipeline_elapsed_s="float", timeout_s="float"),
    "internet": dict(query="str", query_in="str", corrected_query="str",
                     summary="str", summary_out="str", topic="str", stored=STORED,
                     returned_item_count="int", origin="str", volatility="str",
                     confidence="str", grounded="bool", needs_search="bool",
                     dropped="bool", reason="str", retrying="bool",
                     year_corrected="bool", error_category="str", plans=[PLAN],
                     history_len="int", phase_summary_count="int", latency_s="float",
                     tick_latency_s="float", searches_run="int", entries_written="int",
                     replaced="bool", total_entries_after="int",
                     expired_fact_count="int", stale_search_count="int",
                     remaining_count="int", snapshot_revision="int", pruned_count="int"),
    "persona": dict(claims=["str"], count="int", own_msgs="int",
                    first_person="int", latency_s="float", facts=FACTS,
                    parse_ok="bool", status="str"),
    "social_state": dict(suspects_me=["str"], i_pointed_at=["str"], voted_out=["str"],
                         prime_suspect="str", open_threads=["str"], group_mood="str",
                         players=[PLAYER], room_norm="str", room_vibe="str",
                         pressure_type="str", interrogation_pressure="float",
                         attention_on_me="float", msgs="int", had_signal="bool",
                         parse_ok="bool", latency_s="float", status="str"),
    "behavior": dict(advice="str", players_left="int", suspicion_on_self="int",
                     talk_share="float", llm_refreshed="bool", warnings=["str"],
                     recommendation="str", participation_assessment="str",
                     agreement_ratio="float", deflection_ratio="float",
                     latency_s="float", cadence=CADENCE, status="str"),
    "repetition": dict(words=["str"], overused=["str"], note="str", own_msgs="int",
                       latency_s="float", status="str"),
    "round_summary": dict(phase="str", phase_label="str", n_messages="int",
                          summary="str", pending_msgs_count="int", latency_s="float",
                          tick_latency_s="float", status="str"),
    "system": dict(snapshot_revision="int", entry_count="int", turn_index="int",
                   scheduler_decision="str", send_reason="str", final="str",
                   review_risk="str", pipeline_latency_s="float",
                   final_message_len="int", dropped="bool", message_history_len="int",
                   status="str", reason="str", mode="str", generator_ok="bool",
                   reviewer_ok="bool", scheduler_ok="bool"),
}


@dataclass(frozen=True)
class EventSpec:
    component: str
    branch: str
    fields: tuple[str, ...]
    # Output field -> input field. Only explicitly approved aliases are read.
    aliases: tuple[tuple[str, str], ...] = ()


def _spec(component, branch, fields=None, aliases=()):
    return EventSpec(component, branch, tuple(SCHEMAS[branch] if fields is None
                                            else fields.split()), tuple(aliases))


EVENT_SPECS = {
    "scheduler_decision": _spec("Scheduler", "scheduler", "decision send_reason latency_ms last_send_ago_s nudges"),
    "scheduler_blocked": _spec("Scheduler", "scheduler", "reason"),
    "generator_candidates": _spec("Generator", "candidates", "count candidates"),
    "generator_parse_drop": _spec("Generator", "candidates", "reason"),
    "generator_skip": _spec("Generator", "candidates", "reason"),
    "generator_input": _spec("Generator", "candidates", "history_lines current_info_chars"),
    "reviewer_analysis": _spec("Reviewer", "review", "num_candidates selected_index verdicts"),
    "reviewer_selected": _spec("Reviewer", "review", "num_candidates selected_index selected_text selected_word_count source review_status review_risk review_reason_codes"),
    "reviewer_drop": _spec("Reviewer", "review", "num_candidates source review_status review_risk review_reason_codes"),
    "reviewer_demote_response_required": _spec("Reviewer", "filter", "demoted tag pool_size"),
    "reviewer_demote_same_dodge_mode": _spec("Reviewer", "filter", "demoted tag pool_size mode dodge_streak"),
    "reviewer_demote_time_answer": _spec("Reviewer", "filter", "demoted tag pool_size"),
    "state_guard_filter": _spec("Reviewer", "filter", "removed kept"),
    "reviewer_filter_known_accusation": _spec("Reviewer", "filter", "removed kept"),
    "reviewer_filter_agreement_pileup": _spec("Reviewer", "filter", "removed_count streak", (("removed_count", "removed"),)),
    "delivery_sent": _spec("System", "delivery", "message outcome"),
    "delivery_phase_drop": _spec("System", "delivery", "message outcome reason stage"),
    "delivery_dedup_drop": _spec("System", "delivery", "message outcome reason"),
    "delivery_unconfirmed": _spec("System", "delivery", "message outcome reason"),
    "stale_send_drop": _spec("System", "delivery", "reason new_messages suppressed_message"),
    "pipeline_timeout_drop": _spec("System", "delivery", "pipeline_elapsed_s timeout_s suppressed_message"),
    "dedup_blocked": _spec("System", "delivery", "suppressed_message reason"),
    "pipeline_complete": _spec("System", "system", "scheduler_decision send_reason final review_risk"),
    "generate_begin": _spec("Generator", "system", "turn_index message_history_len"),
    "generate_end": _spec("Generator", "system", "turn_index pipeline_latency_s final_message_len dropped"),
    "snapshot_observed": _spec("System", "system", "snapshot_revision entry_count turn_index"),
    "turn_context": _spec("System", "system", "snapshot_revision turn_index"),
    "agent_models_status": _spec("System", "system", "generator_ok reviewer_ok scheduler_ok status"),
    "agent_ready": _spec("System", "system", "status"),
    "orchestration_mode": _spec("System", "system", "mode"),
    "internet_context_tick": _spec("Internet Context", "internet", "needs_search query returned_item_count summary dropped reason", (("returned_item_count", "n_results"),)),
    "internet_tick_begin": _spec("Internet Context", "internet", "history_len topic phase_summary_count"),
    "internet_tick_end": _spec("Internet Context", "internet", "tick_latency_s searches_run entries_written"),
    "internet_decision_response": _spec("Internet Context", "internet", "plans latency_s", (("plans", "parsed"),)),
    "internet_search_request": _spec("Internet Context", "internet", "query volatility"),
    "internet_search_response": _spec("Internet Context", "internet", "query query_in year_corrected returned_item_count summary latency_s grounded", (("returned_item_count", "n_results"), ("summary", "summary_text"))),
    "internet_search_error": _spec("Internet Context", "internet", "query error_category"),
    "internet_search_rejected": _spec("Internet Context", "internet", "query query_in reason grounded latency_s retrying"),
    "internet_search_trace": _spec("Internet Context", "internet", "query_in corrected_query summary_out stored"),
    "internet_seed": _spec("Internet Context", "internet", "topic query summary returned_item_count", (("returned_item_count", "n_results"),)),
    "internet_fact_stored": _spec("Internet Context", "internet", "topic query summary volatility confidence returned_item_count origin replaced total_entries_after", (("returned_item_count", "source_count"),)),
    "internet_facts_pruned": _spec("Internet Context", "internet", "expired_fact_count stale_search_count remaining_count snapshot_revision pruned_count"),
    "round_summary_tick": _spec("Round Summarizer", "round_summary", "phase n_messages summary"),
    "round_summary_response": _spec("Round Summarizer", "round_summary", "phase summary latency_s", (("summary", "summary_text"),)),
    "round_summary_tick_begin": _spec("Round Summarizer", "round_summary", "phase_label pending_msgs_count"),
    "round_summary_tick_end": _spec("Round Summarizer", "round_summary", "phase_label pending_msgs_count tick_latency_s"),
    "persona_monitor_tick": _spec("Persona Monitor", "persona", "own_msgs first_person claims latency_s parse_ok status"),
    "persona_claims_updated": _spec("Persona Monitor", "persona", "claims count"),
    "fixed_persona_facts_set": _spec("Persona Monitor", "persona", "facts"),
    "social_state_tick": _spec("Social State Tracker", "social_state"),
    "social_state_updated": _spec("Social State Tracker", "social_state"),
    "behavior_tick": _spec("Behavior Adapter", "behavior", "players_left suspicion_on_self talk_share llm_refreshed warnings recommendation participation_assessment agreement_ratio deflection_ratio latency_s status"),
    "behavior_advice_updated": _spec("Behavior Adapter", "behavior", "advice"),
    "cadence_updated": _spec("Behavior Adapter", "behavior", "cadence"),
    "repetition_monitor_tick": _spec("Repetition Monitor", "repetition", "own_msgs overused status"),
    "overused_words_updated": _spec("Repetition Monitor", "repetition", "words"),
    "repetition_note_updated": _spec("Repetition Monitor", "repetition", "note"),
    "repetition_llm_pass": _spec("Repetition Monitor", "repetition", "note"),
}

# These neutral inputs are deliberately omitted, not relabelled as live workers.
EXCLUDED_EVENTS = {name: "retired_diagnostic" for name in (
    "live_eval_cleared", "live_eval_feedback_gate", "live_eval_result",
    "self_weaknesses_updated", "diagnostic_evaluator_result",
)}


def classify_event(row: Mapping[str, Any] | str) -> str:
    """Return included or a stable omission reason (never a private identifier)."""
    name = row if isinstance(row, str) else row.get("event_type")
    if not isinstance(name, str):
        return "malformed_input"
    if name in EXCLUDED_EVENTS:
        return EXCLUDED_EVENTS[name]
    return "included" if name in EVENT_SPECS else "unknown_event"


def _typed(value, schema, redact, path="payload"):
    if value is None:
        return None
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        return {key: _typed(value.get(key), child, redact, f"{path}.{key}")
                for key, child in schema.items()}
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected list")
        # Null list elements are not an alternative to an absent list.
        if any(item is None for item in value):
            raise ValueError(f"{path}: null list element")
        return [_typed(item, schema[0], redact, path + "[]") for item in value]
    if schema == "str" and type(value) is str:
        result = redact(value)
        if type(result) is not str:
            raise ValueError(f"{path}: redactor must return text")
        return result
    if schema == "int" and type(value) is int:
        return value
    if schema == "bool" and type(value) is bool:
        return value
    if schema == "float" and type(value) in (int, float):
        try:
            result = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"{path}: expected finite number") from None
        if math.isfinite(result):
            return result
    raise ValueError(f"{path}: expected nullable {schema}")


def _identity(text):
    return text


def _error_category(row):
    # Free-form errors without an initial identifier AND colon are not safe.
    value = row.get("error_category", row.get("error"))
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("error category must be text")
    match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value)
    if "error_category" not in row:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]{0,79}):", value)
    return match[0].rstrip(":") if match else None


def _project_branch(row, spec, redact):
    aliases = dict(spec.aliases)
    approved = {}
    for key in spec.fields:
        source = key if key in row else aliases.get(key, key)
        if source in row:
            approved[key] = row[source]
    if spec.branch == "filter" and isinstance(approved.get("removed"), list):
        # Different guards record either text or explicit text/reason structs.
        # Normalize both into one stable list shape without inventing reasons.
        approved["removed"] = [dict(text=item, reason=None) if type(item) is str
                               else item for item in approved["removed"]]
    if spec.branch == "internet":
        if "error_category" in spec.fields:
            approved["error_category"] = _error_category(row)
        stored = approved.get("stored")
        if isinstance(stored, dict):
            approved["stored"] = {k: stored[k] for k in STORED if k in stored}
            if "returned_item_count" not in stored and "source_count" in stored:
                approved["stored"]["returned_item_count"] = stored["source_count"]
    if spec.branch == "social_state":
        parsed = row.get("parsed")
        if parsed is not None:
            if not isinstance(parsed, dict):
                raise ValueError("social parsed output must be an object")
            for key in spec.fields:
                if key not in approved and key in parsed:
                    approved[key] = parsed[key]
        players = approved.get("players")
        if isinstance(players, dict):
            # Sort source pseudonyms before redaction for deterministic ordering.
            if any(type(name) is not str for name in players):
                raise ValueError("player pseudonyms must be text")
            approved["players"] = [dict(pseudonym=name, observation=players[name])
                                   for name in sorted(players)]
    if spec.branch == "behavior" and "cadence" in spec.fields and "cadence" not in row:
        if any(key in row for key in CADENCE):
            approved["cadence"] = {k: row[k] for k in CADENCE if k in row}
    return _typed(approved, SCHEMAS[spec.branch], redact)


ENVELOPE_FIELDS = frozenset({
    "event_schema", "game_id", "event_id", "event_order", "component", "action",
    "turn_id", "elapsed_seconds", "time_basis", "time_precision_seconds",
    "event_text", "payload",
})


def _display(event):
    return (f'{event["component"]}: {event["action"]}\n'
            + json.dumps(event["payload"], ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False))


def project_event(row, game_id, event_order, redact: Callable[[str], str],
                  turns: Mapping[str, str], timing=None):
    """Project a neutral row; return None only for an explicit classified omission.

    Unmapped non-null turns fail closed. Caller supplies independent public game
    and turn identifiers. No timestamp or raw identifier is read into output.
    timing has only elapsed_seconds, time_basis, time_precision_seconds; absent
    timing means unknown, not zero. Redaction applies to ALL retained text.
    """
    if not isinstance(row, dict):
        raise ValueError("event row must be an object")
    if classify_event(row) != "included":
        return None
    if not callable(redact):
        raise ValueError("text redactor is required")
    if type(event_order) is not int or event_order < 1:
        raise ValueError("event order must be a positive integer")
    spec = EVENT_SPECS[row["event_type"]]
    payload = dict.fromkeys(SCHEMAS)
    payload[spec.branch] = _project_branch(row, spec, redact)
    turn = row.get("turn_id")
    if turn is not None:
        if not isinstance(turn, str) or turn not in turns:
            raise ValueError("unmapped turn identifier")
        turn = turns[turn]
    if timing is None:
        timing = {}
    if not isinstance(timing, dict) or set(timing) - {
        "elapsed_seconds", "time_basis", "time_precision_seconds"
    }:
        raise ValueError("unapproved timing fields")
    event = dict(event_schema=1, game_id=game_id,
                 event_id=f"{game_id}-event{event_order:06d}", event_order=event_order,
                 component=spec.component, action=row["event_type"], turn_id=turn,
                 elapsed_seconds=_typed(timing.get("elapsed_seconds"), "float", _identity),
                 time_basis=timing.get("time_basis", "unknown"),
                 time_precision_seconds=_typed(timing.get("time_precision_seconds"),
                                               "float", _identity),
                 event_text="", payload=payload)
    event["event_text"] = _display(event)
    validate_event(event)
    return event


def _validate_struct(value, schema, path):
    if value is None:
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict) or set(value) != set(schema):
            raise ValueError(f"{path}: missing or unapproved fields")
        for key, child in schema.items():
            _validate_struct(value[key], child, path + "." + key)
    elif isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected list")
        for child in value:
            if child is None:
                raise ValueError(f"{path}: null list element")
            _validate_struct(child, schema[0], path + "[]")
    else:
        _typed(value, schema, _identity, path)
        if schema == "float" and type(value) is not float:
            raise ValueError(f"{path}: canonical float required")


def validate_event(event):
    """Raise ValueError for noncanonical fields/types/semantics; otherwise True."""
    if not isinstance(event, dict) or set(event) != ENVELOPE_FIELDS:
        raise ValueError("missing or unapproved envelope fields")
    if type(event["event_schema"]) is not int or event["event_schema"] != 1:
        raise ValueError("unsupported event schema")
    gid, order = event["game_id"], event["event_order"]
    if type(gid) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", gid):
        raise ValueError("invalid public game identifier")
    if type(order) is not int or order < 1:
        raise ValueError("event order must be a positive integer")
    if event["event_id"] != f"{gid}-event{order:06d}":
        raise ValueError("invalid public event identifier")
    turn = event["turn_id"]
    if turn is not None and (type(turn) is not str or not re.fullmatch(
            re.escape(gid) + r"-turn[0-9]+", turn)):
        raise ValueError("invalid public turn identifier")
    action = event["action"]
    if type(action) is not str or action not in EVENT_SPECS:
        raise ValueError("unsupported action")
    spec = EVENT_SPECS[action]
    if event["component"] != spec.component:
        raise ValueError("action/component mismatch")
    payload = event["payload"]
    if not isinstance(payload, dict) or set(payload) != set(SCHEMAS):
        raise ValueError("missing or unapproved payload branches")
    for branch, schema in SCHEMAS.items():
        if branch == spec.branch:
            if payload[branch] is None:
                raise ValueError("active branch is missing")
            _validate_struct(payload[branch], schema, branch)
            for key, value in payload[branch].items():
                if key not in spec.fields and value is not None:
                    raise ValueError("field not approved for action")
        elif payload[branch] is not None:
            raise ValueError("inactive branch must be null")
    decision = (payload["scheduler"] or {}).get("decision")
    if decision is not None and decision not in {"send", "wait"}:
        raise ValueError("unsupported scheduler decision")
    outcome = (payload["delivery"] or {}).get("outcome")
    if outcome is not None and outcome not in {
        "sent", "suppressed_stale", "suppressed_timeout", "phase_drop",
        "reviewer_drop", "parse_drop", "dedup_suppressed", "delivery_unconfirmed"
    }:
        raise ValueError("unsupported delivery outcome")
    if outcome == "sent" and action != "delivery_sent":
        raise ValueError("only authoritative delivery may claim sent")
    category = (payload["internet"] or {}).get("error_category")
    if category is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", category):
        raise ValueError("unsafe error category")
    basis = event["time_basis"]
    if type(basis) is not str or basis not in TIME_BASES:
        raise ValueError("unsupported time basis")
    for key in ("elapsed_seconds", "time_precision_seconds"):
        _validate_struct(event[key], "float", key)
    if basis == "unknown" and (event["elapsed_seconds"] is not None
                                or event["time_precision_seconds"] is not None):
        raise ValueError("unknown timing cannot claim measured time")
    if basis != "unknown" and event["elapsed_seconds"] is None:
        raise ValueError("known timing needs elapsed seconds")
    precision = event["time_precision_seconds"]
    if precision is not None and precision <= 0:
        raise ValueError("time precision must be positive")
    if type(event["event_text"]) is not str or event["event_text"] != _display(event):
        raise ValueError("event text must match full canonical payload")
    return True


def render_text(events):
    """Full deterministic readable view of one event or an ordered iterable.

    JSON string escaping makes embedded newlines/quotes unambiguous; no content
    is shortened. This is derived display, never a second editable authority.
    """
    if isinstance(events, dict):
        events = [events]
    blocks = []
    for event in events:
        validate_event(event)
        timing = json.dumps({key: event[key] for key in (
            "turn_id", "elapsed_seconds", "time_basis", "time_precision_seconds")},
            ensure_ascii=False, sort_keys=True, allow_nan=False)
        blocks.append(f'{event["event_id"]} [{event["event_order"]}] {timing}\n'
                      + event["event_text"])
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def make_manifest(events, omitted, privacy_status="pending"):
    """Aggregate only public events and reason->count omissions, never raw IDs.

    omitted is a mapping of OMISSION_REASONS to nonnegative integer counts, or
    an iterable of those reason strings. Adapter-owned source/type inventories
    stay private. No events means unobserved coverage, not proof of inactivity.
    privacy_status is pending/reviewed/withheld, supplied by the privacy owner.
    """
    if privacy_status not in {"pending", "reviewed", "withheld"}:
        raise ValueError("unsupported privacy status")
    events = list(events)
    counts = dict(omitted) if isinstance(omitted, Mapping) else dict(Counter(omitted))
    for reason, count in counts.items():
        if reason not in OMISSION_REASONS or type(count) is not int or count < 0:
            raise ValueError("omissions must use safe reasons and integer counts")
    components = Counter({name: 0 for name in COMPONENTS})
    actions, bases, precisions = Counter(), Counter(), Counter()
    identities, last_order = set(), {}
    for event in events:
        validate_event(event)
        if event["event_id"] in identities:
            raise ValueError("duplicate event identifier")
        identities.add(event["event_id"])
        gid, order = event["game_id"], event["event_order"]
        if order <= last_order.get(gid, 0):
            raise ValueError("events not in retained write order")
        last_order[gid] = order
        components[event["component"]] += 1
        actions[event["action"]] += 1
        bases[event["time_basis"]] += 1
        precision = event["time_precision_seconds"]
        if precision is not None:
            precisions[precision] += 1
    return dict(
        event_schema=1, privacy_status=privacy_status, event_count=len(events),
        component_counts=dict(sorted(components.items())),
        action_counts=dict(sorted(actions.items())),
        time_quality=dict(basis_counts=dict(sorted(bases.items())),
                          known_elapsed_count=sum(e["elapsed_seconds"] is not None for e in events),
                          unknown_elapsed_count=sum(e["elapsed_seconds"] is None for e in events),
                          precision_counts=[dict(seconds=p, count=n)
                                            for p, n in sorted(precisions.items())],
                          order_basis="retained_write_order_not_causal_order"),
        input_omission_counts={key: counts.get(key, 0) for key in sorted(OMISSION_REASONS)},
        input_omission_count=sum(counts.values()),
        limitations=["Missing events do not establish inactivity.",
                     "Returned item counts are not independent cited-source counts.",
                     "Grounding is recorded only when explicitly provided.",
                     "Selection and context availability do not establish delivery or use.",
                     "Projection and redaction do not constitute privacy clearance."],
    )