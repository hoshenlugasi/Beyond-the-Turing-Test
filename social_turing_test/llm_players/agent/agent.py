import hashlib
import json
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List


# Context-local correlation id for the CURRENT synchronous
# pipeline turn. A ContextVar is per-execution-context, so the synchronous graph
# (which runs in the calling thread/context) sees the active turn_id, while
# background sub-agent threads run in their own context and read the default ""
# - they therefore never inherit an unrelated turn's id. Set at pipeline entry
# and reset via the returned token in `finally` (even on exceptions).
_CURRENT_TURN_ID: ContextVar[str] = ContextVar("current_turn_id", default="")


def _agent_history_signature(history: list) -> str:
    """Hash the tail of the chat — used to dedup successive responses to the
    same history. Mirrors the helper in scheduler_node so the two signatures
    agree on what counts as "no change"."""
    tail = "\n".join(history[-5:])
    return hashlib.md5(tail.encode("utf-8", errors="replace")).hexdigest()


def _chronological(history: list) -> list:
    """Defensively stable-sort the chat by its [HH:MM:SS] stamp.

    message_history is assembled by appending the manager-file block then the
    daytime-file block each poll (llm_interface), so the two streams can end up
    mildly out of timestamp order across a poll boundary. A stable sort by the
    parsed timestamp restores true chronology while preserving insertion order
    for equal stamps. If ANY line lacks a parseable stamp we return the list
    unchanged — never risk scrambling history on a parsing edge case.
    """
    import re as _re
    from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN

    def _key(line: str):
        m = _re.match(MESSAGE_PARSING_PATTERN, line)
        if not m:
            return None
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))

    keys = [_key(l) for l in history]
    if any(k is None for k in keys):
        return history  # unparseable line present - leave untouched
    # Stable sort: decorate with original index so equal stamps keep order.
    return [l for _, _, l in sorted((k, i, l) for i, (k, l) in enumerate(zip(keys, history)))]

from social_turing_test.llm_players.pipeline_base.pipeline_player import PipelinePlayer
from social_turing_test.llm_players.agent.agent_state import DEFAULT_STATE
from social_turing_test.llm_players.agent.shared_state import SharedSnapshot
from social_turing_test.llm_players.agent.sub_agents.internet_context import InternetContextAgent
from social_turing_test.llm_players.agent.sub_agents.round_summarizer import RoundSummarizerAgent
from social_turing_test.llm_players.agent.sub_agents.persona_monitor import PersonaMonitorAgent
from social_turing_test.llm_players.agent.sub_agents.social_state_tracker import SocialStateTrackerAgent
from social_turing_test.llm_players.agent.sub_agents.behavior_adapter import BehaviorAdapterAgent
from social_turing_test.llm_players.agent.sub_agents.repetition_monitor import RepetitionMonitorAgent

# Persona timezone map - used for local-time injection (Fix A)
_PERSONA_TIMEZONES = {
    "israeli_student":       "Asia/Jerusalem",
    "swiss_student":         "Europe/Zurich",
    "american_student":      "America/New_York",
    "british_student":       "Europe/London",
    "australian_student":    "Australia/Sydney",
    "european_student":      "Europe/Berlin",
    "canadian_student":      "America/Toronto",
    "tech_professional":     "America/Los_Angeles",
    "creative_professional": "America/New_York",
}
_PERSONA_LOCATIONS = {
    "israeli_student":       "Israel",
    "swiss_student":         "Switzerland",
    "american_student":      "United States",
    "british_student":       "United Kingdom",
    "australian_student":    "Australia",
    "european_student":      "Europe",
    "canadian_student":      "Canada",
    "tech_professional":     "United States",
    "creative_professional": "United States",
}


def _compute_local_time(persona_id: str) -> str:
    """Return current local time string for the given persona's timezone."""
    try:
        import pytz
        tz_str = _PERSONA_TIMEZONES.get(persona_id, "UTC")
        tz = pytz.timezone(tz_str)
        from datetime import datetime as _dt
        return _dt.now(tz).strftime("%I:%M %p")  # e.g. "07:33 AM"
    except Exception:
        from datetime import datetime as _dt
        return _dt.now().strftime("%I:%M %p")


def _compute_local_date(persona_id: str) -> str:
    """Return the current calendar date in the persona's timezone."""
    try:
        import pytz
        tz_str = _PERSONA_TIMEZONES.get(persona_id, "UTC")
        tz = pytz.timezone(tz_str)
        from datetime import datetime as _dt
        return _dt.now(tz).strftime("%A, %B %d, %Y")
    except Exception:
        from datetime import datetime as _dt
        return _dt.now().strftime("%A, %B %d, %Y")


def _convert_history_timezone(
    history: list,
    target_timezone: str,
    source_timezone: str = "Asia/Jerusalem",
    *,
    now=None,
) -> list:
    """Return a prompt-only copy whose leading chat stamps use target time.

    The stored game transcript is written in the server's Israel timezone and
    remains untouched. Only the copy shown to an agent is converted, so chat
    ordering, scheduling, persistence, logs, and the admin UI keep their
    existing semantics.
    """
    import re as _re
    from zoneinfo import ZoneInfo

    copied = list(history or [])
    try:
        source_tz = ZoneInfo(source_timezone)
        target_tz = ZoneInfo(target_timezone)
        reference = now or datetime.now(source_tz)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=source_tz)
        else:
            reference = reference.astimezone(source_tz)
    except Exception:
        return copied

    converted = []
    stamp_pattern = _re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")
    for line in copied:
        match = stamp_pattern.match(line)
        if not match:
            converted.append(line)
            continue
        try:
            hour, minute, second = (int(match.group(i)) for i in range(1, 4))
            stamped = reference.replace(
                hour=hour, minute=minute, second=second, microsecond=0
            ).astimezone(target_tz)
            converted.append(
                f"[{stamped.strftime('%H:%M:%S')}]" + line[match.end():]
            )
        except (TypeError, ValueError):
            converted.append(line)
    return converted


class SocialTuringTestAgent(PipelinePlayer):
    """Social Turing Test agent with a synchronous conversation pipeline.

    Six background workers update a shared-memory snapshot. The foreground
    scheduler controls participation; the generator proposes candidates and
    the reviewer selects one verbatim. Delivery remains subject to output
    gates. Persona-local time, social signals and strategic voting use the
    configured settings without changing the stored transcript.
    """

    TYPE_NAME = "social_turing_test_agent"

    # File for full per-turn memory snapshots (Part B of the diagnostics plan).
    _MEMORY_TRACE_FILE = "memory_trace.jsonl"

    def _memory_trace_filename(self) -> str:
        """Per-agent trace filename in multi-agent mode, shared otherwise.

        `_log_suffix` is set in LLMPlayer.__init__ ("" for single-AI games,
        "_<name>" when `multi_agent_logs` is enabled).
        """
        suffix = getattr(self, "_log_suffix", "")
        if suffix:
            return f"memory_trace{suffix}.jsonl"
        return self._MEMORY_TRACE_FILE

    def _history_for_prompt(self, history: list) -> list:
        """Build the agent-facing history without mutating the real transcript."""
        target_timezone = _PERSONA_TIMEZONES.get(self._persona_id, "UTC")
        return _convert_history_timezone(history, target_timezone)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.snapshot = SharedSnapshot()
        # Wire the observability callback so every snapshot upsert/prune emits
        # a structured event into game_events.jsonl via this player's writer.
        self.snapshot.set_event_callback(
            lambda event_type, fields: self.write_game_event(event_type, **fields)
        )
        self._turn_index: int = 0
        cfg0 = getattr(self, "llm_config", {}) or {}

        # Optional DEDICATED scheduler model. By default the scheduler shares
        # the generator's wrapper (self.scheduler = self.llm, set by the parent).
        # When llm_config["scheduler_model"] is provided, build a separate wrapper
        # for the send/wait decision so timing can run on a cheaper/faster model
        # than generation. Provider is auto-detected from the model name. On any
        # failure we keep the parent's shared scheduler rather than break the player.
        sched_model = cfg0.get("scheduler_model")
        if sched_model and str(sched_model).strip():
            sched_model = str(sched_model).strip()
            try:
                from social_turing_test.llm_players.llm_wrapper import LLMWrapper
                sched_cfg = dict(cfg0)
                sched_cfg["model_name"] = sched_model
                _n = sched_model.lower()
                if "gemini" in _n or _n.startswith("google/"):
                    _provider = "gemini"
                elif _n.startswith(("openai/", "anthropic/", "x-ai/", "deepseek/",
                                    "mistralai/", "cohere/", "perplexity/")):
                    _provider = "openrouter"
                else:
                    _provider = "together"
                sched_cfg["use_gemini"] = (_provider == "gemini")
                sched_cfg["use_together"] = (_provider == "together")
                sched_cfg["use_openrouter"] = (_provider == "openrouter")
                sched_cfg["use_pipeline"] = False
                self.scheduler = LLMWrapper(self.logger, **sched_cfg)
                self.logger.log(
                    "scheduler_model",
                    f"dedicated scheduler model={sched_model} provider={_provider}")
            except Exception as _e:  # noqa: BLE001 - never break init over the override
                self.logger.log("scheduler_model_error",
                                f"falling back to shared scheduler: {_e}")

        # Optional DEDICATED sub-agent model (llm_config["sub_agent_model"]).
        # By default every background worker (social state, behavior, persona,
        # phase, repetition, internet - except its own detector
        # override) shares the generator's wrapper. This knob lets the admin run
        # them on a cheaper/faster model (or a different family) without touching
        # the main agent. Sub-agents read player.sub_llm via their base class.
        self.sub_llm = self.llm
        sub_model = cfg0.get("sub_agent_model")
        if sub_model and str(sub_model).strip():
            sub_model = str(sub_model).strip()
            try:
                from social_turing_test.llm_players.llm_wrapper import LLMWrapper
                sub_cfg = dict(cfg0)
                sub_cfg["model_name"] = sub_model
                _n = sub_model.lower()
                if "gemini" in _n or _n.startswith("google/"):
                    _provider = "gemini"
                elif _n.startswith(("openai/", "anthropic/", "x-ai/", "deepseek/",
                                    "mistralai/", "cohere/", "perplexity/")):
                    _provider = "openrouter"
                else:
                    _provider = "together"
                sub_cfg["use_gemini"] = (_provider == "gemini")
                sub_cfg["use_together"] = (_provider == "together")
                sub_cfg["use_openrouter"] = (_provider == "openrouter")
                sub_cfg["use_pipeline"] = False
                self.sub_llm = LLMWrapper(self.logger, **sub_cfg)
                self.logger.log(
                    "sub_agent_model",
                    f"dedicated sub-agent model={sub_model} provider={_provider}")
            except Exception as _e:  # noqa: BLE001
                self.logger.log("sub_agent_model_error",
                                f"falling back to shared llm: {_e}")
        # Run id, unique per process start so turn ids do
        # not collide across restarts within the same game log. Injectable via
        # llm_config["run_id"] for deterministic tests.
        self._run_id: str = str(cfg0.get("run_id") or uuid.uuid4().hex[:8])
        # Log-noise reduction: by default, routine no-op WAIT polls do NOT emit
        # the per-poll generate_begin / generate_end pair (hundreds of
        # dropped=true lines per short game). Real sends and meaningful drops
        # (timeout / dedup) are always recorded. Set verbose_poll_logging=true
        # to include per-poll traces for debugging.
        self._verbose_poll_logging: bool = bool(cfg0.get("verbose_poll_logging", False))
        # Stable per-turn correlation id, set at the top of every generate_message
        # as `{name}-{run_id}-t{index}` and published to the _CURRENT_TURN_ID
        # ContextVar for the duration of the synchronous pipeline. Empty between
        # turns. (The ContextVar - not this attribute - is what stamps events, so
        # background-thread events never inherit it.)
        self._turn_id: str = ""

        # Scheduling, generation and review run synchronously.
        self.requested_orchestration_mode: str = "synchronous"
        self.effective_orchestration_mode: str = "synchronous"
        self.orchestration_mode: str = "synchronous"
        self._trace_lock = Lock()
        # Fix 2 + Fix 5: scheduler/dedup state shared between scheduler_node
        # and generate_message. Set lazily on first decision/response.
        self._last_scheduler_history_sig: str = ""
        self._last_responded_history_sig: str = ""
        self._last_send_ts: float = 0.0

        # Seed immutable persona facts (location + timezone) from the
        # persona id, so the generator has fixed ground truth before the agent
        # has said anything. PersonaMonitor is told NOT to capture these.
        persona_id = getattr(self, "persona_id", "") or ""
        self._persona_id = persona_id
        fixed_facts = {}
        loc = _PERSONA_LOCATIONS.get(persona_id, "")
        if loc:
            fixed_facts["location"] = loc
        tz = _PERSONA_TIMEZONES.get(persona_id, "")
        if tz:
            fixed_facts["timezone"] = tz
        # Shared event context (config: prepare_config --event-context) uses the
        # fixed-facts mechanism as ground truth for the generator and reviewer.
        event_ctx = str((getattr(self, "llm_config", {}) or {}).get("event_context", "") or "").strip()
        if event_ctx:
            fixed_facts["event_context"] = event_ctx
        if fixed_facts:
            self.snapshot.set_fixed_persona_facts(fixed_facts)

        # Persona-consistency: a STABLE handful of concrete interests for this
        # whole game (seeded by game id), so the agent recurringly references the
        # SAME tastes instead of staying generically blank. Fed to the generator.
        try:
            from social_turing_test.llm_players.llm_constants import select_persona_interests
            self._persona_interests = select_persona_interests(
                persona_id, seed=str(getattr(self.game_dir, "name", "") or ""))
        except Exception:
            self._persona_interests = []

        # Vote strategically by default (remove the accuser / prime
        # suspect via the Social State Tracker slice) instead of random. Still
        # overridable via llm_config["voting_strategy"].
        try:
            if isinstance(getattr(self, "llm_config", None), dict):
                self.llm_config.setdefault("voting_strategy", "strategic")
        except Exception:
            pass

        # Rebuild the system prompt with the dual-role framing -
        # single AI (real base rate) but the agent ALSO hunts the AI like a human
        # player, removing the passive-target asymmetry. On by default,
        # overridable via llm_config["investigator"].
        cfg = getattr(self, "llm_config", {}) or {}
        investigator = bool(cfg.get("investigator", True))
        try:
            from social_turing_test.llm_players.llm_constants import build_system_prompt as _bsp
            self.system_prompt = _bsp(
                self.persona_id, experimental=self.experimental,
                investigator=investigator,
                event_context=event_ctx or None,
            )
            self.logger.log("system_prompt_built",
                            f"investigator={investigator} event_context={bool(event_ctx)}")
        except Exception as exc:
            self.logger.log("system_prompt_error", str(exc))

        # Six background workers maintain independent shared-memory slices.
        self.sub_agents: List = [
            InternetContextAgent(self, self.snapshot),
            RoundSummarizerAgent(self, self.snapshot),
            PersonaMonitorAgent(self, self.snapshot),
            # Social State Tracker before Behavior Adapter so the latter can read the
            # former's suspects_me slice (threads are eventually consistent).
            SocialStateTrackerAgent(self, self.snapshot),
            BehaviorAdapterAgent(self, self.snapshot),
            # Deterministic (no-LLM) word-frequency monitor → overused_words.
            RepetitionMonitorAgent(self, self.snapshot),
        ]
        for agent in self.sub_agents:
            agent.start()

        # Record the pipeline configuration once at startup.
        self.logger.log("orchestration_mode", "requested=synchronous effective=synchronous")
        self.write_game_event(
            "orchestration_mode",
            requested_orchestration_mode="synchronous",
            effective_orchestration_mode="synchronous",
            shadow_evaluation_enabled=False,
            active_authoritative_orchestration_enabled=False,
        )

    def record_public_append(self, message: str = "", *, send_id: str = None) -> bool:
        """No pending append acknowledgement; personal-chat append is authoritative."""
        return False

    # ── Observability ────────────────────────────────────────────────────

    def write_game_event(self, event_type: str, **fields):
        """Auto-stamp the current ``turn_id`` (and ``turn_index``)
        onto every event so the dashboard/analyzer can correlate scheduler,
        generator, reviewer and latency rows belonging to the same turn —
        without threading turn_id through every call site.

        Isolation: this override lives on ``SocialTuringTestAgent`` only;
        the base ``write_game_event`` is untouched.
        Explicit ``turn_id``/``turn_index`` kwargs always win.
        """
        # Item 3: stamp the turn id from the CONTEXT-LOCAL var, not a mutable
        # player attribute. Synchronous-pipeline events (same context) get the
        # active turn id; background-thread events (their own context) read the
        # default "" and are NOT mis-attributed to an unrelated turn.
        tid = _CURRENT_TURN_ID.get()
        if tid and "turn_id" not in fields:
            fields["turn_id"] = tid
        if tid and "turn_index" not in fields:
            fields["turn_index"] = self._turn_index
        # Item 6: label the lane. Defaults to authoritative; shadow events set it
        # explicitly via record_shadow_event.
        fields.setdefault("orchestration_lane", "authoritative")
        super().write_game_event(event_type, **fields)

    # ── Memory-trace persistence ──────────────────────────────────────────

    def _write_memory_trace(self, snap: Dict[str, Any]) -> None:
        """Append one JSON line per turn to memory_trace.jsonl.

        Captures the full snapshot view the graph received, plus per-entry
        age + remaining TTL so the post-game analyzer can plot memory
        evolution turn by turn.
        """
        try:
            # Item 4: derive everything from the passed `snap` (the metadata and
            # read_at captured in the SAME atomic read the graph received) - NOT
            # a fresh active_entries() call that could observe a newer revision.
            now = snap.get("read_at") or time.time()
            entries_meta = snap.get("active_entries_metadata", [])
            entries_view = [
                {
                    "topic":            e.get("topic"),
                    "query":            e.get("query"),
                    "summary":          e.get("summary"),
                    "volatility":       e.get("volatility"),
                    "confidence":       e.get("confidence"),
                    "source_count":     e.get("source_count"),
                    # "seed" (one-time background baseline) vs "search"
                    # (chat-driven verified lookup) - displayed differently.
                    "origin":           e.get("origin", "search"),
                    "fetched_ago_s":    round(now - (e.get("fetched_at") or now), 2),
                    "ttl_remaining_s":  round((e.get("expires_at") or now) - now, 2),
                }
                for e in entries_meta
            ]
            line = {
                "schema_version":    1,
                "timestamp_iso":     datetime.now().isoformat(timespec="seconds"),
                "turn_index":        self._turn_index,
                "snapshot_revision": snap.get("snapshot_revision"),
                "current_info_text": snap.get("current_info", ""),
                "current_info_chars": len(snap.get("current_info", "")),
                "hedge_advice":      snap.get("hedge_advice", ""),
                "round_summaries":   list(snap.get("round_summaries", [])),
                "overused_words":    list(snap.get("overused_words", [])),
                "repetition_note":   snap.get("repetition_note", "") or "",
                "open_threads":      list(
                    (snap.get("social_state_raw") or {}).get("open_threads", []) or []
                ),
                "entries":           entries_view,
            }
            path = self.game_dir / self._memory_trace_filename()
            with self._trace_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass   # observability must never break the game loop

    # ── Main entry point (overrides PipelinePlayer) ───────────────────────

    def generate_message(self, message_history) -> str:
        """Inject the latest snapshot slices then run the pipeline graph."""
        if self.message_in_progress:
            self.logger.log("generate_message", "BLOCKED: message already in progress")
            return ""

        # Ensure the history the whole pipeline sees is in true
        # chronological order before anything reads it (scheduler target,
        # generator, dedup sig).
        message_history = _chronological(message_history)

        # Fix 5 - post-graph dedup uses this signature. Computed BEFORE the
        # graph so the dedup decision is based on the inputs that produced the
        # response, not on any draft mutation inside the graph.
        history_sig = _agent_history_signature(message_history)

        self.message_in_progress = True
        self._turn_index += 1
        # Run-scoped, restart-unique per-turn id. Publish it
        # to the context-local var so every synchronous-pipeline event this turn
        # is correlated; reset it in `finally` (token-based) so no later or
        # background event inherits a stale turn id - even if the graph raises.
        self._turn_id = f"{self.name}-{self._run_id}-t{self._turn_index:04d}"
        _turn_token = _CURRENT_TURN_ID.set(self._turn_id)
        turn_start = time.time()
        # Per-turn flag: set True when this turn produces a meaningful event
        # (a real send, or a timeout/dedup drop). Routine WAIT polls leave it
        # False so the noisy begin/end pair is skipped unless verbose logging.
        self._turn_had_meaningful_drop = False
        if self._verbose_poll_logging:
            self.write_game_event(
                "generate_begin",
                turn_index=self._turn_index,
                message_history_len=len(message_history),
                wall_t=turn_start,
            )

        typing_flag = self.game_dir / f"{self.name}_typing.txt"
        final_message = ""
        try:
            typing_flag.touch()

            # Lazy graph compilation (same as PipelinePlayer)
            if self._graph is None:
                from social_turing_test.llm_players.agent.graph import build_graph
                self._graph = build_graph(self)
                self.logger.log("graph_compiled", "LangGraph StateGraph compiled")

            # Item 2/4: prune expired state FIRST (explicit mutator, advances the
            # revision if anything changed + emits the prune event), THEN take ONE
            # atomic snapshot. Every per-turn observable below is derived from this
            # single `snap` - no second snapshot read at a possibly-newer revision.
            self.snapshot.prune_expired()
            snap = self.snapshot.read()
            ci = snap["current_info"]
            hedge = snap.get("hedge_advice", "")
            ph = snap["round_summaries"]
            snap_rev = snap.get("snapshot_revision")
            if ci and hedge:
                ci_for_graph = f"{ci}\n{hedge}"
            else:
                ci_for_graph = ci
            # Log dilution: this runs on EVERY poll tick (~1/s) and the snapshot
            # is usually unchanged - logging it each tick produced 5-7K log
            # lines per minute of game and drowned the real events. Log only
            # when the revision advances (plus the first read).
            snap_rev_changed = snap_rev != getattr(self, "_last_logged_snap_rev", object())
            if snap_rev_changed:
                self._last_logged_snap_rev = snap_rev
                self.logger.log(
                    "snapshot_read",
                    f"rev={snap_rev} | current_info={'<empty>' if not ci else ci!r} | "
                    f"round_summaries={len(ph)} item(s): {ph}",
                )
            # Full snapshot dump for the dashboard + analyzer - entry_count and the
            # recorded revision both come from the SAME `snap` (item 4). Emitted
            # only on revision change (same dilution as above): the admin panel
            # reads the LATEST event, whose content is identical while the
            # revision is unchanged.
            if snap_rev_changed:
                self.write_game_event(
                    "snapshot_observed",
                    turn_index=self._turn_index,
                    snapshot_revision=snap_rev,
                    current_info=ci,
                    hedge_advice=hedge,
                    round_summaries=list(ph),
                    self_claims=list(snap.get("self_claims", [])),
                    social_state=snap.get("social_state", ""),
                    behavior_advice=snap.get("behavior_advice", ""),
                    entry_count=snap.get("active_entry_count", 0),
                    # Per-entry provenance so the admin panel can separate the
                    # one-time seed baseline from chat-driven verified lookups.
                    info_entries=[
                        {"topic": e.get("topic"), "summary": e.get("summary"),
                         "origin": e.get("origin", "search"),
                         "confidence": e.get("confidence"),
                         "source_count": e.get("source_count")}
                        for e in snap.get("active_entries_metadata", [])
                    ],
                    # Full SharedSnapshot slices so the admin panel can show the
                    # WHOLE shared memory the agent saw this turn (not just a subset).
                    room_vibe=snap.get("room_vibe", ""),
                    room_norm=snap.get("room_norm", ""),
                    pressure_type=snap.get("pressure_type", ""),
                    interrogation_pressure=snap.get("interrogation_pressure", 0.0),
                    attention_on_me=snap.get("attention_on_me", 0.0),
                    overused_words=list(snap.get("overused_words", []) or []),
                    repetition_note=snap.get("repetition_note", "") or "",
                    awaiting_search=bool(snap.get("awaiting_search")),
                    pending_search_queries=list(snap.get("pending_search_queries", [])),
                    open_threads=list((snap.get("social_state_raw") or {}).get("open_threads", []) or []),
                    # 9054 balance - per-player behaviour map (for the dashboard + mirroring).
                    social_players=(snap.get("social_state_raw") or {}).get("players", {}) or {},
                )
                self._write_memory_trace(snap)

            initial_state = {
                **DEFAULT_STATE,
                "message_history": message_history,
                "current_info":    ci_for_graph,
                "round_summaries": ph,
                # Fix 3c - scheduler reads this to WAIT instead of answering
                # from training data while a relevant search is in flight.
                "awaiting_search": bool(snap.get("awaiting_search")),
                # Audit P1: in-flight queries so the gate can MATCH the search
                # to the question instead of requiring empty current_info.
                "pending_search_queries": list(snap.get("pending_search_queries", [])),
                # Fix 6 - Persona Monitor's self-claims so the generator stays
                # consistent (no "android" after "iphone").
                "self_claims":     list(snap.get("self_claims", [])),
                # Social State Tracker (who suspects the agent) + Behavior
                # Evolution (play-style directive for this game stage).
                "social_state":   snap.get("social_state", ""),
                "behavior_advice": snap.get("behavior_advice", ""),
                # Repetition Monitor's overused words, so the GENERATOR
                # proactively avoids them (previously only the reviewer saw them,
                # reactively - and even re-introduced "lol" in a fix; 9057).
                "overused_words":  list(snap.get("overused_words", [])),
                # Repetition Monitor LLM pass: shape/approach repetition note.
                "repetition_note": snap.get("repetition_note", "") or "",
                # Fresh unanswered questions aimed at us (Social State Tracker).
                # The scheduler's own heuristic only scans the last 2 lines for our
                # name + a question word, so it misses directed questions phrased
                # without those - this lets it SEND an answer instead of WAITing.
                # Derive open_threads from the SAME atomic
                # read() above (social_state_raw) instead of a second
                # get_social_state() call that could land on a newer revision.
                "open_threads":    list((snap.get("social_state_raw") or {}).get("open_threads", []) or []),
                # Immutable persona facts (location/timezone).
                "fixed_persona_facts": snap.get("fixed_persona_facts", {}) or {},
                # Deterministic local time - computed now, no thread,
                # no race; always correct at the moment the pipeline runs.
                "persona_local_time": _compute_local_time(self._persona_id),
                "persona_local_date": _compute_local_date(self._persona_id),
                # Numeric social signals for downstream nodes.
                "interrogation_pressure": float(snap.get("interrogation_pressure", 0.0) or 0.0),
                "attention_on_me":        float(snap.get("attention_on_me", 0.0) or 0.0),
                "room_norm":              snap.get("room_norm", "") or "",
                "room_vibe":              snap.get("room_vibe", "") or "",
                # Per-player behaviour map from Social State Tracker so
                # the generator can mirror how each specific player engages.
                "social_players":         (snap.get("social_state_raw") or {}).get("players", {}) or {},
                # Stable per-game persona interests (recurring-themes consistency).
                "persona_interests":      list(getattr(self, "_persona_interests", []) or []),
            }

            result = self._graph.invoke(initial_state)
            final_message = result.get("final_message", "")
            ready_time = result.get("pipeline_ready_time") or time.time()

            # Safety timeout. A response delay is natural (we do NOT
            # drop just because someone else spoke meanwhile - target_message
            # handles relevance). But if the actual WORK (scheduler+generator+
            # reviewer LLM calls) took absurdly long (a hang / stuck retry), the
            # reply is genuinely stale - drop it. We measure up to the moment
            # generation+review finished (pipeline_ready_time), EXCLUDING the
            # deliberate typing-delay sleep, so a long-but-legitimate message is
            # never dropped.
            # Game 9084: the old 20s ceiling killed two GOOD messages at 20.96s
            # and 24.3s (reviewer latency spiked to 8.4s) - a third of the
            # agent's output in a 3-player game, right while it was being
            # grilled. True staleness is the post-delay gate's job (it
            # re-reads the live chat at send time); this timeout only needs to
            # catch real hangs, so 35s.
            pipeline_elapsed = ready_time - turn_start
            timeout_s = float(getattr(self, "llm_config", {}).get("pipeline_timeout_s", 35.0))
            if final_message and pipeline_elapsed > timeout_s:
                # Include the suppressed text so the text-log metrics parser can
                # mark the row sent=False (events already carry it; the text log
                # didn't - 9107 showed 6 "sent" for 1 real chat message).
                self.logger.log(
                    "pipeline_timeout_drop",
                    f"pipeline took {pipeline_elapsed:.1f}s > {timeout_s}s — dropping; "
                    f"suppressed: '{final_message[:120]}'",
                )
                self.write_game_event(
                    "pipeline_timeout_drop",
                    pipeline_elapsed_s=round(pipeline_elapsed, 2),
                    timeout_s=timeout_s,
                    suppressed_message=final_message[:200],
                )
                final_message = ""
                self._turn_had_meaningful_drop = True

            # Fix 5 - post-graph dedup. Even if the scheduler / dedup guards
            # missed, never respond twice to the same chat-history state.
            if final_message:
                if history_sig == getattr(self, "_last_responded_history_sig", None):
                    self.logger.log(
                        "dedup_blocked",
                        f"history_sig={history_sig} — already responded to this state",
                    )
                    self.write_game_event(
                        "dedup_blocked",
                        history_sig=history_sig,
                        suppressed_message=final_message[:200],
                    )
                    final_message = ""
                    self._turn_had_meaningful_drop = True
                else:
                    self._last_responded_history_sig = history_sig
                    self._last_send_ts = time.time()
                    # One compact per-SEND record of the snapshot slices
                    # that actually shaped this message (low volume vs the
                    # per-poll snapshot_read_full). Makes the generator/reviewer
                    # context auditable from the event log.
                    self.write_game_event(
                        "turn_context",
                        turn_index=self._turn_index,
                        snapshot_revision=snap.get("snapshot_revision"),
                        final_message=final_message[:200],
                        current_info=ci,
                        self_claims=list(snap.get("self_claims", [])),
                        social_state=snap.get("social_state", ""),
                        behavior_advice=snap.get("behavior_advice", ""),
                        pressure_type=snap.get("pressure_type", ""),
                        overused_words=list(snap.get("overused_words", [])),
                    )
            return final_message

        finally:
            self.message_in_progress = False
            typing_flag.unlink(missing_ok=True)
            self.logger.log("generate_message", "UNLOCKED: message generation complete")
            # Emit the per-turn end event only when something meaningful happened
            # (a real send, or a timeout/dedup drop) - or when verbose poll
            # logging is on. This drops the hundreds of routine WAIT dropped=true
            # lines while preserving every real turn for the dashboard's
            # turn-timings panel.
            if (final_message or getattr(self, "_turn_had_meaningful_drop", False)
                    or self._verbose_poll_logging):
                self.write_game_event(
                    "generate_end",
                    turn_index=self._turn_index,
                    pipeline_latency_s=round(time.time() - turn_start, 3),
                    final_message_len=len(final_message),
                    dropped=(final_message == ""),
                )
            # Clear the turn correlation id LAST, so generate_end is
            # still attributed to this turn but nothing afterwards is.
            self._last_completed_turn_id = self._turn_id
            self._last_completed_turn_index = self._turn_index
            _CURRENT_TURN_ID.reset(_turn_token)
            self._turn_id = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the background worker threads cleanly."""
        for agent in self.sub_agents:
            agent.stop()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
