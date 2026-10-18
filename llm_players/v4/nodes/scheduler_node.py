import hashlib
import random
import re
import time

from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
from llm_players.v4.agent_state import AgentState
from llm_players.schedule_then_generate_player import no_one_has_talked_yet_in_current_phase


# Minimum gap between two consecutive successful sends — prevents the "every
# 11 s" burst the user observed. Override via llm_config["min_send_interval_s"].
_DEFAULT_MIN_SEND_INTERVAL_S: float = 8.0

# Lull-breaker: if nobody has spoken for this long, the conversation has
# stalled and a human would re-engage. Override via llm_config["lull_threshold_s"].
# 30s (not 25s): at 25s + 0.35 prob the agents re-engaged a touch too eagerly
# (9.7 msg/min vs the ~7 human norm); 30s pulls cadence back into range.
_DEFAULT_LULL_THRESHOLD_S: float = 30.0
# Probability an agent breaks a detected lull on a given tick — mirrors the
# opener so 4 agents don't all jump in at once. Override "lull_breaker_probability".
_DEFAULT_LULL_PROBABILITY: float = 0.3


def _seconds_since_last_message(history: list) -> float:
    """Wall-clock seconds since the last non-self, non-Game-Manager chat line,
    derived from the trailing ``[HH:MM:SS]`` timestamp vs the current time.

    Returns inf if no parseable timestamp is found (treated as "very stale").
    """
    from core.game_constants import get_current_timestamp
    last_ts = None
    for line in reversed(history):
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if not m:
            continue
        ts = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
        last_ts = ts
        break
    if last_ts is None:
        return float("inf")
    try:
        def _secs(t):
            h, mn, s = (int(x) for x in t.split(":"))
            return h * 3600 + mn * 60 + s
        gap = _secs(get_current_timestamp()) - _secs(last_ts)
        if gap < 0:               # crossed midnight
            gap += 86400
        return float(gap)
    except (ValueError, AttributeError):
        return float("inf")


def _history_signature(history: list) -> str:
    """Hash the tail of the chat so the scheduler can detect "we already
    looked at this exact state". Cheap, deterministic, collision-safe enough."""
    tail = "\n".join(history[-5:])
    return hashlib.md5(tail.encode("utf-8", errors="replace")).hexdigest()


def _consecutive_self_sends(player, history: list) -> int:
    """Count the agent's own trailing messages with no other player's message
    in between. High counts mean the agent is talking to itself."""
    count = 0
    for msg in reversed(history):
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not m:
            continue
        speaker = m.group(4)
        if speaker == GAME_MANAGER_NAME:
            continue  # manager announcements don't count as "someone replied"
        if speaker == player.name:
            count += 1
        else:
            break  # another player spoke → streak ends
    return count


def _agent_addressed_recently(player, history: list) -> bool:
    """True if the agent was mentioned or asked something in the last 2 msgs
    (so it's legitimate to keep talking even after a self-send streak)."""
    name = player.name.lower()
    for msg in history[-2:]:
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not m or m.group(4) in {GAME_MANAGER_NAME, player.name}:
            continue
        if name in m.group(5).lower():
            return True
    return False


def _emit_blocked(player, reason: str, history_sig: str,
                  last_send_ago_s: float) -> None:
    """Structured event so the replay analyzer can rank WAIT reasons."""
    try:
        player.write_game_event(
            "v5_scheduler_blocked",
            reason=reason,
            history_sig=history_sig,
            last_send_ago_s=round(last_send_ago_s, 2),
        )
    except Exception:
        pass


def _extract_send_reason(player, history: list) -> str:
    """
    Classify *why* the agent is choosing to speak.

    Mirrors the mention/question/follow-up detection already in
    ScheduleThenGeneratePlayer.create_scheduling_prompt(), but returns a
    category string instead of mutating a task prompt.

    Returns one of:
        "direct_question" – someone asked a question and the agent was recently active
        "mention"         – agent's name appears in a recent message
        "follow_up"       – follow-up connector/question after agent spoke
        "proactive"       – none of the above; agent chose to speak unprompted
    """
    if not history:
        return "proactive"

    ai_was_recently_active = any(
        re.match(MESSAGE_PARSING_PATTERN, m) and re.match(MESSAGE_PARSING_PATTERN, m).group(4) == player.name
        for m in history[-3:]
    )

    recent = history[-2:]
    mentioned = False
    asked_question = False
    question_words = {"where", "what", "who", "how", "why", "when", "which"}

    for msg in recent:
        msg_lower = msg.lower()
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if m and m.group(4) in {GAME_MANAGER_NAME, player.name}:
            continue
        if player.name.lower() in msg_lower:
            mentioned = True
            if any(qw in msg_lower for qw in question_words):
                asked_question = True

    # Follow-up detection: only relevant when the agent was recently active
    if ai_was_recently_active and recent:
        last = recent[-1]
        m = re.match(MESSAGE_PARSING_PATTERN, last)
        if m and m.group(4) not in {GAME_MANAGER_NAME, player.name}:
            # Use the parsed content (group 5), not the whole line with its timestamp.
            content = m.group(5).lower().strip()
            is_question = (
                "?" in content
                or any(content.startswith(qw) for qw in question_words)
                or content in {"really", "really?", "why", "how come"}
            )
            is_follow_up = any(
                content.startswith(p)
                for p in ["and ", "so ", "but ", "then ", "really", "why", "how come"]
            )
            if is_question:
                return "direct_question"
            if is_follow_up:
                return "follow_up"

    if asked_question:
        return "direct_question"
    if mentioned:
        return "mention"
    return "proactive"


def make_scheduler_node(player):
    """
    LLM call: decides SEND or WAIT, and classifies the send_reason.

    Reuses:
        player.create_scheduling_prompt()
        player.scheduler.generate()
        player.interpret_scheduling_decision()
        player._log_cost()
    """
    def _node(state: AgentState) -> dict:
        history = state["message_history"]

        # Simulation mode: bypass scheduler LLM call, always speak
        if getattr(player, '_simulation_mode', False):
            return {
                "scheduler_decision": "SEND",
                "scheduler_reason":   "simulation_bypass",
                "send_reason":        _extract_send_reason(player, history),
            }

        if no_one_has_talked_yet_in_current_phase(history):
            # Production (single AI + humans): the AI should never open the
            # conversation before a human speaks — keep waiting.
            cfg = getattr(player, "llm_config", {}) or {}
            if not bool(cfg.get("allow_conversation_opener", False)):
                return {
                    "scheduler_decision": "WAIT",
                    "scheduler_reason":   "no_messages_yet",
                    "send_reason":        "none",
                }
            # Agent-vs-agent mode: there are no humans to break the ice, so an
            # agent must open. Open probabilistically each tick so the agents
            # don't all post a first line simultaneously — once any agent
            # opens, this guard is False for everyone else.
            open_prob = float(cfg.get("opener_probability", 0.25))
            if random.random() < open_prob:
                return {
                    "scheduler_decision": "SEND",
                    "scheduler_reason":   "conversation_opener",
                    "send_reason":        "proactive",
                }
            _emit_blocked(player, "waiting_to_open", _history_signature(history), float("inf"))
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "waiting_to_open",
                "send_reason":        "none",
            }

        # NOTE: a `message_in_progress` guard used to live here, but it is a no-op
        # that silently disabled the scheduler entirely. `generate_message`
        # (langgraph_v5_player / langgraph_player) sets `message_in_progress=True`
        # BEFORE invoking the graph, so by the time this node runs the flag is
        # always True — the guard returned WAIT on every turn and the scheduler
        # LLM never ran in live games. Re-entry is already prevented by
        # `generate_message`'s own top-level `if self.message_in_progress: return ""`,
        # and the graph runs synchronously inside that locked call, so the guard
        # here is both redundant and harmful. Removed.

        # Fix 2 — three cheap guards BEFORE the scheduler LLM call. Each maps
        # to a `scheduler_blocked` event so the analyzer can rank them.
        sig = _history_signature(history)
        now = time.time()
        last_send_ts = getattr(player, "_last_send_ts", 0.0) or 0.0
        last_send_ago = now - last_send_ts if last_send_ts else float("inf")

        # 2a — same history we already evaluated AND already sent recently.
        if (sig == getattr(player, "_last_scheduler_history_sig", None)
                and last_send_ago < 30.0):
            _emit_blocked(player, "history_unchanged", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "history_unchanged",
                "send_reason":        "none",
            }

        # 2b — hard floor between two sends so the bot never feels frantic.
        min_interval = float(getattr(player, "llm_config", {}).get(
            "min_send_interval_s", _DEFAULT_MIN_SEND_INTERVAL_S))
        if last_send_ago < min_interval:
            _emit_blocked(player, "min_interval_floor", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "min_interval_floor",
                "send_reason":        "none",
            }

        # 2c — Fix 3: a relevant web search is in flight, so wait briefly
        # rather than answer from training data. Only applies when the chat
        # appears to be asking a fact-question (state-builder/context-builder
        # would have flagged it via `awaiting_search` propagated from the
        # SharedSnapshot read).
        if state.get("awaiting_search") and not state.get("current_info"):
            _emit_blocked(player, "awaiting_search", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "awaiting_search",
                "send_reason":        "none",
            }

        # 2d — Fix 7a: stop talking to yourself. If the agent already sent the
        # last 2+ messages with nobody else replying, WAIT — unless someone
        # just addressed it. This kills the "lets go / the game / ready" spam.
        max_streak = int(getattr(player, "llm_config", {}).get(
            "max_consecutive_sends", 2))
        if (_consecutive_self_sends(player, history) >= max_streak
                and not _agent_addressed_recently(player, history)):
            _emit_blocked(player, "talking_to_self", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "talking_to_self",
                "send_reason":        "none",
            }

        # 2e — lull-breaker: the conversation has stalled (dead air). A human
        # would re-engage — ask a question, revive the topic. The LLM scheduler
        # is wait-biased and won't, so we break the lull ourselves. Only when
        # the agent didn't just speak (talking_to_self covers that case) and
        # past the min-interval floor. Probabilistic so 4 agents don't all jump.
        cfg = getattr(player, "llm_config", {}) or {}
        lull_threshold = float(cfg.get("lull_threshold_s", _DEFAULT_LULL_THRESHOLD_S))
        silence = _seconds_since_last_message(history)
        agent_spoke_last = _consecutive_self_sends(player, history) >= 1
        if silence > lull_threshold and not agent_spoke_last and last_send_ago >= min_interval:
            lull_prob = float(cfg.get("lull_breaker_probability", _DEFAULT_LULL_PROBABILITY))
            if random.random() < lull_prob:
                _emit_blocked(player, "lull_breaker_fired", sig, last_send_ago)
                return {
                    "scheduler_decision":  "SEND",
                    "scheduler_reason":    "lull_breaker_fired",
                    "send_reason":         "proactive",
                    "revive_conversation": True,
                }
            _emit_blocked(player, "lull_waiting", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "lull_waiting",
                "send_reason":        "none",
            }

        player._last_scheduler_history_sig = sig

        prompt = player.create_scheduling_prompt(history)
        player.logger.log("v4_scheduler_prompt", prompt)

        import time as _time
        sched_start = _time.time()
        try:
            raw = player.scheduler.generate(
                prompt, True, player.get_system_info_message(only_special_tokens=True)
            )
        except Exception as exc:
            # The scheduler LLM call failed (e.g. a 429 that exhausted its short
            # retry budget, or a transient API error). Do NOT let it propagate and
            # crash/freeze the turn — default to WAIT, the safe conservative choice.
            # The agent simply stays quiet this tick and re-evaluates on the next
            # game-loop iteration, which reads as natural human silence.
            player.logger.log("scheduler_error", str(exc))
            _emit_blocked(player, "scheduler_error", sig, last_send_ago)
            return {
                "scheduler_decision": "WAIT",
                "scheduler_reason":   "scheduler_error",
                "send_reason":        "none",
            }
        player._log_cost("scheduler")
        player.logger.log("scheduler_latency", f"{_time.time() - sched_start:.2f}s")
        player.logger.log("v4_scheduler_raw", raw)

        sends = player.interpret_scheduling_decision(raw)
        decision = "SEND" if sends else "WAIT"
        send_reason = _extract_send_reason(player, history) if sends else "none"
        player.logger.log("v4_scheduler_decision",
                          f"decision={decision}, send_reason={send_reason}")

        # V5.9 — directed-question override. Social Dynamics extracts unanswered
        # questions aimed at us into `open_threads` (it catches ones the narrow
        # last-2-lines heuristic misses). If the LLM chose WAIT but there's a
        # fresh unanswered thread aimed at us — and we're not mid-search and
        # didn't just speak — answer it. Not answering a direct question is a
        # top human-detectable tell. The downstream history-sig dedup prevents
        # answering the same state twice.
        open_threads = state.get("open_threads") or []
        if (decision == "WAIT"
                and open_threads
                and not state.get("awaiting_search")
                and _consecutive_self_sends(player, history) == 0):
            # Fire at most ONCE per distinct directed-question state. Social
            # Dynamics keeps open_threads populated across ticks, so without this
            # the override re-fired every tick on the SAME lingering thread →
            # the agent answered far too often (V5.9 over-answering regression).
            ov_sig = f"{sig}|{sorted(open_threads)}"
            if ov_sig != getattr(player, "_last_override_sig", None):
                player._last_override_sig = ov_sig
                player.logger.log(
                    "v4_scheduler_open_thread_override",
                    f"WAIT->SEND; open_threads={open_threads!r}")
                return {
                    "scheduler_decision": "SEND",
                    "scheduler_reason":   "open_thread_override",
                    "send_reason":        "direct_question",
                }

        return {
            "scheduler_decision": decision,
            "scheduler_reason":   raw,
            "send_reason":        send_reason,
        }

    return _node
