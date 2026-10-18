import hashlib
import random
import re
import time

from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
from llm_players.v6.agent_state import AgentState
from llm_players.llm_constants import PASS_TURN_TOKEN_OPTIONS
from llm_players.schedule_then_generate_player import no_one_has_talked_yet_in_current_phase
from llm_players.v6.conversation_analysis import recent_reply_mode_streak as _recent_reply_mode_streak


def _own_recent_texts(player, history: list, n: int = 4) -> list:
    """The agent's own last ``n`` messages (chronological, raw text) — used to
    detect a dodge streak so the scheduler knows it has gone quiet/evasive."""
    msgs = []
    for line in history:
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if m and m.group(4) == player.name:
            msgs.append(m.group(5))
    return msgs[-n:]


def _build_scheduler_social_signals(player, state) -> str:
    """Build the optional 'social signals for timing' block from the shared
    snapshot. ONLY the timing-relevant subset of memory (attention, interrogation
    pressure, room vibe, dodge streak) — content/persona fields are deliberately
    excluded, since the scheduler decides WHEN to speak, not WHAT to write. Each
    line names its source sub-agent and why it matters, and empty/zero fields are
    skipped so the block stays lean."""
    lines = []
    try:
        attention = float(state.get("attention_on_me", 0.0) or 0.0)
    except (TypeError, ValueError):
        attention = 0.0
    try:
        pressure = float(state.get("interrogation_pressure", 0.0) or 0.0)
    except (TypeError, ValueError):
        pressure = 0.0
    room_vibe = (state.get("room_vibe") or "").strip()
    dodge_streak = 0
    try:
        _, dodge_streak = _recent_reply_mode_streak(
            _own_recent_texts(player, state.get("message_history", []), n=4))
    except Exception:
        dodge_streak = 0

    if attention > 0:
        lines.append(
            f"Attention on you right now (Social-Dynamics sub-agent, 0=ignored .. 1=full "
            f"spotlight): {attention:.2f}. If people are looking at you or just addressed "
            "you, going silent is suspicious — lean toward sending.")
    if pressure > 0:
        lines.append(
            f"Direct-questioning pressure on you (Social-Dynamics, 0=none .. 1=being grilled): "
            f"{pressure:.2f}. Under real pressure a human usually answers rather than vanishing "
            "— lean toward sending when this is high.")
    if room_vibe:
        lines.append(
            f"Current room mood (Conversation-Analysis — casual/probing/hunting/tense): "
            f"{room_vibe}. In a hunting or probing room a well-timed reply helps you blend in; "
            "needless silence stands out.")
    if dodge_streak >= 1:
        lines.append(
            f"Your recent dodge streak (your last {dodge_streak} message(s) in a row were "
            "deflections / questions-back, not real answers). If this is high, staying quiet "
            "now compounds the tell — prefer to engage with something substantive.")
    if not lines:
        return ""
    return ("=== SOCIAL SIGNALS FOR YOUR TIMING DECISION ===\n"
            + "\n".join(f"- {l}" for l in lines))


def _strict_interpret_decision(player, raw) -> bool:
    """V6-scoped STRICT scheduler parse (Fix 4): SEND only on an explicit send
    token; an explicit wait token, prose, a full message, or empty output → WAIT
    (fail-toward-WAIT). The shared ``interpret_scheduling_decision`` keeps its
    fail-toward-SEND behaviour for V3/V4/V5 — only the V6 scheduler node uses this."""
    if not raw or not isinstance(raw, str):
        return False
    text = raw.strip().lower()
    pass_tok = str(getattr(player, "pass_turn_token", "<wait>") or "").lower()
    send_tok = str(getattr(player, "use_turn_token", "<send>") or "").lower()
    if pass_tok and pass_tok in text:
        return False
    if any(str(opt).lower() in text for opt in PASS_TURN_TOKEN_OPTIONS):
        return False
    if send_tok and send_tok in text:
        return True
    # Unrecognised output (a full message with no token, prose, empty) → WAIT, not
    # SEND. A token-less message must never be treated as a send decision.
    player.logger.log("v4_scheduler_schema_failure",
                      f"unrecognised scheduler output -> WAIT: {str(raw)[:120]!r}")
    return False


# Minimum gap between two consecutive successful sends — a real anti-burst floor.
# Raised 3.5s -> 6.0s: in live game 9068 the agent fired roughly one message every
# 10-15s and humans reported "it all happened a bit quickly, hard to notice the AI's
# timing". 6s spaces sends out so the cadence reads as a relaxed human, while the LLM
# scheduler still owns the send/wait call. Override via llm_config["min_send_interval_s"].
_DEFAULT_MIN_SEND_INTERVAL_S: float = 6.0

# Lull-breaker: if nobody has spoken for this long, the conversation has
# stalled and a human would re-engage. Override via llm_config["lull_threshold_s"].
# 30s→20s (post-hackathon tuning): across 13 live games the breaker fired only
# once — real chats rarely stayed silent 30s, so the agent almost never got the
# chance to initiate. 20s + 0.4 makes it a real participant in reviving the room
# while the LLM still makes the final send/wait call.
_DEFAULT_LULL_THRESHOLD_S: float = 20.0
# Probability an agent breaks a detected lull on a given tick — mirrors the
# opener so 4 agents don't all jump in at once. Override "lull_breaker_probability".
_DEFAULT_LULL_PROBABILITY: float = 0.4


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
    """True if the agent was mentioned or asked something in the last 5 msgs
    (so it's legitimate to keep talking even after a self-send streak).
    Widened 2->5 (post-hackathon): in a fast 5-player chat a direct question
    was pushed out of a 2-message window before the scheduler even ran, so
    the agent ignored people who addressed it by name."""
    name = player.name.lower()
    for msg in history[-5:]:
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not m or m.group(4) in {GAME_MANAGER_NAME, player.name}:
            continue
        if name in m.group(5).lower():
            return True
    return False


def _talk_share_ratio(player, history: list) -> float:
    """The agent's share of recent chat vs the per-speaker average.
        ratio == 1.0  -> exactly its fair share
        ratio  > 1.0  -> talking MORE than its fair share (over-participating)
        ratio  < 1.0  -> quieter than average
    Counts the last ~24 non-Game-Manager messages. Returns 0.0 when there isn't
    enough history to judge (so callers treat the agent as 'not over-talking')."""
    counts: dict = {}
    for line in reversed(history):
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if not m:
            continue
        speaker = m.group(4)
        if speaker == GAME_MANAGER_NAME:
            continue
        counts[speaker] = counts.get(speaker, 0) + 1
        if sum(counts.values()) >= 24:
            break
    total = sum(counts.values())
    n_speakers = len(counts)
    if total == 0 or n_speakers == 0:
        return 0.0
    avg = total / n_speakers
    return (counts.get(player.name, 0) / avg) if avg > 0 else 0.0


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

        # Lull state, computed up-front: when the chat has gone quiet the LLM should
        # be allowed to DECIDE to re-engage even if nothing new was said — so a lull
        # relaxes the history-unchanged short-circuit below (it must never block the
        # agent from ever initiating on silence).
        cfg = getattr(player, "llm_config", {}) or {}
        lull_threshold = float(cfg.get("lull_threshold_s", _DEFAULT_LULL_THRESHOLD_S))
        silence = _seconds_since_last_message(history)
        in_lull = silence > lull_threshold

        # 2a — same history we already evaluated AND already sent recently. Skipped
        # during a lull so the LLM still gets consulted and may choose to initiate.
        if (sig == getattr(player, "_last_scheduler_history_sig", None)
                and last_send_ago < 30.0 and not in_lull):
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

        # 2c — Fix 3 (rebuilt per audit P1): a web search RELEVANT to what the
        # room is discussing is in flight — wait briefly rather than answer
        # from training data. The old gate required current_info to be EMPTY,
        # which the one-time seed facts permanently satisfied ~30s into every
        # game, so the agent never waited for the search it actually needed
        # (9086: answered about "the game" before the FIFA lookup landed).
        # New rule: wait iff a pending query shares a content word with the
        # recent chat tail — unrelated seed/stale searches never block.
        if state.get("awaiting_search"):
            try:
                from llm_players.v6.conversation_analysis import _w9_content_words
                tail_words = set()
                for _ln in state.get("message_history", [])[-3:]:
                    tail_words |= _w9_content_words(_ln)
                pending = state.get("pending_search_queries") or []
                _matched = next(
                    (q for q in pending if _w9_content_words(q) & tail_words), None)
            except Exception:
                _matched = None
            if _matched:
                _emit_blocked(player, "awaiting_search", sig, last_send_ago)
                player.logger.log(
                    "v6_awaiting_search_match",
                    f"waiting for in-flight search {_matched!r}")
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

        # 2e — lull-breaker (now LLM-first): the conversation has stalled (dead air).
        # A human would re-engage. We give a small probabilistic nudge to revive, but
        # we NO LONGER force a WAIT when it doesn't fire — instead we fall through to
        # the LLM scheduler, which can decide for itself whether to initiate. This
        # guarantees no guard fully blocks the agent from sending on silence.
        agent_spoke_last = _consecutive_self_sends(player, history) >= 1
        if in_lull and not agent_spoke_last and last_send_ago >= min_interval:
            lull_prob = float(cfg.get("lull_breaker_probability", _DEFAULT_LULL_PROBABILITY))
            if random.random() < lull_prob:
                _emit_blocked(player, "lull_breaker_fired", sig, last_send_ago)
                return {
                    "scheduler_decision":  "SEND",
                    "scheduler_reason":    "lull_breaker_fired",
                    "send_reason":         "proactive",
                    "revive_conversation": True,
                }
            # else: fall through to the LLM (no hard WAIT) — it owns the decision.

        player._last_scheduler_history_sig = sig

        sched_signals = _build_scheduler_social_signals(player, state)
        prompt = player.create_scheduling_prompt(history, extra_context=sched_signals)
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

        sends = _strict_interpret_decision(player, raw)
        decision = "SEND" if sends else "WAIT"
        send_reason = _extract_send_reason(player, history) if sends else "none"
        player.logger.log("v4_scheduler_decision",
                          f"decision={decision}, send_reason={send_reason}")
        # Structured event (post-hackathon observability fix): the send/wait call
        # used to live only in the scheduler text log, forcing dashboards to parse
        # raw prompts. One compact JSONL row per LLM decision.
        try:
            player.write_game_event(
                "v6_scheduler_decision",
                decision=decision.lower(),
                raw_decision=str(raw)[:60],
                send_reason=send_reason,
                latency_ms=int((_time.time() - sched_start) * 1000),
                last_send_ago_s=round(last_send_ago, 1),
                nudges=sched_signals[:400] if isinstance(sched_signals, str) else "",
            )
        except Exception:
            pass

        # 9068/9072 — room-wide-question gate. Humans flagged the agent as "too
        # happy to answer" (9068): it replied to nearly every question, including
        # ones thrown to the whole room rather than aimed at it by name. BUT the
        # first version of this gate over-fired — it skipped on independent per-tick
        # coin flips even when the agent had been SILENT for 70-140s and the cadence
        # said "too quiet" (9072: whole rounds with 0 messages). So the gate now only
        # bites when the agent is genuinely OVER-participating: it spoke recently AND
        # its share of the chat is at/above its fair share. When it's already quiet,
        # or under the spotlight, a room question is answered normally.
        if decision == "SEND" and not _agent_addressed_recently(player, history):
            try:
                pressure = float(state.get("interrogation_pressure", 0.0) or 0.0)
            except (TypeError, ValueError):
                pressure = 0.0
            try:
                attention = float(state.get("attention_on_me", 0.0) or 0.0)
            except (TypeError, ValueError):
                attention = 0.0
            try:
                _, dodge_streak = _recent_reply_mode_streak(
                    _own_recent_texts(player, history, n=4))
            except Exception:
                dodge_streak = 0
            under_spotlight = pressure >= 0.5 or attention >= 0.5 or dodge_streak >= 2
            # Only gate an agent that is actually talking too much right now:
            #  - spoke recently (within the recency window) — not already quiet, and
            #  - is at/above its fair share of the recent chat.
            recent_window = float(cfg.get("room_gate_recent_window_s", 30.0))
            spoke_recently = last_send_ago < recent_window
            try:
                talk_ratio = _talk_share_ratio(player, history)
            except Exception:
                talk_ratio = 0.0
            over_participating = spoke_recently and talk_ratio >= 1.0
            if over_participating and not under_spotlight:
                answer_prob = float(cfg.get("room_question_answer_probability", 0.5))
                if random.random() >= answer_prob:
                    _emit_blocked(player, "room_question_skip", sig, last_send_ago)
                    player.logger.log(
                        "v6_room_question_skip",
                        f"SEND->WAIT (over-participating, ratio={talk_ratio:.2f}, "
                        f"last_send_ago={last_send_ago:.1f}s, p={answer_prob})")
                    return {
                        "scheduler_decision": "WAIT",
                        "scheduler_reason":   "room_question_skip",
                        "send_reason":        "none",
                    }

        # V5.9 — directed-question override. Social Dynamics extracts unanswered
        # questions aimed at us into `open_threads` (it catches ones the narrow
        # last-2-lines heuristic misses). If the LLM chose WAIT but there's a
        # fresh unanswered thread aimed at us — and we're not mid-search and
        # didn't just speak — answer it. Not answering a direct question is a
        # top human-detectable tell. The downstream history-sig dedup prevents
        # answering the same state twice. 9068: only fire when the thread is
        # genuinely directed at us (we were named recently, or our name is in the
        # thread text) so the override can't rescue a room-wide question the gate
        # above just skipped.
        open_threads = state.get("open_threads") or []
        _threads_name_me = any(player.name.lower() in str(t).lower() for t in open_threads)
        if (decision == "WAIT"
                and open_threads
                and (_agent_addressed_recently(player, history) or _threads_name_me)
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
