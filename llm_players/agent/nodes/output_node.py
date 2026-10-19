import random
import re
import time

from llm_players.agent.agent_state import AgentState

# Delivery controls after pipeline processing:
#  - MIN_GAP: enforce a small floor since the latest chat message, independently
#    of the typing delay measured from pipeline start.
#  - STALE GATE: re-read live chat after the delay and suppress a send when the
#    conversation has moved on. Deterministic only — no extra LLM call.
_DEFAULT_MIN_GAP_AFTER_LAST_MSG_S = 2.5
_DEFAULT_STALE_NEW_MSGS_DROP = 3


def _read_live_chat_lines(player) -> list:
    """Latest public chat lines straight from disk (not the turn-start history)."""
    try:
        from core.game_constants import PUBLIC_DAYTIME_CHAT_FILE
        f = player.game_dir / PUBLIC_DAYTIME_CHAT_FILE
        if not f.exists():
            return []
        return [ln for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()]
    except Exception:
        return []


def _new_messages_since(history: list, live_lines: list) -> list:
    """Live lines that arrived AFTER the last line the pipeline saw."""
    if not live_lines:
        return []
    if not history:
        return live_lines
    last_seen = history[-1].strip()
    for i in range(len(live_lines) - 1, -1, -1):
        if live_lines[i].strip() == last_seen:
            return live_lines[i + 1:]
    # Last-seen line not found (e.g. history line came from the manager file) —
    # fall back to a length diff so we never misreport the whole chat as new.
    if len(live_lines) > len(history):
        return live_lines[len(history):]
    return []


def _post_delay_gate(player, state: AgentState, final: str) -> str:
    """After the typing delay: enforce a minimum gap since the newest chat
    message (no flash replies) and drop the send if the conversation clearly
    moved on while we were 'typing'. Returns the (possibly emptied) final."""
    from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    from llm_players.agent.nodes.scheduler_node import _seconds_since_last_message
    from llm_players.agent.conversation_analysis import (
        classify_question,
        is_directly_addressed,
    )

    cfg = getattr(player, "llm_config", {}) or {}
    min_gap = float(cfg.get("min_gap_after_last_msg_s", _DEFAULT_MIN_GAP_AFTER_LAST_MSG_S))
    stale_n = int(cfg.get("stale_new_msgs_drop", _DEFAULT_STALE_NEW_MSGS_DROP))

    live = _read_live_chat_lines(player)

    # 1. Flash-reply floor: never appear <min_gap after the newest message.
    gap = _seconds_since_last_message(live) if live else float("inf")
    if gap < min_gap:
        pause = (min_gap - gap) + random.uniform(0.2, 1.0)
        player.logger.log("flash_reply_floor",
                          f"last msg {gap:.1f}s ago < {min_gap}s — pausing {pause:.1f}s")
        time.sleep(pause)
        live = _read_live_chat_lines(player)     # re-read again after the pause

    # 2. Staleness: what arrived while we were generating/typing?
    new_msgs = _new_messages_since(state.get("message_history") or [], live)
    others_new = []
    for ln in new_msgs:
        m = re.match(MESSAGE_PARSING_PATTERN, ln)
        if m and m.group(4) not in (GAME_MANAGER_NAME, player.name):
            others_new.append((m.group(4), m.group(5)))
    if not others_new:
        return final

    reason = ""
    target = state.get("target_message") or {}
    target_speaker = str(target.get("speaker", "") or "")
    # A NEW direct question to us — our prepared line answers the wrong thing;
    # better to stay quiet this tick and answer the fresh question next poll.
    for speaker, text in others_new:
        is_question, _ = classify_question(text)
        if is_directly_addressed(player.name, text) and is_question:
            reason = f"new_direct_question_from_{speaker}"
            break
        # A new question or request from the target speaker invalidates the
        # prepared reply even without a name mention or question mark.
        if target_speaker and speaker == target_speaker and is_question:
            reason = f"target_superseded_by_question_from_{speaker}"
            break
    # Suppress a question that another player already asked while the agent
    # was processing this turn; let the next poll react to the updated chat.
    if not reason:
        from llm_players.agent.conversation_analysis import is_duplicate_of_recent_question
        dup = is_duplicate_of_recent_question(final, live[-(len(new_msgs) + 2):],
                                              player.name)
        if dup:
            reason = f"duplicates_question_from_{dup}"
    # The room moved on substantially while we were typing.
    if not reason and len(others_new) >= stale_n:
        reason = f"conversation_moved_on_{len(others_new)}_new_msgs"

    if reason:
        player.logger.log("stale_send_drop",
                          f"{reason}; suppressed: {final[:80]!r}")
        try:
            player.write_game_event("stale_send_drop", reason=reason,
                                    new_messages=len(others_new),
                                    target_message_id=target.get("message_id"),
                                    suppressed_message=final[:200])
        except Exception:
            pass
        return ""
    return final


def make_output_node(player):
    """
    Final node (no LLM call).

    Applies the human-like typing delay, then the post-delay gate:
    flash-reply floor + stale-send drop. Emits a structured
    pipeline_complete event to game_events.jsonl.
    """
    def _node(state: AgentState) -> dict:
        final          = state.get("final_message", "")
        pipeline_start = state.get("pipeline_start_time", time.time())

        # Stamp the moment generation+review finished, BEFORE
        # the deliberate typing-delay sleep. The player's safety-timeout uses this
        # so a long (but legitimate) typing delay is never mistaken for a hang.
        ready_time = time.time()

        if final:
            player._apply_typing_delay(final, ready_time - pipeline_start)

        # Real-time only — simulation replays have no live wall clock.
        if final and not getattr(player, "_simulation_mode", False):
            try:
                final = _post_delay_gate(player, state, final)
            except Exception as exc:                     # gate must never kill a send
                player.logger.log("post_delay_gate_error", str(exc))

        player.write_game_event(
            "pipeline_complete",
            scheduler_decision = state.get("scheduler_decision"),
            send_reason        = state.get("send_reason"),
            why_speaking       = state.get("why_speaking"),
            style_notes        = state.get("style_notes"),
            draft              = state.get("draft_message"),
            must_check         = state.get("must_check", []),
            risks              = state.get("risks", []),
            final              = final,
            review_risk        = state.get("review_risk", "none"),
            current_info_used  = bool(state.get("current_info", "")),
        )

        # Carry the pre-typing-delay timestamp out so the player can measure
        # actual pipeline work time (excluding the typing sleep). final_message
        # is re-emitted because the gate may have emptied it.
        return {"pipeline_ready_time": ready_time, "final_message": final}

    return _node
