import time

from llm_players.v4.agent_state import AgentState


def make_reviewer_node(player):
    """
    Reviewer node (LLM call, optional).

    Delegates to player._review_and_fix_message() — the full V3 review loop
    (budget guard, self-consistency, SELECTED_BEST, drop logic) is unchanged.

    Returns:
        review_approved – True if a message should be sent
        final_message   – reviewed/fixed message, or "" if dropped
        review_risk     – "low"|"medium"|"high"|"drop"|"none"
    """
    def _node(state: AgentState) -> dict:
        draft          = state.get("draft_message", "")
        history        = state["message_history"]
        pipeline_start = state.get("pipeline_start_time", time.time())

        # No reviewer configured → pass through
        if not getattr(player, "reviewer", None):
            return {
                "review_approved": True,
                "final_message":   draft,
                "review_risk":     "none",
            }

        # Short-message skip (config: reviewer_skip_short_words)
        skip_words = getattr(player, "_reviewer_skip_words", 0)
        if skip_words > 0 and len(draft.split()) <= skip_words:
            player.logger.log("v4_reviewer_skipped_short",
                              f"{len(draft.split())} words ≤ {skip_words}")
            return {
                "review_approved": True,
                "final_message":   draft,
                "review_risk":     "low",
            }

        # Full review loop (V3 logic reused verbatim)
        final_message, review_result = player._review_and_fix_message(
            draft, history, pipeline_start=pipeline_start
        )

        # Preserve full observability (review_outcome + message_modified logs)
        # and expose the draft + ReviewResult to the live-game logger via
        # generate_message_with_review_info().
        try:
            player._log_review_result(draft, final_message, review_result)
        except Exception:
            pass
        player._last_review_info = {
            "raw_message": draft,
            "review_result": review_result,
        }

        # Post-reviewer provenance log
        try:
            from core.chat_logging import append_ai_post_reviewer
            append_ai_post_reviewer(player.game_dir, player.name, final_message)
        except Exception:
            pass

        if not final_message:
            return {
                "review_approved": False,
                "final_message":   "",
                "review_risk":     "drop",
            }

        score = getattr(review_result, "suspicion_score", 0) if review_result else 0
        risk  = "low" if score < 4 else ("medium" if score < 7 else "high")
        return {
            "review_approved": True,
            "final_message":   final_message,
            "review_risk":     risk,
        }

    return _node
