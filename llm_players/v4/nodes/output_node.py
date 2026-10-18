import time

from llm_players.v4.agent_state import AgentState


def make_output_node(player):
    """
    Final node (no LLM call).

    Applies the human-like typing delay and emits a structured
    v4_pipeline_complete event to game_events.jsonl.
    The final_message is already set in state — this node returns {} so
    the graph carries the value through to the caller unchanged.
    """
    def _node(state: AgentState) -> dict:
        final          = state.get("final_message", "")
        pipeline_start = state.get("pipeline_start_time", time.time())

        if final:
            player._apply_typing_delay(final, time.time() - pipeline_start)

        player.write_game_event(
            "v4_pipeline_complete",
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

        return {}  # final_message already in state; graph returns it to caller

    return _node
