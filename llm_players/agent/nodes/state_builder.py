import time

from core.game_constants import (
    REMAINING_PLAYERS_FILE,
    CONVERSATION_TOPIC_FILE,
    get_current_timestamp,
)
from llm_players.agent.agent_state import AgentState


def make_state_builder_node(player):
    """
    Reads game-dir metadata and stamps pipeline_start_time.
    Runs with no LLM call.
    """
    def _node(state: AgentState) -> dict:
        game_dir = player.game_dir

        remaining_file = game_dir / REMAINING_PLAYERS_FILE
        active_players = remaining_file.read_text(encoding="utf-8").splitlines() if remaining_file.exists() else []

        topic_file = game_dir / CONVERSATION_TOPIC_FILE
        topic = topic_file.read_text(encoding="utf-8").strip() if topic_file.exists() else ""

        return {
            "player_name":        player.name,
            "current_timestamp":  get_current_timestamp(),
            "active_players":     active_players,
            "conversation_topic": topic,
            "pipeline_start_time": time.time(),
            # Initialise output fields so downstream nodes never hit KeyError
            # on the WAIT path (graph terminates before these are written).
            "final_message":   "",
            "review_approved": False,
            "review_risk":     "none",
            "must_check":      [],
            "risks":           [],
        }

    return _node
