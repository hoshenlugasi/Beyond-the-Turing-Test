from typing import List

from social_turing_test.llm_players.pipeline_base.agent_state import AgentState as PipelineAgentState


class AgentState(PipelineAgentState):
    # Actual in-flight search queries let the scheduler match pending research
    # against the current question independently of cached seed facts.
    pending_search_queries: List[str]
    # Repetition Monitor: words and message-shape patterns to avoid.
    overused_words: List[str]
    repetition_note: str
    # Social State Tracker: unanswered questions aimed at us.
    open_threads: List[str]

    # The specific message the agent should answer, as a small dict
    # {speaker, timestamp, text, target_type} - so the generator answers the
    # right message and the player can
    # do a lightweight freshness/log check. None when nothing specific.
    target_message: dict

    # ── Immutable / deterministic context (injected by the player) ──────────
    fixed_persona_facts: dict        # {location, timezone, ...} - never contradict
    persona_local_time: str          # "07:33 AM" computed deterministically per turn
    persona_local_date: str          # "Sunday, May 31, 2026" in persona timezone
    # Numeric social signals, read straight from SharedSnapshot.
    interrogation_pressure: float
    attention_on_me: float
    room_norm: str                   # "everyone answers" | "selective" | ""
    room_vibe: str                   # "casual" | "probing" | "hunting" | "tense" | ""
    social_players: dict             # {player_name: behaviour note} - mirror map
    persona_interests: list          # stable per-game interests (recurring themes)

    # Generator may emit several candidate messages; the reviewer
    # then RANKS and selects one instead of rewriting (avoids semantic drift).
    candidates: List[str]

    # Wall-clock when generation+review finished, stamped by
    # output_node BEFORE the typing-delay sleep, so the safety timeout measures
    # real work time (LLM/pipeline) and not the deliberate typing delay.
    pipeline_ready_time: float


# Default state values - used to initialise the state dict before graph.invoke()
DEFAULT_STATE: AgentState = {
    "message_history": [],
    "player_name": "",
    "current_timestamp": "",
    "active_players": [],
    "conversation_topic": "",
    "pipeline_start_time": 0.0,
    "current_info": "",
    "round_summaries": [],
    "current_info_status": "irrelevant",
    "awaiting_search": False,
    "pending_search_queries": [],
    "self_claims": [],
    "revive_conversation": False,
    "social_state": "",
    "behavior_advice": "",
    "overused_words": [],
    "repetition_note": "",
    "open_threads": [],
    "scheduler_decision": "WAIT",
    "scheduler_reason": "",
    "send_reason": "",
    "target_message": None,
    "fixed_persona_facts": {},
    "persona_local_time": "",
    "persona_local_date": "",
    "interrogation_pressure": 0.0,
    "attention_on_me": 0.0,
    "room_norm": "",
    "room_vibe": "",
    "social_players": {},
    "persona_interests": [],
    "why_speaking": "",
    "style_notes": "",
    "memory_notes": "",
    "draft_message": "",
    "candidates": [],
    "must_check": [],
    "risks": [],
    "review_approved": False,
    "review_risk": "none",
    "final_message": "",
    "review_attempts": 0,
    "pipeline_ready_time": 0.0,
}
