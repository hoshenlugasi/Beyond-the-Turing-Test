from typing import TypedDict, Literal, List


class AgentState(TypedDict):
    # ── Input (set by caller before graph.invoke) ──────────────────────────
    message_history: List[str]

    # ── Runtime context (state_builder) ────────────────────────────────────
    player_name: str
    current_timestamp: str
    active_players: List[str]
    conversation_topic: str
    pipeline_start_time: float       # time.time() stamped by state_builder

    # ── Shared state (empty in V4; populated by sub-agents in V5) ───────────
    current_info: str                # Internet Context Agent writes here (V5)
    phase_summaries: List[str]       # Phase Summarizer Agent writes here (V5)
    # V5 fact-awareness: tells the generator whether the snapshot has info
    # relevant to the LAST incoming chat question. Set by context_builder_gen.
    current_info_status: Literal["present", "missing_for_question", "irrelevant"]
    # V5 awaiting-search signal from SharedSnapshot — scheduler uses this to
    # WAIT when a relevant search is in flight.
    awaiting_search: bool
    # V5 Persona Monitor: durable first-person claims the agent already made
    # about itself this game. Generator + reviewer use these to stay consistent.
    self_claims: List[str]
    # V5 lull-breaker: set when the scheduler fired to revive a stalled chat.
    # The generator then writes a conversation-reviving line (a question / fresh
    # topic) instead of a reaction, and the anti-filler <skip> is disabled.
    revive_conversation: bool
    # V5 Social Memory: one line on the agent's own standing (who suspected it,
    # who it pointed at, who's out). Generator reacts to it & stays consistent.
    social_memory: str
    # V5 Behavior Evolution: one play-style directive for the current game stage.
    behavior_advice: str

    # ── Scheduler (scheduler_node) ─────────────────────────────────────────
    scheduler_decision: Literal["SEND", "WAIT"]
    scheduler_reason: str            # raw LLM output token
    send_reason: str                 # "direct_question"|"mention"|"follow_up"|"proactive"|"none"

    # ── Generator context (context_builder_gen) ────────────────────────────
    why_speaking: str                # human-readable reason, e.g. "you were asked a direct question"
    style_notes: str                 # "caps=lowercase, periods=none"
    memory_notes: str                # player's recent messages to avoid repeating

    # ── Generator (generator_node) ─────────────────────────────────────────
    draft_message: str

    # ── Reviewer context (context_builder_rev) ─────────────────────────────
    must_check: List[str]            # e.g. ["repetition", "formality"]
    risks: List[str]                 # e.g. ["possible repetition of: 'lol yeah'"]

    # ── Reviewer (reviewer_node) ────────────────────────────────────────────
    review_approved: bool
    review_risk: str                 # "low"|"medium"|"high"|"drop"|"none"
    final_message: str               # "" if WAIT or dropped

    # ── Metadata ────────────────────────────────────────────────────────────
    review_attempts: int


# Default state values — used to initialise the state dict before graph.invoke()
DEFAULT_STATE: AgentState = {
    "message_history": [],
    "player_name": "",
    "current_timestamp": "",
    "active_players": [],
    "conversation_topic": "",
    "pipeline_start_time": 0.0,
    "current_info": "",
    "phase_summaries": [],
    "current_info_status": "irrelevant",
    "awaiting_search": False,
    "self_claims": [],
    "revive_conversation": False,
    "social_memory": "",
    "behavior_advice": "",
    "scheduler_decision": "WAIT",
    "scheduler_reason": "",
    "send_reason": "",
    "why_speaking": "",
    "style_notes": "",
    "memory_notes": "",
    "draft_message": "",
    "must_check": [],
    "risks": [],
    "review_approved": False,
    "review_risk": "none",
    "final_message": "",
    "review_attempts": 0,
}
