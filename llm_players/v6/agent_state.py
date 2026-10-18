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
    # Audit P1: the actual in-flight search queries, so the scheduler can
    # match the pending search against the question being asked instead of
    # gating on a boolean that the seed facts permanently neutralised.
    pending_search_queries: List[str]
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
    # V5 Repetition Monitor: words and message-shape patterns to avoid.
    overused_words: List[str]
    repetition_note: str
    # V5 Social Memory: unanswered questions aimed at us.
    open_threads: List[str]

    # ── Scheduler (scheduler_node) ─────────────────────────────────────────
    scheduler_decision: Literal["SEND", "WAIT"]
    scheduler_reason: str            # raw LLM output token
    send_reason: str                 # "direct_question"|"mention"|"follow_up"|"proactive"|"none"
    # V6 (Fix D): the specific message the agent should answer, as a small dict
    # {speaker, timestamp, text, target_type} — so the generator answers the
    # RIGHT message (9045: Terry answered a stale question) and the player can
    # do a lightweight freshness/log check. None when nothing specific.
    target_message: dict

    # ── V6 immutable / deterministic context (injected by the player) ───────
    fixed_persona_facts: dict        # {location, timezone, ...} — never contradict
    persona_local_time: str          # "07:33 AM" computed deterministically per turn
    persona_local_date: str          # "Sunday, May 31, 2026" in persona timezone
    # V6 (Fix F) numeric social signals, read straight from SharedSnapshot.
    interrogation_pressure: float
    attention_on_me: float
    room_norm: str                   # "everyone answers" | "selective" | ""
    room_vibe: str                   # "casual" | "probing" | "hunting" | "tense" | ""
    social_players: dict             # {player_name: behaviour note} — mirror map
    persona_interests: list          # stable per-game interests (recurring themes)

    # ── Generator context (context_builder_gen) ────────────────────────────
    why_speaking: str                # human-readable reason, e.g. "you were asked a direct question"
    style_notes: str                 # "caps=lowercase, periods=none"
    memory_notes: str                # player's recent messages to avoid repeating

    # ── Generator (generator_node) ─────────────────────────────────────────
    draft_message: str
    # V6 (Fix C): generator may emit several candidate messages; the reviewer
    # then RANKS and selects one instead of rewriting (avoids semantic drift).
    candidates: List[str]

    # ── Reviewer context (context_builder_rev) ─────────────────────────────
    must_check: List[str]            # e.g. ["repetition", "formality"]
    risks: List[str]                 # e.g. ["possible repetition of: 'lol yeah'"]

    # ── Reviewer (reviewer_node) ────────────────────────────────────────────
    review_approved: bool
    review_risk: str                 # "low"|"medium"|"high"|"drop"|"none"
    final_message: str               # "" if WAIT or dropped

    # ── Metadata ────────────────────────────────────────────────────────────
    review_attempts: int
    # V6 (Fix L): wall-clock when generation+review finished, stamped by
    # output_node BEFORE the typing-delay sleep, so the safety timeout measures
    # real work time (LLM/pipeline) and not the deliberate typing delay.
    pipeline_ready_time: float


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
    "pending_search_queries": [],
    "self_claims": [],
    "revive_conversation": False,
    "social_memory": "",
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
