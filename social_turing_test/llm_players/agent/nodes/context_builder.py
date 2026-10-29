import re

from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
from social_turing_test.llm_players.llm_constants import analyze_capitalization_style, analyze_period_usage
from social_turing_test.llm_players.agent.agent_state import AgentState


# Words that signal a fact-checkable question in the last incoming message.
# Detection is intentionally simple - false positives are fine (the worst case
# is the agent waiting an extra tick); false negatives are what hurt because
# they let the LLM hallucinate from training data.
_FACT_QUESTION_WORDS = {
    "who", "what", "when", "where", "which", "how",
    "latest", "current", "newest", "recent",
}
# Stop-words ignored when matching question words against snapshot topics.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "with", "you", "your", "do", "does", "did",
    "this", "that", "it", "its", "i", "we", "us", "they", "them", "lol",
    "tbh", "im", "i'm", "u", "ur",
}


def _classify_current_info_status(history: list, current_info: str,
                                  player_name: str) -> str:
    """Decide whether the snapshot has relevant info for the last chat question.

    Returns one of:
      - "present"               — snapshot has info that overlaps the question
      - "missing_for_question"  — last message is a fact question but snapshot is empty
      - "irrelevant"            — last message isn't a fact question (small talk)
    """
    last_non_self = None
    for msg in reversed(history):
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not m:
            continue
        speaker = m.group(4)
        if speaker == player_name or speaker == GAME_MANAGER_NAME:
            continue
        last_non_self = m.group(5).strip().lower()
        break
    if not last_non_self:
        return "irrelevant"

    tokens = re.findall(r"[a-z0-9']+", last_non_self)
    is_question = "?" in last_non_self or any(t in _FACT_QUESTION_WORDS for t in tokens[:4])
    if not is_question:
        return "irrelevant"

    if not current_info:
        return "missing_for_question"

    # Question keywords (≥4 chars, not stop-words) → check overlap with snapshot.
    q_keywords = {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}
    info_words = {w for w in re.findall(r"[a-z0-9']+", current_info.lower())
                  if len(w) >= 4 and w not in _STOPWORDS}
    if q_keywords & info_words:
        return "present"
    # Snapshot has facts but none look related - treat as missing for this Q.
    return "missing_for_question"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_player_messages(player, history: list, n: int = 5) -> list:
    """Return the player's last n messages as a list of content strings."""
    result = []
    for msg in reversed(history):
        m = re.match(MESSAGE_PARSING_PATTERN, msg)
        if m and m.group(4) == player.name:
            result.append(m.group(5).strip())
            if len(result) >= n:
                break
    return list(reversed(result))


def _extract_player_recent_messages(player, history: list, n: int = 3) -> str:
    """Return the player's last n messages as a comma-separated quoted string."""
    msgs = _get_player_messages(player, history, n)
    return ", ".join(f"'{m}'" for m in msgs) if msgs else ""


def _normalized_overlap(a: str, b: str) -> float:
    """Word-set overlap ratio — quick proxy for message similarity."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


# ── Why-speaking map ───────────────────────────────────────────────────────────

_WHY_MAP = {
    "direct_question": "you were asked a direct question",
    "mention":         "your name was mentioned in the chat",
    "follow_up":       "someone followed up on your last message",
    "proactive":       "you want to contribute to the conversation",
}

# ── Nodes ──────────────────────────────────────────────────────────────────────

def make_context_builder_gen_node(player):
    """
    Pre-generation context builder (no LLM call).

    Writes:
        why_speaking  – human-readable send reason for the generator prompt
        style_notes   – room capitalisation + period style
        memory_notes  – player's recent messages to avoid repeating
    """
    def _node(state: AgentState) -> dict:
        history     = state["message_history"]
        send_reason = state.get("send_reason", "proactive")

        why_speaking = _WHY_MAP.get(send_reason, "you want to contribute")

        cap_style   = analyze_capitalization_style(history)
        per_style   = analyze_period_usage(history)
        style_notes = f"caps={cap_style}, periods={per_style}"

        memory_notes = _extract_player_recent_messages(player, history, n=3)

        # Tell the generator whether the snapshot has info relevant to
        # the most-recent incoming question. Generator branches on this so it
        # never hallucinates a factual answer the snapshot doesn't back.
        current_info_status = _classify_current_info_status(
            history, state.get("current_info", ""), player.name,
        )

        return {
            "why_speaking": why_speaking,
            "style_notes":  style_notes,
            "memory_notes": memory_notes,
            "current_info_status": current_info_status,
        }

    return _node


def make_context_builder_rev_node(player):
    """
    Pre-reviewer context builder (no LLM call).

    Runs a lightweight scan of the draft message to surface obvious AI tells
    so the reviewer prompt can be more targeted.

    Writes:
        must_check – list of categories the reviewer should pay attention to
        risks      – human-readable descriptions of detected issues
    """
    _AI_TELLS = [
        "however,", "moreover,", "furthermore,",
        "in conclusion", "in summary", "to summarize",
    ]

    def _node(state: AgentState) -> dict:
        draft   = state.get("draft_message", "")
        history = state["message_history"]
        must_check: list = []
        risks: list      = []

        # Repetition check against the player's own recent messages
        prev = _get_player_messages(player, history, n=5)
        for p in prev:
            if _normalized_overlap(draft, p) > 0.7:
                risks.append(f"possible repetition of: '{p}'")
                if "repetition" not in must_check:
                    must_check.append("repetition")
                break

        # Formal connector check (strong AI tells)
        draft_lower = draft.lower()
        for tell in _AI_TELLS:
            if tell in draft_lower:
                risks.append(f"formal connector: '{tell}'")
                if "formality" not in must_check:
                    must_check.append("formality")
                break

        return {"must_check": must_check, "risks": risks}

    return _node
