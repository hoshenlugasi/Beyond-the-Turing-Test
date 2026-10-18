"""V6 — neutral conversation-analysis helpers (Batch 2 correction, item 10).

A dependency-free home for the chat-reading primitives shared by BOTH the legacy
generator node and the new TurnContextResolver. Previously these lived in
``nodes/generator_node.py`` and the resolver imported private helpers upward into
a graph node — the wrong direction, and a circular-import risk once the active
generator consumes TurnContext. This module has no dependency on any node or
player class, so both sides import downward into it.

Two tiers live here:

* **Relocated legacy primitives** — ``find_target_message`` and
  ``deterministic_vibe`` are behaviour-preserving moves (the generator node now
  re-exports them), with only ADDITIVE fields (``line_index``/``raw_line``/
  ``message_id``) so legacy selection is byte-for-byte unchanged.
* **New deterministic classifiers** used only by the resolver: boundary-aware
  direct-address matching, question detection that tolerates a missing '?',
  AI-probe SCOPE (direct vs group vs other-player), and a collision-safe stable
  message id. None of these touch the legacy path.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME


# ── Broad AI-mention regex (room-vibe temperature; relocated verbatim) ────────
VIBE_AI_RE = re.compile(
    r"\b(ai|a\.i\.|bot|robot|sus|suspicious|suspect|accuse|accusing|vote|voting|"
    r"voted|not human|imposter|the ai|prove|chain of thought|are you (a |an )?(ai|bot|human))\b",
    re.IGNORECASE,
)


def _parse(line: str):
    return re.match(MESSAGE_PARSING_PATTERN, line)


# ── Stable, collision-safe message identity (item 3) ──────────────────────────
def make_message_id(raw_line: str, line_index: int) -> str:
    """Deterministic id for a chat line.

    Combines the line's STABLE position in the history with the normalized line
    text and hashes with BLAKE2b — so two messages from the same speaker in the
    same second get different ids (different positions), the same history always
    yields the same id, and the value never depends on Python's salted built-in
    ``hash()``.
    """
    norm = re.sub(r"\s+", " ", (raw_line or "").strip())
    digest = hashlib.blake2b(f"{line_index}\x00{norm}".encode("utf-8"),
                             digest_size=8).hexdigest()
    return f"m_{digest}"


# ── Target selection ─────────────────────────────────────────────────────────
_AI_ACCUSATION_TEMPLATE = (
    r"(?:\b(?:you(?:'re|re| are)|u\s*(?:r|are))\s+(?:definitely\s+|for\s+sure\s+)?"
    r"(?:an?\s+|the\s+)?(?:ai|bot|robot|llm)\b|"
    r"\b{player}\s+(?:is|must\s+be|has\s+to\s+be)\s+(?:an?\s+|the\s+)?"
    r"(?:ai|bot|robot|llm)\b)"
)


def is_ai_accusation(player_name: str, text: str) -> bool:
    """Whether ``text`` explicitly accuses this player of being the AI.

    Deliberately narrower than :data:`VIBE_AI_RE`: generic room discussion such
    as "who is the ai" is not an accusation, while "you are an ai" and
    "Tyler is the bot" are.  The narrow scope keeps target selection stable.
    """
    if not text:
        return False
    player = re.escape((player_name or "").strip()) or r"(?!)"
    return bool(re.search(_AI_ACCUSATION_TEMPLATE.format(player=player), text,
                          re.IGNORECASE))


def find_recent_ai_accusation(player_name: str, history: List[str],
                              lookback: int = 6) -> Dict[str, Any]:
    """Return the newest explicit accusation aimed at the agent, if any."""
    for idx in range(len(history) - 1, max(-1, len(history) - lookback - 1), -1):
        m = _parse(history[idx])
        if not m or m.group(4) in {GAME_MANAGER_NAME, player_name}:
            continue
        if is_ai_accusation(player_name, m.group(5)):
            return {
                "speaker": m.group(4),
                "timestamp": f"{m.group(1)}:{m.group(2)}:{m.group(3)}",
                "text": m.group(5).strip(),
                "target_type": "accusation",
                "line_index": idx,
                "raw_line": history[idx],
                "message_id": make_message_id(history[idx], idx),
            }
    return {}


def is_bare_name_ping(player_name: str, text: str) -> bool:
    """True for a bare check-in such as ``tyler?`` (not a real question)."""
    norm = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    name = re.sub(r"[^a-z0-9]+", "", (player_name or "").lower())
    return bool(name) and norm == name


def is_time_question(text: str) -> bool:
    """Recognize ordinary requests for the current time, including terse chat."""
    t = re.sub(r"[^a-z0-9']+", " ", (text or "").lower()).strip()
    return bool(re.search(
        r"\b(?:what(?:'s|s| is)?\s+(?:the\s+)?time(?:no[we]?)?|"
        r"what\s+the\s+time(?:no[we]?)?|"
        r"time\s+(?:is\s+it|now)|current\s+time)\b", t))


def looks_like_time_answer(text: str) -> bool:
    """Conservative signal that a reply contains an actual clock-time value."""
    t = (text or "").lower()
    if re.search(
        # W15: the old trailing \b made '1:30pm' fail ('0' -> 'p' is not a word
        # boundary), so a correct clock answer was scored as no-answer and the
        # reviewer's time filter deleted it (game 9074).
        r"\b(?:[01]?\d|2[0-3])[:.]\d{2}(?![\d:.])|"
        r"\b(?:[1-9]|1[0-2])\s*(?:am|pm|ish|o'?clock)\b|"
        r"\b(?:around|about|nearly)\s+(?:[1-9]|1[0-2])\b", t
    ):
        return True
    hours = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve")
    qualifiers = ("oh five|ten|fifteen|twenty|twenty five|thirty|forty|"
                  "forty five|fifty|fifty five|ish|o'?clock|am|pm")
    return bool(re.search(rf"\b(?:{hours})(?:\s+(?:{qualifiers}))\b", t))


def asks_who_is_ai(text: str) -> bool:
    """Question that asks for an AI identity already supplied by an accuser."""
    t = (text or "").lower()
    return bool(re.search(
        r"\bwho\s+do\s+(?:you|u)\s+think\s+(?:it\s+is|is\s+(?:the\s+)?(?:ai|bot))\b|"
        r"\bwho\s+is\s+(?:the\s+)?(?:ai|bot)\b", t))


def find_target_message(player_name: str, history: List[str]) -> Dict[str, Any]:
    """Identify the specific message the agent should answer (V6 Fix D).

    Priority: (1) the newest direct named question or explicit AI accusation in
    the last 5 lines, (2) the most recent question/request in the last 3 lines,
    (3) the last non-self, non-manager line.  Treating a fresh accusation as a
    target prevents an older question from hijacking the reply (game 9085).

    Additive (new) fields: ``line_index`` (absolute index in ``history``),
    ``raw_line`` (the full source line), and ``message_id`` (collision-safe).
    """
    name = (player_name or "").lower()
    n = len(history)

    def _mk(idx: int, m, ttype: str) -> Dict[str, Any]:
        return {
            "speaker": m.group(4),
            "timestamp": f"{m.group(1)}:{m.group(2)}:{m.group(3)}",
            "text": m.group(5).strip(),
            "target_type": ttype,
            "line_index": idx,
            "raw_line": history[idx],
            "message_id": make_message_id(history[idx], idx),
        }

    # W9c (game 9079): a line that addresses ANOTHER player by name ("Wyd
    # Emerson") must not become the agent's answer target — answering a
    # question meant for someone else is a state-grounding tell (the agent
    # replied "just chilling too" to a question aimed at Emerson). Rule 1 is
    # untouched (it only matches questions naming the AGENT); rules 2/3 skip
    # other-addressed lines, falling back to the old pick if nothing else fits.
    other_speakers = set()
    for line in history:
        mm = _parse(line)
        if mm and mm.group(4) not in {GAME_MANAGER_NAME, player_name}:
            other_speakers.add(mm.group(4))

    def _names_other_speaker(content: str) -> bool:
        return any(is_directly_addressed(sp, content) for sp in other_speakers)

    # 1. Newest explicit question naming the agent OR explicit accusation.
    # They share a scan so recency wins between them.
    for idx in range(n - 1, max(-1, n - 6), -1):
        m = _parse(history[idx])
        if not m or m.group(4) in {GAME_MANAGER_NAME, player_name}:
            continue
        content = m.group(5)
        is_q, _ = classify_question(content)
        if name and is_directly_addressed(player_name, content) and is_q:
            return _mk(idx, m, "direct_question")
        if not _names_other_speaker(content) and is_ai_accusation(player_name, content):
            return _mk(idx, m, "accusation")
    # 2. Most recent question/request in the last 3 lines (not aimed elsewhere).
    for idx in range(n - 1, max(-1, n - 4), -1):
        m = _parse(history[idx])
        if not m or m.group(4) in {GAME_MANAGER_NAME, player_name}:
            continue
        is_q, _ = classify_question(m.group(5))
        if is_q and not _names_other_speaker(m.group(5)):
            return _mk(idx, m, "recent_question")
    # 3. last non-self, non-manager line (prefer one not aimed at someone else).
    fallback = None
    for idx in range(n - 1, -1, -1):
        m = _parse(history[idx])
        if m and m.group(4) not in {GAME_MANAGER_NAME, player_name}:
            if fallback is None:
                fallback = (idx, m)
            if not _names_other_speaker(m.group(5)):
                return _mk(idx, m, "last_message")
    if fallback is not None:
        return _mk(fallback[0], fallback[1], "last_message")
    return {}


# ── Room-vibe temperature (relocated verbatim) ────────────────────────────────
def deterministic_vibe(history: List[str], player_name: str) -> str:
    """Cheap always-on room mode: casual|probing|hunting, or "" when unreadable."""
    recent = []
    for line in reversed(history):
        m = _parse(line)
        if not m or m.group(4) in (GAME_MANAGER_NAME, player_name):
            continue
        recent.append(m.group(5))
        if len(recent) >= 6:
            break
    if not recent:
        return ""
    ai_hits = len(VIBE_AI_RE.findall(" ".join(recent)))
    questions = sum(1 for r in recent if "?" in r)
    if ai_hits >= 2:
        return "hunting"
    if ai_hits >= 1 or questions >= 3:
        return "probing"
    return "casual"


# ── NEW classifiers (resolver-only; do not affect legacy) ─────────────────────

def is_directly_addressed(name: str, text: str) -> bool:
    """Boundary-aware: is the agent's name mentioned as a WHOLE word?

    ``Ari`` must not match inside ``Paris``. Uses a word-boundary regex on the
    escaped name, so punctuation around the name still counts.
    """
    if not name or not text:
        return False
    return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


# ── W9: deterministic state-grounding guards ─────────────────────────────────
# The manual 13-game review found the final agent's remaining failures share one
# root: locally-fluent messages that contradict the actual conversation STATE
# (who said what, what was just asked, what I myself just said, who a question
# was addressed to). These helpers make the checkable parts of that state
# enforceable in code — same pattern as the identity-risk regex.

_W9_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "do", "does",
    "did", "what", "whats", "what's", "who", "whos", "who's", "which", "of",
    "to", "for", "in", "on", "at", "and", "or", "so", "then", "now", "it",
    "its", "it's", "that", "this", "you", "your", "yours", "u", "ur", "me",
    "my", "mine", "i", "im", "i'm", "we", "us", "our", "he", "she", "they",
    "them", "hey", "please", "pls", "tell", "say", "guys",
}


def _w9_content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    out = set()
    for w in words:
        if w in _W9_STOPWORDS:
            continue
        if w.endswith("'s"):
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3:   # crude plural/possessive fold
            w = w[:-1]
        out.add(w)
    return out


def is_duplicate_of_recent_question(candidate: str, history: List[str],
                                    agent_name: str,
                                    lookback: int = 12) -> Optional[str]:
    """W9a (game 9068): another player asked Tyler his mother's name and one
    second later the agent sent the SAME question — an echo tell. True when the
    candidate is a question whose content words are (nearly) contained in a
    question another player asked recently. Returns that speaker, else None.

    Conservative on purpose: candidate must be a question with >=2 content
    words; overlap thresholds keep normal short banter ("why me though") safe.
    """
    is_q, _ = classify_question(candidate)
    if not is_q:
        # "tyler whats your mothers name" — a leading vocative name hides the
        # interrogative from the leader-word check; retry without it.
        rest = (candidate or "").strip().split(None, 1)
        is_q, _ = classify_question(rest[1]) if len(rest) == 2 else (False, 0.0)
    if not is_q:
        return None
    cand = _w9_content_words(candidate)
    if len(cand) < 2:
        return None
    for line in history[-lookback:]:
        m = _parse(line)
        if not m or m.group(4) in (GAME_MANAGER_NAME, agent_name):
            continue
        other_q, _ = classify_question(m.group(5))
        if not other_q:
            continue
        other = _w9_content_words(m.group(5))
        if not other:
            continue
        inter = len(cand & other)
        if inter / len(cand) >= 0.8 and inter / max(len(other), 1) >= 0.6:
            return m.group(4)
    return None


def is_own_template_repeat(candidate: str, own_recent: List[str]) -> bool:
    """W9b (game 9080): "i like cars more" then "i like electronic more" 40s
    later — same sentence template, one word swapped, and the room called it.
    True when the candidate differs from a recent own message by at most one
    token at the same length (>=3 tokens; exact repeats are caught earlier)."""
    cand_toks = re.findall(r"[a-z']+", (candidate or "").lower())
    if len(cand_toks) < 3:
        return False
    for prev in own_recent:
        prev_toks = re.findall(r"[a-z']+", (prev or "").lower())
        if len(prev_toks) != len(cand_toks):
            continue
        diffs = sum(1 for a, b in zip(prev_toks, cand_toks) if a != b)
        if diffs <= 1:
            return True
    return False


def shares_opener_ngram(candidate: str, own_recent: List[str], n: int = 3) -> bool:
    """9091: a repeated OPENER that ``is_own_template_repeat`` cannot see.

    That game's agent sent "what makes you say that", "what makes you say that
    taylor", "what makes you so sure taylor" and "what makes you think riley
    keeps repeating sentences" — the "what makes you ..." family was 5 of its 19
    messages. Every human survey named it, and the Pro evaluation scored
    structural_repetition 2/10.

    ``is_own_template_repeat`` misses all of them: it requires EQUAL token length
    (5 vs 6 tokens) and at most ONE differing token ("say that" vs "so sure" is
    two). The shared signature is the opener, not the whole shape, so compare
    just the first ``n`` tokens.

    Deliberately advisory: used to tag a candidate and, at most, to swap in a
    clean alternative — never as a hard prefilter, because openers legitimately
    recur in casual chat ("yeah i think ...").
    """
    cand_toks = re.findall(r"[a-z']+", (candidate or "").lower())
    if len(cand_toks) < n:
        return False
    opener = cand_toks[:n]
    for prev in own_recent:
        prev_toks = re.findall(r"[a-z']+", (prev or "").lower())
        if len(prev_toks) >= n and prev_toks[:n] == opener:
            return True
    return False


# Interrogatives that make a line a question even without a '?'.
_Q_LEADERS = (
    "who", "what", "where", "when", "why", "how", "which", "whats", "what's",
    "hows", "how's", "wheres", "where's",
)
_Q_PHRASES = re.compile(
    r"\b(tell us|tell me|let us know|how come|what about (you|u)|"
    r"are (you|u)|do (you|u)|did (you|u)|have (you|u)|whats? (your|ur)|"
    r"watching any|following any|seen any|heard about)\b",
    re.IGNORECASE,
)


def classify_question(text: str) -> Tuple[bool, float]:
    """Return ``(is_question, confidence)`` tolerating a missing '?'.

    High confidence for an explicit '?'; medium for a clear interrogative
    leader/phrase; low/zero otherwise. Confidence is exposed so the plan's later
    ambiguity model can revisit borderline lines — we never silently treat every
    punctuation-less line as a statement.
    """
    if not text:
        return False, 0.0
    t = text.strip().lower()
    if "?" in t:
        return True, 0.95
    first = re.split(r"\s+", t, 1)[0].strip(",.!") if t else ""
    if first in _Q_LEADERS:
        return True, 0.7
    if _Q_PHRASES.search(t):
        return True, 0.65
    return False, 0.1


# ── Trivial "test" probes (item: 2+2 fix) ────────────────────────────────────
# A trivially-obvious question almost nobody would ask sincerely mid-game — it is
# a SOCIAL test of whether you answer like an eager bot. Earnestly answering
# "2+2 -> four" or "capital of france -> paris" is itself a tell, regardless of
# the current room vibe. Deterministic so detection never lags the vibe agent.
# Operators limited to +, x, *, × on purpose: '-' and '/' collide with sports
# scores ("3-1") and dates ("9/11"), which are NOT maths tests.
_TRIVIAL_MATH_RE = re.compile(r"\b\d{1,2}\s*[+x*×]\s*\d{1,2}\b", re.IGNORECASE)
_TRIVIAL_TRIVIA_RE = re.compile(
    r"\b(capital of (france|england|italy|spain|japan|the (us|usa|uk))|"
    r"what colou?r is the (sky|sun|grass)|how many (legs|fingers|days in a week)|"
    r"what('?s| is) (\d{1,2}\s*[+x*×]\s*\d{1,2})|"
    r"spell (cat|dog|your name))\b",
    re.IGNORECASE,
)


# ── Reply-mode streak (balance: stop the agent dodging every single turn) ─────
# Vague-deflection markers (kept in sync with generator_node._DEFLECTION_MARKERS).
_DEFLECT_MARKERS = (
    "idk", "no clue", "not sure", "i forget", "no idea", "dunno",
    "have to think", "pass", "beats me", "who knows", "cant remember",
    "cba", "meh",
)


# Leading filler that precedes a question-back but adds no answer ("so what…").
_REPLY_FILLER = {
    "so", "and", "but", "ok", "okay", "well", "hmm", "lol", "haha", "nah",
    "yeah", "ya", "eh", "oh", "hey", "wait", "umm", "um", "lmao",
}
# Aux verbs that, followed by a pronoun, signal a question-back ("are you…").
_AUX_Q = {"are", "do", "did", "does", "is", "have", "has", "can", "would", "will"}
_AUX_TARGET = {"you", "u", "that", "it", "they", "we"}


def classify_reply_mode(text: str) -> str:
    """Classify ONE of the agent's own messages as how it engaged:
    ``answer`` | ``deflect`` | ``counter_question``.

    - counter_question: lobs a question back with no real content of its own
      ('why what do you like to do', 'so what do you think ziggy') — the 9054
      dodge pattern.
    - deflect: a vague non-answer ('idk', 'not sure', 'meh').
    - answer: anything that actually contributes (the default). A reply that gives
      info AND then asks ("i game with friends, you?") is an ANSWER — only a
      content-free question-back counts as counter_question."""
    if not text:
        return "answer"
    t = text.strip().lower()
    words = t.split()
    if not words:
        return "answer"
    # Short + a vague marker → deflect.
    if len(words) <= 6 and any(m in t for m in _DEFLECT_MARKERS):
        return "deflect"
    # An answer clause before the question (a comma) means it actually said
    # something first → treat as an answer, not a pure dodge.
    if "," in t:
        return "answer"
    # Strip leading filler, then see if it OPENS with a question-back.
    i = 0
    while i < len(words) and words[i].strip(",.!?") in _REPLY_FILLER:
        i += 1
    rest = [w.strip(",.!?") for w in words[i:]]
    if not rest:
        return "answer"
    opens_q = (
        rest[0] in _Q_LEADERS
        or (rest[0] in _AUX_Q and len(rest) > 1 and rest[1] in _AUX_TARGET)
    )
    if opens_q and len(rest) <= 9:
        return "counter_question"
    # A short PURE question (no answer clause before it) is a question-back
    # even when the opener list misses it — dropped-subject forms like "you
    # asking about the time or 2+2?" (9086) or "we doing math now?" (9065)
    # slipped through as "answer" and kept the dodge streak at 0, so the
    # same-dodge-mode guard never fired.
    inferred_question, _ = classify_question(t)
    if inferred_question and len(rest) <= 9 and rest and rest[0].isalpha():
        return "counter_question"
    return "answer"


def recent_reply_mode_streak(own_recent_msgs: list) -> tuple:
    """Given the agent's own recent messages in chronological order, return
    ``(last_mode, dodge_streak)`` where ``dodge_streak`` is how many of the most
    recent messages in a row were NOT real answers (deflect / counter_question).

    A high streak means the agent has been dodging every turn — itself a glaring
    tell — so the reviewer should swing back to a substantive answer."""
    if not own_recent_msgs:
        return ("answer", 0)
    streak = 0
    last_mode = "answer"
    for i, msg in enumerate(reversed(own_recent_msgs)):
        mode = classify_reply_mode(msg)
        if i == 0:
            last_mode = mode
        if mode == "answer":
            break
        streak += 1
    return (last_mode, streak)


def is_trivial_test_question(text: str) -> bool:
    """True if ``text`` is a trivially-obvious test probe (basic maths, kindergarten
    trivia, "are you the ai"). These are social tests, not real questions — an
    earnest literal answer reads as robotic. Vibe-independent on purpose."""
    if not text:
        return False
    t = text.strip().lower()
    # Short line + trivial maths (require the expression to roughly BE the message,
    # so a real "i did 3 x 3 sets at the gym" sentence doesn't trip it).
    if _TRIVIAL_MATH_RE.search(t) and len(t.split()) <= 5:
        return True
    if _TRIVIAL_TRIVIA_RE.search(t):
        return True
    # "are you the ai / a bot" directed at the agent is a probe too.
    if _AI_DIRECT_RE.search(t):
        return True
    return False


# Sharp "are YOU the ai / prove it" tokens (directed-probe candidates).
_AI_DIRECT_RE = re.compile(
    r"\b(are (you|u) (a |an |the )?(ai|bot|human|robot)|you('?re| are) (a |an |the )?(ai|bot|robot)|"
    r"prove (you|u)('?re| are)?( not)? (a |an |the )?(ai|human|bot))\b",
    re.IGNORECASE,
)
# Generic AI-talk that may be group-level, not aimed at anyone in particular.
_AI_GENERIC_RE = re.compile(
    r"\b(the (ai|bot|robot)|imposter|not (a )?human|chain of thought|"
    r"sus(picious)?|who('?s| is) the (ai|bot))\b",
    re.IGNORECASE,
)


def classify_ai_probe_scope(text: str, agent_name: str,
                            known_players: Optional[List[str]] = None) -> str:
    """Classify an AI-related line by WHO it targets (item 5):

    ``direct_to_agent`` | ``about_other_player`` | ``group_level`` | ``none``.

    Only ``direct_to_agent`` should drive the agent's AI-probe policy; group-level
    AI chatter influences room vibe, not a personal interrogation response.
    """
    if not text:
        return "none"
    has_direct_form = bool(_AI_DIRECT_RE.search(text))
    has_generic = bool(_AI_GENERIC_RE.search(text))
    if not (has_direct_form or has_generic):
        return "none"

    names_agent = is_directly_addressed(agent_name, text)
    other = None
    for p in (known_players or []):
        if p and p != agent_name and is_directly_addressed(p, text):
            other = p
            break

    if names_agent:
        return "direct_to_agent"
    # A directed "are you an ai" with no name is still a probe at whoever is being
    # spoken to → treat as direct only when it uses the 2nd-person direct form AND
    # no OTHER player is named.
    if has_direct_form and other is None:
        return "direct_to_agent"
    if other is not None:
        return "about_other_player"
    return "group_level"


# Persona-disclosure questions ("how old", "where you from", "what you study").
PERSONA_Q_RE = re.compile(
    r"\b(how old|your age|where (are|r) (you|u) from|where (are|r) (you|u) based|"
    r"where do (you|u) (live|study)|where (you|u) from|what (do )?(you|u) (study|studying|do for work)|"
    r"what year|which (city|country|uni|university)|your major)\b",
    re.IGNORECASE,
)

# Maps a persona question to the fixed-fact key that would answer it, so the
# policy can tell "the agent knows this about itself" from "it would invent it".
_PERSONA_FACT_MAP = (
    (re.compile(r"\b(where (are|r) (you|u) (from|based|live)|which (city|country)|where (you|u) from)\b", re.IGNORECASE), "location"),
    (re.compile(r"\b(time|timezone|what time)\b", re.IGNORECASE), "timezone"),
)


def persona_fact_key(text: str) -> Optional[str]:
    """Which fixed-fact key a persona question asks about, or None."""
    for rx, key in _PERSONA_FACT_MAP:
        if rx.search(text or ""):
            return key
    return None


# Deterministic "this MIGHT be about something current" hint. NON-authoritative
# (item 7): it only raises a flag; it never decides that a claim is verified and
# never triggers research (that is Batch 3's async model classifier).
FRESHNESS_HINT_RE = re.compile(
    r"\b(today|right now|currently|latest|breaking|news|score|who won|winning|"
    r"world cup|olympics|election|results?|happening now|this week)\b",
    re.IGNORECASE,
)
