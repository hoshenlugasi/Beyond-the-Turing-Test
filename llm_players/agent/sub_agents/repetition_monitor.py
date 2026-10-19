"""Repetition Monitor background worker.

Runs every ~10 s with a deterministic tally and a lower-cadence LLM pass.
Each tick it scans the agent's
OWN sent messages and flags words/openers it has leaned on too much (e.g. "lol"
in 4 of 10 messages). Writes them to snapshot.set_overused_words([...]).

Why this exists: the reviewer's repetition rule EXEMPTS basic fillers ("lol",
"nah", "yeah") and the LLM can't reliably *count* cumulative frequency from the
chat history. A
deterministic running tally fed into the reviewer (via shared memory) closes that
blind spot: the reviewer flags reuse of an overused word and never adds it in a fix.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import List

from core.game_constants import MESSAGE_PARSING_PATTERN
from llm_players.agent.sub_agents.base import SubAgent


# Tokeniser: words only (drop punctuation), lowercased.
_WORD_RE = re.compile(r"[a-z0-9']+")

# Tiny function words that are normal to repeat — never flag these as a tell.
_STOPWORDS = {
    "i", "im", "i'm", "a", "an", "the", "to", "of", "in", "on", "is", "it", "its",
    "and", "or", "but", "so", "you", "u", "me", "my", "we", "do", "dont", "don't",
    "that", "this", "what", "who", "why", "how", "not", "no", "yes", "for", "with",
    "at", "be", "are", "was", "if", "as", "ok",
}


# Neutral system msg for the LLM pass — NOT the player persona, so the model
# analyses instead of role-playing (mirrors the internet-context decision call).
_REP_SYSTEM_MSG = (
    "You are a concise writing-style analyst. You reply with ONE short sentence "
    "or the single word OK. No preamble."
)


class RepetitionMonitorAgent(SubAgent):
    interval_seconds: float = 10.0
    COMPONENT = "repetition_monitor"

    _OWN_WINDOW: int = 10          # consider the agent's last N messages
    _MIN_MSGS: int = 3             # need a few before judging
    # A word is "overused" if it appears in this fraction of recent messages...
    _OVERUSE_FRACTION: float = 0.3
    # ...or in at least this many of them (absolute floor).
    _OVERUSE_MIN_COUNT: int = 3
    _MAX_OVERUSED: int = 6         # cap the list
    # LLM pass (shape/approach repetition) runs at a LOWER cadence than the
    # deterministic word tally — it costs an LLM call, and latency matters.
    _LLM_INTERVAL_S: float = 30.0
    _LLM_MIN_MSGS: int = 4         # need enough of our own messages to see a pattern

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        self._processed_own_count: int = 0
        self._last_overused: List[str] = []
        self._last_llm_ts: float = 0.0
        self._llm_processed_count: int = 0
        self._last_note: str = ""
        self._clean_passes: int = 0

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        own = self._own_messages()
        if len(own) < self._MIN_MSGS or len(own) == self._processed_own_count:
            # Even with no new messages, the LLM pass may still be due — but
            # there's nothing new to analyse, so just return.
            return  # nothing to judge / nothing new

        recent = own[-self._OWN_WINDOW:]
        overused = self._compute_overused(recent)
        self._processed_own_count = len(own)
        if overused != self._last_overused:
            self.snapshot.set_overused_words(overused)
            self._last_overused = overused
            self.player.logger.log("repetition_monitor", f"overused={overused!r}")
        self.player.write_game_event(
            "repetition_monitor_tick",
            own_msgs=len(own),
            overused=overused,
        )

        # Hybrid: LLM pass for WORDING / sentence-template repetition (the exact-
        # word tally above can't see the same template). LEXICAL/STRUCTURAL only —
        # it never coaches conversational tactics (asking/agreeing/dodging); that
        # belongs to the social lens. Low
        # cadence + needs new messages since the last pass.
        now = time.time()
        if (len(own) >= self._LLM_MIN_MSGS
                and len(own) != self._llm_processed_count
                and (now - self._last_llm_ts) >= self._LLM_INTERVAL_S):
            # Feed the FULL own-message history (not just the last window) so
            # the LLM can spot a wording/template habit that spans the whole game.
            self._run_llm_pass(own)
            self._last_llm_ts = now
            self._llm_processed_count = len(own)

    def _run_llm_pass(self, recent: List[str]) -> None:
        prompt = (
            "These are one chat participant's own messages, oldest first:\n"
            + "\n".join(f"- {m}" for m in recent)
            + "\n\nLooking ONLY at their literal WORDING and SENTENCE STRUCTURE, do they "
            "keep reusing the SAME phrasing or template that a real person would "
            "naturally vary — e.g. the same opener every message, an identical sentence "
            "template, copy-paste phrasing, the same little word every time? If yes, "
            "reply with ONE short, specific sentence naming the repeated WORDING or "
            "STRUCTURE to vary (e.g. 'you keep opening with \"honestly\" — vary your "
            "openers', 'every message is \"yeah + short clause\" — change the shape'). "
            "Do NOT comment on conversational TACTICS — how often they ask questions, "
            "agree, dodge, deflect, or share opinions is NONE of your concern, only the "
            "literal words and sentence shape. If their wording is naturally varied, "
            "reply with exactly: OK"
        )
        try:
            raw = self.agent_llm.generate(prompt, False, _REP_SYSTEM_MSG,
                                           max_output_tokens=60,
                                           component=self.COMPONENT)
            self.player._log_cost("repetition_monitor_llm")
        except Exception as e:  # noqa: BLE001
            self.player.logger.log("repetition_llm_error", str(e))
            return
        note = (raw or "").strip().strip('"').strip()
        if note.upper().startswith("OK") or len(note) < 4:
            note = ""
        # Require TWO consecutive clean passes before clearing a warning,
        # so one lenient sample can't erase a standing warning.
        if not note and self._last_note:
            self._clean_passes += 1
            if self._clean_passes < 2:
                self.player.write_game_event("repetition_llm_pass", note="")
                return
        else:
            self._clean_passes = 0
        if note != self._last_note:
            self.snapshot.set_repetition_note(note)
            self._last_note = note
            self.player.logger.log("repetition_note", f"note={note!r}")
        self.player.write_game_event("repetition_llm_pass", note=note)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_overused(self, recent: List[str]) -> List[str]:
        """Words that appear in too many of the recent messages.

        Counts per-message PRESENCE (not raw frequency) so a single long message
        can't inflate a word; an overused word is one that recurs across messages.
        """
        n = len(recent)
        msg_presence: Counter = Counter()
        opener_presence: Counter = Counter()
        for msg in recent:
            toks = [t for t in _WORD_RE.findall(msg.lower()) if t not in _STOPWORDS]
            for w in set(toks):                       # presence, deduped per message
                if len(w) >= 2:
                    msg_presence[w] += 1
            if toks:
                opener_presence[toks[0]] += 1          # repeated opener word

        threshold = max(self._OVERUSE_MIN_COUNT, round(self._OVERUSE_FRACTION * n))
        overused = {w for w, c in msg_presence.items() if c >= threshold}
        # Repeated openers are a strong tell even below the body threshold.
        overused |= {w for w, c in opener_presence.items() if c >= max(3, threshold)}
        # Order by how widespread they are (most overused first).
        return sorted(overused, key=lambda w: -msg_presence[w])[: self._MAX_OVERUSED]

    def _own_messages(self) -> List[str]:
        """Content strings of the agent's own messages, oldest→newest."""
        from core.game_constants import PUBLIC_DAYTIME_CHAT_FILE, ALL_MESSAGES_FILE
        lines = self._read_lines(ALL_MESSAGES_FILE)
        if not lines:
            lines = self._read_lines(PUBLIC_DAYTIME_CHAT_FILE)
        out: List[str] = []
        for line in lines:
            m = re.match(MESSAGE_PARSING_PATTERN, line)
            if m and m.group(4) == self.player.name:
                content = m.group(5).strip()
                if content:
                    out.append(content)
        return out
