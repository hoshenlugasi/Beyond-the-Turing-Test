"""Persona Monitor Sub-Agent (the 5th V5 sub-agent).

Runs every ~10 s.  Each tick it:
  1. Collects the agent's OWN sent messages from the public chat.
  2. If they contain first-person self-references it hasn't processed yet,
     asks the LLM to extract durable factual claims the agent made about
     itself (phone/OS, location, age, occupation, hobbies, stated opinions).
  3. Writes the deduped claim list into snapshot.set_self_claims(...).

The generator and reviewer read `self_claims` from the snapshot so the agent
never contradicts something it already said (e.g. "i'm on android" after
"15 pro i think"). This is the architectural fix for the fatal self-
contradiction tell observed in game 9007.

Cheap: it only calls the LLM when the agent has produced NEW self-referential
messages since the last extraction, so most ticks are no-ops.
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
from llm_players.v6.sub_agents.base import SubAgent


# First-person markers — a cheap pre-filter so we only call the LLM when the
# agent actually said something about itself.
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i've|ive|my|mine|me|myself)\b", re.IGNORECASE
)


class PersonaMonitorAgent(SubAgent):
    interval_seconds: float = 10.0
    COMPONENT = "persona_monitor"

    # Cap how many of the agent's own messages we feed the extractor.
    _MAX_OWN_MSGS: int = 40

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        # Number of own-messages already folded into the claim set; lets us
        # skip the LLM call when nothing new was said.
        self._processed_own_count: int = 0

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        own_msgs = self._own_messages()
        if len(own_msgs) <= self._processed_own_count:
            return  # nothing new since last extraction

        # Only the self-referential ones are worth extracting from.
        first_person = [m for m in own_msgs if _FIRST_PERSON_RE.search(m)]
        if not first_person:
            self._processed_own_count = len(own_msgs)
            return

        tick_start = time.time()
        prompt = self._build_extract_prompt(first_person)
        self.player.logger.log("v5_persona_extract_prompt", prompt)
        raw = self.agent_llm.generate(
            prompt, False, self.player.get_system_info_message(),
            max_output_tokens=256, component=self.COMPONENT,
        )
        self.player._log_cost("persona_monitor")
        claims = self._parse_claims(raw)

        # Persist even an empty list is fine — but only overwrite when we got
        # a usable parse (avoid clobbering good claims with a parse failure).
        if claims is not None:
            self.snapshot.set_self_claims(claims)
            self.player.logger.log(
                "v5_persona_claims", f"claims={claims!r}",
            )
        self._processed_own_count = len(own_msgs)
        self.player.write_game_event(
            "v5_persona_monitor_tick",
            own_msgs=len(own_msgs),
            first_person=len(first_person),
            claims=claims or [],
            latency_s=round(time.time() - tick_start, 3),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

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
        return out[-self._MAX_OWN_MSGS:]

    def _build_extract_prompt(self, own_msgs: List[str]) -> str:
        joined = "\n".join(f"- {m}" for m in own_msgs)
        return (
            "Below are messages YOU sent earlier in a casual group chat.\n"
            f"{joined}\n\n"
            "Extract durable IDENTITY facts you stated about YOURSELF — concrete "
            "things you must stay consistent about: city/country/origin (even when "
            "said plainly like 'from tel aviv'), phone/OS, age, job/studies, hobbies, "
            "pets, real likes/dislikes ('i hate X', 'i'm a Y fan').\n"
            "CRITICAL — do NOT include EVASIONS or COVER-STORIES as facts: "
            "'i don't follow the news', 'not a big sports fan', 'idk', 'i missed it', "
            "'not really into that'. Those are dodges, not identity — recording them "
            "would lock you into not being able to answer later. Skip them entirely. "
            "Also ignore greetings, reactions, and anything about other players.\n"
            # V6 (P0b): the agent's timezone / local time / country are FIXED by
            # the persona and tracked separately as immutable facts. Extracting a
            # spoken time-of-day as a 'claim' is what locked game 9043 into a
            # wrong "2:30pm" for an Israeli at 07:33 AM. Never capture these.
            "Do NOT extract timezone, current local time, time-of-day, or which "
            "country/city you're physically in right now as claims — those are "
            "fixed persona facts handled elsewhere, not things to record here. "
            "(A durable origin like 'from Tel Aviv' is still fine; a momentary "
            "'it's 2:30pm here' is NOT.)\n"
            "Reply with ONLY a JSON array of short identity strings, e.g.:\n"
            '["from Tel Aviv", "uses Android", "studies CS", "i\'m 23"]\n'
            "If you stated no durable identity facts, reply with: []"
        )

    def _parse_claims(self, raw: str) -> Optional[List[str]]:
        """Parse the LLM's JSON array. Returns None on unparseable output so
        the caller leaves existing claims untouched."""
        raw = (raw or "").strip()
        # Direct array
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
        # Extract the first [...] block
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return None
