"""Social State Tracker background worker.

Runs every 6 seconds. Each tick it:
  1. Reads the recent PUBLIC chat.
  2. If there are new messages carrying a "suspicion" signal (the agent's own
     name, or words like ai/bot/sus/accuse/vote), asks the LLM to extract the
     agent's own social standing as a compact JSON object:
         {suspects_me, i_pointed_at, voted_out, prime_suspect}
    3. Writes it via snapshot.set_social_state(...).

The generator reads the rendered one-liner ("Riley has suspected you; you
pointed at Jordan") so the agent reacts to its accusers and stays consistent
with the stance it already took — instead of acting oblivious.

Distinct from the Round Summarizer (one per-round recap): Social State Tracker
is ABOUT THE AGENT ITSELF and is actionable every turn.

Human-fair: reads only PUBLIC information — the public chat and the vote tally
the Game-Manager already echoes into chat (which respects anonymous_voting). It
never reads other players' private {name}_vote.txt files.

Cheap: most ticks are no-ops (no new messages, or nothing suspicion-related).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from social_turing_test.llm_players.agent.sub_agents.base import SubAgent


# Cheap pre-filter: only spend an LLM call when the recent chat actually looks
# social/suspicion-related. The agent's own name is added at runtime.
# Includes indirect AI-probe phrasings ("describe your chain of thought",
# "how would an ai say hi", "prove you're human"). The LLM prompt's own guard
# still prevents these from being mis-logged as explicit suspects_me - this
# regex only decides WHETHER to spend an LLM call, so the slice (pressure_type,
# group_mood, numeric signals) stays fresh under indirect interrogation.
_SUSPICION_RE = re.compile(
    r"\b(ai|a\.i\.|bot|robot|sus|suspicious|suspect|accuse|accusing|vote|voting|"
    r"voted|fake|not human|the ai|it'?s\s+\w+|"
    r"chain of thought|how would (a|an) (ai|bot)|prove you('?re| are)|"
    r"describe your (thinking|reasoning|process|chain)|are you (really )?(human|a bot|an ai)|"
    r"sounds (robotic|like a bot|like an ai)|programmed|turing)\b",
    re.IGNORECASE,
)


class SocialStateTrackerAgent(SubAgent):
    # Poll every 6s so an explicit accusation ("you are the ai") is
    # reflected in suspects_me / room_vibe within a couple of turns instead of
    # lagging a whole tick behind. Most ticks are still no-ops (the pre-filter
    # only spends an LLM call on new suspicion-bearing messages), so the extra
    # frequency is nearly free.
    interval_seconds: float = 6.0
    COMPONENT = "social_state_tracker"

    # How many recent chat lines to feed the extractor. The high cap lets it
    # read the FULL game history (real games stay well
    # under this); the cap is only a defensive guard against runaway token cost.
    _RECENT_MSGS: int = 500

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        # Track the TOTAL number of chat lines processed (cumulative),
        # NOT the capped recent-window size, so change detection continues
        # after the context window fills.
        self._processed_line_count: int = 0
        # Eliminations already folded in - so an elimination updates voted_out
        # even with no accusation chatter (without spending an LLM call).
        self._last_eliminated: List[str] = []

    def _all_lines(self) -> list:
        """Full (uncapped) chat history, oldest→newest. Source of truth for
        'is there anything new', independent of the LLM context window."""
        from social_turing_test.core.game_constants import PUBLIC_DAYTIME_CHAT_FILE, ALL_MESSAGES_FILE
        return self._read_lines(ALL_MESSAGES_FILE) or self._read_lines(PUBLIC_DAYTIME_CHAT_FILE)

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        all_lines = self._all_lines()
        total = len(all_lines)
        eliminated = self._eliminated_names()
        new_msgs = total > self._processed_line_count
        new_elim = eliminated != self._last_eliminated
        if not new_msgs and not new_elim:
            return  # nothing new since last tick

        # Only the lines we have NOT yet considered drive the suspicion check.
        new_lines = all_lines[self._processed_line_count:] if new_msgs else []
        # Recent window fed to the LLM (token budget) - separate from tracking.
        msgs = all_lines[-self._RECENT_MSGS:]

        had_signal = new_msgs and (
            self._has_suspicion_signal(new_lines)
            or self._high_question_pressure(all_lines)
        )

        if not had_signal:
            # No accusation chatter worth an LLM call. But if a player was just
            # voted out, fold that into the slice deterministically (public info).
            if new_elim:
                mem = self.snapshot.get_social_state()
                mem["voted_out"] = eliminated
                self.snapshot.set_social_state(mem)
            self._processed_line_count = total
            self._last_eliminated = eliminated
            self.player.write_game_event(
                "social_state_tick", msgs=len(msgs), had_signal=False,
                parsed={"voted_out": eliminated} if new_elim else {}, latency_s=0.0,
            )
            return

        tick_start = time.time()
        prompt = self._build_extract_prompt(msgs)
        self.player.logger.log("social_state_prompt", prompt)
        # Needs room for a full 6-key JSON object; the chat generator's tiny
        # max_tokens would truncate it mid-object and every parse would fail.
        raw = self.agent_llm.generate(
            prompt, False, self.player.get_system_info_message(),
            max_output_tokens=300, component=self.COMPONENT,
        )
        self.player._log_cost("social_state_tracker")
        parsed = self._parse_memory(raw)

        if parsed is None:
            # Parse failed - log the raw output so the failure is debuggable
            # instead of vanishing (the event below records parse_ok=False).
            self.player.logger.log(
                "social_state_parse_failed", f"raw={(raw or '')[:600]!r}"
            )

        if parsed is not None:
            # Eliminations are reliable public facts - trust the regex over the
            # LLM for voted_out (union, so the LLM can still add any it inferred).
            parsed["voted_out"] = sorted(set(parsed.get("voted_out", [])) | set(eliminated))
            self.snapshot.set_social_state(parsed)
            self.player.logger.log("social_state_tracker", f"memory={parsed!r}")
        self._processed_line_count = total
        self._last_eliminated = eliminated
        self.player.write_game_event(
            "social_state_tick",
            msgs=len(msgs),
            had_signal=had_signal,
            parse_ok=parsed is not None,
            parsed=parsed or {},
            latency_s=round(time.time() - tick_start, 3),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _eliminated_names(self) -> List[str]:
        """Names voted out so far, parsed from the public vote tally the
        Game-Manager echoes into chat_displayed_full.txt (always public, even
        under anonymous voting). Order preserved, deduped."""
        from social_turing_test.core.game_constants import CHAT_DISPLAYED_FULL_FILE
        out: List[str] = []
        for line in self._read_lines(CHAT_DISPLAYED_FULL_FILE):
            m = re.search(r"eliminated:\s*(.+?)\s*$", line)
            if m:
                name = m.group(1).strip()
                if name and name not in out:
                    out.append(name)
        return out

    def _has_suspicion_signal(self, new_lines) -> bool:
        """True if any of the NEW (unprocessed) lines carry an explicit or
        indirect AI-probe signal, or mention the agent by name. Caller passes
        only the new lines (no window-index slicing)."""
        name = (self.player.name or "").lower()
        for line in new_lines:
            low = line.lower()
            if _SUSPICION_RE.search(low):
                return True
            if name and name in low:
                return True
        return False

    def _high_question_pressure(self, all_lines) -> bool:
        """Focused-interrogation trigger. True when the agent's
        name appears in 3+ of the last 10 messages — that volume signals the
        room is zeroing in on it
        even with no explicit AI keyword. Still only gates the LLM call; the
        extractor's own guard keeps suspects_me from false positives."""
        name = (self.player.name or "").lower()
        if not name:
            return False
        recent = all_lines[-10:]
        return sum(1 for m in recent if name in m.lower()) >= 3

    def _build_extract_prompt(self, msgs) -> str:
        chat_text = "\n".join(msgs[-self._RECENT_MSGS:])
        name = self.player.name
        return (
            "This is the recent chat from a Social Turing Test game (players try "
            "to find the hidden AI by accusing and voting).\n"
            f"YOUR name in this game is: {name}\n\n"
            f"{chat_text}\n\n"
            "From YOUR perspective as " + name + ", extract the current social "
            "picture. Reply with ONLY a JSON object, no prose:\n"
            '{"suspects_me": [names who voiced suspicion of ' + name + '], '
            '"i_pointed_at": [names ' + name + ' accused or leaned toward], '
            '"voted_out": [names already eliminated], '
            '"prime_suspect": "the player most suspected right now, or empty", '
            '"open_threads": [short questions/requests aimed at ' + name + ' that '
            + name + ' has NOT answered yet, e.g. "Dana asked where you would travel"], '
            '"group_mood": "one short phrase for the room mood, e.g. relaxed / '
            'mild suspicion toward Alex / tense", '
            '"pressure_type": "HOW ' + name + ' is being tested right now and how '
            'it is going — e.g. \'being grilled on local/current-events knowledge '
            'and keeps dodging\', \'asked for opinions and staying too neutral\', '
            '\'none\'", '
            '"players": {"PlayerName": "one short read on THIS person: their stance '
            'toward ' + name + ' (suspicious / friendly / neutral), their writing/'
            'communication style (message length, tone, formality, punctuation, and '
            'whether they answer directly or deflect — so ' + name + ' can mirror them), '
            'and any notable thing they claimed about themselves — e.g. \'thinks you\\\'re '
            'the AI, short lowercase msgs, deflects, said he\\\'s from haifa\'"}, '
            '"room_norm": "is the chat one where EVERYONE answers every question, or '
            'do people pick and choose what to respond to? answer \'everyone answers\' '
            'or \'selective\'", '
            # The CURRENT mode of the room, so the agent can mirror it.
            '"room_vibe": "the room\'s current mode, EXACTLY one of: '
            '\'casual\' (just chatting / joking / small talk), '
            '\'probing\' (people asking testing or clever questions to feel out who the AI is), '
            '\'hunting\' (actively naming suspects / pushing to vote someone out), '
            '\'tense\' (heated, confrontational, argument)", '
            # Numeric signals so Behavior Adapter reads numbers, not
            # fragile regex over free-text pressure_type.
            '"interrogation_pressure": <float 0.0-1.0: how intensely is ' + name +
            ' being tested/grilled/probed right now (0 = not at all, 1 = relentless)>, '
            '"attention_on_me": <float 0.0-1.0: what fraction of the recent conversation '
            'is focused on ' + name + '>}\n'
            "Use the exact display names from the chat. Use [] / \"\" when none. "
            "Only include an open thread if it is genuinely unanswered. "
            "Do not invent anyone or anything not in the chat.\n"
            # Precision guard: addressing the agent or quizzing it is NOT suspicion;
            # only count an EXPLICIT doubt that the agent is human.
            "IMPORTANT — \"suspects_me\" is ONLY for players who EXPLICITLY doubt "
            "that " + name + " is human: e.g. they call " + name + " the AI/bot/"
            "fake/not-human, say they will vote " + name + ", or directly accuse "
            + name + ". Merely asking " + name + " a question, addressing "
            + name + " by name, quizzing general knowledge (scores, the time, the "
            "date), or being curious is NOT suspicion — do NOT list those players. "
            "When in doubt, leave them out."
        )

    def _parse_memory(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM's JSON object. Returns None on unparseable output so
        the caller leaves the existing slice untouched."""
        raw = (raw or "").strip()
        obj = self._loads_obj(raw)
        if obj is None:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                obj = self._loads_obj(m.group())
        if obj is None:
            return None
        return self._normalize(obj)

    @staticmethod
    def _loads_obj(text: str) -> Optional[dict]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    # Cap how many open threads we keep - only the freshest are actionable.
    _MAX_OPEN_THREADS: int = 3
    # Cap per-player sketches (typical games have 3-6 others).
    _MAX_PLAYERS: int = 6

    @classmethod
    def _normalize(cls, obj: dict) -> Dict[str, Any]:
        def _strs(key: str):
            val = obj.get(key, [])
            if not isinstance(val, list):
                return []
            return [str(x).strip() for x in val if str(x).strip()]

        def _clamp01(key: str) -> float:
            """Parse a 0.0-1.0 float; tolerate strings/ints/garbage → 0.0."""
            try:
                v = float(obj.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, v))

        prime = obj.get("prime_suspect", "")
        mood = obj.get("group_mood", "")
        pressure = obj.get("pressure_type", "")
        pressure = str(pressure).strip() if pressure else ""
        if pressure.lower() in ("none", "n/a", "nothing"):
            pressure = ""
        # Compact per-player sketch: {name: "one short read"} - cap to keep it
        # token-cheap and drop empty/placeholder entries.
        players_raw = obj.get("players", {})
        players: Dict[str, str] = {}
        if isinstance(players_raw, dict):
            for k, v in players_raw.items():
                name_k = str(k).strip()
                read = str(v).strip()
                if name_k and read and read.lower() not in ("none", "n/a", ""):
                    players[name_k] = read[:220]
                if len(players) >= cls._MAX_PLAYERS:
                    break
        return {
            "suspects_me":   _strs("suspects_me"),
            "i_pointed_at":  _strs("i_pointed_at"),
            "voted_out":     _strs("voted_out"),
            "prime_suspect": str(prime).strip() if prime else "",
            "open_threads":  _strs("open_threads")[-cls._MAX_OPEN_THREADS:],
            "group_mood":    str(mood).strip() if mood else "",
            "pressure_type": pressure,
            "players":       players,
            "room_norm":     (str(obj.get("room_norm", "")).strip().lower() or ""),
            # Validated against the small enum; "" when unknown.
            "room_vibe":     (lambda v: v if v in ("casual", "probing", "hunting", "tense") else "")(
                                 str(obj.get("room_vibe", "")).strip().lower()),
            # Numeric signals consumed by BehaviorAdapterAgent._heat().
            "interrogation_pressure": _clamp01("interrogation_pressure"),
            "attention_on_me":        _clamp01("attention_on_me"),
        }
