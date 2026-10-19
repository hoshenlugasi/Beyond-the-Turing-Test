"""Round Summarizer background worker.

Runs every ~10 s.  Each tick it checks whether the current game phase has
changed (Daytime → Voting, or Voting → Daytime).  When a Daytime phase
ends (i.e. voting begins), it summarises the messages from that phase into
one short line and appends it to snapshot.round_summary_state.

Phase detection works purely from `phase_status.txt` (which contains the
string "Daytime", "Voting", or "Daytime_Voting") and the public chat file.
No timestamps or phase-number files are required.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from llm_players.agent.sub_agents.base import SubAgent


# Neutral summariser system message — feeding the player persona here makes the
# model role-play and emit chat prose instead of summarising. Keep it on-task.
_PS_SUMMARY_SYSTEM_MSG = (
    "You summarise a chat discussion into ONE short factual sentence. You do NOT "
    "role-play any chat persona and you do NOT write a chat message — you only "
    "report what happened in the discussion."
)


class RoundSummarizerAgent(SubAgent):
    interval_seconds: float = 10.0
    COMPONENT = "round_summarizer"

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        self._last_phase: Optional[str] = None   # "Daytime" | "Voting" | None
        self._daytime_round: int = 0             # how many daytime rounds seen
        self._msgs_at_last_summary: int = 0      # chat length when we last summarised
        self._pending_msgs: List[str] = []       # messages accumulated during current daytime
        self._last_elim_count: int = 0           # eliminations seen (round-end signal)

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        current_phase = self._read_phase_label()
        all_msgs = self._read_all_messages()

        if current_phase is None:
            return

        tick_start = time.time()
        self.player.write_game_event(
            "round_summary_tick_begin",
            phase_label=current_phase,
            pending_msgs_count=len(self._pending_msgs),
            wall_t=tick_start,
        )
        # Use try/finally so every begin pairs with an end, even when the
        # first-tick init early-returns.
        try:
            self._tick_body(current_phase, all_msgs)
        finally:
            self.player.write_game_event(
                "round_summary_tick_end",
                phase_label=current_phase,
                pending_msgs_count=len(self._pending_msgs),
                tick_latency_s=round(time.time() - tick_start, 3),
            )

    def _tick_body(self, current_phase: str, all_msgs: List[str]) -> None:
        # ── First tick: initialise ──────────────────────────────────────
        # When the agent starts mid-game (e.g. in simulation, or after a
        # restart), seed the pending accumulator with the messages that were
        # already in the chat. Otherwise the first round-end would have nothing
        # to summarise. We do NOT emit anything yet.
        if self._last_phase is None:
            self._last_phase = current_phase
            self._msgs_at_last_summary = len(all_msgs)
            self._last_elim_count = self._elimination_count()
            self._pending_msgs.extend(all_msgs)
            return

        # ── Accumulate any new chat since the last checkpoint ───────────
        new_msgs = all_msgs[self._msgs_at_last_summary:]
        if new_msgs:
            self._pending_msgs.extend(new_msgs)
            self._msgs_at_last_summary = len(all_msgs)

        # ── Round-end detection ────────────────────────────────────────
        # The "Voting" phase flag is written only briefly (stt_main strips it
        # back out after the tally), so polling can miss it. The elimination announcement is
        # persistent, so use a new elimination as the robust round-end signal;
        # keep the phase flag as a secondary trigger for when it is caught.
        elim_count = self._elimination_count()
        round_ended = (
            elim_count > self._last_elim_count
            or (self._last_phase == "Daytime" and "Voting" in current_phase)
        )
        self._last_phase = current_phase
        if round_ended:
            self._last_elim_count = max(elim_count, self._last_elim_count)
            msgs_to_summarise = list(self._pending_msgs)
            self._pending_msgs = []
            self._msgs_at_last_summary = len(all_msgs)
            if len(msgs_to_summarise) >= 3:
                self._daytime_round += 1
                self._summarise_and_store(
                    f"Discussion {self._daytime_round}", msgs_to_summarise)

    # ── LLM call ──────────────────────────────────────────────────────────

    def _summarise_and_store(self, label: str, msgs: List[str]) -> None:
        prompt = self._build_summary_prompt(label, msgs)
        self.player.logger.log("round_summary_prompt", prompt)
        gen_start = time.time()
        summary = self.agent_llm.generate(
            prompt,
            False,
            _PS_SUMMARY_SYSTEM_MSG,
            max_output_tokens=200,
            component=self.COMPONENT,
        )
        gen_latency = time.time() - gen_start
        self.player._log_cost("round_summarizer")
        summary = summary.strip()
        self.player.logger.log(
            "round_summary_response",
            f"summary={summary!r} latency={gen_latency:.2f}s",
        )
        self.player.write_game_event(
            "round_summary_response",
            phase=label,
            raw=summary[:1500],
            summary_text=summary,
            latency_s=round(gen_latency, 3),
        )

        self.snapshot.append_round_summary(label, summary)
        self.player.logger.log(
            "round_summary_update",
            f"phase={label!r} summary={summary!r}",
        )
        self.player.write_game_event(
            "round_summary_tick",
            phase=label,
            n_messages=len(msgs),
            summary=summary,
        )

    # ── Prompt builder ────────────────────────────────────────────────────

    def _build_summary_prompt(self, label: str, msgs: List[str]) -> str:
        chat_text = "\n".join(msgs[-30:])   # cap to last 30 to keep prompt small
        name = self.player.name
        return (
            f"This is the chat log from {label} of a Social Turing Test game.\n"
            f"The game is a chat where players try to identify the AI among them.\n"
            f"YOU are the player named {name} (the hidden AI). Summarise FROM YOUR "
            f"perspective in ONE short sentence (max 25 words):\n"
            f"{chat_text}\n\n"
            f"Include, agent-centric: (1) is {name} (you) under suspicion, and by "
            f"whom; (2) WHAT are they testing you on (e.g. local/current-events "
            f"knowledge, opinions, consistency); (3) the overall mood / any key "
            f"moment. Example: 'They grilled you on local current events and you "
            f"kept dodging; Skylar suspicious; mood tense.'\n"
            "Reply with ONLY the summary sentence, nothing else."
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _elimination_count(self) -> int:
        """How many players have been voted out so far — a persistent round-end
        signal (the brief 'Voting' phase flag is easily missed by a 15 s poll).
        Counts the Game-Manager's '<name> was eliminated' announcements."""
        from core.game_constants import PUBLIC_MANAGER_CHAT_FILE
        n = 0
        for line in self._read_lines(PUBLIC_MANAGER_CHAT_FILE):
            if re.search(r"was eliminated", line):
                n += 1
        return n

    def _read_phase_label(self) -> Optional[str]:
        """Return 'Daytime', 'Voting', or 'Daytime_Voting' from phase_status.txt."""
        from core.game_constants import PHASE_STATUS_FILE, DAYTIME, VOTING_TIME
        raw = self._read_text(PHASE_STATUS_FILE)
        if not raw:
            return None
        if VOTING_TIME in raw and DAYTIME in raw:
            return "Daytime_Voting"
        if VOTING_TIME in raw:
            return "Voting"
        return "Daytime"

    def _read_all_messages(self) -> List[str]:
        """All non-empty public chat lines (both file layout variants)."""
        from core.game_constants import PUBLIC_DAYTIME_CHAT_FILE, ALL_MESSAGES_FILE
        lines = self._read_lines(ALL_MESSAGES_FILE)
        if not lines:
            lines = self._read_lines(PUBLIC_DAYTIME_CHAT_FILE)
        return lines
