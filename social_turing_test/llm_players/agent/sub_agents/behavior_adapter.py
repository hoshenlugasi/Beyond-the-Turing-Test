"""Behavior Adapter background worker.

Runs every ~10 s. It synthesises ONE evolving play-style directive for the
current stage of the game and writes it via snapshot.set_behavior_advice(...).
The generator injects it as a "play-style note for this stage of the game".

Hybrid design (cheap):
  • Every tick it recomputes deterministic SIGNALS from public state —
      - players_left      (remaining_players.txt, fallback = distinct speakers)
    - suspicion_on_self (len of Social State Tracker's suspects_me)
      - talk_share        (the agent's share of the last ~20 messages)
    and builds a rule-based baseline directive from thresholds.
  • The LLM is called to POLISH that directive into one natural line ONLY at a
    major game-stage transition — when players_left drops (someone was voted
    out) — or for the very first directive. This bounds LLM cost to ~once per
    round while keeping the signals fresh every tick.

If a threshold crosses between transitions (e.g. suspicion just reached 2) the
rule-based baseline is refreshed without an LLM call, so the directive still
reacts promptly to rising heat.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional

from social_turing_test.core.game_constants import (
    MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME, REMAINING_PLAYERS_FILE,
    PLAYER_NAMES_FILE,
)
from social_turing_test.llm_players.agent.sub_agents.base import SubAgent
# Reuse the generator's tell-detectors so the game-long diagnostic and the
# per-turn generator guard agree on what counts as a bare agreement / deflection.
from social_turing_test.llm_players.agent.nodes.generator_node import _is_bare_agreement, _DEFLECTION_MARKERS


# Suspicion count at/above which the agent should actively lay low.
_HEAT_THRESHOLD: int = 2
# Player count at/below which every message is heavily scrutinised (endgame).
# "Endgame" = the final few players, when every message is heavily scrutinised.
# This is an ABSOLUTE floor (3 or fewer remaining) AND requires that at least one
# player has already been eliminated (players_left < starting roster) - otherwise a
# small 4-player game's OPENING would be misread as endgame from turn 1.
_ENDGAME_PLAYERS: int = 3
# Talk-share bands over the recent window.
_TALK_SHARE_HIGH: float = 0.40
_TALK_SHARE_LOW: float = 0.10
# How many recent messages define "talk share".
_TALK_WINDOW: int = 20
# Diagnostic window over the agent's OWN recent messages, + thresholds.
_OWN_WINDOW: int = 10
_MIN_OWN_FOR_DIAG: int = 3        # need a few messages before judging a pattern
_AGREEMENT_RATIO_HI: float = 0.5  # half+ of recent messages are bare agreement
_DEFLECTION_RATIO_HI: float = 0.4
_QUESTION_RATE_LO: float = 0.1    # rarely asks/initiates
# Interrogation pressure arrives from Social State Tracker as a numeric
# field (interrogation_pressure 0.0-1.0), avoiding fragile free-text matching.
# These thresholds map the float onto the discrete heat scale.
_INTERROGATION_PRESSURE_HIGH: float = 0.6
_INTERROGATION_PRESSURE_MILD: float = 0.3


class BehaviorAdapterAgent(SubAgent):
    interval_seconds: float = 10.0
    COMPONENT = "behavior_adapter"

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        self._last_players_left: Optional[int] = None
        self._last_baseline: str = ""
        self._cached_directive: str = ""
        # Starting roster size, used to tell "opening" from "endgame". Resolved
        # lazily (and cached) from PLAYER_NAMES_FILE, with a max-observed fallback.
        self._start_players: Optional[int] = None

    # ── Main tick ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        tick_start = time.time()
        players_left = self._players_left()
        suspicion = self._heat()
        talk_share = self._talk_share()
        signals = self._behavior_signals(talk_share)

        # Cadence signal for the SCHEDULER. "over" when the agent's
        # talk-share is above its fair share (1/N), "under" when well below.
        # The scheduler folds this into its prompt to hold back when
        # over-participating and not directly addressed. talk_share is a ratio
        # (deterministic) - the room-norm half comes from Social State Tracker.
        part = signals.get("participation_assessment", "balanced")
        level = ("over" if part == "talking a lot"
                 else "under" if part == "slightly too quiet"
                 else "balanced")
        self.snapshot.set_cadence({
            "level": level,
            "talk_share": round(talk_share, 2),
            "reason": signals.get("recommendation", "") or part,
        })

        # Composite baseline = stage/heat/talk directive + top diagnostic warnings
        # + one recommendation. Diagnostics fold into the SAME one-line directive
        # so the generator prompt doesn't bloat. A diagnostic threshold crossing
        # changes this string → cheap refresh between transitions (no LLM).
        baseline = self._baseline_directive(players_left, suspicion, talk_share)
        # Warnings already carry their own corrective ("...- add something specific"),
        # so fold the top 1-2 in directly; the standalone recommendation stays in
        # the structured event (research/UI) to avoid duplicating it in the line.
        diag_bits = list(signals["warnings"][:2])
        baseline = "; ".join([baseline] + diag_bits) if diag_bits else baseline

        # Major transition: player count dropped (elimination) or first run.
        is_transition = (
            self._last_players_left is None
            or (players_left is not None
                and self._last_players_left is not None
                and players_left < self._last_players_left)
        )

        llm_refreshed = False
        directive = self._cached_directive
        if is_transition:
            polished = self._llm_polish(players_left, suspicion, talk_share, baseline)
            if polished:
                directive = polished
                llm_refreshed = True
            else:
                directive = baseline  # fall back to the rule-based line
        elif baseline != self._last_baseline:
            # Threshold crossed between transitions - refresh cheaply.
            directive = baseline

        if directive and directive != self._cached_directive:
            self.snapshot.set_behavior_advice(directive)
            self.player.logger.log("behavior_advice", f"advice={directive!r}")

        self._cached_directive = directive
        self._last_baseline = baseline
        if players_left is not None:
            self._last_players_left = players_left

        self.player.write_game_event(
            "behavior_tick",
            players_left=players_left,
            suspicion_on_self=suspicion,
            talk_share=round(talk_share, 3),
            llm_refreshed=llm_refreshed,
            warnings=signals["warnings"],
            recommendation=signals["recommendation"],
            participation_assessment=signals["participation_assessment"],
            agreement_ratio=signals["agreement_ratio"],
            deflection_ratio=signals["deflection_ratio"],
            latency_s=round(time.time() - tick_start, 3),
        )

    # ── Signals ─────────────────────────────────────────────────────────────

    def _players_left(self) -> Optional[int]:
        lines = self._read_lines(REMAINING_PLAYERS_FILE)
        if lines:
            return len(lines)
        # No authoritative remaining-players file yet (early game): everyone is
        # still in, so fall back to the FULL roster, not distinct speakers.
        # Distinct-speaker counting undercounts before everyone has spoken.
        # Prefer the roster file to avoid misclassifying the opening as endgame.
        names = self._read_lines(PLAYER_NAMES_FILE)
        if names:
            return len(names)
        # Last resort only: distinct non-manager speakers seen in chat so far.
        speakers = set()
        for line in self._recent_messages(200):
            m = re.match(MESSAGE_PARSING_PATTERN, line)
            if m and m.group(4) != GAME_MANAGER_NAME:
                speakers.add(m.group(4))
        return len(speakers) or None

    def _starting_player_count(self) -> Optional[int]:
        """Authoritative starting roster size (cached).

        Prefers PLAYER_NAMES_FILE (the full roster written at game setup). Falls
        back to the most players ever seen remaining, so a tiny game's opening is
        never mistaken for its endgame. Defensive against missing game context so
        unit tests can call _baseline_directive on a bare instance.
        """
        cached = getattr(self, "_start_players", None)
        try:
            names = self._read_lines(PLAYER_NAMES_FILE)
        except Exception:
            names = None
        if names:
            cached = max(cached or 0, len(names))
        else:
            try:
                current = self._players_left()
            except Exception:
                current = None
            if current is not None:
                cached = max(cached or 0, current)
        self._start_players = cached
        return cached

    def _talk_share(self) -> float:
        msgs = self._recent_messages(_TALK_WINDOW)
        mine = total = 0
        for line in msgs:
            m = re.match(MESSAGE_PARSING_PATTERN, line)
            if not m or m.group(4) == GAME_MANAGER_NAME:
                continue
            total += 1
            if m.group(4) == self.player.name:
                mine += 1
        return (mine / total) if total else 0.0

    def _heat(self) -> int:
        """Suspicion-on-self that counts ACTIVE INTERROGATION, not just explicit
        accusers. Active testing / unanswered directed questions
        are heat in their own right.
        """
        sm = self.snapshot.get_social_state()
        heat = len([s for s in (sm.get("suspects_me") or []) if s])
        open_threads = len([t for t in (sm.get("open_threads") or []) if t])
        # Read the numeric interrogation_pressure instead of regex-
        # matching the free-text pressure_type. Falls back to 0.0 when absent.
        try:
            interrog = float(sm.get("interrogation_pressure", 0.0) or 0.0)
        except (TypeError, ValueError):
            interrog = 0.0
        if interrog >= _INTERROGATION_PRESSURE_HIGH:
            heat = max(heat, _HEAT_THRESHOLD)
        elif interrog >= _INTERROGATION_PRESSURE_MILD:
            heat = max(heat, 1)
        # Open threads are unanswered conversational hooks, NOT inherently
        # suspicion. A generic non-directed greeting ("how's everyone") creates an
        # open thread but means nobody is eyeing you. Only let open threads RAISE
        # heat when at least one player already actually suspects this agent;
        # otherwise they shouldn't manufacture a "someone's eyeing you" signal.
        if heat >= 1:
            if open_threads >= 2:
                heat = max(heat, _HEAT_THRESHOLD)
            elif open_threads == 1:
                heat = max(heat, 1)
        return heat

    def _own_recent(self) -> List[str]:
        """The agent's own recent message contents, oldest→newest."""
        out: List[str] = []
        for line in self._recent_messages(60):
            m = re.match(MESSAGE_PARSING_PATTERN, line)
            if m and m.group(4) == self.player.name:
                content = m.group(5).strip()
                if content:
                    out.append(content)
        return out[-_OWN_WINDOW:]

    def _behavior_signals(self, talk_share: float) -> dict:
        """Game-long behavioural diagnostics over the agent's own messages.

        Complements the per-turn generator guards (same detectors) by giving a
        higher-level 'you've been doing X all game' read → warnings/rec/assessment.
        """
        own = self._own_recent()
        n = len(own)
        if n >= _MIN_OWN_FOR_DIAG:
            agreement_ratio = sum(1 for m in own if _is_bare_agreement(m)) / n
            deflection_ratio = sum(
                1 for m in own if any(d in m.lower() for d in _DEFLECTION_MARKERS)
            ) / n
            question_rate = sum(1 for m in own if "?" in m) / n
        else:
            agreement_ratio = deflection_ratio = question_rate = 0.0

        warnings: List[str] = []
        if n >= _MIN_OWN_FOR_DIAG:
            if agreement_ratio >= _AGREEMENT_RATIO_HI:
                warnings.append("too many one-word agreements - add something specific")
            if deflection_ratio >= _DEFLECTION_RATIO_HI:
                warnings.append("deflecting too often - just answer or guess casually")
            if talk_share <= _TALK_SHARE_LOW and question_rate <= _QUESTION_RATE_LO:
                warnings.append("barely engaging - start a topic or ask someone something")

        # Judge participation RELATIVE to fair share (1/N), not an
        # absolute 0.40. "over" = meaningfully
        # above fair; "under" = well below. Absolute floor/ceiling keep a tiny
        # 2-player endgame from being hair-trigger.
        try:
            n_players = self._players_left() or 4
        except Exception:
            n_players = 4
        fair = 1.0 / max(2, n_players)
        over_thresh = max(fair * 1.2, _TALK_SHARE_LOW + 0.05)
        under_thresh = min(fair * 0.5, _TALK_SHARE_LOW)
        if talk_share >= over_thresh:
            participation = "talking a lot"
        elif talk_share <= under_thresh:
            participation = "slightly too quiet"
        else:
            participation = "balanced"

        recommendation = warnings[0].split(" - ")[-1].strip() if warnings else ""
        return {
            "agreement_ratio":  round(agreement_ratio, 2),
            "deflection_ratio": round(deflection_ratio, 2),
            "question_rate":    round(question_rate, 2),
            "own_msgs":         n,
            "warnings":         warnings,
            "recommendation":   recommendation,
            "participation_assessment": participation,
        }

    # ── Rule-based baseline ─────────────────────────────────────────────────

    def _baseline_directive(self, players_left: Optional[int],
                            suspicion: int, talk_share: float) -> str:
        parts: List[str] = []
        # Game stage. "Endgame" only once we're at/under the floor AND at least one
        # player has been eliminated (so a 4-player opening isn't flagged as
        # endgame). See _ENDGAME_PLAYERS.
        start = self._starting_player_count()
        is_endgame = (
            players_left is not None
            and players_left <= _ENDGAME_PLAYERS
            and (start is None or players_left < start)
        )
        if is_endgame:
            parts.append("few players left, so every message is scrutinised: be "
                         "concise and mirror the others closely")
        else:
            parts.append("plenty of players still in — blend in and contribute "
                         "lightly without standing out")
        # Heat. NOTE: under suspicion, going vague/evasive reads as GUILT and gets
        # you voted out. The human move is to
        # stay engaged: give real, specific short answers and opinions, brush off
        # the accusation casually - do NOT dodge or go neutral.
        if suspicion >= _HEAT_THRESHOLD:
            parts.append("you're drawing suspicion: stay engaged and natural, give "
                         "real short answers and opinions, brush off accusations "
                         "casually — do NOT go vague/dodge AND do NOT get defensive "
                         "or argumentative, both read as guilt")
            # Under real heat, a human deflects attention by turning
            # the lens on someone else - not by going quiet. If the room already
            # has another prime suspect, nudge the agent to probe THEM (light,
            # not an accusation). Mirrors what every human player does.
            prime = (self.snapshot.get_social_state().get("prime_suspect") or "").strip()
            if prime and prime.lower() not in ("you", "me", "myself", (self.player.name or "").lower()):
                parts.append(f"redirect some heat: ask {prime} a specific question or "
                             f"share a light read on them — shift attention naturally")
            else:
                parts.append("redirect some heat: turn a question back on whoever's "
                             "pressing you, or probe someone who's been quiet")
        elif suspicion == 1:
            parts.append("someone's eyeing you: stay relaxed and keep giving real "
                         "answers, don't get defensive or evasive")
        # Participation. Under real heat we DON'T tell it to go quiet - "let others
        # lead" dilutes the stay-engaged/redirect push above and reads as the
        # withdrawal that gets a suspect voted out. Only nudge back when calm.
        if talk_share >= _TALK_SHARE_HIGH and suspicion < _HEAT_THRESHOLD:
            parts.append("you've been talking a lot, let others lead for a bit")
        elif talk_share <= _TALK_SHARE_LOW:
            parts.append("you've been quiet, chime in a bit more so you don't seem absent")
        return "; ".join(parts)

    # ── Hybrid LLM polish (only at transitions) ─────────────────────────────

    def _llm_polish(self, players_left: Optional[int], suspicion: int,
                    talk_share: float, baseline: str) -> Optional[str]:
        prompt = (
            "You are coaching a player in a Social Turing Test game (humans try "
            "to vote out the hidden AI). Based on the current situation, give ONE "
            "short, natural coaching line (max 25 words) on how they should play "
            "right now to seem human and avoid getting voted out.\n\n"
            f"Players left: {players_left}\n"
            f"How many players currently suspect this player: {suspicion}\n"
            f"This player's share of recent messages: {round(talk_share * 100)}%\n"
            f"Rule-of-thumb guidance: {baseline}\n\n"
            "Reply with ONLY the coaching line, no quotes, no preamble."
        )
        try:
            self.player.logger.log("behavior_polish_prompt", prompt)
            raw = self.agent_llm.generate(
                prompt, False, self.player.get_system_info_message(),
                max_output_tokens=120, component=self.COMPONENT,
            )
            self.player._log_cost("behavior_adapter")
        except Exception as exc:
            self.player.logger.log("behavior_polish_error", str(exc))
            return None
        line = (raw or "").strip().strip('"').strip()
        # Keep it to a single line.
        line = line.splitlines()[0].strip() if line else ""
        return line or None
