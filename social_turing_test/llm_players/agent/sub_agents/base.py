from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from social_turing_test.llm_players.agent.agent import SocialTuringTestAgent
    from social_turing_test.llm_players.agent.shared_state import SharedSnapshot


class SubAgent(ABC):
    """Background worker that polls at a fixed interval and writes Shared State slices.

    Lifecycle:
        agent.start()  — spin up daemon thread
        agent.tick()   — called by the thread; subclasses implement this
        agent.stop()   — signal stop + join (waits up to 2 s)

    In simulation mode the thread is never started; the simulation player calls
    tick() synchronously between messages instead.
    """

    # Subclasses override to set their polling cadence.
    interval_seconds: float = 30.0

    # Log routing: each sub-agent's notes and LLM calls land in a focused
    # per-component file (logs/{name}_{COMPONENT}.txt). Subclasses override.
    COMPONENT: str = "subagent"

    @property
    def agent_llm(self):
        """The LLM wrapper background sub-agents should use: the dedicated
        sub-agent model when llm_config["sub_agent_model"] is set (admin UI
        "Sub-agents" card), else the player's main wrapper."""
        return getattr(self.player, "sub_llm", None) or self.player.llm

    def __init__(self, player: SocialTuringTestAgent, snapshot: SharedSnapshot) -> None:
        self.player = player        # for .llm, .logger, .game_dir, .name, etc.
        self.snapshot = snapshot
        self._stop_event: Event = Event()
        self._thread: Optional[Thread] = None
        # ── Throttling (reduces 429 RESOURCE_EXHAUSTED in multi-agent games) ──
        # In agent-vs-agent runs, N agents × 2 sub-agents + N generators share
        # one Vertex quota. Scale the polling cadence and stagger the first
        # tick so the workers don't all fire at t=0.
        cfg = getattr(player, "llm_config", {}) or {}
        scale = float(cfg.get("subagent_interval_scale", 1.0))
        if scale > 0:
            self.interval_seconds = type(self).interval_seconds * scale
        self._startup_jitter = bool(cfg.get("subagent_startup_jitter", True))

    # ── Logging ─────────────────────────────────────────────────────────────

    def log(self, operation: str, content) -> None:
        """Write a note to this sub-agent's focused log file (and the master)."""
        self.player.logger.log(operation, content, component=self.COMPONENT)

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def tick(self) -> None:
        """One unit of work: observe state, optionally call LLM, write slice."""

    def lobby_tick(self) -> None:
        """One-time bounded work allowed while the game is still in the lobby
        (pre-START). Default: no-op — sub-agents stay idle until the admin
        starts the game, so a pre-spawned agent doesn't burn API quota during
        the lobby. Internet Context overrides this to seed world headlines so
        the agent has common-knowledge baseline ready before the game begins.
        """
        return

    def _game_started(self) -> bool:
        """True once the game has actually started (admin clicked START and the
        manager wrote the game-start time). While False, the agent is in the
        lobby and must not run its normal tick loop. Fail-open on error so a
        missing helper never permanently silences the agent.
        """
        try:
            from social_turing_test.core.game_status_checks import all_players_joined
            return bool(all_players_joined(self.player.game_dir))
        except Exception:
            return True

    # ── Thread lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            daemon=True,
            name=type(self).__name__,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        # Lobby gate: an agent may be pre-spawned while the admin lobby is
        # still open (so the admin can see "AI ready" before START). During the
        # lobby the sub-agents stay idle - they do NOT run their normal tick
        # loop, which avoids burning API quota while players are still joining.
        # Each agent gets exactly one chance to run `lobby_tick()` (default
        # no-op; Internet Context seeds world headlines), then waits for START.
        if not self._game_started():
            try:
                self.lobby_tick()
            except Exception as exc:
                self.log(f"{type(self).__name__}_lobby_error", str(exc))
            while not self._game_started():
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

        # Stagger the first tick by a random fraction of one interval so that,
        # across many agents, the sub-agents' LLM calls don't all land at once
        # at game start (the worst burst for Vertex quota / 429 errors).
        if self._startup_jitter and self.interval_seconds > 0:
            initial_delay = random.uniform(0.0, self.interval_seconds)
            slept = 0.0
            while slept < initial_delay:
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)
                slept += 0.5

        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.log(f"{type(self).__name__}_error", str(exc))
            # Sleep in 0.5 s slices so stop() is responsive
            slices = max(1, int(self.interval_seconds / 0.5))
            for _ in range(slices):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

    # ── Shared helpers (available to all sub-agents) ───────────────────────

    def _read_text(self, filename: str) -> Optional[str]:
        """Read a file from game_dir; return None on any error."""
        path: Path = self.player.game_dir / filename
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None

    def _read_lines(self, filename: str) -> List[str]:
        text = self._read_text(filename)
        if text is None:
            return []
        return [l for l in text.splitlines() if l.strip()]

    def _recent_messages(self, n: int = 10) -> List[str]:
        """Last n non-empty lines from public_daytime_chat.txt (or all_messages.txt)."""
        from social_turing_test.core.game_constants import PUBLIC_DAYTIME_CHAT_FILE, ALL_MESSAGES_FILE
        lines = self._read_lines(ALL_MESSAGES_FILE)
        if not lines:
            lines = self._read_lines(PUBLIC_DAYTIME_CHAT_FILE)
        return lines[-n:] if lines else []

    def _read_conversation_topic(self) -> Optional[str]:
        from social_turing_test.core.game_constants import CONVERSATION_TOPIC_FILE
        return self._read_text(CONVERSATION_TOPIC_FILE)

    def _read_current_phase(self) -> Optional[str]:
        """Return the current phase string from phase_status.txt, e.g. 'Discussion 2'."""
        from social_turing_test.core.game_constants import PHASE_STATUS_FILE
        text = self._read_text(PHASE_STATUS_FILE)
        return text if text else None
