import re
from pathlib import Path

from social_turing_test.core.game_constants import LLM_LOG_FILE_FORMAT, SUBDIR_LOGS, get_current_timestamp
from social_turing_test.llm_players.log_format import render_note

# Free-form entry used by .log() for one-off notes.
NEW_LOG_FORMAT = "# NEW LOG\n## TIME: {time}\n## OPERATION: {operation}\n## CONTENT: {content}\n\n"

# Grouped, human-readable entry for a single LLM call. One block shows the
# system prompt, the user prompt, and the model's output together so the log
# can be read top-to-bottom and learned from.
_RULE = "=" * 80
_SUBRULE = "-" * 80


def _sanitize_component(component: str) -> str:
    """Turn a component label into a safe filename fragment."""
    frag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(component)).strip("_")
    return frag or "misc"


class Logger:
    """Per-AI logger.

    Writes a combined chronological master log (``logs/{name}_log.txt``) AND,
    when a ``component`` is given, a focused per-component file
    (``logs/{name}_{component}.txt``) so each sub-agent / the generator /
    the scheduler / the reviewer can be read in isolation.
    """

    def __init__(self, name: str, game_dir: Path):
        self.name = name
        self.game_dir = Path(game_dir)
        self.logs_dir = self.game_dir / SUBDIR_LOGS
        self.master_file = self.game_dir / LLM_LOG_FILE_FORMAT.format(name)
        # Public alias for the master log path.
        self.log_file = self.master_file
        # Ensure the logs/ subfolder exists so the
        # first append doesn't fail on a freshly-copied/sim game directory.
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ── internals ──────────────────────────────────────────────────────────

    def _component_file(self, component: str) -> Path:
        return self.logs_dir / f"{self.name}_{_sanitize_component(component)}.txt"

    @staticmethod
    def _append(path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            # Observability must never break the game loop.
            pass

    # ── public API ─────────────────────────────────────────────────────────

    def log(self, operation, content, component: str | None = None) -> None:
        """Free-form note. Always goes to the master log in the raw
        ``# NEW LOG`` form (the offline analysis parses it); when a
        ``component`` is given it is ALSO appended to that component file as a
        compact one-liner so the readable per-component view stays uncluttered."""
        ts = get_current_timestamp()
        entry = NEW_LOG_FORMAT.format(time=ts, operation=operation, content=content)
        self._append(self.master_file, entry)
        if component:
            self._append(self._component_file(component), render_note(ts, operation, content))

    def log_call(
        self,
        component: str,
        system_prompt: str,
        user_prompt: str,
        output: str,
        *,
        latency_s: float | None = None,
        cost: float | None = None,
        meta: dict | None = None,
    ) -> None:
        """Record one LLM call as a single grouped, readable block in both the
        component file and the master log."""
        bits = []
        if latency_s is not None:
            bits.append(f"latency {latency_s:.2f}s")
        if cost is not None:
            bits.append(f"cost ${cost:.4f}")
        if meta:
            bits.extend(f"{k}={v}" for k, v in meta.items())
        header_extra = f"  ({' | '.join(bits)})" if bits else ""

        entry = (
            f"{_RULE}\n"
            f"[{get_current_timestamp()}]  {component.upper()}{header_extra}\n"
            f"{_SUBRULE}\n"
            f"### SYSTEM PROMPT\n{(system_prompt or '').strip()}\n\n"
            f"### USER PROMPT\n{(user_prompt or '').strip()}\n\n"
            f"### OUTPUT\n{(output or '').strip()}\n"
            f"{_RULE}\n\n"
        )
        self._append(self._component_file(component), entry)
        self._append(self.master_file, entry)
