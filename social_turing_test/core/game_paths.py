"""Runtime paths: games/real/<id>/ and games/test/<id>/.

Published dataset IDs (game_001, etc.) are never runnable game IDs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Tuple

from social_turing_test.core.game_constants import DIRS_PREFIX
from social_turing_test.core.game_identity import CURRENT_AGENT_IDENTITY, is_test_game

# Top-level category folders.
DIRS_REAL = "real"
DIRS_TEST = "test"
DIRS_SIM = "simulations"

# Folders that exist at games/ top level but are NOT category folders.
_NON_CATEGORY_TOP = {DIRS_SIM}

# Match the public web runtime; never resolve published dataset directories.
_VALID_ID_RE = re.compile(r'[0-9]{4}')


def _games_root() -> Path:
    return Path(DIRS_PREFIX)


# ── Per-game internal subfolder layout ───────────────────────────────────
# Single source of truth for which subfolder a file inside a game directory
# belongs to. Files that come in groups (per-player, or several of the same
# role) live in a subfolder; single-instance files stay at the top level.
# game_constants.py embeds these same subfolders into its filename constants
# so writers land in the right place; readers/migration use the helpers below.
SUBDIR_CHATS = "chats"
SUBDIR_VOTES = "votes"
SUBDIR_SURVEYS = "surveys"
SUBDIR_STATUS = "status"
SUBDIR_LOGS = "logs"
SUBDIR_TRANSCRIPTS = "transcripts"
SUBDIR_MESSAGES = "messages"
SUBDIR_EVENTS = "events"

GAME_SUBDIRS = (
    SUBDIR_CHATS, SUBDIR_VOTES, SUBDIR_SURVEYS, SUBDIR_STATUS,
    SUBDIR_LOGS, SUBDIR_TRANSCRIPTS, SUBDIR_MESSAGES, SUBDIR_EVENTS,
)

# Exact filenames that map to a subfolder (overrides suffix rules).
_FILE_SUBDIR_EXACT = {
    "public_daytime_chat.txt": SUBDIR_TRANSCRIPTS,
    "public_manager_chat.txt": SUBDIR_TRANSCRIPTS,
    "chat_displayed.txt": SUBDIR_TRANSCRIPTS,
    "chat_displayed_full.txt": SUBDIR_TRANSCRIPTS,
    "human_raw_messages.txt": SUBDIR_MESSAGES,
    "ai_generator_raw_messages.txt": SUBDIR_MESSAGES,
    "ai_generator_processed_messages.txt": SUBDIR_MESSAGES,
    "ai_post_reviewer_messages.txt": SUBDIR_MESSAGES,
    "elimination_audit.jsonl": SUBDIR_EVENTS,
}

# Filenames that look like a grouped file but must stay at the top level.
_FILE_FORCE_FLAT = {"phase_status.txt"}


def subdir_for(filename: str) -> str | None:
    """Return the subfolder a game file belongs in, or None for top-level.

    ``filename`` is a bare name (no directory component). memory_trace*.jsonl
    and all other single-instance files return None (stay at the top level).
    """
    filename = Path(filename).name
    if filename in _FILE_FORCE_FLAT:
        return None
    if filename in _FILE_SUBDIR_EXACT:
        return _FILE_SUBDIR_EXACT[filename]
    if filename.endswith(".jsonl") and (
        filename.startswith("game_events") or filename.startswith("cost_log")
    ):
        return SUBDIR_EVENTS
    if filename.endswith("_chat.txt"):
        return SUBDIR_CHATS
    if filename.endswith("_vote.txt"):
        return SUBDIR_VOTES
    if filename.endswith("_survey.txt"):
        return SUBDIR_SURVEYS
    if filename.endswith("_status.txt"):
        return SUBDIR_STATUS
    if filename.endswith("_log.txt"):
        return SUBDIR_LOGS
    return None


def ensure_game_subdirs(game_dir: Path) -> None:
    """Create the per-game subfolders if missing (idempotent)."""
    for sub in GAME_SUBDIRS:
        (game_dir / sub).mkdir(parents=True, exist_ok=True)


def game_file(game_dir, filename: str, *, for_write: bool = False) -> Path:
    """Resolve a bare game filename to its location inside ``game_dir``.

    For writes, returns the subfolder path and creates the subfolder. For
    reads, prefers the subfolder path but falls back to a flat path
    when the subfolder copy doesn't exist, so un-migrated / simulation game
    directories remain readable.
    """
    game_dir = Path(game_dir)
    name = Path(filename).name
    sub = subdir_for(name)
    if sub is None:
        return game_dir / name
    sub_path = game_dir / sub / name
    if for_write:
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        return sub_path
    if sub_path.exists():
        return sub_path
    flat = game_dir / name
    return flat if flat.exists() else sub_path


def resolve_game_dir(game_id: str) -> Path:
    """Resolve a real/test runtime game; default to the real-game location."""
    if not _VALID_ID_RE.fullmatch(game_id):
        raise ValueError(f"Invalid game_id: {game_id!r}")
    root = _games_root()
    for cat in (DIRS_REAL, DIRS_TEST):
        path = root / cat / game_id
        if path.exists():
            return path
    return root / DIRS_REAL / game_id


def target_dir_for_config(config: dict, game_id: str) -> Path:
    """Compute the target directory for a *new* game from its config.

    Routes only by ``is_test`` flag (test vs real).
    Creates parent dirs eagerly so the caller can ``mkdir`` the leaf.
    """
    if not _VALID_ID_RE.fullmatch(game_id):
        raise ValueError(f"Invalid game_id: {game_id!r}")
    cat = DIRS_TEST if is_test_game(config) else DIRS_REAL
    target = _games_root() / cat / game_id
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def iter_all_games(include_simulations: bool = False) -> Iterator[Tuple[str, str, Path]]:
    """Yield (category, agent_identity, path), retaining the three-item API.

    Only four-digit runtime directories directly inside real/test are games.
    Optional simulation enumeration does not select another agent identity.
    """
    root = _games_root()
    if not root.exists():
        return

    for cat in (DIRS_REAL, DIRS_TEST):
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        for game_dir in sorted(cat_dir.iterdir()):
            if game_dir.is_dir() and _VALID_ID_RE.fullmatch(game_dir.name):
                yield cat, CURRENT_AGENT_IDENTITY, game_dir

    if include_simulations:
        sim = root / DIRS_SIM
        if sim.exists():
            for gd in sorted(sim.iterdir()):
                if gd.is_dir():
                    yield DIRS_SIM, "", gd


def next_game_id() -> str:
    """Return the next 4-digit game id ('0001', '0002', ...).

    Scans every game directory under the games root (excluding simulations
    and category folders themselves) for the largest numeric id.
    """
    max_id = 0
    for _category, _identity, gd in iter_all_games(include_simulations=False):
        try:
            n = int(gd.name)
        except ValueError:
            continue
        if n > max_id:
            max_id = n
    return f"{max_id + 1:04d}"
