"""Single source of truth for current-round votes and their write lock.

Every vote file is append-only and the manager consumes one line per round, so
the file alone cannot say which round its last line belongs to. File mtimes
cannot answer it either: get_voted_out_name returns the instant the last vote
lands and run_daytime writes the next round's anchor a few milliseconds later,
so the last voter's file is always LESS than a second older than the new
anchor. Any tolerance window reads their round-N vote as a round-N+1 vote,
which locks the last voter of every round out of the next one and leaves the
game waiting for a vote it refuses to accept.

Instead, stt_main rewrites discussion_start_time.txt at the top of every round,
so its CONTENT is a per-round token. Each writer records the token its vote
belongs to, and every reader compares tokens exactly.

Writers that must call record_vote_round:
  - web_player /vote          (human votes)
  - llm_interface.update_vote (the AI's vote)
  - web_player force_tally    (admin <ABSTAIN>)
"""

import os
import time
from contextlib import contextmanager

from core.game_constants import (
    DISCUSSION_START_TIME_FILE,
    PERSONAL_VOTE_ROUND_FILE_FORMAT,
)


@contextmanager
def vote_round_lock(game_dir, name, timeout=10.0, stale_after=30.0):
    """Serialize one player's vote across the web and AI processes.

    ``threading.Lock`` only protects requests inside the Gunicorn process.  The
    AI runs in a separate process, so Force Tally could previously slip between
    its vote append and round-token write and append ``<ABSTAIN>``.  Exclusive
    lock-file creation is atomic on both Windows and Linux and covers every
    writer without adding a deployment dependency.
    """
    lock_path = _round_file(game_dir, name).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout))
    fd = None

    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for vote lock for {name}")
            time.sleep(0.01)

    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def current_round_token(game_dir) -> str:
    """Return the token identifying the round in progress, or "" if unknown."""
    try:
        return (game_dir / DISCUSSION_START_TIME_FILE).read_text(
            encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _round_file(game_dir, name):
    return game_dir / PERSONAL_VOTE_ROUND_FILE_FORMAT.format(name)


def has_voted_this_round(game_dir, name) -> bool:
    """True when a vote from this player is recorded for the current round."""
    token = current_round_token(game_dir)
    if not token:
        return False
    try:
        recorded = _round_file(game_dir, name).read_text(
            encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    return recorded == token


def record_vote_round(game_dir, name, token=None) -> None:
    """Record which round this player's just-written vote belongs to.

    Call immediately after appending to the vote file. Pass the token captured
    before the append when the caller holds a lock, so a round that turns over
    mid-write cannot mislabel the vote.
    """
    if token is None:
        token = current_round_token(game_dir)
    if not token:
        return
    path = _round_file(game_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
