"""Per-game provenance logs.

Six files are written per game (under the game directory). All filenames
are prefixed by audience: `human_*`, `ai_*`, or `chat_*`.

* `human_raw_messages.txt`             — HUMAN players only. Literal text
                                         each human typed into the input
                                         box, before any normalization.
* `ai_generator_raw_messages.txt`      — AI only. Literal text returned
                                         by the LLM generator, BEFORE our
                                         forced edits and BEFORE the
                                         reviewer.
* `ai_generator_processed_messages.txt`— AI only. Generator output AFTER
                                         our forced edits (trim +
                                         make_more_human_like) and BEFORE
                                         the reviewer. This is exactly
                                         what the reviewer is given.
* `ai_post_reviewer_messages.txt`      — AI only. Literal reviewer output,
                                         the final text the agent will
                                         send. Empty body "[ts] Name: "
                                         means the reviewer dropped the
                                         message.
* `chat_displayed.txt`                 — Full conversation (AI + human)
                                         exactly as seen on screen, body
                                         lowercased to mirror the
                                         frontend formatChatMessage.
* `chat_displayed_full.txt`            — Same as `chat_displayed.txt`
                                         PLUS Game-Manager announcements
                                         (eliminations, voting time, win
                                         banner) and per-player vote
                                         tallies, all interleaved in
                                         chronological order. This is the
                                         file that best matches what
                                         users actually saw and is what
                                         the manager UI's Live Chat Feed
                                         should display.

All files are append-only and tolerant of write failures: if the disk is
full or the directory disappears mid-game we silently skip rather than
crash the game loop. Every file gets a one-line `# ...` description
header on first write.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.game_constants import (
    HUMAN_RAW_MESSAGES_FILE,
    AI_GENERATOR_RAW_MESSAGES_FILE,
    AI_GENERATOR_PROCESSED_MESSAGES_FILE,
    AI_POST_REVIEWER_MESSAGES_FILE,
    CHAT_DISPLAYED_FILE,
    CHAT_DISPLAYED_FULL_FILE,
    GAME_MANAGER_NAME,
    VOTING_MESSAGE_FORMAT,
    format_message,
    get_current_timestamp,
)

# Matches "[HH:MM:SS] Name: body" so we can lowercase only the body.
_TIMESTAMPED_LINE = re.compile(r"^(\[[^\]]+\])\s*([^:]+):\s*(.*)$", re.DOTALL)


_LOG_HEADERS = {
    HUMAN_RAW_MESSAGES_FILE: (
        "HUMAN messages only — the literal text each human player typed "
        "into the input box, before any normalization or lowercase. "
        "AI messages are NOT recorded here."
    ),
    AI_GENERATOR_RAW_MESSAGES_FILE: (
        "AI only — literal generator output. Verbatim text returned by "
        "the LLM, BEFORE trim_message_if_too_long, make_more_human_like "
        "and the reviewer."
    ),
    AI_GENERATOR_PROCESSED_MESSAGES_FILE: (
        "AI only — generator output AFTER our forced edits "
        "(trim_message_if_too_long + make_more_human_like) and BEFORE "
        "the reviewer. Exactly what the reviewer receives."
    ),
    AI_POST_REVIEWER_MESSAGES_FILE: (
        "AI only — literal reviewer output (final text the agent will "
        "send, with no further forced edits). Empty body \"[ts] Name: \" "
        "means the reviewer dropped the message and the turn was skipped."
    ),
    CHAT_DISPLAYED_FILE: (
        "Full chat (AI + human) exactly as seen on screen — body "
        "lowercased to mirror the frontend formatChatMessage transform. "
        "Excludes manager announcements and votes."
    ),
    CHAT_DISPLAYED_FULL_FILE: (
        "Full chat AS USERS SAW IT — interleaves chat messages "
        "(body lowercased), Game-Manager announcements (eliminations, "
        "voting prompts, win banner) and per-round vote tallies. "
        "This is the canonical \"what did the players actually see\" log."
    ),
}

# The filename constants above embed a subfolder (e.g. "messages/..."), but
# `_safe_append` looks up headers by ``path.name`` (the bare filename). Re-key
# the dict by basename so the header lookup keeps working.
_LOG_HEADERS = {Path(_k).name: _v for _k, _v in _LOG_HEADERS.items()}


def _safe_append(path: Path, text: str) -> None:
    if not text:
        return
    if not text.endswith("\n"):
        text = text + "\n"
    try:
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            if not existed:
                header = _LOG_HEADERS.get(path.name)
                if header:
                    f.write(f"# {header}\n")
            f.write(text)
    except OSError:
        pass


# ─── Per-stage hooks ──────────────────────────────────────────────────────

def append_human_raw(game_dir: Path, name: str, raw_text: str) -> None:
    """A HUMAN message exactly as typed (pre-normalization)."""
    if raw_text is None:
        return
    _safe_append(game_dir / HUMAN_RAW_MESSAGES_FILE, format_message(name, raw_text))


def append_ai_generator_raw(game_dir: Path, name: str, text: str) -> None:
    """Literal AI generator output (pre-trim, pre-humanizer, pre-reviewer)."""
    if text is None:
        return
    _safe_append(game_dir / AI_GENERATOR_RAW_MESSAGES_FILE, format_message(name, text))


def append_ai_generator_processed(game_dir: Path, name: str, text: str) -> None:
    """AI generator output after trim + humanize, just before the reviewer."""
    if text is None:
        return
    _safe_append(game_dir / AI_GENERATOR_PROCESSED_MESSAGES_FILE, format_message(name, text))


def append_ai_post_reviewer(game_dir: Path, name: str, text: str) -> None:
    """Literal AI reviewer output. Empty body = reviewer dropped the message."""
    if text is None:
        return
    _safe_append(game_dir / AI_POST_REVIEWER_MESSAGES_FILE, format_message(name, text))


# ─── Displayed views ──────────────────────────────────────────────────────

def to_displayed(line: str) -> str:
    """Mirror templates/game.html `formatChatMessage`: lowercase body only."""
    stripped = line.rstrip("\n")
    m = _TIMESTAMPED_LINE.match(stripped)
    if not m:
        return line
    return f"{m.group(1)} {m.group(2)}: {m.group(3).lower()}\n"


def append_chat_displayed(game_dir: Path, line: str) -> None:
    """A chat line as seen on screen — body lowercased.

    Writes to BOTH `chat_displayed.txt` (chat-only) and
    `chat_displayed_full.txt` (chat + manager + votes).
    """
    if not line:
        return
    rendered = to_displayed(line)
    _safe_append(game_dir / CHAT_DISPLAYED_FILE, rendered)
    _safe_append(game_dir / CHAT_DISPLAYED_FULL_FILE, rendered)


def append_manager_announcement(game_dir: Path, message: str) -> None:
    """A Game-Manager announcement, written to chat_displayed_full.txt only.

    `message` is the raw announcement body (without timestamp/name). The
    body is rendered in italics-style brackets so it visually stands out
    when reading the log alongside chat messages.
    """
    if not message:
        return
    # Collapse escaped newlines back to real newlines for a more readable log.
    body = message.replace("\\n", " | ").strip()
    line = format_message(GAME_MANAGER_NAME, f"📣 {body}")
    _safe_append(game_dir / CHAT_DISPLAYED_FULL_FILE, line)


def append_vote_tally(game_dir: Path, vote_details, tally,
                      voted_out_name: str, anonymous: bool) -> None:
    """Append a round's vote outcome to chat_displayed_full.txt.

    `vote_details` is a list of (voter_name, voted_for) tuples. When
    `anonymous` is True we hide the voter→target mapping and only show
    aggregate counts, matching what players saw in the UI.
    """
    ts = get_current_timestamp()
    lines = [f"[{ts}] {GAME_MANAGER_NAME}: 🗳️ Vote results:"]
    if anonymous:
        for cand, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
            if count:
                lines.append(f"    {cand}: {count}")
    else:
        for voter, target in vote_details:
            lines.append(f"    {VOTING_MESSAGE_FORMAT.format(voter, target)}")
    lines.append(f"    → eliminated: {voted_out_name}")
    _safe_append(game_dir / CHAT_DISPLAYED_FULL_FILE, "\n".join(lines) + "\n")
