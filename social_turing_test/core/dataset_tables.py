"""Deterministic stdlib access tables derived from the public four-file games.

These are access views, not replacement scientific analysis inputs. Transcript
occurrences are never deduplicated, sorted by clock, or assigned inferred rounds.
The recorded clock is not aligned to event clocks: elapsed_seconds stays null.
Untimestamped text before the first message is a document header, not a message;
all subsequent non-record lines (including blanks) continue the preceding text.
Line terminators delimit records and are normalized to LF within multiline text.
Only Game-Manager is a system speaker in the canonical transcript format.

build_tables reads only canonical core files and an optional events/manifest.json.
Missing event coverage is unknown (null), not zero. write_tables writes only the
four root tables; it does not rebuild data, update manifests, or edit source files.
No runtime, model, network, third-party or analysis imports are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


TABLE_NAMES = ("games", "messages", "votes", "surveys")
GAME_FIELDS = {
    "game_id": str, "ai_pseudonym": str, "num_human_players": int,
    "total_rounds": int, "ai_eliminated": bool, "ai_elimination_round": int,
}
VOTE_FIELDS = {"round": int, "voter": str, "voted_for": str, "is_ai_voter": bool}
SURVEY_FIELDS = {
    "participant": str, "survey_schema": str, "human_likeness": int,
    "timing": int, "relevance": int, "enjoyment": int, "comments": str,
    "suspicion_reason": str, "additional_comments": str,
}
MESSAGE_FIELDS = (
    "game_id", "message_order", "recorded_time", "speaker", "speaker_role",
    "text", "elapsed_seconds", "time_basis",
)
_RECORD = re.compile(r"^\[([0-2][0-9]:[0-5][0-9]:[0-5][0-9])\] ([^:\r\n]+): ?(.*)$")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _read_json(path: Path):
    if path.is_symlink():
        raise ValueError("Canonical input must not be a symlink")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _project(record, fields, label):
    """Project fixed nullable types; never coerce bool to int or stringify data."""
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    result = {}
    for name, kind in fields.items():
        value = record.get(name)
        if value is not None and type(value) is not kind:
            raise ValueError(f"{label}.{name} has an invalid type")
        result[name] = value
    return result


def _records(path: Path):
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError("Canonical records must be an array")
    return records


def _messages(path: Path, game_id: str, agent: str) -> list[dict]:
    if path.is_symlink():
        raise ValueError("Canonical transcript must not be a symlink")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _RECORD.fullmatch(line)
        if match:
            clock, speaker, text = match.groups()
            if int(clock[:2]) > 23:
                raise ValueError("Invalid transcript clock")
            role = ("system" if speaker == "Game-Manager" else
                    "agent" if speaker == agent else "human")
            rows.append(dict(game_id=game_id, message_order=len(rows) + 1,
                             recorded_time=clock, speaker=speaker,
                             speaker_role=role, text=text,
                             elapsed_seconds=None, time_basis="unknown"))
        elif rows:
            rows[-1]["text"] += "\n" + line
        # Explicitly exclude free-form document headers before the first record.
    return rows


def build_tables(games_root, *, include_event_coverage=True) -> dict[str, list[dict]]:
    """Read games_root/<id>/ and return four fixed-schema lists in stable order.

    Game directories are lexicographically ordered; arrays and transcript
    occurrences retain canonical source order. Only public relative game paths
    are emitted. Unknown input properties are not copied into dynamic columns.
    Historical public data omits event coverage; local diagnostic exports retain it.
    """
    games_root = Path(games_root)
    if games_root.is_symlink() or not games_root.is_dir():
        raise ValueError("games_root must be a real directory")
    tables = {name: [] for name in TABLE_NAMES}
    for base in sorted(games_root.iterdir(), key=lambda p: p.name):
        if base.is_symlink():
            raise ValueError("Game entries must not be symlinks")
        if not base.is_dir():
            continue
        if not _IDENTIFIER.fullmatch(base.name):
            raise ValueError("Invalid public game identifier")
        metadata = _project(_read_json(base / "metadata.json"), GAME_FIELDS, "metadata")
        if metadata["game_id"] != base.name:
            raise ValueError("Metadata game_id must match its directory")
        if not metadata["ai_pseudonym"]:
            raise ValueError("Metadata must identify the agent pseudonym")
        coverage = {}
        if include_event_coverage:
            event_count = None
            events = base / "events"
            if events.is_symlink():
                raise ValueError("Event directory must not be a symlink")
            manifest = events / "manifest.json"
            if manifest.is_symlink():
                raise ValueError("Event manifest must not be a symlink")
            if manifest.exists():
                event_count = _project(_read_json(manifest), {"event_count": int},
                                       "manifest")["event_count"]
                if event_count is not None and event_count < 0:
                    raise ValueError("Event count must be nonnegative")
            coverage["event_count"] = event_count
        tables["games"].append(dict(
            **metadata, **coverage,
            human_start_count=metadata["num_human_players"],
            path=f"games/{base.name}",
        ))
        tables["messages"].extend(_messages(base / "transcript.txt", base.name,
                                              metadata["ai_pseudonym"]))
        for record in _records(base / "votes.json"):
            tables["votes"].append(dict(game_id=base.name,
                                         **_project(record, VOTE_FIELDS, "vote")))
        for record in _records(base / "surveys.json"):
            tables["surveys"].append(dict(game_id=base.name,
                                           **_project(record, SURVEY_FIELDS, "survey")))
    return tables


def write_tables(root, *, include_event_coverage=True) -> dict:
    """Write root/tables/*.jsonl from root/games and return counts and SHA-256.

    UTF-8, LF, fixed key order, compact JSON, and a final LF per row make bytes
    deterministic. Empty tables are zero-byte files. All inputs are read and
    validated before any output is written. Existing derived tables are replaced;
    callers must serialize rebuilds and own any whole-release atomic publication.
    """
    root = Path(root)
    tables = build_tables(root / "games", include_event_coverage=include_event_coverage)
    destination = root / "tables"
    if destination.is_symlink():
        raise ValueError("Table directory must not be a symlink")
    for name in TABLE_NAMES:
        if (destination / f"{name}.jsonl").is_symlink():
            raise ValueError("Table output must not be a symlink")
    payloads = {
        name: "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
        for name, rows in tables.items()
    }
    destination.mkdir(exist_ok=True)
    for name, payload in payloads.items():
        (destination / f"{name}.jsonl").write_bytes(payload)
    return {
        "counts": {name: len(rows) for name, rows in tables.items()},
        "sha256": {name: hashlib.sha256(data).hexdigest()
                   for name, data in payloads.items()},
    }