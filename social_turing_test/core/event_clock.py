"""Private operational clocks, not publication-ready timeline metadata.

Call ``append_event`` under the caller's process-local write lock. One writer
process per file is required: this is NOT an inter-process sequence allocator.
Elapsed time starts at the first event in a file's process/cache segment, NOT
game START. Never merge different segment clocks or publish UUIDs/dates.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


MAX_RECORD_BYTES = 256 * 1024
MAX_TAIL_SCAN_BYTES = MAX_RECORD_BYTES + 1
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Segment:
    pid: int
    identity: tuple[int, int]
    size: int
    sequence: int
    segment_id: str
    anchor_ns: int
    last_ns: int


_CACHE: dict[str, _Segment] = {}


def diagnostic_duration(value: float, digits: int = 2) -> float | None:
    """Unknown durations serialize as null; caller comparisons remain untouched."""
    return round(value, digits) if math.isfinite(value) else None


def _iso_utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("event UTC timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _reject_constant(value: str):
    raise ValueError("non-finite JSON constant")


def _last_sequence(path: Path) -> int:
    """Inspect only the final bounded record, never skip a malformed tail.

    A final record without a sequence starts the next sequence at one;
    its historical rows do not acquire invented sequence numbers. Missing or
    empty files likewise start at one. Oversized/truncated tails are refused.
    """
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            if not size:
                return 0
            start = max(0, size - MAX_TAIL_SCAN_BYTES)
            stream.seek(start)
            tail = stream.read(MAX_TAIL_SCAN_BYTES)
    except FileNotFoundError:
        return 0
    if not tail.endswith(b"\n"):
        raise ValueError("event log has an unterminated tail")
    record = tail[:-1].rsplit(b"\n", 1)[-1]
    if (start and b"\n" not in tail[:-1]) or len(record) + 1 > MAX_RECORD_BYTES:
        raise ValueError("event log tail exceeds record bound")
    entry = json.loads(record.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(entry, dict):
        raise ValueError("event log tail is not an object")
    # JSON numeric exponents (e.g. 1e999) can overflow without invoking
    # parse_constant; reject those too, including nested payloads.
    json.dumps(entry, allow_nan=False)
    if "append_sequence" not in entry and entry.get("schema_version") != 2:
        return 0
    sequence = entry.get("append_sequence")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("event log tail has an invalid append sequence")
    return sequence


def append_event(path, entry: dict, *, now_utc=None, mono_ns=None) -> dict:
    """Serialize completely before append; commit cache only after close.

    UTC and monotonic samples are taken here, inside the caller's lock. UTC
    clock adjustments do not change segment elapsed time. Fork/PID changes,
    cache resets and detected file replacement/truncation start new UUIDs;
    persisted sequence numbers continue across ordinary restarts. External
    writers and edits are unsupported; a size/identity change triggers a fresh
    bounded tail check rather than trusting cached sequence state. A partial
    failed OS write is not repaired: its malformed tail blocks later appends.
    """
    path = Path(path).resolve()
    # Reject NaN/Infinity/unsupported values before touching the file/cache.
    json.dumps(entry, ensure_ascii=False, allow_nan=False)
    utc = datetime.now(timezone.utc) if now_utc is None else now_utc
    occurred_at = _iso_utc(utc)
    mono = time.monotonic_ns() if mono_ns is None else mono_ns
    if type(mono) is not int or mono < 0:
        raise ValueError("monotonic_ns must be a nonnegative integer")
    pid = os.getpid()
    try:
        stat = path.stat()
        identity, size = (stat.st_dev, stat.st_ino), stat.st_size
    except FileNotFoundError:
        identity, size = (0, 0), 0
    previous = _CACHE.get(str(path))
    if previous is None or (previous.pid, previous.identity, previous.size) != (pid, identity, size):
        sequence = _last_sequence(path)
        segment_id, anchor = str(uuid.uuid4()), mono
    else:
        if mono < previous.last_ns:
            raise ValueError("monotonic clock moved backwards within segment")
        sequence, segment_id, anchor = previous.sequence, previous.segment_id, previous.anchor_ns
    result = dict(entry)
    # These reserved fields cannot be spoofed by caller payloads.
    result.update(schema_version=2, timestamp=utc.astimezone().strftime("%H:%M:%S"),
                  occurred_at_utc=occurred_at, monotonic_ns=mono,
                  segment_id=segment_id, append_sequence=sequence + 1,
                  elapsed_seconds=round((mono - anchor) / 1_000_000_000, 3),
                  time_basis="monotonic_segment", time_precision_seconds=0.001)
    encoded = (json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("event exceeds record bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        written = stream.write(encoded)
        if written != len(encoded):
            raise OSError("short event append")
    stat = path.stat()
    _CACHE[str(path)] = _Segment(pid, (stat.st_dev, stat.st_ino), stat.st_size,
                               sequence + 1, segment_id, anchor, mono)
    return result


def write_game_start_anchor(game_dir, *, epoch_ns=None) -> bool:
    """Publish a complete UTC-only anchor once, without replacing a winner.

    Same-directory temporary + hard-link publication is atomic/no-clobber.
    Failure (including unavailable hard links) propagates to the best-effort
    manager hook. No monotonic or derived game elapsed is asserted. UTC sample
    precision is not an accuracy/uncertainty guarantee or a chat-clock bridge.
    """
    destination = Path(game_dir) / "game_start_anchor.json"
    if destination.exists():
        return False
    epoch = time.time_ns() if epoch_ns is None else epoch_ns
    if type(epoch) is not int or epoch < 0:
        raise ValueError("UTC epoch nanoseconds must be a nonnegative integer")
    utc = _UTC_EPOCH + timedelta(microseconds=epoch // 1000)
    text = json.dumps(dict(schema_version=1, occurred_at_utc=_iso_utc(utc),
                           utc_epoch_ns=epoch, time_basis="utc_wall_clock",
                           clock_accuracy_seconds=None), allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)