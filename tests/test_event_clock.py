"""Synthetic clock/append tests; no SDK, runtime, paid calls or study data."""
import json
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from social_turing_test.core import event_clock as clock


class EventClockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "events" / "game_events.jsonl"
        self.utc = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
        clock._CACHE.clear()
        self.addCleanup(clock._CACHE.clear)

    def append(self, mono=1_000_000_000, utc=None, **fields):
        return clock.append_event(self.path, dict(event_type="test", **fields),
                                  now_utc=self.utc if utc is None else utc, mono_ns=mono)

    def rows(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_midnight_wall_backstep_and_rounding(self):
        first = self.append()
        second = self.append(2_234_560_000, self.utc + timedelta(seconds=2))
        third = self.append(3_000_000_000, self.utc - timedelta(hours=1))
        self.assertEqual(first["elapsed_seconds"], 0.0)
        self.assertEqual(second["elapsed_seconds"], 1.235)
        self.assertEqual(third["elapsed_seconds"], 2.0)
        self.assertEqual(second["occurred_at_utc"], "2026-01-02T00:00:01.000Z")
        self.assertEqual(first["segment_id"], third["segment_id"])
        self.assertEqual(first["time_basis"], "monotonic_segment")
        self.assertEqual(first["time_precision_seconds"], 0.001)
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(str(uuid.UUID(first["segment_id"])), first["segment_id"])
        self.assertEqual(self.rows(), [first, second, third])

    def test_cache_reset_restart_and_pid_change(self):
        first = self.append()
        clock._CACHE.clear()
        second = self.append(5)
        with patch.object(clock.os, "getpid", return_value=-1):
            third = self.append(0)
        self.assertEqual([r["append_sequence"] for r in (first, second, third)], [1, 2, 3])
        self.assertEqual([r["elapsed_seconds"] for r in (second, third)], [0.0, 0.0])
        self.assertEqual(len({r["segment_id"] for r in (first, second, third)}), 3)

    def test_separate_files_separate_segments(self):
        first = self.append()
        other = clock.append_event(self.path.with_name("other.jsonl"), {},
                                   now_utc=self.utc, mono_ns=1)
        self.assertNotEqual(first["segment_id"], other["segment_id"])
        self.assertEqual(other["append_sequence"], 1)

    def test_concurrent_threads_with_required_external_lock(self):
        lock = threading.Lock()
        def write(index):
            with lock:
                return clock.append_event(self.path, dict(index=index))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(80)))
        rows = self.rows()
        self.assertEqual([r["append_sequence"] for r in rows], list(range(1, 81)))
        self.assertEqual(len({r["segment_id"] for r in rows}), 1)
        self.assertEqual(sorted(r["index"] for r in rows), list(range(80)))
        self.assertEqual([r["monotonic_ns"] for r in rows], sorted(r["monotonic_ns"] for r in rows))

    def test_legacy_tail_and_bounded_recovery(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"x" * (clock.MAX_TAIL_SCAN_BYTES * 3) + b'\n{"schema_version":1}\n')
        first = self.append()
        self.assertEqual(first["append_sequence"], 1)
        clock._CACHE.clear()
        with patch.object(clock, "_last_sequence", wraps=clock._last_sequence) as scan:
            self.append(2_000_000_000)
            self.append(3_000_000_000)
        self.assertEqual(scan.call_count, 1)

    def test_invalid_tails_are_not_skipped_or_modified(self):
        tails = [b'{"append_sequence":9}', b'{}\nBAD\n', b'{}\n\n', b'[]\n',
                 b'{"schema_version":2}\n', b'{"append_sequence":true}\n',
                 b'{"append_sequence":0}\n', b'{"x":NaN}\n', b'\xff\n',
                 b'{"nested":{"x":1e999}}\n',
                 b'x' * clock.MAX_TAIL_SCAN_BYTES + b'\n']
        self.path.parent.mkdir()
        for tail in tails:
            with self.subTest(tail=tail[:50]):
                clock._CACHE.clear()
                self.path.write_bytes(tail)
                with self.assertRaises((ValueError, UnicodeError)):
                    self.append()
                self.assertEqual(self.path.read_bytes(), tail)

    def test_invalid_event_does_not_append_or_advance(self):
        first = self.append()
        before = self.path.read_bytes()
        for invalid in (float("nan"), float("inf"), {"nested": float("-inf")}, object()):
            with self.subTest(value=repr(invalid)):
                with self.assertRaises((ValueError, TypeError)):
                    self.append(2_000_000_000, invalid=invalid)
                self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.append(3_000_000_000)["append_sequence"], 2)
        self.assertEqual(first["elapsed_seconds"], 0)

    def test_naive_utc_backwards_mono_and_oversized_event_rejected(self):
        self.append()
        before = self.path.read_bytes()
        for kwargs in (dict(now_utc=datetime(2026, 1, 1), mono_ns=2_000_000_000),
                       dict(now_utc=self.utc, mono_ns=0),
                       dict(now_utc=self.utc, mono_ns=True)):
            with self.assertRaises(ValueError):
                clock.append_event(self.path, {}, **kwargs)
        with self.assertRaises(ValueError):
            self.append(2_000_000_000, text="x" * clock.MAX_RECORD_BYTES)
        self.assertEqual(self.path.read_bytes(), before)

    def test_reserved_fields_and_aware_offset(self):
        row = self.append(utc=self.utc.astimezone(timezone(timedelta(hours=3))),
                          schema_version=999, append_sequence=999, segment_id="spoof",
                          elapsed_seconds=999, occurred_at_utc="spoof")
        self.assertEqual(row["occurred_at_utc"], "2026-01-01T23:59:59.000Z")
        self.assertEqual(row["append_sequence"], 1)
        self.assertEqual(row["schema_version"], 2)
        self.assertNotEqual(row["segment_id"], "spoof")

    def test_truncation_or_external_malformed_append_invalidates_cache(self):
        first = self.append()
        self.path.write_bytes(b"")
        second = self.append(0)
        self.assertNotEqual(first["segment_id"], second["segment_id"])
        self.assertEqual(second["append_sequence"], 1)
        with self.path.open("ab") as stream:
            stream.write(b"bad")
        with self.assertRaises(ValueError):
            self.append(1)

    def test_failed_append_does_not_commit_cache(self):
        first = self.append()
        with patch.object(Path, "open", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                self.append(2_000_000_000)
        second = self.append(3_000_000_000)
        self.assertEqual(second["append_sequence"], 2)
        self.assertEqual(first["segment_id"], second["segment_id"])

    def test_anchor_write_once_utc_only_and_cleanup(self):
        folder = Path(self.temporary.name)
        self.assertTrue(clock.write_game_start_anchor(folder, epoch_ns=1_234_567_890))
        path = folder / "game_start_anchor.json"
        before = path.read_bytes()
        self.assertFalse(clock.write_game_start_anchor(folder, epoch_ns=9_999_999_999))
        self.assertEqual(path.read_bytes(), before)
        value = json.loads(before)
        self.assertEqual(value["occurred_at_utc"], "1970-01-01T00:00:01.234Z")
        self.assertEqual(value["utc_epoch_ns"], 1_234_567_890)
        self.assertIsNone(value["clock_accuracy_seconds"])
        self.assertNotIn("monotonic_ns", value)
        self.assertEqual(list(folder.glob("*.tmp")), [])

    def test_anchor_race_no_clobber_and_failure_cleanup(self):
        folder = Path(self.temporary.name)
        destination = folder / "game_start_anchor.json"
        def winner(source, target):
            destination.write_text("winner", encoding="utf-8")
            raise FileExistsError()
        with patch.object(clock.os, "link", side_effect=winner):
            self.assertFalse(clock.write_game_start_anchor(folder, epoch_ns=0))
        self.assertEqual(destination.read_text(encoding="utf-8"), "winner")
        destination.unlink()
        with patch.object(clock.os, "link", side_effect=OSError("unsupported")):
            with self.assertRaises(OSError):
                clock.write_game_start_anchor(folder, epoch_ns=0)
        self.assertFalse(destination.exists())
        self.assertEqual(list(folder.glob("*.tmp")), [])