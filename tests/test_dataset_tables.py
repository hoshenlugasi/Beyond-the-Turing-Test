"""Offline synthetic and complete-cohort checks; no runtime/model imports."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from collections import Counter

from core.dataset_tables import (
    GAME_FIELDS, MESSAGE_FIELDS, SURVEY_FIELDS, TABLE_NAMES, VOTE_FIELDS,
    build_tables, write_tables,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class SyntheticTableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "games" / "example"
        self.base.mkdir(parents=True)
        self.meta = dict(game_id="example", ai_pseudonym="Agent",
                         num_human_players=2, total_rounds=1,
                         ai_eliminated=False, ai_elimination_round=None,
                         private_path="DO NOT COPY")
        self.save("metadata.json", self.meta)
        self.save("votes.json", [dict(round=1, voter="Agent", voted_for="Human",
                                     is_ai_voter=True),
                                 dict(round=1, voter="Human", voted_for="Agent",
                                      is_ai_voter=False)])
        self.save("surveys.json", [dict(participant="Human", survey_schema="web_v1",
                                       human_likeness=4, timing=None, relevance=5,
                                       enjoyment=5, suspicion_reason="",
                                       additional_comments="line one\nline two"),
                                   dict(participant="Other", survey_schema="legacy_terminal_v1",
                                        human_likeness=3, timing=2, relevance=1,
                                        comments="é 🕵️")])
        (self.base / "transcript.txt").write_text(
            "Readable document header\nNot a speaker: not a message\n\n"
            "[23:59:59] Game-Manager: Welcome\n\nMultiline rules.\n"
            "[23:59:59] Human:  full text  \nsecond line\n\n"
            "[00:00:01] Agent: same\n[00:00:01] Agent: same\n"
            "[00:00:00] Human: backstep\n"
            "[00:00:02] Game-Manager: Votes:\n    Human: 1\n    → eliminated: Human\n",
            encoding="utf-8",
        )

    def save(self, name, value):
        (self.base / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_order_multiline_duplicates_header_and_unknown_time(self):
        tables = build_tables(self.root / "games")
        self.assertEqual(tuple(tables), TABLE_NAMES)
        rows = tables["messages"]
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["message_order"] for r in rows], list(range(1, 7)))
        self.assertEqual(rows[0]["text"], "Welcome\n\nMultiline rules.")
        self.assertEqual(rows[1]["text"], " full text  \nsecond line\n")
        self.assertEqual(rows[-1]["text"], "Votes:\n    Human: 1\n    → eliminated: Human")
        self.assertEqual([r["speaker_role"] for r in rows],
                         ["system", "human", "agent", "agent", "human", "system"])
        self.assertEqual(rows[2]["text"], rows[3]["text"])
        self.assertEqual([r["recorded_time"] for r in rows],
                         ["23:59:59", "23:59:59", "00:00:01", "00:00:01", "00:00:00", "00:00:02"])
        for row in rows:
            self.assertEqual(tuple(row), MESSAGE_FIELDS)
            self.assertIsNone(row["elapsed_seconds"])
            self.assertEqual(row["time_basis"], "unknown")
            self.assertNotIn("round", row)

    def test_fixed_projection_missingness_flags_and_manifest(self):
        tables = build_tables(self.root / "games")
        game = tables["games"][0]
        self.assertEqual(set(game), set(GAME_FIELDS) | {"event_count", "human_start_count", "path"})
        self.assertIsNone(game["event_count"])
        self.assertEqual(game["human_start_count"], 2)
        self.assertEqual(game["path"], "games/example")
        self.assertIs(game["ai_eliminated"], False)
        self.assertIs(tables["votes"][1]["is_ai_voter"], False)
        modern, legacy = tables["surveys"]
        for row in (modern, legacy):
            self.assertEqual(set(row), {"game_id"} | set(SURVEY_FIELDS))
        self.assertIsNone(legacy["enjoyment"])
        self.assertIsNone(legacy["suspicion_reason"])
        self.assertEqual(legacy["comments"], "é 🕵️")
        self.assertIsNone(modern["comments"])
        self.assertEqual(modern["suspicion_reason"], "")
        (self.base / "events").mkdir()
        self.save("events/manifest.json", {"event_count": 0, "source_path": "DO NOT COPY"})
        self.assertEqual(build_tables(self.root / "games")["games"][0]["event_count"], 0)

    def test_deterministic_writes_hashes_and_source_immutability(self):
        before = hashes(self.root / "games")
        first = write_tables(self.root)
        payloads = hashes(self.root / "tables")
        self.assertEqual(first, write_tables(self.root))
        self.assertEqual(payloads, hashes(self.root / "tables"))
        self.assertEqual(before, hashes(self.root / "games"))
        expected = build_tables(self.root / "games")
        self.assertEqual(first["counts"], {k: len(v) for k, v in expected.items()})
        for name in TABLE_NAMES:
            data = (self.root / "tables" / f"{name}.jsonl").read_bytes()
            self.assertEqual(first["sha256"][name], hashlib.sha256(data).hexdigest())
            self.assertEqual([json.loads(line) for line in data.splitlines()], expected[name])
            self.assertNotIn(b"\r", data)
            self.assertNotIn(b"DO NOT COPY", data)
            self.assertNotIn(str(self.root).encode(), data)

    def test_empty_games_and_header_only_transcript(self):
        (self.base / "transcript.txt").write_text("Header only\n", encoding="utf-8")
        self.assertEqual(build_tables(self.root / "games")["messages"], [])
        empty = self.root / "empty" / "games"
        empty.mkdir(parents=True)
        result = write_tables(empty.parent)
        self.assertEqual(result["counts"], dict.fromkeys(TABLE_NAMES, 0))
        for name in TABLE_NAMES:
            self.assertEqual((empty.parent / "tables" / f"{name}.jsonl").read_bytes(), b"")

    def test_invalid_types_fail_before_writing(self):
        for field, value in (("num_human_players", True), ("ai_eliminated", 0),
                             ("game_id", "../escape")):
            with self.subTest(field=field):
                self.save("metadata.json", {**self.meta, field: value})
                with self.assertRaises(ValueError):
                    write_tables(self.root)
                self.assertFalse((self.root / "tables").exists())
        self.save("metadata.json", self.meta)
        (self.base / "events").mkdir()
        for value in (True, -1, "0"):
            self.save("events/manifest.json", {"event_count": value})
            with self.assertRaises(ValueError):
                build_tables(self.root / "games")


class CompleteCohortTableTests(unittest.TestCase):
    def test_all_fifty_games_match_canonical_core(self):
        games = ROOT / "games"
        before = hashes(games)
        tables = build_tables(games)
        self.assertEqual(before, hashes(games))
        expected_ids = [f"game_{i:03d}" for i in range(1, 51)]
        self.assertEqual([r["game_id"] for r in tables["games"]], expected_ids)
        self.assertEqual(len(tables["votes"]), 618)
        self.assertEqual(len(tables["surveys"]), 229)
        self.assertEqual(Counter(r["speaker_role"] for r in tables["messages"] if
                                 r["speaker_role"] != "system"), {"agent": 519, "human": 1895})
        self.assertEqual(sum(r["enjoyment"] is not None for r in tables["surveys"]), 224)
        self.assertEqual(sum(not r["ai_eliminated"] for r in tables["games"]), 15)
        self.assertEqual(Counter(r["human_start_count"] for r in tables["games"]), {4: 16, 5: 34})
        # The existing public analysis reader preserves these record occurrences;
        # frozen message-count inputs already reflect original-source deduplication.
        # Do not reapply a deduplication rule to the canonical readable transcript.
        starts = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ([^:\n]+): (.*)$", re.MULTILINE)
        for game in tables["games"]:
            gid = game["game_id"]
            with self.subTest(game_id=gid):
                base = games / gid
                meta = load(base / "metadata.json")
                self.assertEqual({k: game[k] for k in GAME_FIELDS}, meta)
                self.assertEqual(game["path"], f"games/{gid}")
                manifest = base / "events" / "manifest.json"
                self.assertEqual(game["event_count"], load(manifest).get("event_count")
                                 if manifest.exists() else None)
                for name, fields in (("votes", VOTE_FIELDS), ("surveys", SURVEY_FIELDS)):
                    expected = [dict(game_id=gid, **{k: row.get(k) for k in fields})
                                for row in load(base / f"{name}.json")]
                    self.assertEqual([r for r in tables[name] if r["game_id"] == gid], expected)
                rows = [r for r in tables["messages"] if r["game_id"] == gid]
                text = (base / "transcript.txt").read_text(encoding="utf-8")
                boundaries = list(starts.finditer(text))
                self.assertEqual(len(rows), len(boundaries))
                self.assertEqual([r["message_order"] for r in rows], list(range(1, len(rows) + 1)))
                for index, (row, boundary) in enumerate(zip(rows, boundaries)):
                    end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(text)
                    # Remove precisely one record-delimiting LF, not text whitespace.
                    full = text[boundary.start(3):end]
                    if full.endswith("\n"):
                        full = full[:-1]
                    self.assertEqual(row["text"], full)
                    self.assertEqual(row["recorded_time"], boundary.group(1))
                    self.assertEqual(row["speaker"], boundary.group(2))
                    self.assertIsNone(row["elapsed_seconds"])
                    self.assertEqual(row["time_basis"], "unknown")


if __name__ == "__main__":
    unittest.main()