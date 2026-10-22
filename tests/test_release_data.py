"""Read-only checks of the complete approved public exports, using stdlib only.

ROOT is this source bundle, never the research repository. No history, private
mapping, original corpus, API or network is consulted. Published data contains
only core records and four access views, not historical event traces.
These structural checks do not certify anonymity or consent.
"""

import csv
import hashlib
import json
import re
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from social_turing_test.core.dataset_tables import build_tables
from social_turing_test.core.hub_features import dataset_infos


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "data"
INPUTS = REPOSITORY / "analysis" / "inputs"
GAME_IDS = {f"game_{i:03d}" for i in range(1, 51)}
CORE_FILES = {"metadata.json", "transcript.txt", "votes.json", "surveys.json"}
META_KEYS = {"game_id", "ai_pseudonym", "num_human_players", "total_rounds",
             "ai_eliminated", "ai_elimination_round"}
ROUND_KEYS = {"game_id", "round", "timestamp", "cutoff_seconds", "active_players",
              "tally", "voted_out"}
COUNT_KEYS = {"game_id", "round", "player", "word_count", "timestamp_seconds"}
LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ([^:\n]+): (.*)$", re.MULTILINE)
REDACTION = re.compile(r"\[REDACTED(?:_[A-Z_]+)?\]")
# Patterns are generic, not a list of private source IDs or removed quotations.
UNSAFE_VALUE = re.compile(
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
    r"|\b[a-f0-9]{24}\b|\bAIza[A-Za-z0-9_-]{30,}\b"
    r"|\bsk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|[A-Z]:[\\/](?:Users|home)[\\/]|/home/[^/\s]+/"
    r"|[?&](?:api_key|token|access_token|password)=[^\s&]+", re.IGNORECASE)
PRIVATE_KEY = re.compile(
    r"(?:^|_)(?:prolific|participant_id|real_name|email|api_key|password|secret|"
    r"access_token|source_game_id|original_game_id|private_path)(?:$|_)", re.I)
FIELD_ONLY = object()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    # Empty lines are not silently discarded from an approved JSONL export.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def walk(value, field=""):
    """Yield fields and scalar values, including nested verdicts and CSV cells."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, FIELD_ONLY
            yield from walk(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item, field)
    else:
        yield field, value


def public_identifier(field, value):
    """Check identifier *fields*, not arbitrary four-digit mentions (e.g. years)."""
    if value is FIELD_ONLY:
        return True
    if field == "game_id":
        return isinstance(value, str) and value in GAME_IDS
    if field == "turn_id":
        return value is None or isinstance(value, str) and re.fullmatch(
            r"game_\d{3}-turn\d{3,}", value) is not None and value[:8] in GAME_IDS
    if field == "event_id":
        return isinstance(value, str) and re.fullmatch(
            r"game_\d{3}-event\d{6}", value) is not None and value[:8] in GAME_IDS
    if field in {"item_id", "id"}:
        return isinstance(value, str) and re.fullmatch(
            r"game_\d{3}:(?:m\d+|round\d+:active\d+)", value) is not None and value[:8] in GAME_IDS
    return True


def seconds(clock):
    h, m, s = map(int, clock.split(":"))
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError("invalid public clock")
    return h * 3600 + m * 60 + s


def files_under(base):
    """An empty .gitkeep is packaging scaffolding, never a data record."""
    found = {}
    for path in base.rglob("*"):
        if path.is_symlink():
            raise AssertionError("public data must not contain symlinks")
        if path.is_file():
            if path.name == ".gitkeep" and path.stat().st_size == 0:
                continue
            found[path.relative_to(base).as_posix()] = path
    return found


class PublicCoreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mandatory fixtures: missing public data must fail, not skip.
        cls.rounds = defaultdict(list)
        for row in load_jsonl(INPUTS / "rounds.jsonl"):
            cls.rounds[row["game_id"]].append(row)
        cls.counts = defaultdict(list)
        for row in load_jsonl(INPUTS / "message_counts.jsonl"):
            cls.counts[row["game_id"]].append(row)

    def test_exact_fifty_four_file_games(self):
        base = ROOT / "games"
        self.assertEqual({p.name for p in base.iterdir() if p.is_dir()}, GAME_IDS)
        expected = {f"{gid}/{name}" for gid in GAME_IDS for name in CORE_FILES}
        self.assertEqual(len(expected), 200)
        self.assertEqual(set(files_under(base)), expected)
        for gid in sorted(GAME_IDS):
            self.assertEqual({p.name for p in (base / gid).iterdir()}, CORE_FILES)
        self.assertFalse((ROOT / "internal_outputs").exists())

    def test_core_schema_counts_rosters_and_round_tallies(self):
        totals, sizes, schemas = Counter(), Counter(), Counter()
        self.assertEqual(set(self.rounds), GAME_IDS)
        self.assertEqual(set(self.counts), GAME_IDS)
        for gid in sorted(GAME_IDS):
            with self.subTest(game=gid):
                base = ROOT / "games" / gid
                meta = load_json(base / "metadata.json")
                votes = load_json(base / "votes.json")
                surveys = load_json(base / "surveys.json")
                self.assertEqual(set(meta), META_KEYS)
                self.assertEqual(meta["game_id"], gid)
                self.assertIs(type(meta["num_human_players"]), int)
                self.assertIn(meta["num_human_players"], (4, 5))
                self.assertIs(type(meta["ai_eliminated"]), bool)
                self.assertIs(type(meta["total_rounds"]), int)
                self.assertIsInstance(meta["ai_pseudonym"], str)
                ai = meta["ai_pseudonym"]
                rounds = self.rounds[gid]
                self.assertEqual([r["round"] for r in rounds],
                                 list(range(1, meta["total_rounds"] + 1)))
                roster = set(rounds[0]["active_players"])
                self.assertEqual(len(roster), meta["num_human_players"] + 1)
                self.assertIn(ai, roster)
                ballots = defaultdict(list)
                self.assertIsInstance(votes, list)
                for vote in votes:
                    self.assertEqual(set(vote), {"round", "voter", "voted_for", "is_ai_voter"})
                    self.assertIs(type(vote["round"]), int)
                    self.assertIs(type(vote["is_ai_voter"]), bool)
                    self.assertEqual(vote["is_ai_voter"], vote["voter"] == ai)
                    ballots[vote["round"]].append(vote)
                    totals["ai_votes" if vote["is_ai_voter"] else "human_votes"] += 1
                self.assertEqual(set(ballots), set(range(1, len(rounds) + 1)))
                active = roster.copy()
                for audit in rounds:
                    self.assertEqual(set(audit), ROUND_KEYS)
                    self.assertEqual(audit["game_id"], gid)
                    self.assertEqual(set(audit["active_players"]), active)
                    self.assertEqual(len(audit["active_players"]), len(active))
                    self.assertEqual(set(audit["tally"]), active)
                    self.assertIn(ai, active)
                    # Source clocks are seconds-of-day; crossing midnight is valid.
                    self.assertEqual(audit["cutoff_seconds"], seconds(audit["timestamp"]))
                    current = ballots[audit["round"]]
                    self.assertEqual(len({v["voter"] for v in current}), len(current))
                    self.assertEqual(sum(v["is_ai_voter"] for v in current), 1)
                    for vote in current:
                        self.assertIn(vote["voter"], active)
                        self.assertIn(vote["voted_for"], active)
                    tally = Counter(v["voted_for"] for v in current)
                    self.assertEqual(audit["tally"], {p: tally[p] for p in active})
                    for n in audit["tally"].values():
                        self.assertIs(type(n), int)
                        self.assertGreaterEqual(n, 0)
                    out = audit["voted_out"]
                    self.assertIn(out, active)
                    self.assertEqual(tally[out], max(tally.values()))
                    active.remove(out)
                self.assertEqual(meta["ai_eliminated"], ai not in active)
                if meta["ai_eliminated"]:
                    self.assertEqual(meta["ai_elimination_round"], len(rounds))
                    self.assertEqual(rounds[-1]["voted_out"], ai)
                else:
                    self.assertIsNone(meta["ai_elimination_round"])
                    self.assertEqual(len(active), 2)

                messages = [(clock, speaker, text) for clock, speaker, text in
                            LINE.findall((base / "transcript.txt").read_text(encoding="utf-8"))
                            if speaker != "Game-Manager"]
                self.assertEqual(len(messages), len(self.counts[gid]))
                # The source reader sorts equal-second messages by speaker/text;
                # the displayed transcript preserves append order. Compare the
                # time/speaker multiset, not arbitrary order inside a second.
                self.assertEqual(Counter((seconds(clock), speaker) for clock, speaker, _ in messages),
                                 Counter((r["timestamp_seconds"], r["player"]) for r in self.counts[gid]))
                for count in self.counts[gid]:
                    speaker = count["player"]
                    self.assertIn(speaker, roster)
                    self.assertEqual(set(count), COUNT_KEYS)
                    self.assertEqual(count["game_id"], gid)
                    self.assertIs(type(count["word_count"]), int)
                    self.assertGreaterEqual(count["word_count"], 0)
                    self.assertIs(type(count["round"]), int)
                    self.assertIn(count["round"], ballots)
                    # Do not "correct" historical seconds-of-day window membership.
                    totals["ai_messages" if speaker == ai else "human_messages"] += 1
                # Original word counts deliberately are not recomputed from redactions.
                self.assertIsInstance(surveys, list)
                self.assertEqual(len({s["participant"] for s in surveys}), len(surveys))
                for survey in surveys:
                    schema = survey["survey_schema"]
                    self.assertIn(schema, ("legacy_terminal_v1", "web_v1"))
                    keys = {"participant", "survey_schema", "human_likeness", "timing", "relevance"}
                    keys |= ({"comments"} if schema == "legacy_terminal_v1" else
                             {"enjoyment", "suspicion_reason", "additional_comments"})
                    self.assertEqual(set(survey), keys)
                    self.assertIn(survey["participant"], roster - {ai})
                    for key in keys - {"participant", "survey_schema"}:
                        value = survey[key]
                        if key in {"human_likeness", "timing", "relevance", "enjoyment"}:
                            self.assertIs(type(value), int)
                            self.assertIn(value, range(1, 6))
                        else:
                            self.assertIsInstance(value, str)
                    schemas[schema] += 1
                sizes[meta["num_human_players"]] += 1
                totals["rounds"] += len(rounds)
                totals["losses" if meta["ai_eliminated"] else "wins"] += 1
                totals["surveys"] += len(surveys)
                totals["missing_surveys"] += meta["num_human_players"] - len(surveys)
        self.assertEqual(sizes, {5: 34, 4: 16})
        self.assertEqual(schemas, {"legacy_terminal_v1": 5, "web_v1": 224})
        self.assertEqual(totals, {"rounds": 135, "ai_messages": 519, "human_messages": 1895,
                                 "ai_votes": 135, "human_votes": 483, "surveys": 229,
                                 "missing_surveys": 5, "wins": 15, "losses": 35})

    def test_additional_redactions_are_category_only(self):
        manifest = load_json(ROOT / "public_redactions.json")
        self.assertEqual(set(manifest), {"baseline", "additional_public_review", "effect"})
        rows = manifest["additional_public_review"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row["spans"] for row in rows), 4)
        self.assertEqual({row["category"] for row in rows},
                         {"named_work_project", "echo_of_redacted_location", "family_event_linkage"})
        self.assertEqual(len({(r["game_id"], r["file"]) for r in rows}), 3)
        for row in rows:
            self.assertEqual(set(row), {"game_id", "file", "spans", "category"})
            self.assertIn(row["game_id"], GAME_IDS)
            self.assertIn(row["file"], CORE_FILES)
            self.assertIs(type(row["spans"]), int)
            text = (ROOT / "games" / row["game_id"] / row["file"]).read_text(encoding="utf-8")
            self.assertGreaterEqual(len(REDACTION.findall(text)), row["spans"])


# Fixed study acceptance targets, independent of the generated manifests and
# executable schema. Supporting other actions in the exporter does not imply
# that those actions occurred in this cohort.




class PublicAccessTableTests(unittest.TestCase):
    def test_exact_core_only_loading_schema_and_config_paths(self):
        infos = load_json(ROOT / "dataset_infos.json")
        self.assertEqual(infos, dataset_infos(include_events=False))
        self.assertEqual(set(infos), {"games", "messages", "votes", "surveys"})
        self.assertNotIn("event_count", infos["games"]["features"])
        self.assertFalse((ROOT / "dataset_manifest.json").exists())
        self.assertFalse(list((ROOT / "games").glob("*/events/*")))
        card = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in infos:
            self.assertIn(f"config_name: {name}", card)
            self.assertIn(f'tables/{name}.jsonl', card)
        self.assertNotIn("config_name: events", card)

    def test_exact_tables_and_canonical_derivation(self):
        expected_counts = {"games": 50, "messages": 2954, "votes": 618, "surveys": 229}
        self.assertEqual(set(files_under(ROOT / "tables")),
                         {name + ".jsonl" for name in expected_counts})
        derived = build_tables(ROOT / "games", include_event_coverage=False)
        actual = {}
        for name, count in expected_counts.items():
            path = ROOT / "tables" / (name + ".jsonl")
            actual[name] = load_jsonl(path)
            self.assertEqual(len(actual[name]), count)
            self.assertEqual(actual[name], derived[name], name)
            canonical = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                           separators=(",", ":")) + "\n" for row in derived[name])
            self.assertEqual(path.read_bytes(), canonical.encode("utf-8"), name)
        self.assertEqual(Counter(r["speaker_role"] for r in actual["messages"]),
                         {"system": 540, "agent": 519, "human": 1895})
        self.assertEqual(sum(r["enjoyment"] is None for r in actual["surveys"]), 5)
        self.assertEqual(Counter(r["survey_schema"] for r in actual["surveys"]),
                         {"legacy_terminal_v1": 5, "web_v1": 224})
        for gid in sorted(GAME_IDS):
            base = ROOT / "games" / gid
            meta = load_json(base / "metadata.json")
            game = [r for r in actual["games"] if r["game_id"] == gid]
            self.assertEqual(len(game), 1)
            self.assertEqual(game[0], dict(meta,
                human_start_count=meta["num_human_players"], path=f"games/{gid}"))
            for name in ("votes", "surveys"):
                source = load_json(base / (name + ".json"))
                rows = [r for r in actual[name] if r["game_id"] == gid]
                self.assertEqual(len(rows), len(source))
                for row, original in zip(rows, source):
                    self.assertEqual({k: v for k, v in row.items() if k != "game_id"},
                                     {k: original.get(k) for k in row if k != "game_id"})
            # Independent occurrence parser: preserve multiline text, blanks,
            # duplicate messages and append order, not clock-sorted joins.
            parsed = []
            for line in (base / "transcript.txt").read_text(encoding="utf-8").splitlines():
                match = LINE.fullmatch(line)
                if match:
                    parsed.append(list(match.groups()))
                elif parsed:
                    parsed[-1][2] += "\n" + line
            messages = [r for r in actual["messages"] if r["game_id"] == gid]
            self.assertEqual(len(messages), len(parsed))
            for order, (row, (clock, speaker, text)) in enumerate(zip(messages, parsed), 1):
                self.assertEqual(set(row), {"game_id", "message_order", "recorded_time",
                    "speaker", "speaker_role", "text", "elapsed_seconds", "time_basis"})
                self.assertEqual(row["message_order"], order)
                self.assertEqual((row["recorded_time"], row["speaker"], row["text"]),
                                 (clock, speaker, text))
                seconds(clock)
                self.assertIsNone(row["elapsed_seconds"])
                self.assertEqual(row["time_basis"], "unknown")
                self.assertEqual(row["speaker_role"], "system" if speaker == "Game-Manager"
                                 else "agent" if speaker == meta["ai_pseudonym"] else "human")


class PublicPrivacyAndPackagingTests(unittest.TestCase):
    def test_numeric_auxiliary_tables_have_no_conversation_text_columns(self):
        expected = {
            "games.csv": {"game_id", "initial_humans", "initial_total", "ai_name",
                          "rounds_played", "ai_eliminated_round", "ai_won",
                          "first_round_ballots", "first_round_votes_for_ai",
                          "first_round_chance", "eliminations_survived"},
            "validation_annotations.csv": {"id", "label", "human_1", "human_2", "model"},
            "message_behavior_message_level.csv": {
                "id", "game_id", "round", "speaker_blind", "role", "asks_question",
                "answers_question_or_request", "opens_new_topic",
                "accuses_or_suspects_other_player", "responds_to_ai_accusation",
                "deflects_or_evades", "brief_reaction", "greeting_or_social_ritual",
                "shares_personal_detail", "uses_slang_or_informal_abbreviation"},
        }
        for name, keys in expected.items():
            with self.subTest(table=name):
                with (INPUTS / name).open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(set(reader.fieldnames), keys)
                    rows = list(reader)
                self.assertEqual(len(rows), {"games.csv": 50,
                    "validation_annotations.csv": 1000,
                    "message_behavior_message_level.csv": 2414}[name])
                self.assertTrue(all(set(row) == keys and None not in row.values() for row in rows))
                if name == "games.csv":
                    self.assertEqual({row["game_id"] for row in rows}, GAME_IDS)
                    for row in rows:
                        meta = load_json(ROOT / "games" / row["game_id"] / "metadata.json")
                        self.assertEqual(row["ai_name"], meta["ai_pseudonym"])
                        self.assertEqual(int(row["initial_humans"]), meta["num_human_players"])
                        self.assertEqual(int(row["initial_total"]), meta["num_human_players"] + 1)
                        self.assertEqual(int(row["rounds_played"]), meta["total_rounds"])
                        self.assertEqual(row["ai_won"], str(not meta["ai_eliminated"]))
                        self.assertEqual(row["ai_eliminated_round"],
                                         str(meta["ai_elimination_round"] or ""))

    def test_identifier_checks_are_field_scoped_not_year_scans(self):
        self.assertTrue(public_identifier("worker_output_quote", "November 19, 2026"))
        self.assertTrue(public_identifier("word_count", 2026))
        self.assertFalse(public_identifier("game_id", "2026"))
        self.assertFalse(public_identifier("game_id", 2026))
        self.assertFalse(public_identifier("turn_id", "2026-turn001"))
        self.assertTrue(public_identifier("game_id", "game_001"))
        self.assertTrue(public_identifier("event_id", "game_001-event000001"))
        self.assertTrue(public_identifier("turn_id", None))
        self.assertTrue(public_identifier("turn_id", "game_001-turn001"))
        for invalid in (None, 1, "game_001", "game_051-event000001",
                        "game_001-event1", "game_001-event000001-extra"):
            self.assertFalse(public_identifier("event_id", invalid))
        self.assertFalse(public_identifier("game_id", "game_001-event000001"))
        self.assertFalse(public_identifier("turn_id", "game_001-event000001"))
        # Nested reviewed observations are content, not machine identifiers;
        # their names do not exempt actual private keys or unsafe values.
        payload = {"players": [{"pseudonym": "Taylor", "observation": "brief replies"}],
                   "turn_id": "game_001-turn001"}
        for field, value in walk(payload):
            self.assertTrue(public_identifier(field, value))
            self.assertFalse(PRIVATE_KEY.search(field))
        self.assertTrue(PRIVATE_KEY.search("participant_id"))
        self.assertTrue(UNSAFE_VALUE.search("person@example.org"))

    def test_public_structured_data_contains_no_private_fields_or_identifiers(self):
        bases = (ROOT / "games", INPUTS, ROOT / "tables")
        for base in bases:
            for name, path in files_under(base).items():
                if path.suffix not in {".json", ".jsonl", ".csv", ".txt", ".md"}:
                    continue
                with self.subTest(group=base.name, file=name):
                    text = path.read_text(encoding="utf-8")
                    self.assertFalse(UNSAFE_VALUE.search(text), "potential private value in public data")
                    if path.suffix == ".json":
                        records = load_json(path)
                    elif path.suffix == ".jsonl":
                        records = load_jsonl(path)
                    elif path.suffix == ".csv":
                        records = list(csv.DictReader(text.splitlines()))
                    else:
                        continue
                    for field, value in walk(records):
                        self.assertFalse(PRIVATE_KEY.search(field), "private structured-data field")
                        self.assertTrue(public_identifier(field, value), "invalid released identifier")

    def test_license_environment_template_and_documentation_exist(self):
        for name in ("README.md", "DATA_DICTIONARY.md", "dataset_infos.json", "public_redactions.json"):
            self.assertGreater((ROOT / name).stat().st_size, 0, name)
        for name in ("LICENSE", "DATA_LICENSE", ".env.example", "README.md",
                     "analysis/README.md", "analysis/inputs/PROVENANCE.md",
                     "guides/RELEASE_NOTES.md"):
            self.assertGreater((REPOSITORY / name).stat().st_size, 0, name)
        self.assertIn("MIT License", (REPOSITORY / "LICENSE").read_text(encoding="utf-8"))
        self.assertFalse((REPOSITORY / ".env").exists())
        assignments = {}
        for line in (REPOSITORY / ".env.example").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            self.assertEqual(separator, "=")
            self.assertRegex(key.strip(), r"^[A-Z][A-Z0-9_]*$")
            self.assertTrue(not value.strip(), "environment template contains a nonempty value")
            assignments[key.strip()] = True
        self.assertIn("STT_ADMIN_PASSWORD", assignments)
        self.assertIn("GEMINI_API_KEY", assignments)




if __name__ == "__main__":
    unittest.main()