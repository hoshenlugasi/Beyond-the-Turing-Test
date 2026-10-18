"""Read-only checks of the complete approved public exports, using stdlib only.

ROOT is this source bundle, never the research repository. No history, private
mapping, original corpus, API or network is consulted. Only the optional mirror
test reads the explicitly named sibling Hugging Face bundle.

Nested reviewer ``selected: false`` and ``score: 0`` are valid recorded values.
The exported score is a legacy/default field, NOT a calibrated measure of
human-likeness, risk or quality. Selection is not delivery; absent outcomes are
unknown, not synthesized ``delivery_unconfirmed`` events. Redactions can prevent
verbatim comparisons. These structural checks do not certify anonymity/consent.
"""

import csv
import hashlib
import json
import re
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "analysis" / "inputs"
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
        return isinstance(value, str) and re.fullmatch(
            r"game_\d{3}-turn\d{3,}", value) is not None and value[:8] in GAME_IDS
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
        self.assertEqual(set(files_under(base)), expected)
        self.assertEqual(len(expected), 200)

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


# Exact field allowlists reject source timestamps, prompts, full history/snapshots,
# raw API envelopes, costs, process IDs and private mappings at every event type.
EVENT_FIELDS = {
    "v4_pipeline_complete": "final review_risk scheduler_decision send_reason",
    "v5_behavior_advice_updated": "advice",
    "v5_overused_words_updated": "words",
    "v5_phase_summary_tick": "n_messages phase summary",
    "v5_repetition_llm_pass": "note",
    "v5_repetition_note_updated": "note",
    "v5_scheduler_blocked": "reason",
    "v6_delivery_phase_drop": "message outcome reason stage",
    "v6_delivery_sent": "message outcome",
    "v6_generator_candidates": "candidates count",
    "v6_reviewer_analysis": "num_candidates selected_index verdicts",
    "v6_reviewer_demote_response_required": "demoted pool_size tag",
    "v6_reviewer_demote_same_dodge_mode": "demoted dodge_streak mode pool_size tag",
    "v6_reviewer_drop": "num_candidates review_reason_codes review_status source",
    "v6_reviewer_filter_known_accusation": "kept removed",
    "v6_reviewer_selected": "num_candidates review_reason_codes review_status selected_index selected_text selected_word_count source",
    "v6_scheduler_decision": "decision send_reason",
    "v6_stale_send_drop": "new_messages reason suppressed_message",
    "v6_state_guard_filter": "kept removed",
}
VERDICT_KEYS = {"index", "text", "type", "pros", "cons", "score", "red_flags",
                "verdict", "selected", "pool_index"}


class PublicInternalOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(ROOT / "internal_outputs" / "manifest.json")
        cls.events = {gid: load_jsonl(ROOT / "internal_outputs" / "games" / gid /
                                    "events" / "game_events.jsonl") for gid in sorted(GAME_IDS)}

    def test_exact_files_orders_types_and_sanitized_fields(self):
        base = ROOT / "internal_outputs"
        expected = {"manifest.json", "README.md"} | {
            f"games/{gid}/events/game_events.jsonl" for gid in GAME_IDS}
        self.assertEqual(set(files_under(base)), expected)
        counts = Counter()
        redactions = 0
        false_selected = zero_scores = 0
        for gid, rows in self.events.items():
            self.assertTrue(rows)
            for order, row in enumerate(rows, 1):
                with self.subTest(game=gid, order=order):
                    kind = row["event_type"]
                    self.assertIn(kind, EVENT_FIELDS)
                    self.assertEqual(row["game_id"], gid)
                    self.assertIs(type(row["event_order"]), int)
                    self.assertEqual(row["event_order"], order)
                    allowed = {"game_id", "event_type", "event_order", "turn_id"}
                    allowed.update(EVENT_FIELDS[kind].split())
                    self.assertFalse(set(row) - allowed)
                    if "turn_id" in row:
                        self.assertTrue(public_identifier("turn_id", row["turn_id"]))
                        self.assertTrue(row["turn_id"].startswith(gid + "-turn"))
                    for field, value in walk(row):
                        self.assertFalse(PRIVATE_KEY.search(field), "private field in public event")
                        self.assertTrue(public_identifier(field, value), "unsafe public identifier")
                        if isinstance(value, str):
                            self.assertFalse(UNSAFE_VALUE.search(value), "unsafe value in public event")
                            self.assertFalse(re.search(r"\b\d{2}:\d{2}:\d{2}\b", value),
                                             "source wall-clock in internal output")
                            redactions += value.count("[REDACTED_PRIVATE_CONTEXT]")
                    if kind == "v6_reviewer_analysis":
                        self.assertIsInstance(row["verdicts"], list)
                        for verdict in row["verdicts"]:
                            self.assertFalse(set(verdict) - VERDICT_KEYS)
                            self.assertIn("selected", verdict)
                            self.assertIs(type(verdict["selected"]), bool)
                            self.assertIn("score", verdict)
                            self.assertIn(type(verdict["score"]), (int, float))
                            for key in ("text", "type", "pros", "cons", "verdict"):
                                self.assertIsInstance(verdict[key], str)
                            self.assertIsInstance(verdict["red_flags"], list)
                            false_selected += verdict["selected"] is False
                            zero_scores += verdict["score"] == 0
                    if kind == "v6_state_guard_filter":
                        for removed in row["removed"]:
                            self.assertEqual(set(removed), {"text", "reason"})
                    counts[kind] += 1
        self.assertEqual(sum(counts.values()), 26277)
        self.assertEqual(self.manifest["games"], 50)
        self.assertEqual(self.manifest["events"], 26277)
        self.assertEqual(counts, self.manifest["event_types"])
        self.assertEqual(set(counts), set(EVENT_FIELDS))
        self.assertEqual(counts["v6_delivery_sent"], 519)
        self.assertEqual(counts["v6_generator_candidates"], 582)
        self.assertEqual(counts["v6_reviewer_selected"], 580)
        self.assertGreater(false_selected, 0)
        self.assertGreater(zero_scores, 0)
        review = self.manifest["manual_text_review"]
        self.assertEqual(review["unique_values_reviewed"], 5385)
        self.assertEqual(review["flagged_or_context_ambiguous_values_withheld"], 22)
        self.assertEqual(review["redaction_occurrences"], {
            "core_dataset_redaction_echo": 0, "manually_flagged_or_uncertain_text": 54})
        self.assertEqual(redactions, 54)
        # The released review manifest describes distinct values, not 54 people.
        self.assertTrue(all(isinstance(s, str) for s in self.manifest["limitations"]))

    def test_selection_delivery_and_redaction_aware_candidate_membership(self):
        outcomes = Counter()
        selected_turns, delivered_turns = set(), set()
        for gid, rows in self.events.items():
            candidates = {}
            selected = {}
            for row in rows:
                kind, turn = row["event_type"], row.get("turn_id")
                if kind == "v6_generator_candidates":
                    self.assertIsInstance(row["candidates"], list)
                    self.assertEqual(row["count"], len(row["candidates"]))
                    self.assertTrue(all(isinstance(s, str) for s in row["candidates"]))
                    candidates[turn] = row["candidates"]
                elif kind == "v6_reviewer_selected":
                    selected_turns.add((gid, turn))
                    text = row["selected_text"]
                    self.assertIsInstance(text, str)
                    self.assertIs(type(row["selected_index"]), int)
                    selected[turn] = text
                    pool = candidates.get(turn)
                    # Filtering may reindex a pool. Require exact membership when
                    # known, not equality at an unverified original pool index.
                    if pool is not None and not REDACTION.search(text) and not any(
                            REDACTION.search(candidate) for candidate in pool):
                        self.assertIn(text, pool)
                    elif pool is not None and text == "[REDACTED_PRIVATE_CONTEXT]":
                        self.assertIn(text, pool, "redacted copies must be consistent")
                elif kind in {"v6_delivery_sent", "v6_delivery_phase_drop"}:
                    expected = "sent" if kind == "v6_delivery_sent" else "phase_drop"
                    self.assertEqual(row["outcome"], expected)
                    outcomes[expected] += 1
                    if expected == "sent":
                        delivered_turns.add((gid, turn))
                        if turn in selected:
                            a, b = selected[turn], row["message"]
                            if not REDACTION.search(a) and not REDACTION.search(b):
                                self.assertEqual(a, b)
                elif kind == "v6_stale_send_drop":
                    self.assertNotIn("outcome", row)  # Do not invent a recorded field.
                    outcomes["suppressed_stale"] += 1
                elif kind == "v6_reviewer_drop":
                    self.assertEqual(row["review_status"], "drop")
                    outcomes["reviewer_drop"] += 1
        self.assertEqual(outcomes, {"sent": 519, "phase_drop": 32,
                                    "suppressed_stale": 29, "reviewer_drop": 2})
        self.assertTrue(selected_turns - delivered_turns)
        # In particular, no synthetic timeout/parse/dedup/unconfirmed events.
        self.assertNotIn("delivery_unconfirmed", outcomes)


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

    def test_public_structured_data_contains_no_private_fields_or_identifiers(self):
        bases = (ROOT / "games", INPUTS, ROOT / "internal_outputs")
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
        for name in ("LICENSE", ".env.example", "README.md", "DATASET.md",
                     "analysis/README.md", "analysis/inputs/PROVENANCE.md",
                     "internal_outputs/README.md", "public_redactions.json"):
            self.assertTrue((ROOT / name).is_file(), name)
            self.assertGreater((ROOT / name).stat().st_size, 0, name)
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertFalse((ROOT / ".env").exists(), "actual environment file must not ship")
        assignments = {}
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            self.assertEqual(separator, "=", "invalid environment-template assignment")
            self.assertRegex(key.strip(), r"^[A-Z][A-Z0-9_]*$")
            # Do not echo a failing secret assignment into test output.
            self.assertTrue(not value.strip(), "environment template contains a nonempty value")
            assignments[key.strip()] = True
        self.assertIn("STT_ADMIN_PASSWORD", assignments)
        self.assertIn("GEMINI_API_KEY", assignments)

    def test_optional_huggingface_mirror_hashes(self):
        mirror = ROOT.parent / "huggingface"
        if not mirror.exists():
            self.skipTest("optional sibling Hugging Face bundle is absent in source-only checkout")
        self.assertTrue(mirror.is_dir())
        for relative in ("games", "internal_outputs", "analysis/inputs"):
            a, b = ROOT / relative, mirror / relative
            self.assertTrue(b.is_dir(), "incomplete Hugging Face data mirror")
            original, copied = files_under(a), files_under(b)
            self.assertEqual(set(original), set(copied), relative)
            for name in original:
                self.assertEqual(hashlib.sha256(original[name].read_bytes()).digest(),
                                 hashlib.sha256(copied[name].read_bytes()).digest(),
                                 f"mirror mismatch: {relative}/{name}")
        for relative in ("DATASET.md", "public_redactions.json"):
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).digest(),
                             hashlib.sha256((mirror / relative).read_bytes()).digest(), relative)
        # Upstream software copyright differs intentionally from dataset rights.
        self.assertEqual((ROOT / "DATA_LICENSE").read_bytes(), (mirror / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()