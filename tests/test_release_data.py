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

from core.event_schema import EVENT_SPECS, make_manifest, render_text, validate_event
from core.dataset_tables import build_tables


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
        expected |= {f"{gid}/events/{name}" for gid in GAME_IDS
                     for name in ("timeline.jsonl", "timeline.txt", "manifest.json")}
        self.assertEqual(set(files_under(base)), expected)
        self.assertEqual(len(expected), 350)
        for gid in sorted(GAME_IDS):
            self.assertEqual({p.name for p in (base / gid).iterdir()}, CORE_FILES | {"events"})
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
ACTION_COUNTS = {
    "agent_models_status": 50, "agent_ready": 50,
    "behavior_advice_updated": 407, "behavior_tick": 2245, "cadence_updated": 2248,
    "delivery_phase_drop": 32, "delivery_sent": 519, "fixed_persona_facts_set": 50,
    "generate_end": 551, "generator_candidates": 582, "generator_input": 578,
    "internet_context_tick": 1528, "internet_decision_response": 1259,
    "internet_fact_stored": 699, "internet_facts_pruned": 11,
    "internet_search_error": 54, "internet_search_rejected": 404,
    "internet_search_request": 939, "internet_search_response": 653,
    "internet_search_trace": 599, "internet_seed": 100,
    "internet_tick_begin": 1259, "internet_tick_end": 1255,
    "orchestration_mode": 50, "overused_words_updated": 38,
    "persona_claims_updated": 322, "persona_monitor_tick": 322,
    "pipeline_complete": 580, "repetition_llm_pass": 282,
    "repetition_monitor_tick": 419, "repetition_note_updated": 76,
    "reviewer_analysis": 573, "reviewer_demote_response_required": 115,
    "reviewer_demote_same_dodge_mode": 17, "reviewer_drop": 2,
    "reviewer_filter_known_accusation": 4, "reviewer_selected": 580,
    "round_summary_response": 126, "round_summary_tick": 126,
    "round_summary_tick_begin": 2249, "round_summary_tick_end": 2249,
    "scheduler_blocked": 17905, "scheduler_decision": 4408,
    "snapshot_observed": 3366, "social_state_tick": 1838,
    "social_state_updated": 1059, "stale_send_drop": 29,
    "state_guard_filter": 2, "turn_context": 551,
}
COMPONENT_COUNTS = {
    "Scheduler": 22313, "Generator": 1711, "Reviewer": 1293,
    "Internet Context": 8760, "Round Summarizer": 4750, "Persona Monitor": 694,
    "Social State Tracker": 2897, "Behavior Adapter": 4900,
    "Repetition Monitor": 815, "System": 5227,
}
PREVIOUS_ACTIONS = {
    "pipeline_complete", "behavior_advice_updated", "overused_words_updated",
    "round_summary_tick", "repetition_llm_pass", "repetition_note_updated",
    "scheduler_blocked", "delivery_phase_drop", "delivery_sent",
    "generator_candidates", "reviewer_analysis", "reviewer_demote_response_required",
    "reviewer_demote_same_dodge_mode", "reviewer_drop",
    "reviewer_filter_known_accusation", "reviewer_selected", "scheduler_decision",
    "stale_send_drop", "state_guard_filter",
}
VERDICT_KEYS = {"index", "text", "type", "pros", "cons", "score", "red_flags",
                "verdict", "selected", "pool_index"}


class PublicTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(ROOT / "dataset_manifest.json")
        cls.events = {gid: load_jsonl(ROOT / "games" / gid / "events" / "timeline.jsonl")
                      for gid in sorted(GAME_IDS)}

    def test_exact_files_orders_types_and_sanitized_fields(self):
        counts, components, omissions = Counter(), Counter(), Counter()
        redactions = Counter()
        false_selected = zero_scores = 0
        for gid, rows in self.events.items():
            self.assertTrue(rows)
            base = ROOT / "games" / gid / "events"
            self.assertEqual(set(files_under(base)), {"timeline.jsonl", "timeline.txt", "manifest.json"})
            manifest = load_json(base / "manifest.json")
            expected = make_manifest(rows, manifest["input_omission_counts"], "withheld")
            for key, value in expected.items():
                if key not in {"privacy_status", "limitations"}:
                    self.assertEqual(manifest[key], value, f"{gid}: {key}")
            self.assertEqual(manifest["privacy_status"], "reviewed_with_withholding")
            self.assertEqual(manifest["game_id"], gid)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["retained_event_count"], len(rows))
            self.assertEqual(manifest["source_event_count"], len(rows) + manifest["input_omission_count"])
            omissions.update(manifest["input_omission_counts"])
            self.assertEqual((base / "timeline.txt").read_bytes(), render_text(rows).encode("utf-8"))
            self.assertEqual(rows[0]["elapsed_seconds"], 0.0)
            last_elapsed = 0.0
            for order, row in enumerate(rows, 1):
                with self.subTest(game=gid, order=order):
                    self.assertTrue(validate_event(row))
                    kind = row["action"]
                    self.assertIn(kind, ACTION_COUNTS)
                    self.assertEqual(row["component"], EVENT_SPECS[kind].component)
                    self.assertEqual(row["game_id"], gid)
                    self.assertIs(type(row["event_order"]), int)
                    self.assertEqual(row["event_order"], order)
                    self.assertEqual(row["event_id"], f"{gid}-event{order:06d}")
                    self.assertNotIn("round", row)  # No inferred round annotation.
                    self.assertEqual(row["time_basis"], "since_first_event")
                    self.assertIs(type(row["elapsed_seconds"]), float)
                    self.assertGreaterEqual(row["elapsed_seconds"], last_elapsed)
                    self.assertEqual(row["elapsed_seconds"] % 1, 0)
                    last_elapsed = row["elapsed_seconds"]
                    self.assertEqual(row["time_precision_seconds"], 1.0)
                    delivery = row["payload"]["delivery"]
                    if delivery is not None and delivery["outcome"] == "sent":
                        self.assertEqual(kind, "delivery_sent")
                    if row["turn_id"] is not None:
                        self.assertTrue(public_identifier("turn_id", row["turn_id"]))
                        self.assertTrue(row["turn_id"].startswith(gid + "-turn"))
                    for field, value in walk(row):
                        self.assertFalse(PRIVATE_KEY.search(field), "private field in public event")
                        self.assertNotIn(field, {"timestamp", "wall_t", "epoch", "date", "datetime",
                                                 "source_event_id", "source_order", "source_timestamp"})
                        self.assertTrue(public_identifier(field, value), "unsafe public identifier")
                        if isinstance(value, str):
                            self.assertFalse(UNSAFE_VALUE.search(value), "unsafe value in public event")
                            self.assertFalse(re.search(r"\b\d{2}:\d{2}:\d{2}\b", value),
                                             "source wall-clock in timeline")
                            self.assertFalse(re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value),
                                             "source absolute datetime in timeline")
                    # Count canonical payload only, not its derived text echo.
                    for _, value in walk(row["payload"]):
                        if isinstance(value, str):
                            redactions["prior"] += value.count("[REDACTED_PRIVATE_CONTEXT]")
                            redactions["new"] += value.count("[REDACTED: privacy review]")
                    payload = row["payload"][EVENT_SPECS[kind].branch]
                    if kind == "reviewer_analysis":
                        self.assertIsInstance(payload["verdicts"], list)
                        for verdict in payload["verdicts"]:
                            self.assertEqual(set(verdict), VERDICT_KEYS)
                            self.assertIn("selected", verdict)
                            self.assertIs(type(verdict["selected"]), bool)
                            self.assertIn("score", verdict)
                            self.assertIn(type(verdict["score"]), (int, float))
                            for key in ("text", "type", "pros", "cons", "verdict"):
                                self.assertIsInstance(verdict[key], str)
                            self.assertIsInstance(verdict["red_flags"], list)
                            false_selected += verdict["selected"] is False
                            zero_scores += verdict["score"] == 0
                    if kind == "state_guard_filter":
                        for removed in payload["removed"]:
                            self.assertEqual(set(removed), {"text", "reason"})
                    counts[kind] += 1
                    components[row["component"]] += 1
        self.assertEqual(counts, ACTION_COUNTS)
        self.assertEqual(components, COMPONENT_COUNTS)
        self.assertEqual(sum(counts.values()), 53360)
        self.assertEqual(sum(counts[k] for k in PREVIOUS_ACTIONS), 26277)
        self.assertEqual(self.manifest["games"], 50)
        self.assertEqual(self.manifest["source_event_count"], 54918)
        self.assertEqual(self.manifest["retained_event_count"], 53360)
        self.assertEqual(self.manifest["excluded_diagnostic_event_count"], 1558)
        self.assertEqual(self.manifest["coverage"], {
            "source_records": 54918, "retained_records": 53360,
            "excluded_diagnostic_records": 1558, "clock_vetted_games": 50})
        aggregate = make_manifest((row for rows in self.events.values() for row in rows),
                                  omissions, "withheld")
        for key, value in aggregate.items():
            if key not in {"privacy_status", "limitations"}:
                self.assertEqual(self.manifest[key], value, key)
        self.assertEqual(omissions["retired_diagnostic"], 1558)
        self.assertEqual(sum(omissions.values()), 1558)
        self.assertEqual(self.manifest["privacy_status"], "reviewed_with_withholding")
        self.assertEqual(set(self.manifest["components"]), set(COMPONENT_COUNTS))
        self.assertGreater(false_selected, 0)
        self.assertGreater(zero_scores, 0)
        self.assertEqual(self.manifest["privacy_review"], {
            "new_unique_strings_first_pass_reviewed": 7408,
            "new_unique_strings_withheld": 52, "new_payload_values_withheld": 135,
            "prior_unique_values_withheld": 22, "prior_baseline_withheld_occurrences": 54,
            "anonymity_guaranteed": False})
        self.assertEqual(redactions, {"prior": 111, "new": 135})
        self.assertTrue(all(isinstance(s, str) for s in self.manifest["limitations"]))

    def test_selection_delivery_and_redaction_aware_candidate_membership(self):
        outcomes = Counter()
        selected_turns, delivered_turns = set(), set()
        for gid, rows in self.events.items():
            candidates = {}
            selected = {}
            for row in rows:
                kind, turn = row["action"], row["turn_id"]
                payload = row["payload"][EVENT_SPECS[kind].branch]
                if kind == "generator_candidates":
                    self.assertIsNotNone(turn)
                    self.assertIsInstance(payload["candidates"], list)
                    self.assertEqual(payload["count"], len(payload["candidates"]))
                    self.assertTrue(all(isinstance(s, str) for s in payload["candidates"]))
                    candidates[turn] = payload["candidates"]
                elif kind == "reviewer_selected":
                    self.assertIsNotNone(turn)
                    selected_turns.add((gid, turn))
                    text = payload["selected_text"]
                    self.assertIsInstance(text, str)
                    self.assertIs(type(payload["selected_index"]), int)
                    selected[turn] = text
                    pool = candidates.get(turn)
                    # Filtering may reindex a pool. Require exact membership when
                    # known, not equality at an unverified original pool index.
                    if pool is not None and not REDACTION.search(text) and not any(
                            REDACTION.search(candidate) for candidate in pool):
                        self.assertIn(text, pool)
                    elif pool is not None and text == "[REDACTED_PRIVATE_CONTEXT]":
                        self.assertIn(text, pool, "redacted copies must be consistent")
                elif kind in {"delivery_sent", "delivery_phase_drop"}:
                    expected = "sent" if kind == "delivery_sent" else "phase_drop"
                    self.assertEqual(payload["outcome"], expected)
                    outcomes[expected] += 1
                    if expected == "sent":
                        delivered_turns.add((gid, turn))
                        if turn in selected:
                            a, b = selected[turn], payload["message"]
                            if not REDACTION.search(a) and not REDACTION.search(b):
                                self.assertEqual(a, b)
                elif kind == "stale_send_drop":
                    self.assertIsNone(payload["outcome"])  # Unknown, not synthesized.
                    outcomes["suppressed_stale"] += 1
                elif kind == "reviewer_drop":
                    self.assertEqual(payload["review_status"], "drop")
                    outcomes["reviewer_drop"] += 1
        self.assertEqual(outcomes, {"sent": 519, "phase_drop": 32,
                                    "suppressed_stale": 29, "reviewer_drop": 2})
        self.assertTrue(selected_turns - delivered_turns)
        # In particular, no synthetic timeout/parse/dedup/unconfirmed events.
        self.assertNotIn("delivery_unconfirmed", outcomes)

    def test_delivery_counts_join_authoritative_core_without_clock_merge(self):
        case_only_differences = Counter()
        for gid, rows in self.events.items():
            ai = load_json(ROOT / "games" / gid / "metadata.json")["ai_pseudonym"]
            transcript = (ROOT / "games" / gid / "transcript.txt").read_text(encoding="utf-8")
            messages = [text for _, speaker, text in LINE.findall(transcript) if speaker == ai]
            delivered = [row["payload"]["delivery"]["message"] for row in rows
                         if row["action"] == "delivery_sent"]
            self.assertEqual(len(delivered), len(messages), gid)
            # Compare only intact occurrences; different approved redaction
            # categories in core and timelines need not have identical tokens.
            intact_core = Counter(text for text in messages if "[REDACTED" not in text)
            intact_events = Counter(text for text in delivered if "[REDACTED" not in text)
            common = intact_core & intact_events
            # One recorded message differs only in a single letter's case
            # between the historical event and participant-facing transcript.
            # Preserve both observations; never rewrite either authoritative
            # source or silently normalize arbitrary mismatches.
            if gid == "game_038":
                a, b = list((intact_core - common).elements()), list((intact_events - common).elements())
                self.assertEqual((len(a), len(b)), (1, 1))
                self.assertEqual(a[0].casefold(), b[0].casefold())
                self.assertEqual(len(a[0]), len(b[0]))
                self.assertEqual(sum(x != y for x, y in zip(a[0], b[0])), 1)
                intact_core.subtract(a)
                intact_events.subtract(b)
                common = intact_core & intact_events
                case_only_differences[gid] += 1
            self.assertLessEqual(sum((intact_events - common).values()),
                                 sum("[REDACTED" in text for text in messages), gid)
            self.assertLessEqual(sum((intact_core - common).values()),
                                 sum("[REDACTED" in text for text in delivered), gid)
        self.assertEqual(case_only_differences, {"game_038": 1})


class PublicAccessTableTests(unittest.TestCase):
    def test_exact_tables_and_canonical_derivation(self):
        expected_counts = {"games": 50, "messages": 2954, "votes": 618, "surveys": 229}
        self.assertEqual(set(files_under(ROOT / "tables")),
                         {name + ".jsonl" for name in expected_counts})
        derived = build_tables(ROOT / "games")
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
                event_count=load_json(base / "events" / "manifest.json")["event_count"],
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
        # The global manifest is outside those trees and remains mandatory.
        manifest = load_json(ROOT / "dataset_manifest.json")
        for field, value in walk(manifest):
            self.assertFalse(PRIVATE_KEY.search(field), "private manifest field")
            self.assertTrue(public_identifier(field, value), "invalid manifest identifier")
            if isinstance(value, str):
                self.assertFalse(UNSAFE_VALUE.search(value), "unsafe manifest value")

    def test_license_environment_template_and_documentation_exist(self):
        for name in ("LICENSE", ".env.example", "README.md", "DATASET.md",
                     "analysis/README.md", "analysis/inputs/PROVENANCE.md",
                     "DATA_DICTIONARY.md", "dataset_manifest.json", "public_redactions.json"):
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
        self.assertFalse((mirror / "internal_outputs").exists())
        for relative in ("games", "analysis/inputs", "tables"):
            a, b = ROOT / relative, mirror / relative
            self.assertTrue(b.is_dir(), "incomplete Hugging Face data mirror")
            original, copied = files_under(a), files_under(b)
            self.assertEqual(set(original), set(copied), relative)
            for name in original:
                self.assertEqual(hashlib.sha256(original[name].read_bytes()).digest(),
                                 hashlib.sha256(copied[name].read_bytes()).digest(),
                                 f"mirror mismatch: {relative}/{name}")
        for relative in ("DATASET.md", "DATA_DICTIONARY.md", "dataset_manifest.json",
                         "public_redactions.json"):
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).digest(),
                             hashlib.sha256((mirror / relative).read_bytes()).digest(), relative)
        # Upstream software copyright differs intentionally from dataset rights.
        self.assertEqual((ROOT / "DATA_LICENSE").read_bytes(), (mirror / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()