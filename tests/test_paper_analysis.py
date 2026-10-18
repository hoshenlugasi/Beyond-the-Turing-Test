"""Offline synthetic checks and paper-value checks against the public exports.

All public paper inputs are mandatory: missing or malformed exports fail, never
skip. These tests calculate from frozen decisions only; no inference is invoked.
"""
import hashlib
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import takeaway_evidence as evidence
import external_stage_judge as stage
import message_behavior_analysis as behavior
import analyze_message_behavior_interannotator as validation

INPUTS = ROOT / "analysis/inputs"


class CalculationTests(unittest.TestCase):
    def test_bootstrap_original_percentile_indices_and_seed(self):
        xs, n = [0, 1, 0.25, -0.2], 100
        rng = random.Random(42)
        values = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
        self.assertEqual(evidence.boot_mean(xs, n=n), (values[2], values[97]))
        self.assertEqual(evidence.boot_mean([]), (None, None))

    def test_ladder_eligibility_and_reference(self):
        games = {"a": {"row": {"initial_total": 6, "eliminations_survived": 4, "ai_won": "True"}},
                 "b": {"row": {"initial_total": 5, "eliminations_survived": 0, "ai_won": "False"}}}
        out = evidence.survival(games)
        ladder = out["survival_ladder"]
        self.assertEqual([r["eligible_games"] for r in ladder], [1, 2, 2, 2])
        self.assertEqual(ladder[0]["reference"], 5 / 6)
        self.assertEqual(ladder[-1]["reference"], mean([2 / 6, 2 / 5]))
        self.assertEqual(ladder[-1]["observed"], 0.5)

    def test_stage_human_ballots_and_terminal_split(self):
        games = {"g": {"ai": "A", "audit": [
            {"tally": {"A": 1, "B": 2, "C": 0}, "voted_out": "B"},
            {"tally": {"A": 1, "C": 1}, "voted_out": "A"}]}}
        out = evidence.stage_breakdown(games)
        self.assertEqual(out["totals"], {"rounds": 2, "ballots": 3, "votes_for_ai": 2})
        self.assertEqual(out["by_stage"][0]["nonterminal_ballots"], 2)
        self.assertEqual(out["by_stage"][1]["terminal_rounds"], 1)

    def test_participation_keeps_silent_players_and_boundary(self):
        games = {"g": {"ai": "A", "ends": [10, 20], "audit": [
            {"tally": {"A": 0, "B": 0, "C": 3}}, {"tally": {"A": 1, "B": 1}}],
            "msgs": [{"player": "A", "t": 10, "words": 2},
                     {"player": "B", "t": 11, "words": 4}]}}
        out = evidence.participation(games)
        self.assertEqual(out["agent"]["active_player_rounds"], 2)
        self.assertEqual(out["human"]["active_player_rounds"], 3)
        self.assertEqual(out["agent"]["messages"], 1)
        self.assertEqual(out["agent"]["mean_messages_per_active_round"], 0.5)
        self.assertEqual(out["human"]["mean_messages_per_active_round"], 1 / 3)
        self.assertEqual(out["human"]["mean_words_per_message"], 4)

    def test_behavior_player_keys_include_game_and_human_macro_average(self):
        rows = []
        for gid, speaker, role, positives, count in (
                ("g1", "Player A", "agent", 1, 1),
                ("g1", "Player B", "human", 1, 1),
                ("g1", "Player C", "human", 0, 3),
                ("g2", "Player A", "agent", 0, 1),
                ("g2", "Player B", "human", 0, 1)):
            for i in range(count):
                rows.append({"game_id": gid, "speaker_blind": speaker, "role": role,
                             **{label: i < positives for label in behavior.RETAINED_LABELS}})
        out = behavior.summarize(rows)
        metric = out["paired"]["asks_question_share"]
        self.assertEqual(out["player_instances"], 5)
        self.assertEqual(metric["agent"], 0.5)
        self.assertEqual(metric["humans"], 0.25)

    def test_raw_kappa_contingency_and_undefined(self):
        out = validation.binary_cohen_kappa(
            [True, True, False, False], [True, False, True, False], rater_a="a", rater_b="b")
        self.assertEqual(out["cohen_kappa"], 0)
        self.assertEqual(out["observed_agreement"], 0.5)
        self.assertEqual(out["both_positive"], 1)
        same = validation.binary_cohen_kappa([True, False], [True, False], rater_a="a", rater_b="b")
        self.assertEqual(same["cohen_kappa"], 1)
        constant = validation.binary_cohen_kappa([False] * 4, [False] * 4, rater_a="a", rater_b="b")
        self.assertIsNone(constant["cohen_kappa"])
        self.assertFalse(constant["meets_threshold"])
        with self.assertRaises(ValueError):
            validation.binary_cohen_kappa([1], [True], rater_a="a", rater_b="b")

    def test_validation_requires_same_items_all_labels(self):
        rows = [{"id": str(i), "label": label, "human_1": bool(i),
                 "human_2": bool(i), "model": bool(i)} for label in behavior.LABELS for i in range(2)]
        out = validation.analyze(rows, expected_items=2)
        self.assertEqual(out["decisions_per_rater"], 20)
        with self.assertRaises(ValueError):
            validation.analyze(rows[:-1], expected_items=2)
        with self.assertRaises(ValueError):
            validation.analyze(rows + [rows[0]], expected_items=2)

    def test_surveys_missing_enjoyment_and_both_sd_conventions(self):
        games = {"g": {"surveys": [
            {"survey_schema": "legacy_terminal_v1", "human_likeness": 1},
            {"survey_schema": "web", "human_likeness": 3, "enjoyment": 5}]}}
        out = evidence.surveys(games)
        audit = evidence.survey_sd_audit(games)
        self.assertEqual(out["human_likeness"]["mean"], 2)
        self.assertAlmostEqual(out["human_likeness"]["sd"], 2 ** 0.5)
        self.assertEqual(audit["human_likeness"]["population_sd"], 1)
        self.assertEqual(out["enjoyment"]["n"], 1)

    def test_judge_prompt_payload_and_wilson(self):
        prompt = stage.prompt_for_stage({"roster": ["Player A", "Player B"],
                                         "round_index": 1, "conversation": "Player A: hi"})
        self.assertIn("Do not include reasoning or any other keys.", prompt)
        claude = stage.build_payload(stage.DEFAULT_MODEL, prompt, 42)
        gpt = stage.build_payload("openai/gpt-5", prompt, 42, "minimal")
        self.assertEqual(claude["max_tokens"], 1024)
        self.assertEqual(claude["temperature"], 0)
        self.assertNotIn("temperature", gpt)
        self.assertEqual(gpt["max_completion_tokens"], 8192)
        self.assertEqual(gpt["reasoning"], {"effort": "minimal"})
        self.assertAlmostEqual(stage.wilson(5, 10)[0], 0.2365895936154873)

    def test_four_file_adapter_uses_original_not_displayed_words(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            game, inputs = base / "games/g", base / "inputs"
            game.mkdir(parents=True)
            inputs.mkdir()
            (game / "metadata.json").write_text(json.dumps({
                "game_id": "g", "ai_pseudonym": "A", "num_human_players": 2,
                "total_rounds": 1, "ai_eliminated": False, "ai_elimination_round": None}), encoding="utf-8")
            ballots = [{"round": 1, "voter": "A", "voted_for": "B", "is_ai_voter": True},
                       {"round": 1, "voter": "B", "voted_for": "A", "is_ai_voter": False},
                       {"round": 1, "voter": "C", "voted_for": "B", "is_ai_voter": False}]
            (game / "votes.json").write_text(json.dumps(ballots), encoding="utf-8")
            (game / "surveys.json").write_text("[]", encoding="utf-8")
            (game / "transcript.txt").write_text("[00:00:01] Game-Manager: start\n[00:00:02] A: [redacted]\n", encoding="utf-8")
            audit = {"game_id": "g", "round": 1, "timestamp": "00:00:10",
                     "cutoff_seconds": 10, "tally": {"A": 1, "B": 2, "C": 0},
                     "voted_out": "B", "active_players": ["A", "B", "C"]}
            (inputs / "rounds.jsonl").write_text(json.dumps(audit) + "\n", encoding="utf-8")
            (inputs / "message_counts.jsonl").write_text(json.dumps({"game_id": "g", "round": 1,
                "player": "A", "word_count": 7, "timestamp_seconds": 2}) + "\n", encoding="utf-8")
            row = {"game_id": "g", "initial_humans": 2, "initial_total": 3, "ai_name": "A",
                   "rounds_played": 1, "ai_eliminated_round": "", "ai_won": "True",
                   "first_round_ballots": 2, "first_round_votes_for_ai": 1,
                   "first_round_chance": 0.5, "eliminations_survived": 1}
            games = evidence.load([row], base / "games", inputs)
            self.assertEqual(evidence.participation(games)["agent"]["mean_words_per_message"], 7)
            # A missing zero-tally silent roster member must not silently disappear.
            del audit["tally"]["C"]
            (inputs / "rounds.jsonl").write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaises(ValueError):
                evidence.load([row], base / "games", inputs)


class PublicOutcomeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = evidence.load(evidence.read_cohort(INPUTS / "games.csv"), ROOT / "games", INPUTS)

    def test_survival_and_first_human_votes(self):
        result = evidence.survival(self.games)
        self.assertEqual(result["final_two_n"], 15)
        self.assertEqual(result["final_two_rate"], 0.3)
        # Fixed targets are independent of the bundled result JSON. Retain all
        # ladder points, including eligibility, reference and bootstrap bounds.
        expected = [
            (5, 34, 28, 5 / 6, (23 / 34, 32 / 34), (-15.6862745098, 10.7843137255)),
            (4, 50, 34, (34 * 4 / 6 + 16 * 4 / 5) / 50, (.54, .80), (-16.1333333333, 9.4666666667)),
            (3, 50, 23, (34 * 3 / 6 + 16 * 3 / 5) / 50, (.32, .60), (-20.6, 6.6)),
            (2, 50, 15, (34 * 2 / 6 + 16 * 2 / 5) / 50, (.18, .42), (-17.3333333333, 7.2)),
        ]
        self.assertEqual(len(result["survival_ladder"]), 4)
        for row, (remaining, eligible, reached, reference, observed_ci, excess_ci) in zip(
                result["survival_ladder"], expected):
            with self.subTest(players_remaining=remaining):
                self.assertEqual((row["players_remaining"], row["eligible_games"], row["reached"]),
                                 (remaining, eligible, reached))
                self.assertAlmostEqual(row["observed"], reached / eligible)
                self.assertAlmostEqual(row["reference"], reference)
                self.assertAlmostEqual(row["excess_pp"], 100 * (reached / eligible - reference))
                self.assertEqual(len(row["observed_ci95"]), 2)
                self.assertEqual(len(row["excess_ci95_pp"]), 2)
                for actual, target in zip(row["observed_ci95"], observed_ci):
                    self.assertAlmostEqual(actual, target)
                for actual, target in zip(row["excess_ci95_pp"], excess_ci):
                    self.assertAlmostEqual(actual, target)
        final = result["survival_ladder"][-1]
        self.assertEqual(round(final["reference"] * 100, 1), 35.5)
        self.assertEqual([round(x, 1) for x in final["excess_ci95_pp"]], [-17.3, 7.2])
        first = evidence.identification(self.games)
        self.assertEqual((first["votes_for_ai"], first["human_ballots"]), (51, 227))
        self.assertEqual(round(first["observed_rate"] * 100, 1), 22.5)
        self.assertEqual(round(first["expected_rate"] * 100, 1), 21.4)

    def test_stages_and_participation(self):
        result = evidence.stage_breakdown(self.games)
        self.assertEqual([r["rounds"] for r in result["by_stage"]], [34, 44, 34, 23])
        self.assertEqual([r["ballots"] for r in result["by_stage"]], [165, 170, 102, 46])
        self.assertEqual(result["totals"]["rounds"], 135)
        self.assertEqual(result["totals"]["ballots"], 483)
        # Recount human ballots from the core votes, not the summary or tally.
        expected = [
            (5, 34, 165, 35, 6, 28, 136, 17, 1.2, -7.5),
            (4, 44, 170, 48, 10, 34, 131, 25, 3.2, -5.9),
            (3, 34, 102, 43, 11, 23, 69, 20, 8.8, -4.3),
            (2, 23, 46, 26, 8, 15, 30, 11, 6.5, -13.3),
        ]
        independent = {h: Counter() for h in (5, 4, 3, 2)}
        for gid, game in self.games.items():
            votes = json.loads((ROOT / "games" / gid / "votes.json").read_text(encoding="utf-8"))
            for audit in game["audit"]:
                h = len(audit["active_players"]) - 1
                counts = independent[h]
                ballots = [v for v in votes if v["round"] == audit["round"]
                           and v["voter"] != game["ai"]]
                terminal = audit["voted_out"] == game["ai"]
                hits = sum(v["voted_for"] == game["ai"] for v in ballots)
                counts.update(rounds=1, ballots=len(ballots), votes_for_ai=hits,
                              terminal_rounds=int(terminal))
                if not terminal:
                    counts.update(nonterminal_rounds=1, nonterminal_ballots=len(ballots),
                                  nonterminal_votes_for_ai=hits)
        fields = ("active_humans", "rounds", "ballots", "votes_for_ai", "terminal_rounds",
                  "nonterminal_rounds", "nonterminal_ballots", "nonterminal_votes_for_ai")
        for row, targets in zip(result["by_stage"], expected):
            with self.subTest(active_humans=targets[0]):
                self.assertEqual(tuple(row[key] for key in fields), targets[:8])
                counts = independent[targets[0]]
                self.assertEqual({key: row[key] for key in fields[1:]}, dict(counts))
                self.assertEqual(round(row["excess_pp"], 1), targets[8])
                self.assertEqual(round(row["nonterminal_excess_pp"], 1), targets[9])
                # Terminal-only counts are the disjoint complement, not another
                # survival-conditioned comparison or a new analysis output.
                self.assertEqual(row["ballots"] - row["nonterminal_ballots"],
                                 {5: 29, 4: 39, 3: 33, 2: 16}[targets[0]])
                self.assertEqual(row["votes_for_ai"] - row["nonterminal_votes_for_ai"],
                                 {5: 18, 4: 23, 3: 23, 2: 15}[targets[0]])
        part = evidence.participation(self.games)
        self.assertEqual(part["agent"]["active_player_rounds"], 135)
        self.assertEqual(part["human"]["active_player_rounds"], 494)
        self.assertEqual(sum(len(a["active_players"]) - 1 for g in self.games.values()
                             for a in g["audit"]), 494)
        for role, messages, words in (("agent", 519, 4.44), ("human", 1895, 4.59)):
            self.assertEqual(part[role]["messages"], messages)
            self.assertEqual(round(part[role]["mean_messages_per_active_round"], 2), 3.84)
            self.assertEqual(round(part[role]["mean_words_per_message"], 2), words)

    def test_surveys_and_declared_sample_sd(self):
        result = evidence.surveys(self.games)
        audit = evidence.survey_sd_audit(self.games)
        self.assertEqual(result["n_responses"], 229)
        expected = {
            "human_likeness": (229, 3.93, 0.94, [3, 16, 43, 98, 69]),
            "timing": (229, 3.75, 1.08, [7, 28, 44, 87, 63]),
            "relevance": (229, 4.08, 0.90, [1, 14, 35, 94, 85]),
            "enjoyment": (224, 4.55, 0.78, [2, 5, 14, 49, 154]),
        }
        for key, (n, avg, sd, distribution) in expected.items():
            with self.subTest(key=key):
                self.assertEqual(result[key]["n"], n)
                self.assertEqual(round(result[key]["mean"], 2), avg)
                self.assertEqual(list(result[key]["distribution"].values()), distribution)
                self.assertEqual(round(result[key]["sd"], 2), sd,
                    f"Declared sample SD does not match paper; local SD audit: {audit[key]}")


class PublicCachedIntegrationTests(unittest.TestCase):
    def test_required_public_inputs_and_results_exist(self):
        required = ["games.csv", "rounds.jsonl", "message_counts.jsonl",
                    "message_behavior_message_level.csv", "validation_annotations.csv",
                    "validation_manifest.json", "annotation_settings.json", "message_behavior_prompt.txt"]
        required += [f"external_stage_judges/{name}/{file}" for name in ("claude", "gpt5_minimal")
                     for file in ("ledger.jsonl", "request_settings.json")]
        for name in required:
            with self.subTest(input=name):
                self.assertTrue((INPUTS / name).is_file(), f"Missing required public input: {name}")
        for name in ("takeaway_evidence", "external_stage_judge", "message_behavior_analysis",
                     "analyze_message_behavior_interannotator"):
            path = ROOT / "analysis/results" / f"{name}.json"
            with self.subTest(result=name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_stage_judges(self):
        base = INPUTS / "external_stage_judges"
        # Audit-derived coverage prevents two identically wrong ledgers passing.
        audits = evidence.read_jsonl(INPUTS / "rounds.jsonl")
        items = [{"game_id": a["game_id"], "round_index": a["round"],
                  "active_players": len(a["active_players"]),
                  "item_id": f"{a['game_id']}:round{a['round']}:active{len(a['active_players'])}",
                  "chance": 1 / len(a["active_players"])} for a in audits]
        audit_keys = {(r["game_id"], r["round_index"], r["active_players"]) for r in items}
        self.assertEqual(len(items), 135)
        self.assertEqual(len(audit_keys), 135)
        self.assertEqual(Counter(r["active_players"] for r in items), {6: 34, 5: 44, 4: 34, 3: 23})
        coverage = []
        for name, correct in (("claude", [1, 5, 7, 7]), ("gpt5_minimal", [2, 3, 5, 5])):
            model = stage.DEFAULT_MODEL if name == "claude" else "openai/gpt-5"
            settings = json.loads((base / name / "request_settings.json").read_text(encoding="utf-8"))
            self.assertEqual((settings["model"], settings["seed"], settings["items"]), (model, 42, 135))
            expected_settings = ({"max_tokens": 1024, "temperature": 0.0} if name == "claude" else
                                 {"max_completion_tokens": 8192, "reasoning": {"effort": "minimal"},
                                  "fallback_reasoning_effort_after_attempt": {"attempt": 3, "effort": "low"}})
            self.assertEqual(settings["request_settings"], expected_settings)
            rows = stage.load_ledger(base / name / "ledger.jsonl")
            self.assertEqual(len(rows), 135)
            self.assertEqual({r["item_id"] for r in rows}, {r["item_id"] for r in items})
            coverage.append({(r["game_id"], r["round_index"], r["active_players"]) for r in rows})
            self.assertEqual(coverage[-1], audit_keys)
            for row in rows:
                with self.subTest(judge=name, item=row["item_id"]):
                    self.assertEqual(row["model"], model)
                    self.assertEqual(row["status"], "ok")
                    self.assertIs(row["valid"], True)
                    self.assertEqual(row["item_id"],
                                     f"{row['game_id']}:round{row['round_index']}:active{row['active_players']}")
            out = stage.summarize(items, rows, model)
            stages = list(out["by_active_players"].values())
            self.assertEqual([r["valid"] for r in stages], [34, 44, 34, 23])
            self.assertEqual([r["correct"] for r in stages], correct)
            for active, count, successes in zip((6, 5, 4, 3), (34, 44, 34, 23), correct):
                summary = out["by_active_players"][str(active)]
                self.assertEqual((summary["n"], summary["failed"]), (count, 0))
                self.assertAlmostEqual(summary["accuracy"], successes / count)
                self.assertAlmostEqual(summary["chance"], 1 / active)
                # Independent Wilson expression, rather than calling stage.wilson twice.
                p, z2 = successes / count, 1.96 ** 2
                center = (p + z2 / (2 * count)) / (1 + z2 / count)
                half = (z2 * (p * (1 - p) / count + z2 / (4 * count ** 2))) ** .5 / (1 + z2 / count)
                for actual, target in zip(summary["ci95_wilson"], (center - half, center + half)):
                    self.assertAlmostEqual(actual, target)
            if name == "gpt5_minimal":
                self.assertEqual(Counter(r["reasoning_effort"] for r in rows), {"minimal": 134, "low": 1})
            else:
                self.assertEqual({r["reasoning_effort"] for r in rows}, {None})
        self.assertEqual(coverage[0], coverage[1])

    def test_retained_behavior_rates(self):
        path = INPUTS / "message_behavior_message_level.csv"
        rows = behavior.load_messages(path)
        out = behavior.summarize(rows)
        self.assertEqual((out["games"], out["messages"]), (50, 2414))
        self.assertEqual(out["player_instances"], 279)
        self.assertEqual(len({(r["game_id"], r["speaker_blind"]) for r in rows}), 279)
        expected = {"answers_question_or_request": (30.5, 23.2), "asks_question": (20.6, 27.3),
                    "greeting_or_social_ritual": (9.4, 14.9), "shares_personal_detail": (14.4, 16.2)}
        for label, rates in expected.items():
            row = out["paired"][f"{label}_share"]
            self.assertEqual((round(row["agent"] * 100, 1), round(row["humans"] * 100, 1)), rates)

    def test_validation_100_items_1000_decisions_four_retained(self):
        path = INPUTS / "validation_annotations.csv"
        result = validation.analyze(validation.load_validation(path))
        self.assertEqual((result["items"], result["labels"], result["decisions_per_rater"]), (100, 10, 1000))
        self.assertEqual(set(result["retained_labels"]), set(behavior.RETAINED_LABELS))

    def test_all_thirty_raw_kappas_match_paper(self):
        expected = {
            "asks_question": (.917, .893, .861),
            "answers_question_or_request": (.784, .637, .706),
            "opens_new_topic": (.370, .390, .273),
            "accuses_or_suspects_other_player": (.514, .469, .463),
            "responds_to_ai_accusation": (.657, .556, .755),
            "deflects_or_evades": (.260, .000, .000),
            "brief_reaction": (.522, .532, .598),
            "greeting_or_social_ritual": (.648, .718, .889),
            "shares_personal_detail": (.789, .647, .706),
            "uses_slang_or_informal_abbreviation": (.649, .347, .268),
        }
        result = validation.analyze(validation.load_validation(INPUTS / "validation_annotations.csv"))
        self.assertEqual(set(result["comparisons"]), set(expected))
        pairs = ("human_1_vs_human_2", "model_vs_human_1", "model_vs_human_2")
        for label, targets in expected.items():
            self.assertEqual(set(result["comparisons"][label]), set(pairs))
            for pair, target in zip(pairs, targets):
                with self.subTest(label=label, pair=pair):
                    comparison = result["comparisons"][label][pair]
                    self.assertEqual(comparison["n"], 100)
                    self.assertIs(comparison["kappa_defined"], True)
                    self.assertEqual(round(comparison["cohen_kappa"], 3), target)

    def test_validation_joins_cached_messages_and_recorded_provenance(self):
        rows = validation.load_validation(INPUTS / "validation_annotations.csv")
        messages = behavior.load_messages(INPUTS / "message_behavior_message_level.csv")
        by_id = {r["id"]: r for r in messages}
        self.assertEqual(len(by_id), 2414, "Message IDs must be globally unique for this join")
        manifest = json.loads((INPUTS / "validation_manifest.json").read_text(encoding="utf-8"))
        ids = {r["id"] for r in rows}
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(manifest["item_order"]), 100)
        self.assertEqual(set(manifest["item_order"]), ids)
        self.assertEqual(manifest["items"], 100)
        self.assertEqual(manifest["decisions_per_rater"], 1000)
        self.assertEqual(set(manifest["labels"]), set(behavior.LABELS))
        self.assertEqual(Counter((r["id"], r["label"]) for r in rows),
                         Counter({(item, label): 1 for item in ids for label in behavior.LABELS}))
        self.assertTrue(ids <= set(by_id), "Validation IDs must exist in the actual cached message CSV")
        self.assertEqual(Counter(by_id[item]["role"] for item in ids), {"agent": 50, "human": 50})
        self.assertEqual(manifest["hidden_role_counts"], {"agent": 50, "human": 50})
        for row in rows:
            with self.subTest(item=row["id"], label=row["label"]):
                self.assertEqual(row["model"], by_id[row["id"]][row["label"]])
        # Only the prompt has a recorded digest. No settings hash or sampling
        # strata membership is invented from the manifest's aggregate claims.
        settings = json.loads((INPUTS / "annotation_settings.json").read_text(encoding="utf-8"))
        # The recorded digest hashes UTF-8 prompt text (LF), not checkout bytes;
        # universal-newline reading preserves that contract on Windows/CRLF.
        prompt = (INPUTS / "message_behavior_prompt.txt").read_text(encoding="utf-8")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(settings["prompt_sha256"], digest)
        self.assertEqual(digest, "df2fa557149e24394281a1ed0ae21e20066008f8827baa2fac34748d8207a83f")
        for key, target in {"model": "anthropic/claude-sonnet-4.5", "temperature": 0.0,
                            "targets": 2414, "original_prompt_labels": 13,
                            "validated_conversational_labels": 10, "retained_labels": 4}.items():
            self.assertEqual(settings[key], target)


if __name__ == "__main__":
    unittest.main()