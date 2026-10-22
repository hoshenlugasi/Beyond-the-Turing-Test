"""Offline synthetic contract tests; no runtime/provider/data imports."""
import copy
import json
import math
import unittest

from social_turing_test.core.event_schema import (
    COMPONENTS, ENVELOPE_FIELDS, EVENT_SPECS, EXCLUDED_EVENTS, SCHEMAS,
    classify_event, make_manifest, project_event, render_text, validate_event,
)


class EventSchemaTests(unittest.TestCase):
    def project(self, kind, **fields):
        return project_event(dict(event_type=kind, **fields), "game_001", 1,
                             lambda s: s.replace("PRIVATE", "[withheld]"),
                             {"private-turn": "game_001-turn001"})

    def test_every_registered_event_has_exact_schema(self):
        for name, spec in EVENT_SPECS.items():
            with self.subTest(name=name):
                event = self.project(name)
                self.assertTrue(validate_event(event))
                self.assertEqual(set(event), ENVELOPE_FIELDS)
                self.assertEqual(set(event["payload"]), set(SCHEMAS))
                active = event["payload"][spec.branch]
                self.assertEqual(set(active), set(SCHEMAS[spec.branch]))
                self.assertTrue(all(v is None for v in active.values()))
                self.assertEqual(event["component"], spec.component)

    def test_all_six_worker_actual_outputs(self):
        fixtures = [
            ("internet_search_trace", dict(query_in="PRIVATE question", corrected_query="new question",
                 summary_out="final summary", stored=dict(topic="topic", confidence="medium",
                 volatility="volatile", source_count=0, prompt="DO NOT COPY"))),
            ("round_summary_response", dict(phase="Discussion 1", summary_text="round output")),
            ("persona_claims_updated", dict(claims=["PRIVATE claim"], count=1)),
            ("social_state_tick", dict(had_signal=False, parse_ok=True, parsed=dict(
                 players={"Zed": "quiet", "Ari": "PRIVATE observation"},
                 attention_on_me=0, open_threads=[], pressure_type="questioning",
                 prompt="DO NOT COPY"))),
            ("behavior_tick", dict(players_left=0, talk_share=0, llm_refreshed=False,
                 warnings=[], recommendation="be specific", agreement_ratio=0.0)),
            ("repetition_monitor_tick", dict(own_msgs=0, overused=[])),
        ]
        components = set()
        for name, fields in fixtures:
            event = self.project(name, **fields)
            components.add(event["component"])
            encoded = json.dumps(event, allow_nan=False)
            self.assertNotIn("PRIVATE", encoded)
            self.assertNotIn("DO NOT COPY", encoded)
            self.assertTrue(validate_event(json.loads(encoded)))
        self.assertEqual(components, set(COMPONENTS) - {"Scheduler", "Generator", "Reviewer", "System"})
        social = self.project(fixtures[3][0], **fixtures[3][1])["payload"]["social_state"]
        self.assertEqual(social["players"], [
            dict(pseudonym="Ari", observation="[withheld] observation"),
            dict(pseudonym="Zed", observation="quiet")])
        self.assertIs(social["had_signal"], False)
        self.assertEqual(social["attention_on_me"], 0.0)
        self.assertIsNone(social["group_mood"])

    def test_nested_plans_verdicts_cadence(self):
        plan = self.project("internet_decision_response", parsed=[dict(
            query="PRIVATE", volatility="stable", needs_search=False,
            raw_results=["SECRET"], prompt="SECRET")])["payload"]["internet"]["plans"][0]
        self.assertEqual(plan["query"], "[withheld]")
        self.assertIs(plan["needs_search"], False)
        self.assertEqual(set(plan), {"query", "volatility", "confidence", "reason", "needs_search"})
        verdict = self.project("reviewer_analysis", selected_index=0, verdicts=[dict(
            index=0, score=0, selected=False, pros="", cons="PRIVATE", red_flags=[],
            scores={"raw": math.inf})])["payload"]["review"]["verdicts"][0]
        self.assertEqual(verdict["index"], 0)
        self.assertEqual(verdict["score"], 0.0)
        self.assertIs(type(verdict["score"]), float)
        self.assertIs(verdict["selected"], False)
        self.assertEqual(verdict["pros"], "")
        self.assertEqual(verdict["cons"], "[withheld]")
        cadence = self.project("cadence_updated", level="under", reason="PRIVATE", talk_share=0)["payload"]["behavior"]["cadence"]
        self.assertEqual(cadence, dict(level="under", reason="[withheld]", talk_share=0.0))

    def test_unknown_and_explicit_diagnostic_dispositions(self):
        self.assertIsNone(self.project("future_unreviewed_event", prompt="PRIVATE"))
        self.assertEqual(classify_event("future_unreviewed_event"), "unknown_event")
        self.assertEqual(classify_event({}), "malformed_input")
        for name in EXCLUDED_EVENTS:
            self.assertEqual(classify_event(name), "retired_diagnostic")
            self.assertIsNone(self.project(name, result="PRIVATE"))

    def test_guard_counts_and_text_normalize_to_fixed_types(self):
        state = self.project("state_guard_filter", removed=[dict(
            text="PRIVATE", reason="repeat", raw="SECRET")], kept=0)["payload"]["filter"]
        self.assertEqual(state["removed"], [dict(text="[withheld]", reason="repeat")])
        self.assertEqual(state["kept"], 0)
        accusation = self.project("reviewer_filter_known_accusation", removed=["PRIVATE"], kept=1)["payload"]["filter"]
        self.assertEqual(accusation["removed"], [dict(text="[withheld]", reason=None)])
        pileup = self.project("reviewer_filter_agreement_pileup", removed=0, streak=3)["payload"]["filter"]
        self.assertEqual(pileup["removed_count"], 0)
        self.assertIsNone(pileup["removed"])

    def test_private_discard_and_snapshot_counts_only(self):
        private = dict(prompt="SECRET", history=["SECRET"], raw="SECRET",
                       raw_results=[dict(snippet="SECRET")], titles=["SECRET"],
                       raw_answer="SECRET", error="SECRET", timestamp="SECRET",
                       snapshot={"raw": "SECRET"}, diagnostics=math.inf)
        event = self.project("snapshot_observed", snapshot_revision=0, entry_count=0,
                             self_claims=["SECRET"], current_info="SECRET", **private)
        self.assertNotIn("SECRET", json.dumps(event))
        self.assertEqual(event["payload"]["system"]["entry_count"], 0)
        for name in EVENT_SPECS:
            self.assertNotIn("SECRET", json.dumps(self.project(name, **private)))

    def test_error_category_only_and_grounding_not_inferred(self):
        event = self.project("internet_search_error", error="QuotaError: PRIVATE credential/path")
        self.assertEqual(event["payload"]["internet"]["error_category"], "QuotaError")
        self.assertNotIn("credential", json.dumps(event))
        self.assertIsNone(self.project("internet_search_error", error="PRIVATE details")["payload"]["internet"]["error_category"])
        fact = self.project("internet_fact_stored", source_count=2, summary="fact")["payload"]["internet"]
        self.assertEqual(fact["returned_item_count"], 2)
        self.assertIsNone(fact["grounded"])
        response = self.project("internet_search_response", n_results=0, grounded=False)["payload"]["internet"]
        self.assertIs(response["grounded"], False)
        self.assertEqual(response["returned_item_count"], 0)

    def test_missing_empty_null_are_distinct(self):
        for value in (None, [], [""]):
            event = self.project("persona_claims_updated", claims=value)
            self.assertEqual(event["payload"]["persona"]["claims"], value)
        self.assertIsNone(self.project("persona_claims_updated")["payload"]["persona"]["claims"])
        self.assertEqual(self.project("social_state_updated", players={})["payload"]["social_state"]["players"], [])

    def test_nonfinite_retained_reject_diagnostic_discard(self):
        for number in (math.inf, -math.inf, math.nan):
            with self.subTest(number=number):
                with self.assertRaises(ValueError):
                    self.project("behavior_tick", talk_share=number)
                with self.assertRaises(ValueError):
                    self.project("reviewer_analysis", verdicts=[dict(score=number)])
                self.assertTrue(validate_event(self.project("behavior_tick", diagnostic=number)))

    def test_strict_types_and_unapproved_fields(self):
        for name, fields in [
            ("generator_candidates", dict(count=False)),
            ("generator_candidates", dict(candidates=[1])),
            ("reviewer_analysis", dict(verdicts=[dict(selected=0)])),
            ("reviewer_analysis", dict(verdicts=[None])),
            ("internet_search_response", dict(grounded=1)),
            ("social_state_updated", dict(players={"Ari": {"raw": "no"}})),
            ("social_state_tick", dict(parsed="invalid")),
            ("persona_claims_updated", dict(claims="not a list")),
        ]:
            with self.subTest(name=name, fields=fields), self.assertRaises(ValueError):
                self.project(name, **fields)
        base = self.project("scheduler_decision", decision="wait")
        for mutate in (
            lambda e: e.update(extra=True),
            lambda e: e["payload"]["scheduler"].update(extra=True),
            lambda e: e["payload"].update(extra=None),
            lambda e: e.update(component="Generator"),
            lambda e: e.update(event_order=True),
            lambda e: e.update(event_schema=True),
            lambda e: e.update(elapsed_seconds=0),
            lambda e: e.update(event_text="truncated"),
            lambda e: e["payload"].update(delivery={}),
            lambda e: e["payload"]["scheduler"].pop("reason"),
        ):
            bad = copy.deepcopy(base)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_event(bad)

    def test_selection_and_completion_never_become_delivery(self):
        for name, fields in [
            ("reviewer_selected", dict(selected_text="hello")),
            ("pipeline_complete", dict(final="hello")),
            ("turn_context", dict(final_message="hello")),
        ]:
            event = self.project(name, outcome="sent", **fields)
            self.assertIsNone(event["payload"]["delivery"])
        delivered = self.project("delivery_sent", message="hello", outcome="sent")
        self.assertEqual(delivered["payload"]["delivery"]["outcome"], "sent")
        with self.assertRaises(ValueError):
            self.project("delivery_phase_drop", outcome="sent")

    def test_text_is_full_deterministic_escaped_and_redacted(self):
        message = 'PRIVATE "quoted"\nUnicode café\\' + "x" * 5000
        event = self.project("delivery_sent", message=message)
        rendered = render_text(event)
        self.assertEqual(rendered, render_text([json.loads(json.dumps(event))]))
        self.assertIn("x" * 5000, rendered)
        self.assertIn('\\"quoted\\"\\nUnicode café\\\\', rendered)
        self.assertNotIn("PRIVATE", rendered)
        self.assertEqual(json.loads(event["event_text"].split("\n", 1)[1]), event["payload"])
        calls = []
        row = dict(event_type="reviewer_analysis", verdicts=[dict(text="one", pros="two", cons="three", red_flags=["four"])])
        project_event(row, "game_001", 1, lambda s: calls.append(s) or s, {})
        self.assertEqual(calls, ["one", "two", "three", "four"])
        self.assertEqual(render_text([]), "")

    def test_turn_mapping_and_explicit_relative_timing(self):
        event = self.project("generate_end", turn_id="private-turn")
        self.assertEqual(event["turn_id"], "game_001-turn001")
        self.assertNotIn("private-turn", json.dumps(event))
        with self.assertRaises(ValueError):
            self.project("generate_end", turn_id="unknown-turn")
        event = project_event(dict(event_type="agent_ready", timestamp="PRIVATE"),
            "game_001", 2, lambda s: s, {}, dict(elapsed_seconds=0,
            time_basis="since_first_event", time_precision_seconds=1))
        self.assertEqual(event["elapsed_seconds"], 0.0)
        self.assertIs(type(event["time_precision_seconds"]), float)
        self.assertNotIn("timestamp", event)
        for timing in (dict(elapsed_seconds=0), dict(time_basis="since_game_start"),
                       dict(timestamp="PRIVATE"), dict(elapsed_seconds=math.inf),
                       dict(elapsed_seconds=0, time_basis="since_game_start", time_precision_seconds=0)):
            with self.assertRaises(ValueError):
                project_event(dict(event_type="agent_ready"), "game_001", 1, str, {}, timing)

    def test_manifest_only_safe_aggregates(self):
        event = self.project("reviewer_selected", selected_text="hello")
        report = make_manifest([event], {"unknown_event": 2, "privacy_withheld": 1})
        self.assertEqual(report["privacy_status"], "pending")
        self.assertEqual(report["component_counts"]["Reviewer"], 1)
        self.assertEqual(report["component_counts"]["System"], 0)
        self.assertEqual(report["input_omission_count"], 3)
        self.assertEqual(report["time_quality"]["unknown_elapsed_count"], 1)
        self.assertNotIn("game_001", json.dumps(report))
        self.assertNotIn("hello", json.dumps(report))
        self.assertEqual(make_manifest([], ["retired_diagnostic"])["event_count"], 0)
        for omitted in ({"private-source-id": 1}, {"unknown_event": False}, {"unknown_event": -1}):
            with self.assertRaises(ValueError):
                make_manifest([], omitted)
        with self.assertRaises(ValueError):
            make_manifest([event, event], {})


if __name__ == "__main__":
    unittest.main()