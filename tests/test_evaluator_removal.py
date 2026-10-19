"""Offline constructor regression: real public factory/wrappers, fake SDK only.

No game loop or background worker executes. Constructor warm-ups DO execute
through LLMWrapper.generate and are captured at models.generate_content.
"""

import ast
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


EXPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPORT))
WORKERS = [
    "InternetContextAgent", "RoundSummarizerAgent", "PersonaMonitorAgent",
    "SocialStateTrackerAgent", "BehaviorAdapterAgent", "RepetitionMonitorAgent",
]


def _module(name):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(EXPORT):
        raise AssertionError(f"Tests must import the exported runtime: {name}")
    return module


class EvaluatorRemovalTests(unittest.TestCase):
    def setUp(self):
        self.factory = _module("llm_players.factory")
        self.wrapper = _module("llm_players.llm_wrapper")
        self.reviewer = _module("llm_players.gemini_reviewer")
        self.base = _module("llm_players.agent.sub_agents.base")
        self.constants = _module("core.game_constants")
        self.environment = patch.dict(os.environ, {"GEMINI_API_KEY": "offline-sdk-spy"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.game_dir = Path(self.temp.name) / "synthetic_game"
        self.game_dir.mkdir()
        example = json.loads((EXPORT / "configurations/example_local_agent.json").read_text(encoding="utf-8"))
        self.example = next(p for p in example["players"] if p["is_ai"])
        # Real base constructors and Logger write only to this synthetic game.
        (self.game_dir / "logs").mkdir()
        for relative, text in (
            (self.constants.GAME_START_TIME_FILE, ""),
            (self.constants.REMAINING_PLAYERS_FILE, "Alex\nBlake\nCasey\n"),
            (self.constants.PERSONAL_CHAT_FILE_FORMAT.format(self.example["name"]), ""),
        ):
            path = self.game_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.sdk_calls = []
        self.clients = []

        def generate_content(**kwargs):
            self.sdk_calls.append(kwargs)
            return SimpleNamespace(
                text="<wait>", candidates=[],
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )

        def client(**kwargs):
            instance = Mock()
            instance.models.generate_content.side_effect = generate_content
            self.clients.append(instance)
            return instance

        sdk = patch.object(self.wrapper.genai, "Client", side_effect=client)
        self.client_constructor = sdk.start()
        self.addCleanup(sdk.stop)
        # Fail closed even if a regression accidentally takes another provider path.
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("Network forbidden in offline regression"))
            guard.start()
            self.addCleanup(guard.stop)
        for attr in ("Together", "_ensure_hf_imports"):
            guard = patch.object(self.wrapper, attr, side_effect=AssertionError("Unexpected model provider"))
            guard.start()
            self.addCleanup(guard.stop)

    def construct(self, settings):
        config = copy.deepcopy(self.example)
        config["llm_config"] = copy.deepcopy(settings)
        config["game_dir"] = self.game_dir
        original_wrapper_init = self.wrapper.LLMWrapper.__init__
        original_reviewer_init = self.reviewer.GeminiReviewer.__init__
        wrappers = []
        reviewers = []

        def wrapper_init(instance, *args, **kwargs):
            wrappers.append(instance)
            original_wrapper_init(instance, *args, **kwargs)

        def reviewer_init(instance, *args, **kwargs):
            reviewers.append(instance)
            original_reviewer_init(instance, *args, **kwargs)

        # Spy on actual constructors, not replacements for their objects/logic.
        with patch.object(self.wrapper.LLMWrapper, "__init__", autospec=True, side_effect=wrapper_init), \
                patch.object(self.reviewer.GeminiReviewer, "__init__", autospec=True, side_effect=reviewer_init), \
                patch.object(self.base.Thread, "start", autospec=True) as thread_start:
            player = self.factory.llm_player_factory(config)
        self.assertEqual([type(a).__name__ for a in player.sub_agents], WORKERS)
        self.assertEqual([c.args[0].name for c in thread_start.call_args_list], WORKERS)
        self.assertTrue(all(not a.is_running for a in player.sub_agents))
        self.assertEqual(len(reviewers), 1)
        self.assertIs(player.reviewer, reviewers[0])
        self.assertIsInstance(player, self.factory.SocialTuringTestAgent)
        self.assertIsInstance(player.llm, self.wrapper.LLMWrapper)
        self.assertIsInstance(player.reviewer, self.reviewer.GeminiReviewer)
        self.assertTrue(all(a.agent_llm is player.sub_llm for a in player.sub_agents))
        self.assertFalse(any("evaluator" in type(a).__name__.lower() for a in player.sub_agents))
        self.assertFalse(any("live_evaluator" in key for key in vars(player)))
        models = [w.model_name for w in wrappers] + [r.model_name for r in reviewers]
        self.assertEqual(set(models), {"gemini-2.5-flash"})
        self.assertEqual(len(self.clients), len(wrappers) + len(reviewers))
        self.assertGreaterEqual(len(self.sdk_calls), len(wrappers))
        # Each actual wrapper constructor made its initial generation request.
        for wrapper in wrappers:
            self.assertTrue(wrapper.client.models.generate_content.called)
        self.assert_sdk_flash_only()
        return player, wrappers

    def assert_sdk_flash_only(self):
        self.assertTrue(self.sdk_calls, "Warm-ups must reach the SDK spy, not a mocked wrapper")
        self.assertEqual({call["model"] for call in self.sdk_calls}, {"gemini-2.5-flash"})
        self.assertFalse(any("pro" in call["model"].lower() for call in self.sdk_calls))
        for client in self.clients:
            client.models.generate_content_stream.assert_not_called()

    def test_public_example_real_factory_warmups_and_reviewer_sdk(self):
        player, wrappers = self.construct(self.example["llm_config"])
        self.assertEqual(len(wrappers), 2)  # generator plus explicitly configured scheduler
        self.assertIs(player.sub_llm, player.llm)
        self.assertIsNot(player.scheduler, player.llm)
        self.assertEqual(player.llm_config["num_candidates"], 4)
        self.assertEqual(player.llm_config["pipeline_timeout_s"], 35.0)
        self.assertEqual(player.llm_config["voting_strategy"], "strategic")
        # The reviewer adapter is real too; its executor can now run, but its
        # Google SDK is still intercepted and socket connections remain blocked.
        before = len(self.sdk_calls)
        self.assertEqual(player.reviewer._call_api("Synthetic offline reviewer probe"), "<wait>")
        self.assertEqual(len(self.sdk_calls), before + 1)
        self.assert_sdk_flash_only()

    def test_schema_defaults_real_factory_have_no_default_pro(self):
        player, wrappers = self.construct({})
        self.assertEqual(len(wrappers), 1)
        self.assertIs(player.scheduler, player.llm)
        self.assertIs(player.sub_llm, player.llm)
        self.assertFalse(any(k.startswith("live_evaluator_") for k in player.llm_config))

    def test_dedicated_subagent_wrapper_warmup_is_also_observed(self):
        settings = dict(self.example["llm_config"], sub_agent_model="gemini-2.5-flash")
        player, wrappers = self.construct(settings)
        self.assertEqual(len(wrappers), 3)
        self.assertIsNot(player.sub_llm, player.llm)
        self.assertIsNot(player.sub_llm, player.scheduler)

    def test_retired_settings_rejected_before_schema_fallback_or_any_constructor(self):
        for key in ("live_evaluator_model", "live_evaluator_feedback_enabled", "live_evaluator_future"):
            for value in (False, None, "gemini-2.5-pro"):
                for top_level in (False, True):
                    with self.subTest(key=key, value=value, top_level=top_level):
                        config = copy.deepcopy(self.example)
                        config["game_dir"] = self.game_dir
                        (config if top_level else config["llm_config"])[key] = value
                        with patch.object(self.factory, "apply_defaults", side_effect=self.factory.ConfigError("forced fallback")) as defaults, \
                                patch.dict(self.factory.llm_players_classes, {"social_turing_test_agent": Mock()}, clear=True), \
                                patch.object(self.wrapper.LLMWrapper, "__init__", side_effect=AssertionError("Constructor forbidden")) as wrapper_init, \
                                patch.object(self.reviewer.GeminiReviewer, "__init__", side_effect=AssertionError("Constructor forbidden")) as reviewer_init:
                            with self.assertRaisesRegex(ValueError, "retired"):
                                self.factory.llm_player_factory(config)
                            defaults.assert_not_called()
                            self.factory.llm_players_classes["social_turing_test_agent"].assert_not_called()
                            wrapper_init.assert_not_called()
                            reviewer_init.assert_not_called()
        self.client_constructor.assert_not_called()
        self.assertEqual(self.sdk_calls, [])

    def test_evaluator_modules_state_and_runtime_references_are_absent(self):
        for relative in ("llm_players/agent/sub_agents/live_evaluator.py", "llm_players/evaluation_prompts.py"):
            self.assertFalse((EXPORT / relative).exists(), relative)
        state = _module("llm_players.agent.agent_state")
        snapshot = _module("llm_players.agent.shared_state").SharedSnapshot()
        self.assertNotIn("self_weaknesses", state.DEFAULT_STATE)
        self.assertNotIn("self_weaknesses", state.AgentState.__annotations__)
        self.assertNotIn("self_weaknesses", snapshot.read())
        for attr in ("self_weaknesses_state", "set_self_weaknesses", "get_self_weaknesses"):
            self.assertFalse(hasattr(snapshot, attr), attr)
        for path in (EXPORT / "llm_players").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, {"LiveEvaluatorAgent", "self_weaknesses"}, str(path))
                elif isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, {"self_weaknesses_state", "set_self_weaknesses", "get_self_weaknesses"}, str(path))
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(node.value, {"self_weaknesses", "self_weaknesses_state"}, str(path))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, {"llm_players.evaluation_prompts", "llm_players.agent.sub_agents.live_evaluator"}, str(path))


if __name__ == "__main__":
    unittest.main()