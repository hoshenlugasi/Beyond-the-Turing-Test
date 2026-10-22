"""Offline release checks. SDK clients and the wrapper warm-up are mocked."""

import ast
import hashlib
import importlib
import importlib.util
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REPOSITORY = Path(__file__).resolve().parents[1]
EXPORT = REPOSITORY / "social_turing_test"
sys.path.insert(0, str(REPOSITORY))


def _module(name):
    module = importlib.import_module(name)
    # Do not accidentally test the research-root modules in an existing process.
    if not Path(module.__file__).resolve().is_relative_to(EXPORT):
        raise AssertionError(f"Tests must import the exported runtime: {name}")
    return module


class PromptRestorationTests(unittest.TestCase):
    def test_no_duplicate_top_level_definitions_or_constants(self):
        paths = [EXPORT / "llm_players/llm_constants.py", EXPORT / "llm_players/factory.py"]
        for package in ("agent", "pipeline_base"):
            paths.extend((EXPORT / "llm_players" / package).rglob("*.py"))
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            names = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Assign):
                    names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.append(node.target.id)
            with self.subTest(path=path.relative_to(EXPORT)):
                self.assertEqual(len(names), len(set(names)), "Duplicate module binding")

    def test_prompt_definitions_and_all_builder_branches(self):
        constants = _module("social_turing_test.llm_players.llm_constants")
        self.assertTrue(callable(constants.build_identity_section))
        self.assertTrue(constants.STYLE_AND_BEHAVIOR_GUIDELINES)
        personas = [None, "missing-persona", *constants.load_personas()]
        for persona in personas:
            for investigator in (False, True):
                for experimental in (False, True):
                    for event in (None, "a synthetic local gathering"):
                        with self.subTest(persona=persona, investigator=investigator,
                                          experimental=experimental, event=event):
                            prompt = constants.build_system_prompt(
                                persona, experimental=experimental,
                                investigator=investigator, event_context=event,
                            )
                            self.assertIn(constants.STYLE_AND_BEHAVIOR_GUIDELINES, prompt)
                            self.assertIn(constants.build_identity_section(constants.get_persona(persona)), prompt)
                            rules = (constants.INVESTIGATOR_GAME_RULES_PROMPT if investigator
                                     else constants.GAME_RULES_PROMPT)
                            self.assertIn(rules, prompt)
                            self.assertEqual(constants.INVESTIGATOR_GUIDELINES in prompt, investigator)
                            self.assertEqual(constants._EMOJI_POLICY_BLOCK in prompt, experimental)
                            self.assertEqual("=== SHARED EVENT CONTEXT ===" in prompt, bool(event))
        self.assertEqual(constants.GENERAL_SYSTEM_INFO,
                         constants.build_system_prompt(constants.DEFAULT_PERSONA_ID))

    def test_imported_constant_symbols_exist(self):
        constants = _module("social_turing_test.llm_players.llm_constants")
        for path in (EXPORT / "llm_players").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "social_turing_test.llm_players.llm_constants":
                    for alias in node.names:
                        if alias.name != "*":
                            self.assertTrue(hasattr(constants, alias.name),
                                            f"{path.relative_to(EXPORT)}: missing {alias.name}")


class RenamedRuntimeStateTests(unittest.TestCase):
    def test_explicit_import_symbols_exist_including_lazy_imports(self):
        """A directory existing is insufficient: its initializer may be empty."""
        for package in ("agent", "pipeline_base"):
            for path in (EXPORT / "llm_players" / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if (not isinstance(node, ast.ImportFrom) or not node.module
                            or not node.module.startswith("social_turing_test.llm_players.")):
                        continue
                    module = _module(node.module)
                    for alias in node.names:
                        if alias.name == "*" or hasattr(module, alias.name):
                            continue
                        child = REPOSITORY.joinpath(*node.module.split("."), alias.name)
                        self.assertTrue(child.with_suffix(".py").is_file() or child.is_dir(),
                                        f"{path.relative_to(EXPORT)}: {node.module}.{alias.name}")

    def test_defaults_cover_all_inherited_and_agent_channels(self):
        base = _module("social_turing_test.llm_players.pipeline_base.agent_state")
        agent = _module("social_turing_test.llm_players.agent.agent_state")
        for module in (base, agent):
            self.assertEqual(set(module.AgentState.__annotations__), set(module.DEFAULT_STATE))
        self.assertLess(set(base.AgentState.__annotations__), set(agent.AgentState.__annotations__))
        for key in ("round_summaries", "social_state", "behavior_advice"):
            self.assertIn(key, agent.AgentState.__annotations__)
        for key in ("phase_summaries", "social_memory", "behavior_adapter"):
            self.assertNotIn(key, agent.DEFAULT_STATE)

    def test_snapshot_initial_read_mutators_and_copy_ownership(self):
        snapshot = _module("social_turing_test.llm_players.agent.shared_state").SharedSnapshot()
        before = snapshot.read()
        self.assertEqual(before["round_summaries"], [])
        self.assertEqual(before["social_state"], "")
        self.assertEqual(before["behavior_advice"], "")
        memory = {"suspects_me": ["Blake"], "open_threads": ["a synthetic question"]}
        snapshot.set_social_state(memory)
        memory["suspects_me"].append("Casey")
        self.assertEqual(snapshot.get_social_state()["suspects_me"], ["Blake"])
        snapshot.append_round_summary("Discussion 1", "a synthetic summary")
        snapshot.set_behavior_advice("  vary your phrasing  ")
        current = snapshot.read()
        self.assertEqual(snapshot.last_round_summarized, "Discussion 1")
        self.assertEqual(current["round_summaries"], ["Discussion 1: a synthetic summary"])
        self.assertIn("Blake", current["social_state"])
        self.assertEqual(current["behavior_advice"], "vary your phrasing")
        self.assertEqual(current["snapshot_revision"], before["snapshot_revision"] + 3)
        current["social_state_raw"]["suspects_me"].append("Drew")
        current["round_summaries"].append("not stored")
        self.assertEqual(snapshot.get_social_state()["suspects_me"], ["Blake"])
        self.assertEqual(len(snapshot.read()["round_summaries"]), 1)

    def test_shared_channels_survive_real_langgraph_schema(self):
        """Exercise real channel filtering without constructors, sleeps or models."""
        graph = _module("social_turing_test.llm_players.agent.graph")
        state = _module("social_turing_test.llm_players.agent.agent_state")
        shared = {
            "round_summaries": ["synthetic round context"],
            "social_state": "synthetic social context",
            "behavior_advice": "synthetic style directive",
        }
        seen = []

        def generate(current):
            seen.append({key: current[key] for key in shared})
            return {"draft_message": "synthetic reply"}

        with patch.object(graph, "make_state_builder_node", return_value=lambda s: {}), \
                patch.object(graph, "make_scheduler_node", return_value=lambda s: {"scheduler_decision": "SEND"}), \
                patch.object(graph, "make_context_builder_gen_node", return_value=lambda s: {}), \
                patch.object(graph, "make_generator_node", return_value=generate), \
                patch.object(graph, "make_output_node", return_value=lambda s: {}):
            compiled = graph.build_graph(SimpleNamespace(reviewer=None))
            result = compiled.invoke({**state.DEFAULT_STATE, **shared})
        self.assertEqual(seen, [shared])
        self.assertEqual(result["final_message"], "synthetic reply")

    def test_generator_receives_shared_context_in_actual_prompt(self):
        state = _module("social_turing_test.llm_players.agent.agent_state")
        with tempfile.TemporaryDirectory() as directory:
            for package in ("agent", "pipeline_base"):
                module = _module(f"social_turing_test.llm_players.{package}.nodes.generator_node")
                player = SimpleNamespace(
                    name="Alex", game_dir=Path(directory), logger=Mock(),
                    llm_config={}, llm=SimpleNamespace(generate=Mock(return_value="<skip>")),
                    _apply_memory_window=lambda history: history,
                    get_system_info_message=lambda **kwargs: "synthetic system prompt",
                    _log_cost=Mock(),
                )
                current = {
                    **state.DEFAULT_STATE,
                    "round_summaries": ["ROUND_CONTEXT_SENTINEL"],
                    "social_state": "SOCIAL_CONTEXT_SENTINEL",
                    "behavior_advice": "STYLE_CONTEXT_SENTINEL",
                }
                with self.subTest(package=package):
                    module.make_generator_node(player)(current)
                    prompt = player.llm.generate.call_args.args[0]
                    for sentinel in ("ROUND_CONTEXT_SENTINEL", "SOCIAL_CONTEXT_SENTINEL",
                                     "STYLE_CONTEXT_SENTINEL"):
                        self.assertIn(sentinel, prompt)

    def test_graph_readers_do_not_use_retired_shared_keys(self):
        for package in ("agent", "pipeline_base"):
            for path in (EXPORT / "llm_players" / package / "nodes").glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "get" and node.args
                            and isinstance(node.args[0], ast.Constant)):
                        self.assertNotIn(node.args[0].value, {"phase_summaries", "social_memory"},
                                         str(path.relative_to(EXPORT)))


class CredentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _module("social_turing_test.llm_players.llm_wrapper")
        cls.reviewer = _module("social_turing_test.llm_players.gemini_reviewer")

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        # __init__ normally makes a paid warm-up generation. Never run it here.
        self.warmup = patch.object(self.wrapper.LLMWrapper, "generate", return_value="")
        self.warmup_mock = self.warmup.start()
        self.addCleanup(self.warmup.stop)
        self.client_patch = patch.object(self.wrapper.genai, "Client")
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.no_hf = patch.object(
            self.wrapper, "_ensure_hf_imports",
            side_effect=AssertionError("Cloud authentication must not load HuggingFace"),
        )
        self.no_hf.start()
        self.addCleanup(self.no_hf.stop)

    def make_wrapper(self, **kwargs):
        return self.wrapper.LLMWrapper(Mock(), model_name="gemini-2.5-flash", **kwargs)

    def test_key_loaders_never_read_private_files(self):
        loaders = [self.wrapper.get_api_key, lambda _: self.reviewer._get_gemini_api_key()]
        for name, loader in (
            ("openrouter", "_get_openrouter_api_key"),
            ("together", "_get_together_api_key"),
        ):
            # Together is an optional adapter, but test it whenever installed.
            if name == "together" and importlib.util.find_spec("together") is None:
                continue
            module = _module(f"social_turing_test.llm_players.{name}_reviewer")
            loaders.append(lambda _, m=module, attr=loader: getattr(m, attr)())
        with patch("builtins.open", side_effect=AssertionError("private file read")), \
                patch.object(Path, "read_text", side_effect=AssertionError("private file read")):
            for load in loaders:
                self.assertIsNone(load("GEMINI_API_KEY"))

    def test_gemini_environment_keys_and_no_key_logging(self):
        for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "mock-key-9zQ7"}):
                output = io.StringIO()
                with redirect_stdout(output):
                    instance = self.make_wrapper(use_gemini=True)
                self.client.assert_called_with(api_key="mock-key-9zQ7")
                self.assertFalse(instance.use_pipeline)
                self.assertNotIn("9zQ7", output.getvalue())
                self.warmup_mock.assert_called()

    def test_explicit_vertex_project_and_location(self):
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "operator-project",
            "GOOGLE_CLOUD_LOCATION": "operator-location",
        }):
            self.make_wrapper(use_gemini=True)
            self.client.assert_called_once_with(
                vertexai=True, project="operator-project", location="operator-location"
            )

    def test_missing_or_partial_vertex_auth_fails_before_warmup(self):
        for environment in ({}, {"GOOGLE_CLOUD_PROJECT": "operator-project"},
                            {"GOOGLE_CLOUD_LOCATION": "operator-location"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                with self.assertRaisesRegex(RuntimeError, "authentication missing"):
                    self.make_wrapper(use_gemini=True)
        self.client.assert_not_called()
        self.warmup_mock.assert_not_called()

    def test_invalid_provider_does_not_fall_back_to_local_model(self):
        with self.assertRaisesRegex(ValueError, "No model provider selected"):
            self.make_wrapper()
        self.client.assert_not_called()
        self.warmup_mock.assert_not_called()

    def test_together_and_openrouter_missing_keys_fail(self):
        for flag in ("use_together", "use_openrouter"):
            with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, "key missing"):
                self.make_wrapper(**{flag: True})
        self.warmup_mock.assert_not_called()

    def test_together_and_openrouter_keep_adapters(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "mock-together"}), \
                patch.object(self.wrapper, "Together") as together:
            self.make_wrapper(use_together=True)
            together.assert_called_once_with(api_key="mock-together")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "mock-openrouter"}):
            instance = self.make_wrapper(use_openrouter=True)
            self.assertEqual(instance.openrouter_api_key, "mock-openrouter")
        self.client.assert_not_called()

    def test_reviewer_uses_own_model_and_no_key_suffix(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-key-9zQ7"}):
            output = io.StringIO()
            with redirect_stdout(output):
                instance = self.reviewer.GeminiReviewer(model_name="gemini-2.5-pro")
            self.assertEqual(instance.model_name, "gemini-2.5-pro")
            self.assertNotIn("9zQ7", output.getvalue())
            self.client.assert_called_once_with(api_key="mock-key-9zQ7")

    def test_reviewer_requires_explicit_vertex_settings(self):
        with self.assertRaisesRegex(RuntimeError, "authentication missing"):
            self.reviewer.GeminiReviewer()
        self.client.assert_not_called()
        self.reviewer.GeminiReviewer(project_id="operator-project", location="operator-location")
        self.client.assert_called_once_with(
            vertexai=True, project="operator-project", location="operator-location"
        )

    def test_renamed_search_reaches_sdk_once_without_recursive_alias(self):
        module = _module("social_turing_test.llm_players.agent.sub_agents.web_search")
        sdk = Mock()
        sdk.models.generate_content.return_value = SimpleNamespace(
            text="A synthetic search answer.", candidates=[],
        )
        with patch.object(module, "_get_client", return_value=sdk), \
                patch.object(module._time, "sleep", side_effect=AssertionError("Unexpected retry")):
            result = module.grounded_search("synthetic query", today_str="2026-10-18")
        self.assertEqual(result, [{
            "title": "answer", "snippet": "A synthetic search answer.", "url": "", "grounded": False,
        }])
        sdk.models.generate_content.assert_called_once()
        request = sdk.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-2.5-flash")
        self.assertIn("Today is 2026-10-18", request["contents"])
        self.assertIn("Search for: synthetic query", request["contents"])
        self.assertEqual(request["config"].temperature, 0.3)
        self.assertIsNotNone(request["config"].tools[0].google_search)

    def test_search_helper_authenticates_without_network(self):
        module = _module("social_turing_test.llm_players.agent.sub_agents.web_search")
        with patch.object(module, "_client_cache", None):
            self.client.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "authentication missing"):
                module._get_client()
            self.client.assert_not_called()
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "mock-search"}):
                module._get_client()
                module._get_client()
            self.client.assert_called_once_with(api_key="mock-search")
        with patch.object(module, "_client_cache", None), patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "operator-project",
            "GOOGLE_CLOUD_LOCATION": "operator-location",
        }):
            self.client.reset_mock()
            module._get_client()
            self.client.assert_called_once_with(
                vertexai=True, project="operator-project", location="operator-location"
            )


class RuntimeClosureTests(unittest.TestCase):
    def test_factory_is_single_agent_and_defaults_before_schema(self):
        factory = _module("social_turing_test.llm_players.factory")
        self.assertEqual(set(factory.llm_players_classes), {"social_turing_test_agent"})
        fake_player = Mock()
        with patch.dict(factory.llm_players_classes, {"social_turing_test_agent": fake_player}, clear=True):
            factory.llm_player_factory({"llm_config": {}})
            self.assertEqual(fake_player.call_args.kwargs["llm_config"]["async_type"], "social_turing_test_agent")
            with self.assertRaisesRegex(ValueError, "Unknown async_type"):
                factory.llm_player_factory({"llm_config": {"async_type": "unsupported_agent"}})

    def test_inheritance_chain_is_unchanged(self):
        player = _module("social_turing_test.llm_players.agent.agent").SocialTuringTestAgent
        self.assertEqual([base.__name__ for base in player.__mro__[:5]], [
            "SocialTuringTestAgent", "PipelinePlayer", "ReviewedPlayer",
            "ScheduleThenGeneratePlayer", "LLMPlayer",
        ])

    def test_no_private_key_file_reader_or_author_project_in_retained_loaders(self):
        paths = ["llm_wrapper.py", "gemini_reviewer.py", "openrouter_reviewer.py",
                 "together_reviewer.py", "agent/sub_agents/web_search.py"]
        for relative in paths:
            with self.subTest(path=relative):
                text = (EXPORT / "llm_players" / relative).read_text(encoding="utf-8")
                self.assertNotIn("SECRETS_DICT_FILE_PATH", text)
                self.assertNotIn("project-aec", text)
                self.assertNotIn("api_key[-", text)

    def test_exported_llm_import_targets_exist(self):
        # Includes function-local imports, so missing lazy dependencies are caught.
        for path in (EXPORT / "llm_players").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                names = ([node.module] if isinstance(node, ast.ImportFrom) else
                         [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    if name and name.startswith("social_turing_test.llm_players."):
                        target = REPOSITORY.joinpath(*name.split("."))
                        self.assertTrue(target.with_suffix(".py").exists() or target.is_dir(),
                                        f"{path.relative_to(EXPORT)} imports missing {name}")

    def test_dormant_architectures_and_profiler_are_absent(self):
        removed = [
            "every_x_messages_player.py", "fine_tuned_player.py", "generate_then_schedule_player.py",
            "agent/background_integration.py", "agent/background_worker.py", "agent/langsmith_tracing.py",
            "agent/graphs/background_intelligence_graph.py", "agent/graphs/__init__.py",
            "agent/sub_agents/player_profiler.py", "agent/sub_agents/live_evaluator.py",
            "evaluation_prompts.py",
        ]
        removed += [f"agent/research/{name}.py" for name in (
            "backend", "classifier", "fact_selection", "recovery", "researcher", "searcher",
            "seeder", "shared_snapshot_backend", "source_verifier",
        )]
        for name in removed:
            with self.subTest(name=name):
                self.assertFalse((EXPORT / "llm_players" / name).exists())

if __name__ == "__main__":
    unittest.main()