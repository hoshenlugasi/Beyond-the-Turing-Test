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
import unittest
from unittest.mock import Mock, patch


EXPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPORT))


def _module(name):
    module = importlib.import_module(name)
    # Do not accidentally test the research-root modules in an existing process.
    if not Path(module.__file__).resolve().is_relative_to(EXPORT):
        raise AssertionError(f"Tests must import the exported runtime: {name}")
    return module


def _without_original_evaluator_feedback(source, filename):
    """Normalize ORIGINAL source only; accept precisely five known deletions.

    These fingerprints identify original AST statements, not a new exported
    baseline. Everything else (including all remaining prompt literals) must
    compare structurally with the unmodified exported AST. No original prompt
    text or private source fixture is distributed.
    """
    allowed = {
        "generator_node.py": {
            "f9c3a4799a6cc477c50816dd45848bd5a2266db9c3e7de2c02185d5ee2d538f3",
            "b23af3dfb20386ab54bfb575b2722a4cf980637ab36f44b06848de572e0121bb",
        },
        "reviewer_node.py": {
            "54880157d860732a7fb5fc89a3ae777967cea662ff9853d9f96b1decb9ad3daf",
            "b2aca750dc9fd22dfc0b083be65ed969d8cc3ebfe33281688229e15138f03797",
            "80502c34b5cd4e6fd4ab5542ec6099c814c5bee0378e19806be0464b277956b7",
        },
    }[filename]
    removed = []

    def fingerprint(node):
        return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

    class OriginalFeedbackOnly(ast.NodeTransformer):
        def visit_Assign(self, node):
            return self._statement(node)

        def visit_If(self, node):
            return self._statement(node)

        def _statement(self, node):
            digest = fingerprint(node)
            if digest in allowed:
                removed.append(digest)
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            retained = []
            for key, value in zip(node.keys, node.values):
                item = ast.Tuple(elts=[key, value], ctx=ast.Load()) if key is not None else None
                digest = fingerprint(item) if item is not None else None
                if digest in allowed:
                    removed.append(digest)
                else:
                    retained.append((key, value))
            node.keys = [key for key, _ in retained]
            node.values = [value for _, value in retained]
            return self.generic_visit(node)

    normalized = OriginalFeedbackOnly().visit(ast.parse(source))
    if set(removed) != allowed or len(removed) != len(allowed):
        raise AssertionError(f"{filename}: original evaluator deletion allowlist changed: {removed}")
    return normalized


class CredentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _module("llm_players.llm_wrapper")
        cls.reviewer = _module("llm_players.gemini_reviewer")

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
            module = _module(f"llm_players.{name}_reviewer")
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

    def test_search_helper_authenticates_without_network(self):
        module = _module("llm_players.v6.sub_agents.web_search")
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
    def test_factory_is_v6_only_and_defaults_before_schema(self):
        factory = _module("llm_players.factory")
        self.assertEqual(set(factory.llm_players_classes), {"langgraph_v6"})
        fake_player = Mock()
        with patch.dict(factory.llm_players_classes, {"langgraph_v6": fake_player}, clear=True):
            factory.llm_player_factory({"llm_config": {}})
            self.assertEqual(fake_player.call_args.kwargs["llm_config"]["async_type"], "langgraph_v6")
            with self.assertRaisesRegex(ValueError, "Unknown async_type"):
                factory.llm_player_factory({"llm_config": {"async_type": "langgraph_v5"}})

    def test_inheritance_chain_is_unchanged(self):
        player = _module("llm_players.v6.langgraph_v6_player").LangGraphV6Player
        self.assertEqual([base.__name__ for base in player.__mro__[:5]], [
            "LangGraphV6Player", "LangGraphPlayer", "ReviewedPlayer",
            "ScheduleThenGeneratePlayer", "LLMPlayer",
        ])

    def test_no_private_key_file_reader_or_author_project_in_retained_loaders(self):
        paths = ["llm_wrapper.py", "gemini_reviewer.py", "openrouter_reviewer.py",
                 "together_reviewer.py", "v6/sub_agents/web_search.py"]
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
                    if name and name.startswith("llm_players."):
                        target = EXPORT.joinpath(*name.split("."))
                        self.assertTrue(target.with_suffix(".py").exists() or target.is_dir(),
                                        f"{path.relative_to(EXPORT)} imports missing {name}")

    def test_dormant_architectures_and_profiler_are_absent(self):
        removed = [
            "every_x_messages_player.py", "fine_tuned_player.py", "generate_then_schedule_player.py",
            "v5/sub_agents/web_search.py", "v5/sub_agents/__init__.py", "v5/__init__.py",
            "v5/langgraph_v5_player.py", "v5/shared_state.py", "v5/sub_agents/base.py",
            "v5/sub_agents/behavior_evolution.py", "v5/sub_agents/internet_context.py",
            "v5/sub_agents/persona_monitor.py", "v5/sub_agents/phase_summarizer.py",
            "v5/sub_agents/repetition_monitor.py", "v5/sub_agents/social_memory.py",
            "v6/background_integration.py", "v6/background_worker.py", "v6/langsmith_tracing.py",
            "v6/graphs/background_intelligence_graph.py", "v6/graphs/__init__.py",
            "v6/sub_agents/player_profiler.py", "v6/sub_agents/live_evaluator.py",
            "evaluation_prompts.py",
        ]
        removed += [f"v6/research/{name}.py" for name in (
            "backend", "classifier", "fact_selection", "recovery", "researcher", "searcher",
            "seeder", "shared_snapshot_backend", "source_verifier",
        )]
        for name in removed:
            with self.subTest(name=name):
                self.assertFalse((EXPORT / "llm_players" / name).exists())

    def test_only_exact_evaluator_deletions_when_research_source_available(self):
        research = EXPORT.parents[2]
        paths = [f"llm_players/v6/nodes/{name}_node.py" for name in ("generator", "reviewer")]
        if not all((research / relative).exists() for relative in paths):
            self.skipTest("Original generator/reviewer source is intentionally not distributed")
        for relative in paths:
            with self.subTest(path=relative):
                original = (research / relative).read_text(encoding="utf-8-sig")
                exported = (EXPORT / relative).read_text(encoding="utf-8-sig")
                normalized = _without_original_evaluator_feedback(original, Path(relative).name)
                self.assertEqual(
                    ast.dump(ast.parse(exported), include_attributes=False),
                    ast.dump(normalized, include_attributes=False),
                    "Only the exact original evaluator feedback statements may be deleted",
                )

    def test_scientific_files_hash_parity_when_research_source_available(self):
        research = EXPORT.parents[2]
        if not (research / "llm_players/v6/nodes/reviewer_node.py").exists():
            self.skipTest("Research source is intentionally not distributed")
        paths = [
            "llm_player.py", "prompting_player.py", "schedule_then_generate_player.py",
            "reviewed_player.py", "reviewer_base.py", "v6/graph.py", "v6/conversation_analysis.py",
            "v6/research/query_validator.py", "v6/sub_agents/internet_context.py",
            "v6/sub_agents/phase_summarizer.py", "v6/sub_agents/persona_monitor.py",
            "v6/sub_agents/social_memory.py", "v6/sub_agents/behavior_evolution.py",
            "v6/sub_agents/repetition_monitor.py",
        ]
        paths += [f"v6/nodes/{name}.py" for name in (
            "state_builder", "scheduler_node", "context_builder", "output_node",
        )]
        for relative in paths:
            with self.subTest(path=relative):
                original = (research / "llm_players" / relative).read_bytes()
                exported = (EXPORT / "llm_players" / relative).read_bytes()
                self.assertEqual(hashlib.sha256(exported).hexdigest(), hashlib.sha256(original).hexdigest())
        # Only architecture routing changes in this prompt-bearing module.
        original = (research / "llm_players/llm_constants.py").read_text(encoding="utf-8-sig")
        exported = (EXPORT / "llm_players/llm_constants.py").read_text(encoding="utf-8-sig")
        before = ('ASYNC_TYPES = [SCHEDULE_THEN_GENERATE_TYPE, GENERATE_THEN_SCHEDULE_TYPE,\n'
                  '               FINE_TUNED_TYPE, EVERY_X_MESSAGES_TYPE,\n'
                  '               REVIEWED_SCHEDULE_THEN_GENERATE_TYPE]')
        self.assertIn(before, original)
        self.assertEqual(exported, original.replace(before, 'ASYNC_TYPES = ["langgraph_v6"]', 1))


if __name__ == "__main__":
    unittest.main()