"""Focused regression checks for neutral component names; no runtime imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import re

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"__builtins__": __builtins__}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def test_strategic_vote_uses_social_state_without_model():
    vote = function("llm_interface.py", "_strategic_vote")
    snapshot = SimpleNamespace(get_social_state=lambda: {"suspects_me": ["Blue"]})
    player = SimpleNamespace(snapshot=snapshot, logger=Mock())
    assert vote(player, ["Blue", "Green"], []) == "Blue"


def test_cadence_reads_room_norm_from_the_same_social_state():
    method = function("llm_players/schedule_then_generate_player.py", "_cadence_scheduling_nudge")
    snapshot = SimpleNamespace(get_cadence=lambda: {"level": "over"},
                               get_social_state=lambda: {"room_norm": "everyone answers"})
    player = SimpleNamespace(snapshot=snapshot)
    assert "strongly prefer to WAIT" not in method(player, False)
    snapshot.get_social_state = lambda: {"room_norm": "selective"}
    assert "strongly prefer to WAIT" in method(player, False)


def test_agent_panel_handlers_and_state_use_one_namespace():
    html = (ROOT / "templates/game.html").read_text(encoding="utf-8")
    assert not re.search(r"(?:[vV][56]|_v[56])", html)
    for name in ("setAgentOutcomeFilter", "toggleAgentCandidate"):
        assert len(re.findall(r"function\s+" + name + r"\s*\(", html)) == 1
        assert 'onclick="' + name + "(" in html
    assert "if (_agentOutcomeFilter === 'sent')" in html
    assert "_agentExpandedCandidates.has(candidateKey)" in html
    for absent in ("Agent Legacy", "active/shadow", "agentActiveStatusCard", "agentFactUsageCard"):
        assert absent not in html


def test_single_agent_schema_and_example_agree_on_synchronous_mode():
    import json
    from core.config_schema import CONFIG_SCHEMA
    mode = next(entry for entry in CONFIG_SCHEMA if entry["key"] == "orchestration_mode")
    assert mode["default"] == "synchronous"
    assert mode["choices"] == ["synchronous"]
    config = json.loads((ROOT / "configurations/example_local_agent.json").read_text(encoding="utf-8"))
    settings = next(player["llm_config"] for player in config["players"] if player["is_llm"])
    assert settings["orchestration_mode"] == "synchronous"
    assert settings["async_type"] == "social_turing_test_agent"