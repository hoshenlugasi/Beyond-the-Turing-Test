"""Offline single-agent identity and runtime directory contract."""
from pathlib import Path

import pytest

from social_turing_test.core import game_paths
from social_turing_test.core.game_identity import CURRENT_AGENT_IDENTITY, detect_agent_identity


def test_identity_is_independent_of_model_and_provider():
    for model in ("gemini-2.5-flash", "custom/generator", "Qwen/example"):
        assert detect_agent_identity({"players": [{
            "is_ai": True, "llm_config": {"model_name": model},
        }]}) == CURRENT_AGENT_IDENTITY
    assert detect_agent_identity({"players": []}) == "unknown"


def test_direct_category_layout_and_three_item_iterator(tmp_path, monkeypatch):
    monkeypatch.setattr(game_paths, "DIRS_PREFIX", str(tmp_path))
    real = game_paths.target_dir_for_config({}, "0001")
    test = game_paths.target_dir_for_config({"is_test": True}, "0002")
    assert real == tmp_path / "real" / "0001"
    assert test == tmp_path / "test" / "0002"
    real.mkdir()
    test.mkdir()
    (tmp_path / "game_001").mkdir()
    (tmp_path / "real" / "not_a_game").mkdir()
    assert list(game_paths.iter_all_games()) == [
        ("real", CURRENT_AGENT_IDENTITY, real),
        ("test", CURRENT_AGENT_IDENTITY, test),
    ]
    assert game_paths.resolve_game_dir("0001") == real
    assert game_paths.resolve_game_dir("0002") == test
    assert game_paths.resolve_game_dir("0003") == tmp_path / "real" / "0003"
    assert game_paths.next_game_id() == "0003"


@pytest.mark.parametrize("game_id", ["game_001", "../0001", "123", "12345", ""])
def test_only_four_digit_runtime_ids(game_id):
    with pytest.raises(ValueError):
        game_paths.resolve_game_dir(game_id)
    with pytest.raises(ValueError):
        game_paths.target_dir_for_config({}, game_id)