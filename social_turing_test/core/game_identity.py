"""Single-agent identity and local real/test game classification.

Model/provider choices do not define a different agent or directory layout.
"""
from __future__ import annotations

import json
from pathlib import Path


CURRENT_AGENT_IDENTITY = "social_turing_test_agent"
AGENT_IDENTITY_KEY = "agent_identity"


def detect_agent_identity(config: dict) -> str:
    """Return the runtime identity, or unknown when no AI is configured.

    There is one supported agent. This function deliberately does not infer
    architecture variants from model names or historical configuration markers.
    Configuration validation remains the responsibility of the caller.
    """
    if any(p.get("is_ai") or p.get("is_llm") for p in config.get("players", [])):
        return CURRENT_AGENT_IDENTITY
    return "unknown"


def is_test_game(config: dict) -> bool:
    """Whether a runtime game belongs under games/test/."""
    return bool(config.get("is_test", False))


def load_config(game_dir: Path) -> dict:
    """Read a game configuration, returning an empty mapping on read errors."""
    try:
        return json.loads((game_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}