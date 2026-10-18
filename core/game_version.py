"""Helpers for agent-version detection and test-game routing.

The "agent version" is a property of the AGENT CODE (this branch), not of
the underlying LLM. The public runtime is V6 regardless of which model it
runs (Gemini, Qwen, GPT, …). New games are stamped v6.

Legacy detection (v1/v2) is preserved only for historical configs that
were created before this convention.
"""
from __future__ import annotations

import json
from pathlib import Path

CURRENT_AGENT_VERSION = "v6"

# Marker added to llm_config in newly-prepared games. When present we
# trust it directly instead of guessing from the model name.
AGENT_VERSION_KEY = "agent_version"


def _get_ai_llm_config(config: dict) -> dict:
    for p in config.get("players", []):
        if p.get("is_ai") or p.get("is_llm") or p.get("is_mafia"):
            return p.get("llm_config") or {}
    return {}


def detect_agent_version(config: dict) -> str:
    """Return 'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'unknown'.

    Order:
      1. Explicit `agent_version` field in llm_config (preferred for new games).
      2. Top-level `agent_version` field on the config (fallback).
      3. Heuristic for legacy Qwen/Llama configs (-> v1/v2).
    4. Default: current agent version (v6).
    """
    llm_cfg = _get_ai_llm_config(config)
    if not llm_cfg:
        return "unknown"

    explicit = llm_cfg.get(AGENT_VERSION_KEY) or config.get(AGENT_VERSION_KEY)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    # Infer from the LangGraph async_type so V6/V5/V4 real games route to
    # games/real/v6|v5|v4/ instead of defaulting to v3 (which mislabelled 9029-9034).
    async_type = (llm_cfg.get("async_type") or "").lower()
    if async_type == "langgraph_v6":
        return "v6"
    if async_type == "langgraph_v5":
        return "v5"
    if async_type == "langgraph_v4":
        return "v4"

    # Legacy heuristic: only old Qwen/Llama-via-Together configs without an
    # explicit version are demoted to v1/v2.
    model = (llm_cfg.get("model_name") or "").lower()
    is_legacy_qwen = (
        ("qwen" in model or "llama" in model)
        and llm_cfg.get("use_together")
        and not llm_cfg.get("use_gemini")
    )
    if is_legacy_qwen:
        reviewer = llm_cfg.get("reviewer_backend")
        async_type = (llm_cfg.get("async_type") or "").lower()
        if reviewer or "reviewed" in async_type:
            return "v2"
        return "v1"

    return CURRENT_AGENT_VERSION


def is_test_game(config: dict) -> bool:
    """Whether the game belongs under games/test/<version>/."""
    return bool(config.get("is_test", False))


def load_config(game_dir: Path) -> dict:
    """Read config.json from a game directory; returns {} on error."""
    try:
        return json.loads((game_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


