"""Per-model knowledge-cutoff lookup (V5.7 B2).

The agent needs a *soft* sense of where its own training knowledge ends so it can
reason "Oct 7 2023 I'd obviously know, but today's score I wouldn't unless it came
up." A PRECISE cutoff is unreliable to obtain and rarely the real issue (a human
also misses recent news) — the robust behaviour comes from the human-framed recency
line in the system prompt. This module only supplies the soft boundary phrase.

Strategy (never blocks on the network at game time):
  1. llm_config override `knowledge_cutoff` wins.
  2. cached value in model_knowledge_cutoffs.json (seeded with known models).
  3. optional one-time grounding lookup (refresh_cutoff_via_web) writes back to the
     cache — run offline / lazily, NOT in the hot path.
  4. relative fallback "about a year ago" when nothing is known.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_CACHE_PATH = Path(__file__).resolve().parent.parent / "model_knowledge_cutoffs.json"

# Seed values (best-effort, correctable). Keys are matched as case-insensitive
# substrings of the model name so "gemini-2.5-pro", "gemini-2.5-flash" etc. hit.
_SEED = {
    "gemini-2.5":   "early 2025",
    "gemini-2.0":   "mid 2024",
    "gemini-1.5":   "late 2023",
    "gpt-5":        "late 2024",
    "gpt-4o":       "late 2023",
    "claude":       "early 2025",
    "qwen":         "2024",
    "llama-3":      "early 2024",
}


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(d: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def get_knowledge_cutoff(model_name: Optional[str], override: Optional[str] = None) -> str:
    """Return a human phrase for the model's knowledge cutoff (e.g. 'early 2025'),
    or "" if genuinely unknown (caller then uses the no-date human heuristic only).

    Never hits the network. Caches/looks up by case-insensitive substring match.
    """
    if override and str(override).strip():
        return str(override).strip()
    name = (model_name or "").lower()
    if not name:
        return ""
    cache = _load_cache()
    # exact then substring
    if name in cache:
        return cache[name]
    for key, val in {**_SEED, **cache}.items():
        if key.lower() in name:
            return val
    return ""   # unknown → caller falls back to the relative heuristic


def refresh_cutoff_via_web(model_name: str, web_search_fn) -> str:
    """OPTIONAL network call: ask the grounding backend for a model's cutoff
    and write it to the cache. Not called in the game hot path. `web_search_fn`
    is injected (e.g. llm_players.v6.sub_agents.web_search.gemini_grounded_search)."""
    try:
        results = web_search_fn(f"knowledge cutoff training data date of {model_name}", max_results=1)
        text = " ".join(r.get("snippet", "") for r in (results or []))
        m = re.search(r"((?:early|mid|late)\s+)?(20\d{2})", text, re.IGNORECASE)
        if m:
            val = (m.group(1) or "").strip() + " " + m.group(2)
            val = val.strip()
            cache = _load_cache()
            cache[model_name.lower()] = val
            _save_cache(cache)
            return val
    except Exception:
        pass
    return ""
