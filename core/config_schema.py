"""Single source of truth for tunable LLM/agent config keys.

Everything arbitrary — thresholds, typing speed, memory window, reviewer behavior —
lives here with a declared type, default, and range. `apply_defaults()` fills missing
keys and validates ranges. The admin UI is generated from `describe()` so any knob
added here shows up in the form automatically.

Design goals:
- Defaults stay identical to pre-schema hard-coded values, so existing configs
  behave the same after the switch.
- Invalid values raise loudly (ConfigError) instead of silently accepting.
- Schema is cheap to import — no heavy deps, no network.
"""
from typing import Any


class ConfigError(ValueError):
    pass


def reject_retired_config(config: Any) -> None:
    """Reject retired live_evaluator_* keys in settings or nested game payloads.

    Raises ConfigError (a ValueError) even when a retired value is false or null.
    Does not mutate the input or include configuration values in the error.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(key, str) and key.startswith("live_evaluator_"):
                raise ConfigError(f"{key}: retired setting; remove it from the configuration")
            reject_retired_config(value)
    elif isinstance(config, (list, tuple)):
        for value in config:
            reject_retired_config(value)


# Each entry: {key, type, default, optional fields: label, description, group,
# required, min, max, choices}
# Types: "bool", "int", "float", "string", "choice", "int_or_null",
#        "float_or_null", "list[string]", "opaque" (pass-through, no validation)
CONFIG_SCHEMA: list[dict[str, Any]] = [
    # ── Routing / player type ──
    {"key": "async_type", "type": "choice", "group": "routing", "required": True,
    "choices": ["langgraph_v6"],
    "default": "langgraph_v6",
     "description": "Player class to instantiate."},
    {"key": "persona", "type": "string", "group": "routing", "default": None,
     "description": "Persona ID from configurations/personas.json."},

    # ── Generator / scheduler LLM ──
    {"key": "model_name", "type": "string", "group": "generator", "required": True,
    "default": "gemini-2.5-flash",
     "description": "Same model used for scheduler + generator."},
    {"key": "use_together", "type": "bool", "group": "generator", "default": False},
    {"key": "use_gemini", "type": "bool", "group": "generator", "default": True},
    {"key": "use_pipeline", "type": "bool", "group": "generator", "default": False},
    {"key": "use_openrouter", "type": "bool", "group": "generator", "default": False},
    {"key": "pipeline_task", "type": "string", "group": "generator", "default": "text-generation"},
    {"key": "temperature", "type": "float", "group": "generator",
     "default": 1.3, "min": 0.0, "max": 2.0},
    {"key": "max_tokens", "type": "int", "group": "generator",
     "default": 25, "min": 1, "max": 500},
    {"key": "max_new_tokens", "type": "int", "group": "generator",
     "default": 25, "min": 1, "max": 500},
    {"key": "num_beams", "type": "int", "group": "generator",
     "default": 1, "min": 1, "max": 8},
    {"key": "repetition_penalty", "type": "float", "group": "generator",
     "default": 1.25, "min": 1.0, "max": 2.5},
    {"key": "do_sample", "type": "bool", "group": "generator", "default": True},
    {"key": "no_repeat_ngram_size", "type": "int", "group": "generator",
     "default": 8, "min": 0, "max": 20},

    # ── Scheduler tokens ──
    {"key": "pass_turn_token", "type": "string", "group": "scheduler",
     "default": "<wait>"},
    {"key": "use_turn_token", "type": "string", "group": "scheduler",
     "default": "<send>"},
    {"key": "scheduler_model", "type": "string", "group": "scheduler", "default": None,
     "description": "V6 only: override the model used for the send/wait scheduler. "
                    "Empty = share the generator's model_name. Provider is auto-detected "
                    "from the model name."},
    {"key": "num_words_per_second_to_wait", "type": "float", "group": "scheduler",
     "default": 1.0, "min": 0.1, "max": 10.0,
     "description": "Legacy scheduler pacing knob (pre-typing-rhythm). "
                    "See typing_wps for the new adaptive delay."},

    # ── Reviewer ──
    {"key": "reviewer_enabled", "type": "bool", "group": "reviewer", "default": True},
    {"key": "reviewer_backend", "type": "choice", "group": "reviewer",
     "choices": ["gemini", "together", "openrouter"], "default": "gemini"},
    {"key": "reviewer_model", "type": "string", "group": "reviewer", "default": None,
     "description": "Override reviewer model. Empty = backend default (gemini-2.5-flash)."},
    {"key": "reviewer_threshold", "type": "int", "group": "reviewer",
     "default": 4, "min": 1, "max": 10,
     "description": "Score ≥ threshold triggers rewrite."},
    {"key": "reviewer_revert_threshold", "type": "int", "group": "reviewer",
     "default": 7, "min": 1, "max": 10,
     "description": "If best attempt ≥ this, revert to original message."},
    {"key": "reviewer_max_attempts", "type": "int", "group": "reviewer",
     "default": 3, "min": 1, "max": 5,
     "description": "Max review-and-rewrite attempts per message. "
                    "DEFAULT 3 = generator output + up to 2 rewrites. "
                    "Each attempt is one Gemini API call (~5–15s). "
                    "Bounded by reviewer_budget_sec: if total time runs out, only "
                    "attempts that finished count. Lower = faster but less polished; "
                    "higher = better quality at risk of latency."},
    {"key": "reviewer_self_consistency", "type": "bool", "group": "reviewer",
     "default": True,
     "description": "Re-score the reviewer's own rewrite on flash-lite; abort if rewrite also fails."},
    {"key": "reviewer_skip_short_words", "type": "int", "group": "reviewer",
     "default": 0, "min": 0, "max": 10,
     "description": "Skip reviewer for messages ≤ N words (0 disables shortcut — review every message)."},
    {"key": "reviewer_ensemble", "type": "list[string]", "group": "reviewer",
     "default": [],
     "description": "Extra reviewer models run in parallel; scores averaged. "
                    "Same-latency only when all reviewers are similarly fast."},

    # ── Typing rhythm (adaptive delay) ──
    {"key": "typing_delay_enabled", "type": "bool", "group": "typing", "default": True},
    {"key": "typing_wps", "type": "float", "group": "typing",
     "default": 1.6, "min": 0.5, "max": 5.0,
     "description": "Words-per-second target (1.6 wps ≈ 96 WPM). "
                    "Adaptive — we only wait if generation+review finished faster than ideal."},
    {"key": "typing_jitter_pct", "type": "float", "group": "typing",
     "default": 0.25, "min": 0.0, "max": 1.0},
    {"key": "typing_min_delay_sec", "type": "float", "group": "typing",
     "default": 0.8, "min": 0.0, "max": 10.0},
    {"key": "typing_max_delay_sec", "type": "float", "group": "typing",
     "default": 12.0, "min": 1.0, "max": 60.0},
    {"key": "typing_min_words", "type": "int", "group": "typing",
     "default": 2, "min": 0, "max": 10,
     "description": "Legacy short-message skip threshold (no longer skips — the base "
                    "delay below now applies to all messages)."},
    {"key": "typing_base_delay_sec", "type": "float", "group": "typing",
     "default": 2.0, "min": 0.0, "max": 10.0,
     "description": "Constant reading/thinking delay added on top of ~1s/word, applied "
                    "to ALL messages (incl. short) so the agent never replies instantly."},
    {"key": "reviewer_budget_sec", "type": "float_or_null", "group": "reviewer",
     "default": None, "min": 5.0, "max": 120.0,
     "description": "MAX TOTAL LATENCY (seconds) from scheduler-decision-to-send. "
                    "Includes generator + every reviewer attempt + typing wait. "
                    "DEFAULT null = no budget cap (pre-change behavior). "
                    "Set a number to enforce a hard cap; when exceeded mid-loop, "
                    "the best-scored attempt so far is sent (NEVER dropped just "
                    "because time ran out). Lower = snappier; higher = more time "
                    "for quality fixes."},
    {"key": "reviewer_drop_score", "type": "int", "group": "reviewer",
     "default": 10, "min": 1, "max": 10,
     "description": "DROP THRESHOLD: skip the turn entirely if even our best attempt "
                    "scored ≥ N suspicion. DEFAULT 10 = never drop (always send "
                    "something — pre-change behavior). Lower (e.g. 8) to drop when "
                    "even rewrites still sound very AI."},
    {"key": "reviewer_api_timeout_sec", "type": "float_or_null", "group": "reviewer",
     "default": None, "min": 2.0, "max": 60.0,
     "description": "Per-call Gemini API timeout (seconds). DEFAULT null = no "
                    "timeout (pre-change behavior — call blocks until done). "
                    "Set a number to abort a hung Gemini call mid-attempt; on "
                    "timeout the loop tries the next attempt or falls back to "
                    "SELECTED_BEST."},
    {"key": "reviewer_backoff_enabled", "type": "bool", "group": "reviewer",
     "default": False,
     "description": "If a reviewer attempt took >5s, sleep briefly before retrying "
                    "(0.5s × attempt#). DEFAULT FALSE (pre-change behavior — retry "
                    "immediately). Set true to give a slow Gemini backend time to "
                    "recover instead of hammering it."},

    # ── Memory window ──
    {"key": "memory_window_size", "type": "int_or_null", "group": "memory",
     "default": None, "min": 1, "max": 1000,
     "description": "Last N messages kept verbatim. null (default) = full history."},
    {"key": "memory_decay_mode", "type": "choice", "group": "memory",
     "choices": ["none", "summarize_older", "drop_older"],
     "default": "none"},

    # ── Voting ──
    {"key": "voting_strategy", "type": "choice", "group": "voting",
        "choices": ["random", "strategic"], "default": "strategic",
        "description": "Strategic (final V6 default) asks the LLM who most suspects "
                    "the AI; random ignores context."},

    # ── Experimental (v4-prep bundle) ──
    {"key": "experimental_agent", "type": "bool", "group": "experimental",
     "default": False,
     "description": "Enable v4-prep bundle: emoji policy in system prompt, "
                    "larger max_tokens (see experimental_max_tokens), and an "
                    "anti-shrink clause in the reviewer retry prompt. "
                    "DEFAULT FALSE = legacy agent unchanged."},
    {"key": "experimental_max_tokens", "type": "int", "group": "experimental",
     "default": 80, "min": 0, "max": 400,
     "description": "When experimental_agent=true, override max_tokens / "
                    "max_new_tokens with this value. 0 keeps the original "
                    "max_tokens. DEFAULT 80."},

    # ── V6 (langgraph_v6 only; ignored by earlier agents) ──
    {"key": "multi_candidate_generation", "type": "bool", "group": "v6",
        "default": True,
     "description": "V6: generator emits several candidate messages and the "
                    "reviewer SELECTS one (ranker mode) instead of rewriting. "
                    "Avoids the reviewer drifting a good message into a bad one."},
    {"key": "num_candidates", "type": "int", "group": "v6",
     "default": 4, "min": 2, "max": 8,
     "description": "V6: how many candidates the generator produces when "
                    "multi_candidate_generation=true."},
    {"key": "investigator", "type": "bool", "group": "v6",
     "default": True,
     "description": "V6: give the AI the human players' dual role — it also "
                    "hunts the AI and votes, removing the passive-target "
                    "asymmetry. Swaps in the V6 dual-role system prompt."},
    {"key": "pipeline_timeout_s", "type": "float", "group": "v6",
     "default": 35.0, "min": 5.0, "max": 120.0,
     "description": "V6 safety net: drop a message only if the whole pipeline "
                    "took longer than this (extreme hang). A normal response "
                    "delay is fine and is never dropped."},
    {"key": "orchestration_mode", "type": "choice", "group": "v6",
        "default": "legacy", "choices": ["legacy"],
        "description": "Final V6 legacy pipeline. The local backend locks this mode; "
                    "shadow/active rollout modes are not supported."},
    # ── V6 legacy InternetContext sub-agent tuning (Fix 5) ──
    {"key": "internet_max_queries_per_tick", "type": "int", "group": "v6",
     "default": 5, "min": 1, "max": 10,
     "description": "Max web searches the InternetContext sub-agent runs per tick. "
                    "Raised from 2 so multiple distinct recent-event topics in one "
                    "message are all searched (dedup/cache bound the cost)."},
    {"key": "internet_detector_model", "type": "string", "group": "v6", "default": None,
     "description": "Optional cheaper model for the InternetContext 'should I search?' "
                    "decision call (e.g. gemini-2.5-flash-lite). None → reuse the main "
                    "model."},
    # ── V6 Wave 1 — background research lane + LangSmith (all opt-in) ──
    {"key": "background_intelligence_enabled", "type": "bool", "group": "v6",
     "default": False,
     "description": "V6 Wave 1: enable the separate BackgroundIntelligenceGraph "
                    "research lane (off the authoritative hot path). Default off."},
    {"key": "langsmith_tracing_enabled", "type": "bool", "group": "v6",
     "default": False,
     "description": "V6: optional LangSmith tracing for the graph lanes. Never "
                    "authoritative; gameplay is identical when off. Default off."},
    {"key": "langsmith_project", "type": "string", "group": "v6", "default": None,
     "description": "V6: LangSmith project name (when tracing is enabled)."},
    {"key": "langsmith_sample_rate", "type": "float", "group": "v6",
     "default": 1.0, "min": 0.0, "max": 1.0,
     "description": "V6: fraction of traces to sample when tracing is enabled "
                    "(0..1). 1.0 = trace everything; 0.0 = trace nothing."},
    {"key": "langsmith_include_chat_content", "type": "bool", "group": "v6",
     "default": False,
     "description": "V6: include chat content in traces. Default OFF (privacy)."},
    {"key": "langsmith_redaction_enabled", "type": "bool", "group": "v6",
     "default": True,
     "description": "V6: redact excluded/sensitive fields before any trace emit."},
]

_SCHEMA_BY_KEY = {entry["key"]: entry for entry in CONFIG_SCHEMA}


def _validate(key: str, value: Any, spec: dict) -> Any:
    t = spec["type"]
    if t == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected bool, got {type(value).__name__}")
    elif t == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected int, got {type(value).__name__}")
        if "min" in spec and value < spec["min"]:
            raise ConfigError(f"{key}: {value} < min {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ConfigError(f"{key}: {value} > max {spec['max']}")
    elif t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected number, got {type(value).__name__}")
        value = float(value)
        if "min" in spec and value < spec["min"]:
            raise ConfigError(f"{key}: {value} < min {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ConfigError(f"{key}: {value} > max {spec['max']}")
    elif t == "string":
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{key}: expected string or null, got {type(value).__name__}")
    elif t == "choice":
        if value not in spec["choices"]:
            raise ConfigError(f"{key}: {value!r} not in {spec['choices']}")
    elif t == "int_or_null":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected int or null, got {type(value).__name__}")
        if "min" in spec and value < spec["min"]:
            raise ConfigError(f"{key}: {value} < min {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ConfigError(f"{key}: {value} > max {spec['max']}")
    elif t == "float_or_null":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected number or null, got {type(value).__name__}")
        value = float(value)
        if "min" in spec and value < spec["min"]:
            raise ConfigError(f"{key}: {value} < min {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ConfigError(f"{key}: {value} > max {spec['max']}")
    elif t == "list[string]":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{key}: expected list of strings")
    # "opaque" passes through
    return value


def apply_defaults(user_config: dict | None) -> dict:
    """Return a new dict: user values + defaults for anything missing, all validated.

    Unknown keys are preserved unchanged except retired settings, which are rejected.
    Raises ConfigError if a required key has no value and no default, or if a
    supplied value fails its type/range check.
    """
    if user_config is None:
        user_config = {}
    reject_retired_config(user_config)
    result = dict(user_config)
    # An explicitly selected alternate provider must not gain the Gemini default.
    if "use_gemini" not in result and any(
        result.get(key) for key in ("use_together", "use_openrouter", "use_pipeline")
    ):
        result["use_gemini"] = False
    for spec in CONFIG_SCHEMA:
        key = spec["key"]
        if key in result:
            result[key] = _validate(key, result[key], spec)
        elif "default" in spec:
            result[key] = spec["default"]
        elif spec.get("required"):
            raise ConfigError(f"{key}: required, no default available")
    return result


def describe() -> list[dict]:
    """Return schema entries for UI rendering — defensive copies."""
    return [dict(e) for e in CONFIG_SCHEMA]


def schema_by_group() -> dict[str, list[dict]]:
    """Schema entries grouped for UI sectioning. Order preserved within group."""
    groups: dict[str, list[dict]] = {}
    for entry in CONFIG_SCHEMA:
        groups.setdefault(entry.get("group", "misc"), []).append(dict(entry))
    return groups


def get_default(key: str) -> Any:
    """Return the declared default for `key`, or raise KeyError if unknown."""
    spec = _SCHEMA_BY_KEY[key]
    if "default" not in spec:
        raise KeyError(f"{key}: no default declared")
    return spec["default"]
