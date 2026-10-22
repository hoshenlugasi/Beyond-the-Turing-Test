"""Single source of truth for tunable LLM/agent config keys.

Everything arbitrary — thresholds, typing speed, memory window, reviewer behavior —
lives here with a declared type, default, and range. `apply_defaults()` fills missing
keys and validates ranges. The admin UI is generated from `describe()` so any knob
added here shows up in the form automatically.

Design goals:
- Describe the synchronous agent configuration and its supported options.
- Invalid values raise loudly (ConfigError) instead of silently accepting.
- Schema is cheap to import — no heavy deps, no network.
"""
from typing import Any


class ConfigError(ValueError):
    pass


def reject_retired_config(config: Any) -> None:
    """Reject unsupported live_evaluator_* keys in settings or nested game payloads.

    Raises ConfigError (a ValueError) even when an unsupported value is false or null.
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
    "choices": ["social_turing_test_agent"],
    "default": "social_turing_test_agent",
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
    "description": "Override the model used for the send/wait scheduler. "
                    "Empty = share the generator's model_name. Provider is auto-detected "
                    "from the model name."},
    {"key": "num_words_per_second_to_wait", "type": "float", "group": "scheduler",
     "default": 1.0, "min": 0.1, "max": 10.0,
        "description": "Words-per-second pacing setting, distinct from typing_wps."},

    # ── Reviewer ──
    {"key": "reviewer_enabled", "type": "bool", "group": "reviewer", "default": True},
    {"key": "reviewer_backend", "type": "choice", "group": "reviewer",
     "choices": ["gemini", "together", "openrouter"], "default": "gemini"},
    {"key": "reviewer_model", "type": "string", "group": "reviewer", "default": None,
     "description": "Override reviewer model. Empty = backend default (gemini-2.5-flash)."},
    {"key": "reviewer_threshold", "type": "int", "group": "reviewer",
     "default": 4, "min": 1, "max": 10,
        "description": "Single-draft helper: score ≥ threshold triggers a rewrite. "
                    "Not used by four-candidate selection."},
    {"key": "reviewer_revert_threshold", "type": "int", "group": "reviewer",
     "default": 7, "min": 1, "max": 10,
        "description": "Single-draft helper: if the best attempt scores ≥ this, "
                    "revert to the original draft. Not used by four-candidate selection."},
    {"key": "reviewer_max_attempts", "type": "int", "group": "reviewer",
     "default": 3, "min": 1, "max": 5,
        "description": "Single-draft helper: maximum review-and-rewrite attempts "
                    "per message, subject to reviewer_budget_sec. "
                    "Not used by four-candidate selection."},
    {"key": "reviewer_self_consistency", "type": "bool", "group": "reviewer",
     "default": True,
        "description": "Single-draft helper: re-score the reviewer's rewrite on "
                    "flash-lite; abort if it also fails. "
                    "Not used by four-candidate selection."},
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
        "description": "Does not skip short messages; typing_base_delay_sec applies "
                    "regardless of message length."},
    {"key": "typing_base_delay_sec", "type": "float", "group": "typing",
     "default": 2.0, "min": 0.0, "max": 10.0,
        "description": "Base reading/thinking component of the typing-delay target, "
                    "including for short messages."},
    {"key": "reviewer_budget_sec", "type": "float_or_null", "group": "reviewer",
     "default": None, "min": 5.0, "max": 120.0,
        "description": "Single-draft helper: time budget in seconds for the "
                    "review-and-rewrite loop. null = no budget cap. "
                    "Not used by four-candidate selection."},
    {"key": "reviewer_drop_score", "type": "int", "group": "reviewer",
     "default": 10, "min": 1, "max": 10,
        "description": "Single-draft helper: suspicion-score threshold for dropping "
                    "a reviewed draft. 10 disables score-based dropping. "
                    "Not used by four-candidate selection."},
    {"key": "reviewer_api_timeout_sec", "type": "float_or_null", "group": "reviewer",
     "default": None, "min": 2.0, "max": 60.0,
        "description": "Single-draft helper: per-call reviewer API timeout in "
                    "seconds. null = no configured per-call timeout. "
                    "Not used by four-candidate selection."},
    {"key": "reviewer_backoff_enabled", "type": "bool", "group": "reviewer",
     "default": False,
        "description": "Single-draft helper: pause before retrying after a reviewer "
                    "attempt takes more than 5 seconds (0.5s × attempt number). "
                    "Disabled by default. Not used by four-candidate selection."},

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
        "description": "Strategic (default) asks the LLM who most suspects "
                    "the AI; random ignores context."},

    # ── Experimental emoji and length policy ──
    {"key": "experimental_agent", "type": "bool", "group": "experimental",
     "default": False,
    "description": "Optional emoji and length settings: emoji policy in the "
                    "system prompt, experimental_max_tokens override, and an "
                    "anti-shrink clause for single-draft helper retries. "
                    "Disabled by default."},
    {"key": "experimental_max_tokens", "type": "int", "group": "experimental",
     "default": 80, "min": 0, "max": 400,
     "description": "When experimental_agent=true, override max_tokens / "
                    "max_new_tokens with this value. 0 keeps the original "
                    "max_tokens. DEFAULT 80."},

    # ── Agent pipeline ──
    {"key": "multi_candidate_generation", "type": "bool", "group": "agent",
        "default": True,
    "description": "Generate candidate messages, then select one verbatim. "
                    "The reviewer ranks candidates without rewriting them; "
                    "the default is four candidates."},
    {"key": "num_candidates", "type": "int", "group": "agent",
     "default": 4, "min": 2, "max": 8,
    "description": "How many candidates the generator produces when "
                    "multi_candidate_generation=true."},
    {"key": "investigator", "type": "bool", "group": "agent",
     "default": True,
    "description": "Give the AI the human players' dual role — it also "
                    "hunts the AI and votes, removing the passive-target "
                    "asymmetry. Swaps in the dual-role system prompt."},
    {"key": "pipeline_timeout_s", "type": "float", "group": "agent",
     "default": 35.0, "min": 5.0, "max": 120.0,
    "description": "Safety net: drop a message only if the whole pipeline "
                    "took longer than this (extreme hang). A normal response "
                    "delay is fine and is never dropped."},
    {"key": "orchestration_mode", "type": "choice", "group": "agent",
        "default": "synchronous", "choices": ["synchronous"],
        "description": "Synchronous agent pipeline; the only supported orchestration mode."},
    # ── Internet Context tuning ──
    {"key": "internet_max_queries_per_tick", "type": "int", "group": "agent",
     "default": 5, "min": 1, "max": 10,
     "description": "Max web searches the InternetContext sub-agent runs per tick. "
                    "Deduplication and caching limit repeated searches."},
    {"key": "internet_detector_model", "type": "string", "group": "agent", "default": None,
     "description": "Optional cheaper model for the InternetContext 'should I search?' "
                    "decision call (e.g. gemini-2.5-flash-lite). None → reuse the main "
                    "model."},
    # ── Optional research and tracing settings ──
    {"key": "background_intelligence_enabled", "type": "bool", "group": "agent",
     "default": False,
    "description": "Unsupported option; leave false."},
    {"key": "langsmith_tracing_enabled", "type": "bool", "group": "agent",
     "default": False,
    "description": "Optional LangSmith tracing for the synchronous agent pipeline. Never "
                    "authoritative; gameplay is identical when off. Default off."},
    {"key": "langsmith_project", "type": "string", "group": "agent", "default": None,
     "description": "LangSmith project name (when tracing is enabled)."},
    {"key": "langsmith_sample_rate", "type": "float", "group": "agent",
     "default": 1.0, "min": 0.0, "max": 1.0,
    "description": "Fraction of traces to sample when tracing is enabled "
                    "(0..1). 1.0 = trace everything; 0.0 = trace nothing."},
    {"key": "langsmith_include_chat_content", "type": "bool", "group": "agent",
     "default": False,
    "description": "Include chat content in traces. Default OFF (privacy)."},
    {"key": "langsmith_redaction_enabled", "type": "bool", "group": "agent",
     "default": True,
    "description": "Redact excluded/sensitive fields before any trace emit."},
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

    Unknown keys are preserved unchanged except unsupported settings rejected by
    reject_retired_config().
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
