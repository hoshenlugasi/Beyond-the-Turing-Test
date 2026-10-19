# Ensure project root is on sys.path so `config_schema` (a top-level module)
# resolves whether this file is imported from the host shell or from inside a
# Docker container that sets a different cwd.
import sys as _sys
from pathlib import Path as _Path
_PROJ_ROOT = str(_Path(__file__).resolve().parent.parent)
if _PROJ_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJ_ROOT)

from llm_players.llm_constants import LLM_CONFIG_KEY, ASYNC_TYPE_KEY
from llm_players.agent.agent import SocialTuringTestAgent
from core.config_schema import reject_retired_config

# config_schema is optional — when missing (e.g. stale deployment), fall back
# to identity defaults so the game can still run with whatever the caller
# supplied. This keeps LLM players alive even if the schema module hasn't
# been deployed to the server yet.
try:
    from core.config_schema import apply_defaults, ConfigError  # type: ignore
except ModuleNotFoundError:
    class ConfigError(Exception):
        pass
    def apply_defaults(cfg):
        return dict(cfg) if cfg else {}

llm_players_classes = {
    SocialTuringTestAgent.TYPE_NAME: SocialTuringTestAgent,
}

def llm_player_factory(player_config):
    # Hard rejection must precede the non-fatal schema compatibility fallback.
    reject_retired_config(player_config)
    player_config = dict(player_config)
    raw_llm_cfg = dict(player_config.get(LLM_CONFIG_KEY, {}))
    raw_llm_cfg.setdefault(ASYNC_TYPE_KEY, SocialTuringTestAgent.TYPE_NAME)
    try:
        player_config[LLM_CONFIG_KEY] = apply_defaults(raw_llm_cfg)
    except ConfigError as e:
        # Non-fatal: log and fall back to raw config so the game can still
        # start even if a schema rule is too strict. Server-side crash on
        # validation error made the LLM subprocess die at startup in prod.
        import logging
        logging.warning(
            "[config_schema] Validation failed, using raw config unchanged: %s", e
        )
        player_config[LLM_CONFIG_KEY] = dict(raw_llm_cfg)
    async_type = player_config[LLM_CONFIG_KEY].get(ASYNC_TYPE_KEY, SocialTuringTestAgent.TYPE_NAME)
    if async_type not in llm_players_classes:
        raise ValueError(
            f"Unknown async_type '{async_type}'. "
            f"Valid options: {sorted(llm_players_classes.keys())}"
        )
    player_class = llm_players_classes[async_type]
    return player_class(**player_config)