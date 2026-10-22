from social_turing_test.llm_players.llm_constants import LLM_CONFIG_KEY, ASYNC_TYPE_KEY
from social_turing_test.llm_players.agent.agent import SocialTuringTestAgent
from social_turing_test.core.config_schema import apply_defaults, ConfigError, reject_retired_config

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
        # start even if a schema rule is too strict.
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