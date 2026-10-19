import json
import argparse
import random
from pathlib import Path
from core.game_constants import DIRS_PREFIX, DEFAULT_GAME_CONFIG, GAME_ID_NUM_DIGITS, GAME_CONFIG_FILE, \
    PLAYER_NAMES_FILE, REMAINING_PLAYERS_FILE, AI_PLAYER_FILE, PHASE_STATUS_FILE, DAYTIME, \
    PUBLIC_MANAGER_CHAT_FILE, PUBLIC_DAYTIME_CHAT_FILE, WHO_WINS_FILE, \
    GAME_START_TIME_FILE, NOTES_FILE, REAL_NAME_CODENAME_DELIMITER, REAL_NAMES_FILE, \
    PLAYERS_KEY_IN_CONFIG, PERSONAL_STATUS_FILE_FORMAT, PERSONAL_CHAT_FILE_FORMAT, \
    PERSONAL_VOTE_FILE_FORMAT, LLM_LOG_FILE_FORMAT, PERSONAL_SURVEY_FILE_FORMAT, \
    CONVERSATION_TOPIC_FILE
from prepare_config import PlayerConfig
from core.config_schema import reject_retired_config

CONVERSATION_TOPICS_FILE = Path("configurations/conversation_topics.json")


def get_next_free_game_id():
    from core.game_paths import next_game_id
    return next_game_id()


def get_id_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", default=None, help="explicit new game id")
    parser.add_argument("-c", "--config", default=None, help="non-default configuration path")
    args = parser.parse_args()
    # handle config_path first (validate before generating a game id)
    config_path = args.config
    if config_path is None:
        print(f"(!) Using default config path: {DEFAULT_GAME_CONFIG}")
        config_path = DEFAULT_GAME_CONFIG
    if not Path(config_path).exists():
        raise ValueError(f"Can't use this config because its path doesn't exist: {config_path}")
    # handle game_id
    game_id = args.id
    if game_id is None:
        game_id = get_next_free_game_id()
        print(f"Generated a new game id: {game_id}")
    from core.game_paths import resolve_game_dir
    if resolve_game_dir(game_id).exists():
        raise ValueError(f"Can't create game dir with the following id"
                         f" because it already exists: {game_id}")
    return game_id, config_path


def select_conversation_topic(config):
    """
    Select a conversation topic for the game.
    Returns None if feature is disabled or no topic available.
    """
    # Check if conversation topic feature is enabled (default: False)
    if not config.get("enable_conversation_topic", False):
        print("Conversation topic feature is disabled (enable_conversation_topic=false)")
        return None
    
    # Check if specific topic provided in config
    if "conversation_topic" in config:
        return config["conversation_topic"]
    
    # Otherwise, random from default pool
    if CONVERSATION_TOPICS_FILE.exists():
        try:
            with open(CONVERSATION_TOPICS_FILE, encoding="utf-8") as f:
                topics = json.load(f)
            default_topics = topics.get("default", [])
            if default_topics:
                return random.choice(default_topics)
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to None if file is malformed
    
    return None  # Fallback: no topic

def init_game(game_id, config_path):
    with open(config_path, "r", encoding="utf-8") as original_file:
        config = json.load(original_file)
    reject_retired_config(config)
    # Record the one public agent identity independently of model/provider.
    from core.game_identity import CURRENT_AGENT_IDENTITY, AGENT_IDENTITY_KEY
    for p in config.get(PLAYERS_KEY_IN_CONFIG, []):
        if p.get("is_ai") or p.get("is_llm"):
            llm_cfg = p.setdefault("llm_config", {})
            llm_cfg[AGENT_IDENTITY_KEY] = CURRENT_AGENT_IDENTITY
    # Runtime games are separate from the released study dataset.
    from core.game_paths import target_dir_for_config, ensure_game_subdirs
    game_dir = target_dir_for_config(config, game_id)
    game_dir.mkdir()
    ensure_game_subdirs(game_dir)
    config["config_original_path_when_game_created"] = config_path
    with open(game_dir / GAME_CONFIG_FILE, "w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=4)
    players = [PlayerConfig(**player_config) for player_config in config[PLAYERS_KEY_IN_CONFIG]]
    all_names_str = "\n".join([player.name for player in players])
    (game_dir / PLAYER_NAMES_FILE).write_text(all_names_str, encoding="utf-8")
    (game_dir / REMAINING_PLAYERS_FILE).write_text(all_names_str, encoding="utf-8")
    all_ai_names_str = [player.name for player in players if player.is_ai]
    (game_dir / AI_PLAYER_FILE).write_text("\n".join(all_ai_names_str), encoding="utf-8")
    real_name_to_codename_str = [f"{player.real_name}{REAL_NAME_CODENAME_DELIMITER}{player.name}"
                                 for player in players if not player.is_llm]
    (game_dir / REAL_NAMES_FILE).write_text("\n".join(real_name_to_codename_str), encoding="utf-8")
    (game_dir / PHASE_STATUS_FILE).write_text(DAYTIME, encoding="utf-8")
    (game_dir / PUBLIC_MANAGER_CHAT_FILE).touch()
    (game_dir / PUBLIC_DAYTIME_CHAT_FILE).touch()
    (game_dir / WHO_WINS_FILE).touch()
    (game_dir / GAME_START_TIME_FILE).touch()
    (game_dir / NOTES_FILE).touch()
    
    # Select and store conversation topic
    topic = select_conversation_topic(config)
    if topic:
        (game_dir / CONVERSATION_TOPIC_FILE).write_text(topic, encoding="utf-8")
        print(f"Selected conversation topic: {topic}")
    else:
        (game_dir / CONVERSATION_TOPIC_FILE).touch()  # Initialize the shared topic file.
    for player in players:
        (game_dir / PERSONAL_CHAT_FILE_FORMAT.format(player.name)).touch()
        (game_dir / PERSONAL_VOTE_FILE_FORMAT.format(player.name)).touch()
        (game_dir / PERSONAL_STATUS_FILE_FORMAT.format(player.name)).touch()
        if player.is_llm:
            (game_dir / LLM_LOG_FILE_FORMAT.format(player.name)).touch()
        else:
            (game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(player.name)).touch()
    print(f"Successfully created a new game dir in: {game_dir.absolute()}")


def main():
    game_id, config_path = get_id_and_config()
    init_game(game_id, config_path)


if __name__ == '__main__':
    main()
