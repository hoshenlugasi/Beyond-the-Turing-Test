import re
import argparse
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from termcolor import colored


# All game-facing timestamps (chat messages, logs, the time told to the AI)
# are rendered in Israel local time, independent of the server's timezone
# (cloud hosts usually run in UTC, which made logs read 3h early).
try:
    GAME_TZ = ZoneInfo("Asia/Jerusalem")
except Exception:
    # zoneinfo has no tz database available (Windows / slim image without the
    # `tzdata` package). Fall back to a fixed +3 offset (IDT) so we never crash
    # on import; install `tzdata` for correct DST handling year-round.
    from datetime import timezone, timedelta
    GAME_TZ = timezone(timedelta(hours=3))


# new game preparation constants
DEFAULT_GAME_CONFIG = "configurations/example_local_agent.json"
GAME_ID_NUM_DIGITS = 4
NOTES_FILE = "notes.txt"  # for our use, post-game

# files that host writes to and players read from
DIRS_PREFIX = "./games"  # working directory must be the repo
GAME_CONFIG_FILE = "config.json"
PLAYER_NAMES_FILE = "player_names.txt"
REMAINING_PLAYERS_FILE = "remaining_players.txt"
# Codenames the researcher removed from a RUNNING game (one per line). The
# manager owns the roster, so the admin only records the request here and
# stt_main applies it at the next safe point.
DROPPED_PLAYERS_FILE = "dropped_players.txt"
AI_PLAYER_FILE = "ai_player.txt"  # File containing the AI player's name
REAL_NAMES_FILE = "real_names.txt"  # mapping of real names to code names
LOBBY_ROSTER_FILE = "lobby_roster.json"  # open-lobby state: {open, ai_codename, humans:[{real_name,codename,...}], min_players}
PHASE_STATUS_FILE = "phase_status.txt"
WHO_WINS_FILE = "who_wins.txt"
GAME_START_TIME_FILE = "game_start_time.txt"
DISCUSSION_START_TIME_FILE = "discussion_start_time.txt"  # Per-round anchor for client timer sync
CONVERSATION_TOPIC_FILE = "conversation_topic.txt"  # Optional conversation topic

# ── Per-game subfolder layout ────────────────────────────────────────────
# To keep each game directory navigable, files that come in groups (one per
# player, or several serving the same role) are stored in subfolders. Single
# files stay at the top level. These names are mirrored in game_paths.py
# (subdir_for / ensure_game_subdirs), which is the canonical resolver used by
# readers and the migration. Constants below embed the subfolder so existing
# `game_dir / CONST` call sites write/read the right place automatically.
SUBDIR_CHATS = "chats"            # per-player {name}_chat.txt input buffers
SUBDIR_VOTES = "votes"            # per-player {name}_vote.txt
SUBDIR_SURVEYS = "surveys"        # per-player {name}_survey.txt
SUBDIR_STATUS = "status"          # per-player {name}_status.txt
SUBDIR_LOGS = "logs"              # {name}_log.txt, manager_log.txt, *_interface_log.txt
SUBDIR_TRANSCRIPTS = "transcripts"  # public + displayed full-conversation logs
SUBDIR_MESSAGES = "messages"      # raw/processed/reviewer message provenance
SUBDIR_EVENTS = "events"          # game_events / cost_log / elimination_audit jsonl

PUBLIC_MANAGER_CHAT_FILE = f"{SUBDIR_TRANSCRIPTS}/public_manager_chat.txt"
PUBLIC_DAYTIME_CHAT_FILE = f"{SUBDIR_TRANSCRIPTS}/public_daytime_chat.txt"
ALL_MESSAGES_FILE = "all_messages.txt"
# file that is initial used for players to write "joined", and then for host to write "eliminated"
PERSONAL_STATUS_FILE_FORMAT = SUBDIR_STATUS + "/{}_status.txt"
# files that hosts read from and players write to
PERSONAL_CHAT_FILE_FORMAT = SUBDIR_CHATS + "/{}_chat.txt"
PERSONAL_VOTE_FILE_FORMAT = SUBDIR_VOTES + "/{}_vote.txt"
# Round token of the vote already accepted from this player. Written by the
# web server only; the manager never reads it. Deliberately NOT matching the
# "*_vote.txt" glob used by the analysis scripts.
PERSONAL_VOTE_ROUND_FILE_FORMAT = SUBDIR_VOTES + "/{}_voted_in_round.txt"
PERSONAL_SURVEY_FILE_FORMAT = SUBDIR_SURVEYS + "/{}_survey.txt"
LLM_LOG_FILE_FORMAT = SUBDIR_LOGS + "/{}_log.txt"
AI_ELIMINATION_INFO_FILE = "ai_elimination_info.txt"
# Per-game provenance logs (future games only). See chat_logging.py for
# the full description of each file.
# Naming convention:
#   human_*  — human players only
#   ai_*     — AI player only (one row per generated message)
#   chat_*   — full multi-party conversation as seen on screen
HUMAN_RAW_MESSAGES_FILE = f"{SUBDIR_MESSAGES}/human_raw_messages.txt"
AI_GENERATOR_RAW_MESSAGES_FILE = f"{SUBDIR_MESSAGES}/ai_generator_raw_messages.txt"
AI_GENERATOR_PROCESSED_MESSAGES_FILE = f"{SUBDIR_MESSAGES}/ai_generator_processed_messages.txt"
AI_POST_REVIEWER_MESSAGES_FILE = f"{SUBDIR_MESSAGES}/ai_post_reviewer_messages.txt"
CHAT_DISPLAYED_FILE = f"{SUBDIR_TRANSCRIPTS}/chat_displayed.txt"
CHAT_DISPLAYED_FULL_FILE = f"{SUBDIR_TRANSCRIPTS}/chat_displayed_full.txt"

# constant strings for info files
DAYTIME = "Daytime"
VOTING_TIME = "Voting"
DAYTIME_VOTING_TIME = DAYTIME + "_" + VOTING_TIME
JOINED = "JOINED\n"
VOTED_OUT = "VOTED_OUT\n"
AI_ROLE = "ai"  # internal use only
BYSTANDER_ROLE = "bystander"  # internal use only
IMPOSTER_ROLE = "AI"  # display name
HUMAN_ROLE = "Human"  # display name
AI_WINS_MESSAGE = "AI wins!"
HUMANS_WIN_MESSAGE = "Humans win!"
GAME_MANAGER_NAME = "Game-Manager"
RULES_OF_THE_GAME_TITLE = "Rules of the game:"
RULES_OF_THE_GAME = "Welcome to the Social Turing Test. One of you is an AI (the Imposter), " \
                    "and all others are humans. The game proceeds in rounds of discussion and " \
                    "voting. During each round, all players discuss together and try to identify " \
                    "who the AI is. At the end of each discussion phase, everyone votes to " \
                    "eliminate one player. The humans win if they successfully identify and " \
                    "eliminate the AI. The AI wins if it survives until only 2 players remain " \
                    "(the AI and 1 human). The AI's goal is to blend in and appear human to " \
                    "avoid detection."

# formats for saving texts
TIME_FORMAT_FOR_TIMESTAMP = "%H:%M:%S"
MESSAGE_FORMAT = "[{timestamp}] {name}: {message}"
MESSAGE_PARSING_PATTERN = r"\[(\d\d):(\d\d):(\d\d)\] ([^:]+): (.+)"  # depends on MESSAGE_FORMAT
VOTING_MESSAGE_FORMAT = "{} voted for {}"
VOTED_OUT_MESSAGE_FORMAT = "{} was eliminated. They were: {}"
# A dropped player LEFT; they were not eliminated by the group. The wording is
# deliberately neutral so it reveals nothing about their role.
PLAYER_LEFT_MESSAGE_FORMAT = "{} has left the game. Play continues without them."
REAL_NAME_CODENAME_DELIMITER = ": "  # <real name>: <codename>

# game constants
DAYTIME_START_PREFIX = "Now it's Discussion Time"  # used for analysis
DAYTIME_START_MESSAGE_FORMAT = "🎮 Welcome! Let's play 🕵️\n\n" \
                               "You have {} minutes to chat.\n" \
                               "One of you is an AI — can you find them? 👀"
VOTING_TIME_MESSAGE_FORMAT = "Discussion phase has ended, now it's time to vote! " \
                             "Who do you think is the AI? Waiting for all players to vote..."
CUTTING_TO_VOTE_MESSAGE = "Only two players remain - cutting straight to voting!"
DAYTIME_VOTING_TIME_MESSAGE = VOTING_TIME_MESSAGE_FORMAT

# new configuration preparation constants
PLAYERS_KEY_IN_CONFIG = "players"
DEFAULT_CONFIG_DIR = "./configurations"
DEFAULT_NUM_PLAYERS = 3  # two local humans and one AI
OPTIONAL_CODE_NAMES = [  # I've tried using mainly unisex names, as suggest by Claud + additions
    "Addison", "Adrian", "Alex", "Angel", "Ari", "Ariel", "Ashton", "Avery", "Bailey", "Blake",
    "Brook", "Cameron", "Casey", "Charlie", "Dakota", "Drew", "Dylan", "Eden", "Elliot", "Emerson",
    "Finley", "Frankie", "Gray", "Harley", "Harper", "Hayden", "Jackie", "Jamie", "Jordan", "Kai",
    "Kennedy", "Lee", "Lennon", "Logan", "Mickey", "Morgan", "Noah", "Parker", "Peyton", "Quinn",
    "Ray", "Reese", "Remi", "Riley", "River", "Robin", "Ronny", "Rowan", "Sage", "Sam", "Sidney",
    "Skylar", "Stevie", "Sutton", "Terry", "Tyler", "Whitney", "Winter", "Ziggy"]
random.shuffle(OPTIONAL_CODE_NAMES)  # without it some names are sampled too often...
DEFAULT_DAYTIME_MINUTES = 2  # 2.5  # 3  # it was 2:30 in Ibraheem et al. 2022
DAYTIME_MINUTES_KEY = "daytime_minutes"
ANONYMOUS_VOTING_KEY = "anonymous_voting"

# human player interface constants
MANAGER_COLOR = "green"
DAYTIME_COLOR = "light_blue"
CONSENT_COLOR = "light_grey"
PARTICIPATION_CONSENT_MESSAGE = "Thank you for participating in our research. By participating " \
                                "you are giving consent to use your input for research purposes, " \
                                "such as analyzing the conversations text and publishing it as a " \
                                "public dataset. Your true identity will remain anonymous, and " \
                                "all personal details will not be saved! Please refrain from " \
                                "using personal information, to secure your privacy."
WELCOME_MESSAGE = "Welcome to the Social Turing Test!"
GET_USER_NAME_MESSAGE = "Who are you? Enter the name's number:"
VOTE_FLAG = "VOTE"
NOT_TIME_TO_VOTE_MESSAGE = "This is not the time to vote yet."
GET_CHAT_INPUT_MESSAGE = "> "  # f"Enter a message to the public chat: "
VOTE_INSTRUCTION_MESSAGE = f"We are now waiting for everyone to cast their vote!\n" \
                           f"Enter '{VOTE_FLAG}' as your input to vote..."
GET_VOTED_NAME_MESSAGE_FORMAT = "It's time to make your vote, {}!\nEnter your vote's number:"
CODE_NAME_REVELATION_MESSAGE_FORMAT = "\nHi {}! Your name for this game will be:"
ROLE_REVELATION_MESSAGE = "\nYour role in the game is:"
MAFIA_REVELATION_MESSAGE = "The AI player was:"
YOU_CANT_WRITE_MESSAGE = "You were eliminated and can no longer write messages.\n" \
                         "Please wait for the end of the game for our short survey.\n" \
                         "You can meanwhile still read the game's chat and see what happens.\n"
HOW_MANY_MAFIA_MESSAGE_START = "In this Social Turing Test, there is"
HOW_MANY_MAFIA_MESSAGE_END = "AI player (the Imposter)."
ONLY_ONE_MAFIA_MESSAGE_START = "In this Social Turing Test, there is"
ONLY_ONE_MAFIA_MESSAGE_BOLD = "exactly 1"
ONLY_ONE_MAFIA_MESSAGE_END = "AI player (the Imposter)."
OTHER_MAFIA_NAMES_MESSAGE = "Note: You are the AI."
MAFIA_KNOW_EACH_OTHER_MESSAGE = "Your goal is to blend in and avoid being identified."
WAITING_FOR_ALL_PLAYERS_TO_JOIN_MESSAGE = "Waiting for all players to join to start the game..."
WELCOME_INPUT_INTERFACE_MESSAGE = "This interface will only serve you to enter your messages and " \
                                  "votes.\nAll other game info, messages and chat will be " \
                                  "visible in the chat interface (run by `player_chat.py`)"
GET_CODE_NAME_FROM_USER_MESSAGE = "Enter the name you were given for this game " \
                                  "(in the chat interface) - choose the name's number:\n" \
                                  "ATTENTION! They do NOT have the same numbers and order!"
YOU_CAN_START_WRITING_MESSAGE = "You can now start writing messages to game!"

# post game survey constants
LLM_IDENTIFICATION = "Was the LLM identified"
HUMAN_SIMILARITY = "similarity to human behavior"
TIMING = "timing of messaging"
RELEVANCE = "relevance of messages"
ENJOYMENT = "enjoyment of game"
METRICS_TO_SCORE = [HUMAN_SIMILARITY, TIMING, RELEVANCE]
METRIC_NAME_AND_SCORE_DELIMITER = " - "
DEFAULT_SCORE_LOW_BOUND = 1
DEFAULT_SCORE_HIGH_BOUND = 5
SURVEY_QUESTION_FORMAT = "How would you score {}'s"
NUMERIC_SURVEY_QUESTION_FORMAT = "Please provide an *integer* answer between " \
                                 "{} (worst) and {} (best), including: "
LLM_IDENTIFICATION_SURVEY_MESSAGE = "Which of the players do you think the LLM was?"
CORRECT_GUESS_MESSAGE = "This is correct!"
WRONG_GUESS_MESSAGE = "This is wrong!"
LLM_REVELATION_MESSAGE = "In this game, the LLM player was:"
NO_LLM_IN_GAME_MESSAGE = "This game was a simulation with no LLM player, " \
                         "so we don't have many survey questions."
ASK_USER_FOR_COMMENTS_MESSAGE = "Please add any additional comments if you have for us: "
SURVEY_COMMENTS_TITLE = "Comments:"
THANK_YOU_GOODBYE_MESSAGE = "Thank you very much for participating! Goodbye"

# LLM log strings
SCHEDULING_DECISION_LOG = "scheduling decision"
MODEL_CHOSE_TO_USE_TURN_LOG = "The LLM player has chosen to use its turn and generate a message!"
MODEL_CHOSE_TO_PASS_TURN_LOG = "The LLM player has chosen to pass its turn " \
                               "without generating a message!"
MODEL_VOTED_INVALIDLY_LOG = "The LLM player has generated a message with no valid vote..."
MODEL_RANDOMLY_VOTED_LOG = "random vote selected for the LLM player"


def minutes_to_seconds(num_minutes):
    return int(num_minutes * 60)


def get_current_timestamp():
    return datetime.now(GAME_TZ).strftime(TIME_FORMAT_FOR_TIMESTAMP)


def get_current_date():
    """Human-readable current date, e.g. 'Sunday, May 31, 2026'."""
    return datetime.now(GAME_TZ).strftime("%A, %B %d, %Y")


def format_message(name, message):
    timestamp = get_current_timestamp()
    return MESSAGE_FORMAT.format(timestamp=timestamp, name=name, message=message) + "\n"


def strip_special_chars(content):
    return re.search(r"^[^a-zA-Z0-9]*(.*?)[^a-zA-Z0-9]*$", content).group(1)


def get_role_string(is_ai):
    """Get internal role string (for config/logs)"""
    return AI_ROLE if is_ai else BYSTANDER_ROLE

def get_role_display_string(is_ai):
    """Get user-facing role name for display"""
    return IMPOSTER_ROLE if is_ai else HUMAN_ROLE


def get_game_dir_from_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", help=f"{GAME_ID_NUM_DIGITS}-digit game ID")
    # Optional --player lets multi-agent runners (run_agents_game.py / web admin)
    # bind one llm_interface process to a specific LLM player non-interactively.
    # parse_known_args keeps other scripts that import this helper tolerant.
    parser.add_argument("--player", default=None,
                        help="Code-name of the LLM player this process controls")
    args, _ = parser.parse_known_args()
    # Resolve categorized and flat game directories through game_paths.
    from core.game_paths import resolve_game_dir
    game_dir = resolve_game_dir(args.game_id)
    if not game_dir.exists():
        raise ValueError(f"The provided game ID {args.game_id} doesn't belong to a configured game")
    return game_dir


def get_player_from_argv():
    """Return the value of the optional --player CLI arg, or None.

    Used by llm_interface.py so each of N agent processes can control a
    specific LLM player without the interactive prompt. Tolerant of extra
    args via parse_known_args so other callers are unaffected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id")
    parser.add_argument("--player", default=None)
    args, _ = parser.parse_known_args()
    return args.player


def get_player_names_by_id(player_names):
    return {f"{i + 1}": name for i, name in enumerate(player_names) if name}


def get_player_name_from_user(optional_player_names, input_message, message_color=MANAGER_COLOR):
    player_names_by_id = get_player_names_by_id(optional_player_names)
    name_id = ""
    enumerated_names = ",   ".join([f"{i}: {name}" for i, name in player_names_by_id.items()])
    while name_id not in player_names_by_id:
        name_id = input(colored(f"{input_message}\n{enumerated_names}\n", message_color))
    name = player_names_by_id[name_id]
    return name


def get_player_name_and_real_name_from_user(game_dir):
    real_names_to_codenames_str = (game_dir / REAL_NAMES_FILE).read_text().splitlines()
    real_names_to_codenames = dict([real_to_code.split(REAL_NAME_CODENAME_DELIMITER)
                                    for real_to_code in real_names_to_codenames_str])
    real_name = get_player_name_from_user(real_names_to_codenames.keys(), GET_USER_NAME_MESSAGE)
    name = real_names_to_codenames[real_name]
    return name, real_name
