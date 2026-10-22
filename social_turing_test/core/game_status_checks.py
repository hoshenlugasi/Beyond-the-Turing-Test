from social_turing_test.core.game_constants import PHASE_STATUS_FILE, WHO_WINS_FILE, VOTED_OUT, \
    PERSONAL_STATUS_FILE_FORMAT, VOTING_TIME, GAME_START_TIME_FILE, AI_PLAYER_FILE

def is_game_over(game_dir):
    f = game_dir / WHO_WINS_FILE
    return f.exists() and bool(f.read_text(encoding="utf-8", errors="ignore"))

def is_voted_out(name, game_dir):
    f = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(name)
    return f.exists() and VOTED_OUT in f.read_text(encoding="utf-8", errors="ignore")

def is_time_to_vote(game_dir):
    f = game_dir / PHASE_STATUS_FILE
    return f.exists() and VOTING_TIME in f.read_text(encoding="utf-8", errors="ignore")

def all_players_joined(game_dir):
    f = game_dir / GAME_START_TIME_FILE
    return f.exists() and bool(f.read_text(encoding="utf-8", errors="ignore"))

def get_is_ai(name, game_dir):
    """Check if a player is the AI player"""
    f = game_dir / AI_PLAYER_FILE
    if not f.exists():
        return False
    ai_player_names = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    return name in ai_player_names
