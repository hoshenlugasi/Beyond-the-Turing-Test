"""
Web interface for the Social Turing Test game and agent monitoring.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_cors import CORS
import hmac
import hashlib
import secrets
import re
import os
import sys
import json
import subprocess
import signal
import random
import threading
import time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from urllib.parse import quote
from collections import Counter

from core.game_constants import (
    DIRS_PREFIX,
    PERSONAL_CHAT_FILE_FORMAT,
    PERSONAL_VOTE_FILE_FORMAT,
    PERSONAL_VOTE_ROUND_FILE_FORMAT,
    PERSONAL_STATUS_FILE_FORMAT,
    PERSONAL_SURVEY_FILE_FORMAT,
    PUBLIC_MANAGER_CHAT_FILE,
    PUBLIC_DAYTIME_CHAT_FILE,
    REAL_NAMES_FILE,
    REAL_NAME_CODENAME_DELIMITER,
    REMAINING_PLAYERS_FILE,
    PLAYER_NAMES_FILE,
    AI_PLAYER_FILE,
    LOBBY_ROSTER_FILE,
    LLM_LOG_FILE_FORMAT,
    METRIC_NAME_AND_SCORE_DELIMITER,
    HUMAN_SIMILARITY,
    TIMING,
    RELEVANCE,
    ENJOYMENT,
    SURVEY_COMMENTS_TITLE,
    JOINED,
    RULES_OF_THE_GAME,
    format_message,
    CONVERSATION_TOPIC_FILE,
    OPTIONAL_CODE_NAMES,
    DEFAULT_DAYTIME_MINUTES,
    PLAYERS_KEY_IN_CONFIG,
    DAYTIME_MINUTES_KEY,
    ANONYMOUS_VOTING_KEY,
    GAME_ID_NUM_DIGITS,
    GAME_CONFIG_FILE,
    PHASE_STATUS_FILE,
    DAYTIME,
    WHO_WINS_FILE,
    GAME_START_TIME_FILE,
    DISCUSSION_START_TIME_FILE,
    DROPPED_PLAYERS_FILE,
    NOTES_FILE,
)
from core.game_status_checks import (
    is_game_over,
    is_time_to_vote,
    all_players_joined,
    get_is_ai,
)
from core.game_constants import get_role_display_string
from core.atomic_io import atomic_write_text
from core.vote_rounds import (
    current_round_token,
    has_voted_this_round,
    record_vote_round,
    vote_round_lock,
)
from core.game_paths import resolve_game_dir, target_dir_for_config, iter_all_games, ensure_game_subdirs, game_file, SUBDIR_EVENTS
from core.game_identity import detect_agent_identity
from core.config_schema import ConfigError, reject_retired_config
from core.source_fingerprint import relevant_source_fingerprint
from prepare_config import PlayerConfig
from llm_players.llm_constants import (
    DEFAULT_LLM_CONFIG, load_personas, DEFAULT_PERSONA_ID,
    ASYNC_TYPE_KEY, SCHEDULE_THEN_GENERATE_TYPE,
    REVIEWED_SCHEDULE_THEN_GENERATE_TYPE,
    MODEL_NAME_KEY, USE_GEMINI_KEY, USE_TOGETHER_KEY, USE_OPENROUTER_KEY,
    PERSONA_KEY,
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

def _load_admin_password():
    """Admin access requires the operator's own environment variable."""
    return os.environ.get("STT_ADMIN_PASSWORD") or None

ADMIN_PASSWORD = _load_admin_password()
if not ADMIN_PASSWORD:
    print("ADMIN PASSWORD NOT SET — admin panel disabled. Set STT_ADMIN_PASSWORD.")

# ── Game-ID validation to prevent path traversal ──
_GAME_ID_RE = re.compile(r"^\d{4}$")

def _valid_game_id(game_id) -> bool:
    """Return True only for exactly 4-digit game IDs (e.g. '0064')."""
    return bool(game_id) and bool(_GAME_ID_RE.fullmatch(str(game_id)))

# ── In-memory API key store (never persisted) ──
server_api_keys = {}  # {"TOGETHER_API_KEY": "sk-...", "GOOGLE_CLOUD_PROJECT": "...", ...}

# ── Typing-indicator state ──
# game_id -> {player_name -> last_typing_timestamp}
_typing_states: dict = {}
_TYPING_TIMEOUT_SECS = 5  # stop showing indicator after 5 s of silence

# ── Game orchestration state ──

def _check_admin_auth():
    """Return error response if admin is not authenticated, else None."""
    if not session.get("admin_auth"):
        return jsonify({"error": "Unauthorized — admin login required"}), 401
    return None

def _get_env_with_api_keys():
    """Build subprocess environment with API keys from UI + host env.

    Always sets PYTHONUNBUFFERED=1 so subprocess stdout/stderr is flushed
    to the log file in real-time (critical for seeing crash tracebacks).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for key, val in server_api_keys.items():
        if val:
            env[key] = val
    return env

def _detect_provider(model_name):
    """Auto-detect provider from model name string."""
    name_lower = (model_name or "").lower()
    # OpenRouter-style prefixed models take priority (google/gemini-... is OpenRouter, not Vertex)
    _OPENROUTER_PREFIXES = ("openai/", "anthropic/", "x-ai/", "google/", "meta-llama/",
                            "mistralai/", "deepseek/", "nvidia/", "cohere/", "microsoft/")
    if any(name_lower.startswith(p) for p in _OPENROUTER_PREFIXES):
        return "openrouter"
    # Bare "gemini-..." → use Gemini API directly
    if "gemini" in name_lower:
        return "gemini"
    # Together AI models typically have org/ prefix like Qwen/
    if "/" in name_lower:
        return "together"
    return "together"

# Global state for each player session (in-memory, per-server-process)
player_states = {}

# Persist session metadata to disk so that docker-compose restart / page
# refresh doesn't invalidate in-flight games. The file stores only the
# minimum needed to re-hydrate a WebPlayer object:
#   { "<session_id>": {"game_id": "0082", "code_name": "Alex", "is_ai": false} }
# The chat/vote line cursors are intentionally NOT persisted — after
# re-hydration we rewind to 0 so the client gets the full chat history once.
_SESSIONS_FILE = Path(DIRS_PREFIX).parent / "sessions.json"


def _load_persisted_sessions():
    """Re-hydrate player_states from disk on server startup."""
    try:
        if not _SESSIONS_FILE.exists():
            return
        meta = json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Sessions] Failed to load {_SESSIONS_FILE}: {e}")
        return
    loaded = 0
    for sid, info in meta.items():
        try:
            gid = info.get("game_id")
            code = info.get("code_name")
            is_ai = bool(info.get("is_ai", False))
            if not (gid and code):
                continue
            gdir = resolve_game_dir(gid)
            if not gdir.exists():
                continue
            player_states[sid] = WebPlayer(gdir, code, is_ai)
            loaded += 1
        except Exception as e:
            print(f"[Sessions] Skipped bad entry {sid}: {e}")
    if loaded:
        print(f"[Sessions] Rehydrated {loaded} sessions from {_SESSIONS_FILE}")


def _persist_sessions():
    """Atomically write session metadata to disk (best-effort)."""
    try:
        meta = {}
        for sid, p in player_states.items():
            try:
                meta[sid] = {
                    "game_id": p.game_dir.name,
                    "code_name": p.name,
                    "is_ai": bool(getattr(p, "is_ai", False)),
                }
            except Exception:
                continue
        tmp = _SESSIONS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, _SESSIONS_FILE)
    except Exception as e:
        print(f"[Sessions] Persist failed: {e}")


# ── Open-lobby state (per-game JSON file) ─────────────────────────────
# Open lobby model: admin creates a game with no human roster; players
# type their real name freely; admin clicks START GAME when enough have
# joined. The lobby roster lives at <game_dir>/lobby_roster.json so it
# survives server restarts naturally (no in-memory cache needed).

def _lobby_path(game_dir: Path) -> Path:
    from core.game_constants import LOBBY_ROSTER_FILE
    return game_dir / LOBBY_ROSTER_FILE


def _load_lobby(game_dir: Path):
    """Return the lobby roster dict, or None if no lobby file exists.

    Shape:
      {
        "open": true,
        "ai_codename": "Ashton" | null,
        "min_players": 3,
        "humans": [
            {"real_name": "Alice", "codename": "Riley",
             "session_id": "...", "joined_at": "ISO-8601"}, ...
        ]
      }
    """
    p = _lobby_path(game_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Lobby] Failed to read {p}: {e}")
        return None


def _save_lobby(game_dir: Path, lobby: dict):
    """Atomically write the lobby roster."""
    p = _lobby_path(game_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lobby, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _normalize_real_name(s: str) -> str:
    """Lowercase + collapse whitespace for duplicate-detection only."""
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


# Last-seen timestamps for connectivity heartbeat (admin "inactive" badge).
# Updated every time a player polls /messages or /vote_status. Pure
# in-memory — survives a single process lifetime, which is the lifetime
# of a single game.
_last_seen_by_session = {}  # {session_id: epoch_seconds}
_INACTIVE_THRESHOLD_SECONDS = 30


def _touch_session(session_id: str):
    if session_id:
        _last_seen_by_session[session_id] = time.time()


# Per-game lock guarding lobby-roster mutations. Two players hitting
# /lobby_join concurrently would otherwise race on _load_lobby /
# _save_lobby and could both grab the same codename. Flask's default
# threaded server makes this race real; the lock is one cheap mutex
# per active game.
_lobby_locks = {}
_lobby_locks_master = threading.Lock()


def _get_lobby_lock(game_id: str) -> threading.Lock:
    with _lobby_locks_master:
        lk = _lobby_locks.get(game_id)
        if lk is None:
            lk = threading.Lock()
            _lobby_locks[game_id] = lk
        return lk


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')




def _game_not_started(game_dir: Path) -> bool:
    path = game_dir / GAME_START_TIME_FILE
    return not path.exists() or not path.read_text(encoding='utf-8').strip()


def _is_canonical_game_dir(game_id: str, game_dir: Path) -> bool:
    return _valid_game_id(game_id) and resolve_game_dir(game_id).resolve() == game_dir.resolve()


def _lobby_human_capacity(lobby):
    return None


def _occupied_human_seat_count(lobby: dict) -> int:
    return len(lobby.get('humans', []))


def _game_is_running(game_dir: Path) -> bool:
    return not _game_not_started(game_dir) and not is_game_over(game_dir)


def _admin_dropped_codenames(game_dir: Path) -> set[str]:
    path = game_dir / DROPPED_PLAYERS_FILE
    return set(path.read_text(encoding='utf-8').splitlines()) if path.exists() else set()


class WebPlayer:
    """
    Represents one player connected through the web UI.
    Tracks how many lines were already read from each chat file in this session,
    so we only send *new* messages to the browser.
    """

    def __init__(self, game_dir: Path, name: str, is_ai: bool):
        self.game_dir = game_dir
        self.name = name
        self.is_ai = is_ai

        self.personal_chat_file = game_dir / PERSONAL_CHAT_FILE_FORMAT.format(name)
        self.personal_vote_file = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(name)

        # how many lines we have already read from each public chat file
        self.num_read_manager = 0
        self.num_read_daytime = 0
        
        # Survey submission tracking
        self.did_submit_survey = False

    def get_new_messages(self):
        """
        Read only *new* lines from manager & daytime chat files
        and return them as a list of {text, type} dicts.

        Classify messages by their file of origin:
        - All lines from PUBLIC_MANAGER_CHAT_FILE → type="manager"
        - All lines from PUBLIC_DAYTIME_CHAT_FILE → type="chat"

        Interleave both batches by their [HH:MM:SS] clock values, retaining
        file-order indices to break ties. This is a time-of-day display sort,
        not a reconstruction of cross-day chronology.
        """
        messages = []

        # Local helper: extract seconds-since-midnight from a "[HH:MM:SS] ..."
        # line. Lines without a timestamp keep their relative order via the
        # secondary sort key (file-order index).
        _ts_re = re.compile(r'^\s*\[(\d{2}):(\d{2}):(\d{2})\]')
        def _ts_key(line):
            m = _ts_re.match(line)
            if not m:
                return None
            h, mi, s = map(int, m.groups())
            return h * 3600 + mi * 60 + s

        # Pending list of (ts_or_None, file_order_idx, message_dict).
        pending = []
        idx = 0

        try:
            # Manager messages (rules, phase announcements, results...)
            manager_file = self.game_dir / PUBLIC_MANAGER_CHAT_FILE
            if manager_file.exists():
                with manager_file.open("r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    new_lines = all_lines[self.num_read_manager :]
                    # advance cursor by ALL new lines (including blank)
                    self.num_read_manager += len(new_lines)

                    for raw in new_lines:
                        line = raw.strip()
                        if line:
                            # Strip "Game-Manager:" prefix to avoid duplication
                            # (frontend CSS adds "🛠️ Game Manager:" header)
                            # Pattern: "[HH:MM:SS] Game-Manager: message"
                            cleaned = re.sub(r'Game-Manager:\s*', '', line)
                            pending.append((
                                _ts_key(line), idx,
                                {"text": cleaned, "type": "manager"},
                            ))
                            idx += 1

            # Daytime chat (merged messages from all personal files - player chat only)
            daytime_file = self.game_dir / PUBLIC_DAYTIME_CHAT_FILE
            if daytime_file.exists():
                with daytime_file.open("r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    new_lines = all_lines[self.num_read_daytime :]
                    self.num_read_daytime += len(new_lines)

                    for raw in new_lines:
                        line = raw.strip()
                        if line:
                            # All messages from daytime file are type="chat".
                            # Parse "[HH:MM:SS] Name: text" so the frontend can
                            # reliably tell apart own vs others' messages
                            # without fragile string.startsWith() checks.
                            m = re.match(r'^\[\d\d:\d\d:\d\d\]\s+([^:]+):\s', line)
                            sender = m.group(1).strip() if m else None
                            pending.append((
                                _ts_key(line), idx,
                                {
                                    "text": line,
                                    "type": "chat",
                                    "sender": sender,
                                    "is_own": (sender == self.name) if sender else False,
                                },
                            ))
                            idx += 1

            # Stable sort by timestamp; entries without a timestamp keep their
            # arrival order (idx) and float to wherever they land vs. siblings.
            # Python's sort is stable, so equal-timestamp lines preserve idx.
            pending.sort(key=lambda t: (t[0] if t[0] is not None else float('inf'), t[1]))
            messages = [item[2] for item in pending]
        except Exception as e:
            print(f"[WebPlayer] Error reading messages for {self.name}: {e}")

        return messages

    def send_message(self, message: str):
        """Append a personal chat message that the game manager process will merge."""
        # Capture pre-filter raw input first (humans aren't filtered server-side,
        # but the frontend will still lowercase the body when displaying it).
        try:
            from core.chat_logging import append_human_raw
            append_human_raw(self.game_dir, self.name, message)
        except Exception:
            pass
        formatted = format_message(self.name, message)
        with self.personal_chat_file.open("a", encoding="utf-8") as f:
            f.write(formatted)

    def send_vote(self, voted_name: str):
        """Append a vote for the current voting phase."""
        with self.personal_vote_file.open("a", encoding="utf-8") as f:
            f.write(voted_name + "\n")

def get_llm_player_name(game_dir):
    """
    Find the AI player's name from its marker file, a roster member's log,
    or the AI entry in the game configuration, in that order.
    """
    # Read the explicit AI-name marker first.
    ai_player_file = game_dir / AI_PLAYER_FILE
    if ai_player_file.exists():
        content = ai_player_file.read_text(encoding="utf-8").strip()
        if content:  # Make sure it's not empty
            return content
    
    # Otherwise identify a roster member with an AI log file.
    player_names_file = game_dir / PLAYER_NAMES_FILE
    if player_names_file.exists():
        for player_name in player_names_file.read_text(encoding="utf-8").splitlines():
            player_name = player_name.strip()
            if player_name:
                log_file = game_dir / LLM_LOG_FILE_FORMAT.format(player_name)
                if log_file.exists():
                    return player_name
    
    # Step 3: Read from config.json as final fallback
    config_file = game_dir / "config.json"
    if config_file.exists():
        try:
            import json
            with open(config_file) as f:
                config = json.load(f)
            
            # Find the AI player in config
            for player in config.get("players", []):
                if player.get("is_llm", False) or player.get("is_ai", False):
                    return player.get("name")
        except Exception as e:
            print(f"[AI Name] Error reading from config: {e}")
    
    return None

@app.route("/")
def index():
    return render_template("game.html")


@app.route("/api/release")
def release_marker():
    """Public, non-secret marker for deployment and smoke verification."""
    status = relevant_source_fingerprint(Path(_BASE_DIR))
    return jsonify({
        "release_id": None,
        "agent_preset_id": None,
        "agent_preset_version": None,
        "source_fingerprint_sha256": status.get("sha256"),
        "manifest_matches_source": None,
        "manifest_status": "manifest_disabled",
        "file_differences": {"changed": [], "missing": [], "unexpected": []},
    })



@app.route("/join", methods=["POST"])
def join_game():
    """
    Map real_name → code name, create a WebPlayer, mark player as JOINED,
    and return session_id + metadata to the browser.
    """
    data = request.json or {}
    game_id = data.get("game_id")
    real_name = data.get("real_name")
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    real_names_file = game_dir / REAL_NAMES_FILE
    real_names_to_codenames_str = real_names_file.read_text(encoding="utf-8").splitlines()

    real_to_code = {}
    for line in real_names_to_codenames_str:
        if REAL_NAME_CODENAME_DELIMITER in line:
            real, code = line.split(REAL_NAME_CODENAME_DELIMITER, 1)
            real_to_code[real] = code

    if real_name not in real_to_code:
        return jsonify({"error": "Name not found in game"}), 404

    code_name = real_to_code[real_name]
    is_ai = get_is_ai(code_name, game_dir)

    # Create WebPlayer session
    session_id = secrets.token_hex(8)
    player_states[session_id] = WebPlayer(game_dir, code_name, is_ai)
    _persist_sessions()

    # Record the player's joined status.
    status_file = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(code_name)
    status_file.write_text(JOINED, encoding="utf-8")

    role_display = get_role_display_string(is_ai) if is_ai else None

    return jsonify(
        {
            "session_id": session_id,
            "code_name": code_name,
            "is_ai": is_ai,
            "role": role_display,
            # Provide rules as a separate response field, not a duplicate chat bubble.
            "rules": RULES_OF_THE_GAME,
        }
    )


@app.route("/lobby_join", methods=["POST"])
def lobby_join():
    """Open-lobby join: player types game_id + real_name (free text). The
    server allocates a random codename from OPTIONAL_CODE_NAMES (minus the AI's
    reserved codename and codenames already given out), appends to the lobby
    roster, marks the player as JOINED, and creates a WebPlayer session.

    Rejected if the lobby is closed (admin already clicked START) or the game
    has already started, or the real_name is already taken in this lobby, or
    the codename pool is exhausted.
    """
    data = request.json or {}
    game_id = data.get("game_id")
    real_name = (data.get("real_name") or "").strip()
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    if not real_name:
        return jsonify({"error": "Please enter your name"}), 400
    if len(real_name) > 60:
        return jsonify({"error": "Name too long"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    with _get_lobby_lock(game_id):
        lobby = _load_lobby(game_dir)
        if lobby is None:
            return jsonify({"error": "This game does not use the open lobby"}), 400

        # Reject if game already started.
        gst_file = game_dir / GAME_START_TIME_FILE
        if gst_file.exists() and gst_file.read_text(encoding="utf-8").strip():
            return jsonify({"error": "Game already started"}), 409
        if not lobby.get("open"):
            return jsonify({"error": "Lobby is closed — admin started the game"}), 409

        # Duplicate real_name check (case-insensitive).
        norm = _normalize_real_name(real_name)
        for h in lobby.get("humans", []):
            if _normalize_real_name(h.get("real_name", "")) == norm:
                return jsonify({"error": f"'{real_name}' is already taken in this lobby"}), 409

        # Allocate codename: random from pool minus AI minus already-used.
        used = {h["codename"] for h in lobby.get("humans", [])}
        if lobby.get("ai_codename"):
            used.add(lobby["ai_codename"])
        available = [c for c in OPTIONAL_CODE_NAMES if c not in used]
        if available:
            codename = random.choice(available)
        else:
            suffix = 1
            while f"Player{suffix}" in used:
                suffix += 1
            codename = f"Player{suffix}"

        # Touch per-player files for this codename so /messages, /vote, etc. work.
        (game_dir / PERSONAL_CHAT_FILE_FORMAT.format(codename)).touch()
        (game_dir / PERSONAL_VOTE_FILE_FORMAT.format(codename)).touch()
        (game_dir / PERSONAL_STATUS_FILE_FORMAT.format(codename)).touch()
        (game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(codename)).touch()

        # Mark as JOINED so stt_main.wait_for_players() proceeds immediately
        # once the admin clicks START GAME.
        (game_dir / PERSONAL_STATUS_FILE_FORMAT.format(codename)).write_text(
            JOINED, encoding="utf-8"
        )

        # Create the WebPlayer session.
        session_id = secrets.token_hex(8)
        player_states[session_id] = WebPlayer(game_dir, codename, is_ai=False)
        _touch_session(session_id)

        # Append to lobby roster + persist. `instructions_ack` tracks whether
        # the player has read & confirmed the instructions screen (shown
        # between joining and entering the chat) so the admin can see who is
        # still reading before clicking START.
        lobby.setdefault("humans", []).append({
            "real_name": real_name,
            "codename": codename,
            "session_id": session_id,
            "joined_at": datetime.now().isoformat(timespec="seconds"),
            "instructions_ack": False,
            "instructions_ack_at": None,
        })
        _save_lobby(game_dir, lobby)
        _persist_sessions()

    return jsonify({
        "session_id": session_id,
        "code_name": codename,
        "is_ai": False,
        "role": None,
        "rules": RULES_OF_THE_GAME,
    })


@app.route("/lobby_ack_instructions", methods=["POST"])
def lobby_ack_instructions():
    """Mark a player as having read & confirmed the instructions screen.

    Called by the client after the player checks the agreement box and clicks
    the confirm button on the instructions screen (which sits between joining
    the lobby and entering the chat). Updates the lobby roster so the admin
    lobby card can show who is still reading before clicking START.
    """
    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id"}), 400
    p = player_states.get(session_id)
    if p is None:
        return jsonify({"error": "unknown session"}), 404
    game_dir = getattr(p, "game_dir", None)
    if game_dir is None:
        return jsonify({"error": "no game for session"}), 400
    game_id = game_dir.name
    _touch_session(session_id)

    with _get_lobby_lock(game_id):
        lobby = _load_lobby(game_dir)
        if lobby is None:
            return jsonify({"error": "no lobby file"}), 400
        found = False
        for h in lobby.get("humans", []):
            if h.get("session_id") == session_id:
                h["instructions_ack"] = True
                h["instructions_ack_at"] = datetime.now().isoformat(timespec="seconds")
                found = True
                break
        if found:
            _save_lobby(game_dir, lobby)

    if not found:
        # Never report success for an acknowledgement we did not record —
        # otherwise the participant walks into the chat while the admin
        # lobby shows them as still reading, with no way to recover.
        return jsonify({"error": "player not in lobby", "found": False}), 404
    return jsonify({"success": True, "found": True})


@app.route("/rejoin_session", methods=["POST"])
def rejoin_session():
    """
    Re-hydrate a lost session from (session_id, game_id, code_name) kept in the
    browser's localStorage. Called on page load if the server has forgotten the
    session (e.g. after docker-compose restart).

    Returns the same shape as /join on success so the client can just reuse it.
    """
    data = request.json or {}
    session_id = data.get("session_id")
    game_id = data.get("game_id")
    code_name = data.get("code_name")
    if not (session_id and game_id and code_name):
        return jsonify({"error": "Missing fields"}), 400
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    # Fast path: session already exists in memory.
    if session_id in player_states:
        p = player_states[session_id]
        return jsonify({
            "session_id": session_id,
            "code_name": p.name,
            "is_ai": bool(getattr(p, "is_ai", False)),
            "role": get_role_display_string(getattr(p, "is_ai", False)) if getattr(p, "is_ai", False) else None,
            "rules": RULES_OF_THE_GAME,
            "rehydrated": False,
        })

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    # Verify this code_name is a real player in this game.
    real_names_file = game_dir / REAL_NAMES_FILE
    if not real_names_file.exists():
        return jsonify({"error": "Game not ready"}), 404
    valid_codes = set()
    for line in real_names_file.read_text(encoding="utf-8").splitlines():
        if REAL_NAME_CODENAME_DELIMITER in line:
            _real, code = line.split(REAL_NAME_CODENAME_DELIMITER, 1)
            valid_codes.add(code)
    if code_name not in valid_codes:
        return jsonify({"error": "Unknown player"}), 404

    is_ai = get_is_ai(code_name, game_dir)
    player_states[session_id] = WebPlayer(game_dir, code_name, is_ai)
    _persist_sessions()
    return jsonify({
        "session_id": session_id,
        "code_name": code_name,
        "is_ai": is_ai,
        "role": get_role_display_string(is_ai) if is_ai else None,
        "rules": RULES_OF_THE_GAME,
        "rehydrated": True,
    })


@app.route("/messages", methods=["POST"])
def get_messages():
    """
    Polling endpoint: the browser calls this every second with a session_id.
    We return:
      - new messages since last poll
      - flags: game_over, can_vote, can_chat, game_started
      - remaining players (for voting)
    """
    data = request.json or {}
    session_id = data.get("session_id")
    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    _touch_session(session_id)
    player = player_states[session_id]
    messages = player.get_new_messages()

    game_over = is_game_over(player.game_dir)
    can_vote = is_time_to_vote(player.game_dir)
    game_started = all_players_joined(player.game_dir)

    remaining_players = []
    remaining_file = player.game_dir / REMAINING_PLAYERS_FILE
    if remaining_file.exists():
        # After the first vote: use remaining_players.txt
        remaining_players = [
            line.strip()
            for line in remaining_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        # Before ANY elimination: all players are remaining
        full_list_file = player.game_dir / PLAYER_NAMES_FILE
        if full_list_file.exists():
            remaining_players = [
                line.strip()
                for line in full_list_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    # Get all players for the frontend
    all_players = []
    full_list_file = player.game_dir / PLAYER_NAMES_FILE
    if full_list_file.exists():
        all_players = [
            line.strip()
            for line in full_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # Pre-start (open-lobby): PLAYER_NAMES_FILE may only contain the AI's
    # codename until the admin clicks START. Fall back to the lobby roster
    # so the player's "waiting room" can show all joined codenames.
    if not game_started:
        lobby = _load_lobby(player.game_dir)
        if lobby is not None:
            lobby_codenames = [h["codename"] for h in lobby.get("humans", [])]
            if lobby.get("ai_codename"):
                lobby_codenames.append(lobby["ai_codename"])
            # Merge: lobby codenames take precedence (richer pre-start view).
            seen = set(lobby_codenames)
            for n in all_players:
                if n not in seen:
                    lobby_codenames.append(n)
                    seen.add(n)
            all_players = lobby_codenames

    # Joined players — same check the admin uses (see /game_state route).
    joined_players = []
    for _pname in all_players:
        _sf = player.game_dir / PERSONAL_STATUS_FILE_FORMAT.format(_pname)
        if _sf.exists() and _sf.read_text(encoding="utf-8").strip() == JOINED.strip():
            joined_players.append(_pname)

    # Get game over message if game has ended
    game_over_message = None
    game_duration_minutes = None
    if game_over:
        from core.game_constants import WHO_WINS_FILE
        winner_file = player.game_dir / WHO_WINS_FILE
        if winner_file.exists():
            game_over_message = winner_file.read_text(encoding="utf-8").strip()

        # Calculate game duration deterministically: rounds × discussion_time
        config_file = player.game_dir / "config.json"
        if config_file.exists():
            try:
                import json
                with open(config_file) as f:
                    config = json.load(f)
                from core.game_constants import DAYTIME_MINUTES_KEY
                daytime_minutes = config.get(DAYTIME_MINUTES_KEY, 2.0)
                total_players = len(all_players) if all_players else 0
                final_players = len(remaining_players) if remaining_players else 0
                num_rounds = total_players - final_players
                if num_rounds == 0 and game_over:
                    num_rounds = 1
                game_duration_minutes = round(num_rounds * daytime_minutes, 1)
            except Exception as e:
                print(f"[Duration] Error calculating duration: {e}")
                game_duration_minutes = None

    # chat is only allowed when:
    #   - all players joined
    #   - not voting time
    #   - game not over
    can_chat = game_started and (not can_vote) and (not game_over)

    # Expose server-side game_start_time + daytime_minutes so the player's
    # timer can sync to the server clock instead of running a drifting client-side
    # countdown that resets per browser tab.
    game_start_time_str = None
    gst_file = player.game_dir / GAME_START_TIME_FILE
    if gst_file.exists():
        gst_text = gst_file.read_text(encoding="utf-8").strip()
        if gst_text:
            game_start_time_str = gst_text

    # Per-round discussion anchor (preferred by client for timer sync).
    # If the file exists but is empty (race: player polled before stt_main wrote
    # it), fall back to the file's mtime — close enough for a one-poll anchor
    # until the real HH:MM:SS is written on the next tick.
    #
    # We send BOTH the HH:MM:SS string (human/debug) AND the file's mtime as
    # epoch ms (authoritative anchor for the timer). HH:MM:SS + client-side
    # Date parsing caused a clock-drift midnight-wrap bug where the parsed
    # time went "into the future" vs the client's clock, triggering the
    # "subtract a day" branch and showing 00:00 for the whole round.
    discussion_start_time_str = None
    discussion_start_epoch_ms = None
    dst_file = player.game_dir / DISCUSSION_START_TIME_FILE
    if dst_file.exists():
        dst_text = dst_file.read_text(encoding="utf-8").strip()
        try:
            mtime = dst_file.stat().st_mtime
            discussion_start_epoch_ms = int(mtime * 1000)
        except Exception:
            pass
        if dst_text:
            discussion_start_time_str = dst_text
        else:
            if discussion_start_epoch_ms is not None:
                discussion_start_time_str = time.strftime(
                    "%H:%M:%S", time.localtime(discussion_start_epoch_ms / 1000)
                )

    daytime_minutes_val = None
    cfg_file = player.game_dir / "config.json"
    if cfg_file.exists():
        try:
            import json as _json
            with open(cfg_file, encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            from core.game_constants import DAYTIME_MINUTES_KEY as _DTK
            daytime_minutes_val = _cfg.get(_DTK)
        except Exception:
            pass

    lobby_state = _load_lobby(player.game_dir)
    lobby_closed = bool(
        lobby_state is not None
        and not lobby_state.get("open")
        and not game_started
        and not lobby_state.get("finalized_at")
    )
    lobby_closed_waited = None
    lobby_closed_paid = False

    return jsonify(
        {
            "messages": messages,
            "game_over": game_over,
            "game_over_message": game_over_message,
            "game_duration_minutes": game_duration_minutes,
            "can_vote": can_vote,
            "can_chat": can_chat,
            "game_started": game_started,
            "lobby_closed": lobby_closed,
            "lobby_closed_paid": lobby_closed_paid,
            "lobby_closed_waited_minutes": (
                round(lobby_closed_waited, 1)
                if lobby_closed_waited is not None else None
            ),
            "remaining_players": remaining_players,
            "all_players": all_players,
            "joined_players": joined_players,
            "game_start_time": game_start_time_str,
            "discussion_start_time": discussion_start_time_str,
            "discussion_start_epoch_ms": discussion_start_epoch_ms,
            "daytime_minutes": daytime_minutes_val,
            "server_time_ms": int(time.time() * 1000),
            "typing_users": _get_typing_users(player.game_dir.name, player.name),
            "votes_cast": _count_current_round_votes(player.game_dir, remaining_players, dst_file) if can_vote else 0,
            "votes_needed": len(remaining_players),
            "stuck": _detect_stuck_game(player.game_dir, can_vote, remaining_players, dst_file, game_over, game_id=player.game_dir.name),
        }
    )


def _detect_stuck_game(game_dir, can_vote, remaining_players, dst_file, game_over, game_id=None):
    """Heuristic: detect a game stuck in the voting phase. Two scenarios:

    1. voting_wait_timeout: some players haven't voted after 120s — manager is
       blocked waiting for them (player disconnected).
    2. manager_dead_all_votes_cast: all votes exist but the manager process died
       before it could tally them (e.g. Ctrl+C killed the process group).

    Returns {stuck: True, reason, seconds_idle, waiting_for: [...]} or {stuck: False}.
    """
    out = {"stuck": False}
    if game_over or not can_vote:
        return out
    try:
        now = time.time()
        anchor = dst_file.stat().st_mtime if dst_file and dst_file.exists() else 0.0
        phase_file = game_dir / PHASE_STATUS_FILE
        phase_mtime = phase_file.stat().st_mtime if phase_file.exists() else 0.0
        ref = max(anchor, phase_mtime)
        elapsed = now - ref if ref > 0 else 0
        if elapsed < 120:
            return out
        # Who hasn't voted yet?
        waiting = [name for name in remaining_players
                   if not has_voted_this_round(game_dir, name)]
        if not waiting:
            # All votes exist. Check if the manager is dead — if so, nobody
            # will ever tally these votes without admin intervention.
            if game_id is not None:
                procs = running_games.get(game_id, {})
                manager_proc = procs.get("manager")
                manager_alive = manager_proc is not None and manager_proc.poll() is None
                if not manager_alive:
                    return {
                        "stuck": True,
                        "reason": "manager_dead_all_votes_cast",
                        "seconds_idle": int(elapsed),
                        "waiting_for": [],
                        "votes_complete": True,
                    }
            return out
        out = {
            "stuck": True,
            "reason": "voting_wait_timeout",
            "seconds_idle": int(elapsed),
            "waiting_for": waiting,
        }
    except Exception as e:
        print(f"[StuckDetect] {e}")
    return out


def _count_current_round_votes(game_dir, remaining_players, dst_file) -> int:
    """Count current-round votes using the per-round token each writer records.

    Vote-file existence and modification times do not establish round membership.
    """
    return sum(1 for name in remaining_players
               if has_voted_this_round(game_dir, name))


def _current_round_token(game_dir) -> str:
    return current_round_token(game_dir)


def _has_voted_this_round(game_dir, name) -> bool:
    return has_voted_this_round(game_dir, name)


def _record_vote_round(game_dir, name, token=None) -> None:
    record_vote_round(game_dir, name, token)


_vote_locks: dict = {}
_vote_locks_guard = threading.Lock()


def _get_vote_lock(game_id: str, name: str) -> threading.Lock:
    """Serialize check-then-write for one player's vote.

    Concurrent /vote requests must not both pass the round check and append.
    This lock serializes those operations within the server process.
    """
    key = (str(game_id), str(name))
    with _vote_locks_guard:
        lock = _vote_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _vote_locks[key] = lock
        return lock


def _get_typing_users(game_id: str, exclude_name: str) -> list:
    """Return names currently typing in game_id, excluding the requesting player."""
    now = time.time()
    # Human players via in-memory state
    states = _typing_states.get(game_id, {})
    typing = [
        name for name, ts in states.items()
        if name != exclude_name and (now - ts) < _TYPING_TIMEOUT_SECS
    ]
    # AI player via flag file ({name}_typing.txt written by llm player process)
    game_dir = resolve_game_dir(game_id)
    for flag in game_dir.glob("*_typing.txt"):
        ai_name = flag.stem.replace("_typing", "")
        if ai_name != exclude_name and ai_name not in typing:
            # Only trust flag file if it was modified recently (process might have crashed)
            try:
                if (now - flag.stat().st_mtime) < _TYPING_TIMEOUT_SECS * 2:
                    typing.append(ai_name)
            except OSError:
                pass
    return typing


@app.route("/set_typing", methods=["POST"])
def set_typing():
    """Record that a player is actively typing (called on keypress, debounced client-side)."""
    data = request.json or {}
    session_id = data.get("session_id")
    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401
    player = player_states[session_id]
    game_id = player.game_dir.name
    _typing_states.setdefault(game_id, {})[player.name] = time.time()
    return jsonify({"ok": True})


@app.route("/send", methods=["POST"])
def send_message():
    """
    Append a chat message to this player's personal chat file.
    We assume the front-end already enforces "only during discussion",
    so here אנחנו רק בודקים תקינות בסיסית.
    """
    data = request.json or {}
    session_id = data.get("session_id")
    message = (data.get("message") or "").strip()

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # English-only enforcement (server-side safety net).
    # Allow ASCII plus common smart-punctuation characters from copy-paste.
    _allowed_non_ascii = set("\u2018\u2019\u201C\u201D\u2013\u2014\u2010\u2011")
    if any((ord(c) > 127) and (c not in _allowed_non_ascii) for c in message):
        return jsonify({"error": "Please write in English only."}), 400

    player = player_states[session_id]
    player.send_message(message)
    # Clear typing indicator for this player on send
    game_id = player.game_dir.name
    _typing_states.get(game_id, {}).pop(player.name, None)

    return jsonify({"success": True})

@app.route("/vote", methods=['POST'])
def vote():
    """Append this player's vote for the current round.

    Enforce one vote per player per round so duplicate submissions do not
    leave extra vote-file lines for a later round to consume.
    """
    data = request.json or {}
    session_id = data.get("session_id")
    voted_name = (data.get("voted_name") or "").strip()

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    if not voted_name:
        return jsonify({"error": "No vote"}), 400

    player = player_states[session_id]
    game_dir = player.game_dir

    # The vote panel already enforces all of this, but nothing stops a
    # participant from calling the endpoint directly.
    try:
        phase = (game_dir / PHASE_STATUS_FILE).read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        phase = ""
    if not ("VOTING" in phase.upper() or "Voting" in phase):
        return jsonify({"error": "Voting is not open right now"}), 409

    try:
        remaining = [n.strip() for n in
                     (game_dir / REMAINING_PLAYERS_FILE).read_text(
                         encoding="utf-8", errors="replace").splitlines()
                     if n.strip()]
    except OSError:
        remaining = []
    if remaining:
        if player.name not in remaining:
            return jsonify({"error": "You are no longer in the game"}), 403
        if voted_name not in remaining:
            return jsonify({"error": "That player is no longer in the game"}), 400
    if voted_name == player.name:
        return jsonify({"error": "You cannot vote for yourself"}), 400

    # Check-then-write must be one atomic step: two threads racing a double tap
    # would otherwise both pass the check and both append.
    with _get_vote_lock(game_dir.name, player.name):
        with vote_round_lock(game_dir, player.name):
            if _has_voted_this_round(game_dir, player.name):
                return jsonify({"error": "You have already voted this round"}), 409
            token = _current_round_token(game_dir)
            player.send_vote(voted_name)
            _record_vote_round(game_dir, player.name, token)

    return jsonify({"success": True})

@app.route("/get_game_config", methods=["POST"])
def get_game_config():
    """
    Return game configuration, specifically the discussion time duration.
    Used by frontend to initialize the countdown timer.
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    config_file = player.game_dir / "config.json"
    
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        from core.game_constants import DAYTIME_MINUTES_KEY
        daytime_minutes = config.get(DAYTIME_MINUTES_KEY, 2.0)
        
        return jsonify({
            "daytime_minutes": daytime_minutes
        })
    
    # Default fallback
    return jsonify({"daytime_minutes": 2.0})

@app.route("/get_conversation_topic", methods=["POST"])
def get_conversation_topic():
    """
    Return the conversation topic for this game (if available).
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    
    from core.game_constants import CONVERSATION_TOPIC_FILE
    topic_file = player.game_dir / CONVERSATION_TOPIC_FILE
    
    if topic_file.exists():
        topic = topic_file.read_text(encoding="utf-8").strip()
        if topic:
            return jsonify({"topic": topic})
    
    return jsonify({"topic": None})

@app.route("/lobby_status", methods=["POST"])
def lobby_status():
    """
    Return lobby status: all players and which have joined.
    Used for the waiting lobby screen.

    Open-lobby model: pre-start, the player roster comes from
    `lobby_roster.json` (codenames only — never expose other players' real
    names). Post-start, fall back to PLAYER_NAMES_FILE / status files.
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]

    game_started = all_players_joined(player.game_dir)
    lobby = _load_lobby(player.game_dir)

    if lobby is not None and not game_started:
        # Pre-start: codenames only (privacy — don't leak real names to peers).
        codenames = [h["codename"] for h in lobby.get("humans", [])]
        if lobby.get("ai_codename"):
            codenames.append(lobby["ai_codename"])
        # Everyone in the roster has JOINED (set during /lobby_join).
        return jsonify({
            "players": codenames,
            "joined_players": codenames,
            "game_started": False,
            "lobby_open": bool(lobby.get("open")),
            "min_players": int(lobby.get("min_players", 3)),
            "joined_count": len(lobby.get("humans", [])),
        })

    # Use persisted player state after START or when no open lobby is present.
    player_names_file = player.game_dir / PLAYER_NAMES_FILE
    if not player_names_file.exists():
        return jsonify({"error": "Game not ready"}), 404

    all_players = [
        name.strip()
        for name in player_names_file.read_text(encoding="utf-8").splitlines()
        if name.strip()
    ]

    joined_players = []
    for player_name in all_players:
        status_file = player.game_dir / PERSONAL_STATUS_FILE_FORMAT.format(player_name)
        if status_file.exists() and status_file.read_text(encoding="utf-8").strip() == JOINED.strip():
            joined_players.append(player_name)

    return jsonify({
        "players": all_players,
        "joined_players": joined_players,
        "game_started": game_started,
    })

@app.route("/get_survey_options", methods=["POST"])
def get_survey_options():
    """
    Return all player names for the survey (including eliminated players).
    Excludes the current player from the list.
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    
    # Get all players from PLAYER_NAMES_FILE
    player_names_file = player.game_dir / PLAYER_NAMES_FILE
    if not player_names_file.exists():
        return jsonify({"error": "Player names file not found"}), 404
    
    all_players = [
        name.strip()
        for name in player_names_file.read_text(encoding="utf-8").splitlines()
        if name.strip()
    ]
    
    # Exclude the current player
    survey_options = [p for p in all_players if p != player.name]
    
    return jsonify({"players": survey_options})

@app.route("/get_ai_player_name", methods=["POST"])
def get_ai_player_name():
    """
    Return the actual AI player's name using the reliable detection logic.
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    ai_name = get_llm_player_name(player.game_dir)
    
    if not ai_name:
        return jsonify({"error": "AI player not found"}), 404
    
    return jsonify({"ai_name": ai_name})

@app.route("/get_model_info", methods=["POST"])
def get_model_info():
    """
    Return the AI model name from the game configuration.
    """
    data = request.json or {}
    session_id = data.get("session_id")

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    config_file = player.game_dir / "config.json"
    
    if not config_file.exists():
        return jsonify({"error": "Config not found"}), 404
    
    try:
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        # Find the LLM player and extract model name
        model_name = "Unknown Model"
        for p in config.get("players", []):
            if p.get("is_llm", False):
                llm_config = p.get("llm_config", {})
                model_name = llm_config.get("model_name", "Unknown Model")
                break
        
        return jsonify({"model_name": model_name})
    except Exception as e:
        print(f"[Model Info] Error: {e}")
        return jsonify({"error": "Could not read model info"}), 500

@app.route("/submit_tap", methods=["POST"])
def submit_tap():
    """Record an in-game 'who feels most human' tap to in_game_taps.jsonl."""
    data = request.json or {}
    session_id = data.get("session_id")
    tapped_name = (data.get("tapped_name") or "").strip()
    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401
    if not tapped_name:
        return jsonify({"error": "missing tapped_name"}), 400
    player = player_states[session_id]
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "voter": player.name,
        "tapped": tapped_name,
    }
    taps_file = player.game_dir / "in_game_taps.jsonl"
    with open(taps_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")
    return jsonify({"ok": True})


@app.route("/survey_submit", methods=["POST"])
def survey_submit():
    """
    Submit end-of-game survey results.
    Writes the survey fields defined by game_constants.py:
    TIME - <timestamp>
    Was the LLM identified - <not used, always 0>
    similarity to human behavior - <1-5>
    timing of messaging - <1-5>
    relevance of messages - <1-5>
    enjoyment of game - <1-5>
    Comments:
    Suspicion reasoning: <suspicion_reason>
    Other comments or suggestions: <notes>
    
    The AI is already revealed when this survey is shown. The identification
    field is written as 0 and is not a participant response.
    """
    data = request.json or {}
    session_id = data.get("session_id")
    human_likeness = data.get("human_likeness")
    timing = data.get("timing")
    relevance = data.get("relevance")
    enjoyment = data.get("enjoyment")
    suspicion_reason = (data.get("suspicion_reason") or "").strip()
    notes = (data.get("notes") or "").strip()

    if not suspicion_reason:
        return jsonify({"error": "suspicion_reason is required"}), 400
    if not notes:
        return jsonify({"error": "notes is required"}), 400
    if len(suspicion_reason) > 5000:
        return jsonify({"error": "suspicion_reason too long (max 5000 chars)"}), 400
    if len(notes) > 5000:
        return jsonify({"error": "notes too long (max 5000 chars)"}), 400

    if session_id not in player_states:
        return jsonify({"error": "Invalid session"}), 401

    player = player_states[session_id]
    
    # Check if already submitted
    if player.did_submit_survey:
        return jsonify({"error": "Survey already submitted"}), 400
    
    # Validate all rating fields (1-5)
    try:
        human_likeness = int(human_likeness)
        timing = int(timing)
        relevance = int(relevance)
        enjoyment = int(enjoyment)
        
        for val, name in [(human_likeness, "human_likeness"), 
                          (timing, "timing"), (relevance, "relevance"),
                          (enjoyment, "enjoyment")]:
            if not (1 <= val <= 5):
                return jsonify({"error": f"{name} must be between 1 and 5"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid rating value"}), 400
    
    # Write to survey file in Niv's exact format
    survey_file = player.game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(player.name)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Match exact format from game_constants.py
    # The identification field is 0 (not applicable), not a participant response.
    survey_content = f"TIME{METRIC_NAME_AND_SCORE_DELIMITER}{timestamp}\n"
    survey_content += f"Was the LLM identified{METRIC_NAME_AND_SCORE_DELIMITER}0\n"
    survey_content += f"similarity to human behavior{METRIC_NAME_AND_SCORE_DELIMITER}{human_likeness}\n"
    survey_content += f"timing of messaging{METRIC_NAME_AND_SCORE_DELIMITER}{timing}\n"
    survey_content += f"relevance of messages{METRIC_NAME_AND_SCORE_DELIMITER}{relevance}\n"
    survey_content += f"{ENJOYMENT}{METRIC_NAME_AND_SCORE_DELIMITER}{enjoyment}\n"
    survey_content += "Comments:\n"
    survey_content += f"Suspicion reasoning: {suspicion_reason}\n"
    survey_content += f"Other comments or suggestions: {notes}\n"
    
    survey_file.write_text(survey_content, encoding="utf-8")
    
    # Mark as submitted
    player.did_submit_survey = True
    
    return jsonify({"success": True})

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_AGENT_CODES = frozenset({"current"})
_AGENT_CODE_EXPECTED_COMMITS = {}
_AGENT_BUILD_INFO_FILE = "agent_build.json"
_AGENT_BUILD_WRITE_LOCK = threading.Lock()


def _agent_code_from_game(game_dir):
    """Return the configured agent-code label or the default agent label."""
    try:
        with open(game_dir / GAME_CONFIG_FILE, encoding='utf-8') as f:
            cfg = json.load(f)
        return next((p.get('llm_config', {}).get('agent_code', 'current')
                     for p in cfg.get('players', []) if p.get('is_llm')),
                    'current')
    except Exception:
        return 'current'


def _agent_worktree(agent_code):
    return None


def _agent_spawn_params(game_dir):
    return _BASE_DIR, {}


class AgentCheckoutVerificationError(RuntimeError):
    """A frozen comparison label does not resolve to its promised commit."""


def _annotate_agent_build(info, agent_code):
    """Attach label/commit agreement without treating a dirty patch as failure."""
    annotated = dict(info)
    expected = _AGENT_CODE_EXPECTED_COMMITS.get(agent_code)
    full_commit = str(annotated.get("commit_full") or annotated.get("commit") or "")
    annotated.update({
        "agent_code": agent_code,
        "expected_commit": expected,
        "commit_matches_label": (
            full_commit.startswith(expected) if expected and full_commit != "unknown"
            else None
        ),
    })
    return annotated


def _verify_frozen_agent_checkout(agent_code, checkout):
    return None


def _agent_spawn_preflight(game_dir):
    """Resolve spawn params and verify frozen checkout identity before Popen."""
    agent_code = _agent_code_from_game(game_dir)
    spawn_cwd, extra_env = _agent_spawn_params(game_dir)
    build = _verify_frozen_agent_checkout(agent_code, spawn_cwd)
    return spawn_cwd, extra_env, build


def _record_agent_build_info(game_dir, spawn_cwd, agent_code=None):
    """Persist the exact checkout fingerprint used for an LLM subprocess.

    The configured label alone is not provenance: a worktree can point at the
    wrong commit or carry an operational patch.  Record the runtime-derived
    commit and dirty/patch marker immediately after a successful spawn.
    """
    agent_code = agent_code or _agent_code_from_game(game_dir)
    info = _annotate_agent_build(
        _agent_build_info(spawn_cwd, refresh=True), agent_code,
    )
    info.update({
        "spawn_cwd": str(Path(spawn_cwd).resolve()),
        "recorded_at": datetime.now().isoformat(),
        "source": "spawn_record",
        "recorded": True,
    })
    out_path = Path(game_dir) / _AGENT_BUILD_INFO_FILE
    tmp_path = out_path.with_name(
        f".{out_path.name}.{os.getpid()}.{threading.get_ident()}."
        f"{secrets.token_hex(4)}.tmp"
    )
    try:
        # The lock serializes threads in this web process; the unique temp name
        # also prevents collisions between separate worker processes. replace()
        # is atomic once the complete JSON has been flushed and closed.
        with _AGENT_BUILD_WRITE_LOCK:
            tmp_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            tmp_path.replace(out_path)
    except Exception as exc:
        # A provenance write must never orphan an already-running subprocess.
        info["recorded"] = False
        info["record_error"] = f"{type(exc).__name__}: {exc}"
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return info

# On Windows, Ctrl+C propagates to the entire console process group by default.
# Setting CREATE_NEW_PROCESS_GROUP gives each child its own group so pressing
# Ctrl+C in the web_player terminal does not kill stt_main.py or llm_interface.py.
_POPEN_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32" else {}
)

# ── Admin: process management state ──
running_games = {}  # game_id -> {"manager": Popen, "llm": Popen, "started_at": str}

# Open-lobby AI processes that were pre-spawned at create-game time so the AI
# is already warming up (and its "ready" badge can turn green) while the admin
# lobby is still open. These are NOT in running_games — they hold only the LLM
# subprocess. At finalize/START, the live process is promoted into
# running_games (see _launch_game_processes) instead of spawning a new one.
lobby_ai_procs = {}  # game_id -> {"llm": Popen, "llm_log": str, "llm_log_f": file, "started_at": str}


def _launch_lobby_ai(game_id: str, game_dir: Path):
    """Pre-spawn llm_interface.py for an open-lobby game so the AI begins
    warming up while players are still joining. Best-effort: returns the Popen
    on success, or None if there is no LLM player / spawn fails. The agent's
    own wait loop blocks until the manager writes the game-start time, so the
    pre-spawned agent stays idle (sub-agents gated) until START.
    """
    config_file = game_dir / GAME_CONFIG_FILE
    if not config_file.exists():
        return None
    try:
        with open(config_file, encoding='utf-8') as f:
            config = json.load(f)
    except Exception:
        return None
    if not any(p.get('is_llm') for p in config.get('players', [])):
        return None
    # Don't double-spawn.
    existing = lobby_ai_procs.get(game_id)
    if existing and existing.get('llm') and existing['llm'].poll() is None:
        return existing['llm']

    python_exe = sys.executable
    proc_env = _get_env_with_api_keys()
    proc_env['PYTHONIOENCODING'] = 'utf-8'
    llm_log_path = game_file(game_dir, "llm_interface_log.txt", for_write=True)
    try:
        llm_log_f = open(llm_log_path, 'w', encoding='utf-8', errors='replace')
    except Exception:
        return None
    try:
        spawn_cwd, extra_env, _preflight_build = _agent_spawn_preflight(game_dir)
    except AgentCheckoutVerificationError as exc:
        # Keep the failure visible to the existing Issues/readiness plumbing,
        # but never start an agent whose source does not match its label.
        llm_log_f.write(f"AGENT CHECKOUT PRE-FLIGHT FAILED: {exc}\n")
        llm_log_f.flush()
        llm_log_f.close()
        return None
    proc_env.update(extra_env)
    try:
        llm_proc = subprocess.Popen(
            [python_exe, 'llm_interface.py', game_id],
            stdout=llm_log_f, stderr=subprocess.STDOUT,
            cwd=spawn_cwd, env=proc_env,
            **_POPEN_FLAGS,
        )
    except Exception:
        llm_log_f.close()
        return None
    agent_build = _record_agent_build_info(game_dir, spawn_cwd)
    lobby_ai_procs[game_id] = {
        "llm": llm_proc,
        "llm_log": str(llm_log_path),
        "llm_log_f": llm_log_f,
        "started_at": datetime.now().isoformat(),
        "agent_build": agent_build,
    }
    return llm_proc


def _cleanup_lobby_ai(game_id: str, *, terminate: bool = True):
    """Remove a pre-spawned lobby AI for `game_id`. When `terminate` is True the
    subprocess is killed; when False the entry is dropped WITHOUT touching the
    process (used when promoting it into running_games at START)."""
    entry = lobby_ai_procs.pop(game_id, None)
    if not entry:
        return None
    if terminate:
        proc = entry.get('llm')
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        fh = entry.get('llm_log_f')
        if fh:
            try:
                fh.close()
            except Exception:
                pass
        return None
    return entry


def _get_next_game_id():
    """Get next available 4-digit game ID."""
    from core.game_paths import next_game_id
    Path(DIRS_PREFIX).mkdir(parents=True, exist_ok=True)
    return next_game_id()


# ── Admin: Authentication ──

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Authenticate admin with password. Sets session cookie."""
    data = request.json or {}
    password = data.get('password', '')
    if ADMIN_PASSWORD and hmac.compare_digest(password, ADMIN_PASSWORD):
        session['admin_auth'] = True
        return jsonify({"success": True})
    return jsonify({"error": "Wrong password"}), 403


@app.route('/api/admin/check_auth')
def admin_check_auth():
    """Check if current session is admin-authenticated."""
    if session.get("admin_auth"):
        return jsonify({"authenticated": True})
    return jsonify({"authenticated": False}), 401


# ── Admin: API Key Management ──

@app.route('/api/admin/set_api_keys', methods=['POST'])
def admin_set_api_keys():
    """Store API keys in memory (never persisted to disk)."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    ENV_KEYS = ["TOGETHER_API_KEY", "GOOGLE_CLOUD_PROJECT", "OPENROUTER_API_KEY",
                "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    keys_set = []
    for env_key in ENV_KEYS:
        val = data.get(env_key, "").strip()
        if val:
            server_api_keys[env_key] = val
            keys_set.append(env_key)
    return jsonify({"success": True, "keys_set": keys_set})


@app.route('/api/admin/get_key_status')
def admin_get_key_status():
    """Return which API keys are configured (from UI or env). Never returns actual values."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    status = {}
    for env_key in ["TOGETHER_API_KEY", "GOOGLE_CLOUD_PROJECT", "OPENROUTER_API_KEY",
                    "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        if server_api_keys.get(env_key):
            status[env_key] = "set_ui"
        elif os.environ.get(env_key):
            status[env_key] = "set_env"
        else:
            status[env_key] = "missing"
    return jsonify(status)


# ── Admin: Test Model Connectivity ──

# Realistic scheduler/generator prompts for deep-test mode.
# These approximate what the agent actually sends, so we catch models that
# pass the "Say hi" smoke test but fail on the real prompt (bad format,
# empty output, token truncation, content filters, etc.).
_DEEP_TEST_SYSTEM = (
    "You are one player in a group chat game. You must respond like a normal "
    "human in a casual conversation. Keep messages very short (2-6 words)."
)
_DEEP_TEST_CONVO = (
    "[10:00:01] Alex: hey everyone\n"
    "[10:00:04] Jordan: yooo\n"
    "[10:00:07] Sam: how's it going\n"
)
_DEEP_TEST_SCHEDULER_PROMPT = (
    _DEEP_TEST_SYSTEM + "\n\n"
    "Recent chat:\n" + _DEEP_TEST_CONVO +
    "\nShould you send a message now or wait? "
    "Reply with exactly one token: <send> or <wait>."
)
_DEEP_TEST_GENERATOR_PROMPT = (
    _DEEP_TEST_SYSTEM + "\n\n"
    "Recent chat:\n" + _DEEP_TEST_CONVO +
    "\nWrite your next message (2-6 words, casual, no name prefix):"
)


def _deep_test_call(provider, model_name, prompt, max_tokens):
    """Send one prompt to the provider and return the text response.

    Raises on any error (caller catches and returns formatted error).
    """
    if provider == "gemini":
        from google import genai as _genai
        from google.genai import types as _genai_types
        api_key = (server_api_keys.get("GEMINI_API_KEY")
                   or server_api_keys.get("GOOGLE_API_KEY")
                   or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
        if api_key:
            client = _genai.Client(api_key=api_key)
        else:
            project = server_api_keys.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                raise RuntimeError("No Gemini/Google API key and no Cloud Project set.")
            client = _genai.Client(vertexai=True, project=project, location=location)
        # Gemini 2.5 "thinking" models burn output tokens on internal reasoning before
        # producing text. At small budgets (e.g. 10 tokens for the scheduler) the whole
        # quota is consumed by thinking and the response comes back empty. Mirror the
        # policy already used in llm_wrapper.py:
        #   Gemini 3         → thinking_budget is gone; use thinking_level="low"
        #   Pro (2.5)        → thinking is REQUIRED, minimum budget 128 + extra max_tokens
        #   Flash/Flash-Lite → disable thinking entirely (budget=0)
        config_kwargs = {"temperature": 1.0}
        model_lower = (model_name or "").lower()
        adjusted_max = max_tokens
        if "gemini-3" in model_lower:
            config_kwargs["thinking_config"] = _genai_types.ThinkingConfig(thinking_level="low")
            if "pro" in model_lower and "flash" not in model_lower:
                adjusted_max = max(max_tokens + 200, 256)
            else:
                adjusted_max = max(max_tokens + 100, 200)
        elif "gemini-2.5" in model_lower:
            if "pro" in model_lower:
                config_kwargs["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=128)
                adjusted_max = max(max_tokens + 200, 256)
            else:
                config_kwargs["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=0)
        config_kwargs["max_output_tokens"] = adjusted_max
        resp = client.models.generate_content(
            model=model_name, contents=prompt, config=config_kwargs,
        )
        return resp.text if hasattr(resp, 'text') else str(resp)

    if provider == "together":
        from together import Together as _Together
        api_key = server_api_keys.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            raise RuntimeError("Together API Key not set.")
        client = _Together(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=1.0,
        )
        return resp.choices[0].message.content

    if provider == "openrouter":
        import requests as _req
        api_key = server_api_keys.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OpenRouter API Key not set.")
        # Note: some models (e.g. OpenAI GPT-5.x / o-series on OpenRouter) reject
        # `temperature`; omit it here since the Test button is just a liveness check.
        # Some providers (e.g. Azure-hosted GPT-5.x) also enforce a minimum max_tokens
        # (>= 16); bump small Deep-test budgets to satisfy that floor.
        effective_max = max(max_tokens, 16)
        payload = {"model": model_name,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": effective_max}
        # Reasoning models (GPT-5.x / o-series / Gemini Pro) default to high effort and
        # would burn this tiny budget on internal reasoning → empty reply. Force LOW.
        from llm_players.openrouter_utils import reasoning_payload_for as _reasoning_for
        _reasoning = _reasoning_for(model_name)
        if _reasoning:
            payload["reasoning"] = _reasoning
        r = _req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if not r.ok:
            # Surface the actual OpenRouter error body so the caller sees the real reason
            # (e.g. unsupported parameter, model not available, etc.) instead of a generic 400.
            body = r.text
            try:
                err_json = r.json()
                err = err_json.get("error") if isinstance(err_json, dict) else None
                if isinstance(err, dict):
                    err_msg = err.get("message") or ""
                    meta = err.get("metadata")
                    if meta:
                        # Provider's raw error is usually under metadata.raw or metadata.provider_name
                        err_msg = f"{err_msg} | metadata={meta}"
                else:
                    err_msg = err or body
            except Exception:
                err_msg = body
            raise RuntimeError(f"OpenRouter {r.status_code}: {err_msg}")
        return r.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unknown provider: {provider}")


@app.route('/api/admin/test_internet', methods=['POST'])
def admin_test_internet():
    """Pre-game check that Internet Context grounding actually works in
    THIS environment. Catches the server-side DefaultCredentialsError (Vertex
    with no ADC) BEFORE a game, so the internet sub-agent isn't silently dead."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    import time as _time
    start = _time.time()
    try:
        # Reset the cached client so this reflects the CURRENT auth/env.
        try:
            import llm_players.agent.sub_agents.web_search as ws
        except Exception:
            import llm_players.agent.sub_agents.web_search as ws
        ws._client_cache = None
        results = ws.gemini_grounded_search("biggest world news this week", max_results=2)
        elapsed_ms = int((_time.time() - start) * 1000)
        answer = (results[0].get("snippet") if results else "") or ""
        return jsonify({
            "success": bool(answer),
            "message": (f"Internet OK in {elapsed_ms}ms" if answer
                        else "Search returned no answer"),
            "answer": answer[:200],
            "latency_ms": elapsed_ms,
        })
    except Exception as e:
        elapsed_ms = int((_time.time() - start) * 1000)
        return jsonify({
            "success": False,
            "message": f"Internet search FAILED: {str(e)[:400]}",
            "latency_ms": elapsed_ms,
        })


@app.route('/api/admin/test_model', methods=['POST'])
def admin_test_model():
    """Test that a model is reachable with a tiny API call.

    Modes:
      deep_test=false (default) — single "Say hi" round-trip, ~5 tokens.
      deep_test=true — sends a realistic scheduler prompt + generator prompt,
        verifies non-empty outputs. Catches models that pass the smoke test
        but fail on the real agent prompt.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    model_name = data.get('model_name', '').strip()
    provider = data.get('provider', '').strip() or _detect_provider(model_name)
    deep_test = bool(data.get('deep_test', False))
    if not model_name:
        return jsonify({"success": False, "message": "No model name provided"}), 400

    import time as _time
    start = _time.time()

    if deep_test:
        # Two calls: scheduler + generator
        try:
            sched_text = _deep_test_call(provider, model_name, _DEEP_TEST_SCHEDULER_PROMPT, max_tokens=10)
            gen_text = _deep_test_call(provider, model_name, _DEEP_TEST_GENERATOR_PROMPT, max_tokens=25)
        except Exception as e:
            elapsed_ms = int((_time.time() - start) * 1000)
            return jsonify({
                "success": False,
                "message": f"Deep test failed: {str(e)[:400]}",
                "latency_ms": elapsed_ms,
                "deep_test": True,
            })

        elapsed_ms = int((_time.time() - start) * 1000)
        sched_text = (sched_text or "").strip()
        gen_text = (gen_text or "").strip()

        warnings = []
        if not sched_text:
            warnings.append("Scheduler returned empty output")
        elif "<send>" not in sched_text.lower() and "<wait>" not in sched_text.lower():
            warnings.append(f"Scheduler did not return <send>/<wait> token (got: {sched_text[:60]!r})")
        if not gen_text:
            warnings.append("Generator returned empty output")
        elif len(gen_text) > 300:
            warnings.append(f"Generator output suspiciously long ({len(gen_text)} chars) — may not follow length constraints")

        success = not any("empty" in w for w in warnings)
        return jsonify({
            "success": success,
            "message": f"Deep test completed in {elapsed_ms}ms" + (f" — {len(warnings)} warning(s)" if warnings else ""),
            "latency_ms": elapsed_ms,
            "deep_test": True,
            "scheduler_response": sched_text[:200],
            "generator_response": gen_text[:200],
            "warnings": warnings,
        })

    # ── Simple smoke test (default) ──
    try:
        text = _deep_test_call(provider, model_name, "Say hi", max_tokens=5)
        elapsed_ms = int((_time.time() - start) * 1000)
        return jsonify({"success": True, "message": f"Model responded in {elapsed_ms}ms", "latency_ms": elapsed_ms})
    except Exception as e:
        elapsed_ms = int((_time.time() - start) * 1000)
        return jsonify({"success": False, "message": str(e)[:300], "latency_ms": elapsed_ms})


@app.route('/api/admin/personas')
def admin_personas():
    """Return available personas for the config form."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    personas = load_personas()
    result = {}
    for pid, pdata in personas.items():
        result[pid] = {"name": pdata.get("name", pid), "description": pdata.get("core_identity", "")}
    return jsonify(result)


@app.route('/api/admin/topics')
def admin_topics():
    """Return available conversation topics."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    topics_file = Path("configurations/conversation_topics.json")
    if topics_file.exists():
        with open(topics_file, encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data.get("default", []))
    return jsonify([])




@app.route('/api/admin/create_config', methods=['POST'])
def admin_create_config():
    """Create a game configuration from web form data.

    Open-lobby model: humans supply their names when joining the lobby.
    Humans join via /lobby_join after the game directory is created. This
    config holds only the AI player (when has_llm) and game-wide settings;
    `min_players` is the gate for the admin's START GAME button.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    try:
        # Validate the entire payload, including advanced overrides, before merging.
        reject_retired_config(data)
    except ConfigError as exc:
        return jsonify({'error': str(exc)}), 400
    has_llm = bool(data.get('has_llm', True))
    # Two humans and, when enabled, one AI; joining itself has no upper limit.
    min_players = 3
    daytime_minutes = float(data.get('daytime_minutes', 2.5))
    anonymous_voting = bool(data.get('anonymous_voting', True))
    persona = data.get('persona', 'swiss_student')
    topic = data.get('topic', '').strip() or None
    review_mode = data.get('review_mode', 'gemini')
    pipeline = (data.get('pipeline') or 'agent').strip().lower()
    if pipeline != 'agent':
        return jsonify({'error': 'Only the Social Turing Test agent is available'}), 400
    generator_model = data.get('generator_model', '').strip() or None
    generator_provider = data.get('generator_provider', '').strip() or None
    reviewer_model = data.get('reviewer_model', '').strip() or None
    scheduler_model = data.get('scheduler_model', 'gemini-2.5-flash').strip() or None
    sub_agent_model = (data.get('sub_agent_model') or '').strip() or None
    is_test = bool(data.get('is_test', False))
    normalize_commas = bool(data.get('normalize_commas', False))

    if min_players < 1:
        min_players = 1

    # Build player configs — only the AI is known up-front. Humans are
    # appended later at /api/admin/finalize_and_start once the lobby closes.
    player_configs = []
    ai_codename = None
    if has_llm:
        ai_codename = random.choice(OPTIONAL_CODE_NAMES)
        imposter = PlayerConfig(name=ai_codename)
        imposter.is_ai = True
        imposter.is_llm = True
        imposter.real_name = "LLM0"
        llm_config = DEFAULT_LLM_CONFIG.copy()
        llm_config[PERSONA_KEY] = persona

        # Event context (same key prepare_config --event-context writes): the
        # shared physical setting of the human players, injected into the
        # agent's fixed persona facts for consistent identity answers.
        event_context = (data.get('event_context') or '').strip()
        if event_context:
            llm_config["event_context"] = event_context

        # The public release runs only the current checkout.
        agent_code = (data.get('agent_code') or 'current').strip()
        if agent_code not in SUPPORTED_AGENT_CODES:
            return jsonify({
                "error": f"Unsupported agent_code {agent_code!r}. Expected one of: "
                         + ", ".join(sorted(SUPPORTED_AGENT_CODES)),
            }), 400
        llm_config["agent_code"] = agent_code

        # Public defaults: Flash handles generation/scheduling/review.
        gen_model = generator_model or "gemini-2.5-flash"
        llm_config[MODEL_NAME_KEY] = gen_model
        provider = generator_provider or _detect_provider(gen_model)
        llm_config[USE_GEMINI_KEY] = (provider == "gemini")
        llm_config[USE_TOGETHER_KEY] = (provider == "together")
        llm_config[USE_OPENROUTER_KEY] = (provider == "openrouter")

        # Reviewer mode sets provider fields before the agent pipeline override.
        if review_mode == 'gemini':
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "gemini"
            llm_config["reviewer_model"] = reviewer_model or "gemini-2.5-flash"
        elif review_mode == 'together':
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "together"
            llm_config["reviewer_model"] = reviewer_model or "Qwen/Qwen3-Next-80B-A3B-Instruct"
        elif review_mode == 'openrouter':
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "openrouter"
            llm_config["reviewer_model"] = reviewer_model or "openai/gpt-4o-mini"
        else:
            llm_config[ASYNC_TYPE_KEY] = SCHEDULE_THEN_GENERATE_TYPE

        # Explicitly honor the reviewer toggle despite inherited defaults and
        # the default reviewer model set below. The metrics/UI can then show
        # "reviewer: off" when review_mode is 'none'.
        llm_config["reviewer_enabled"] = (review_mode != 'none')

        # Pipeline override — agent-based players. The reviewer
        # config above is reused (LangGraphPlayer inherits from ReviewedPlayer).
        if pipeline == 'agent':
            llm_config[ASYNC_TYPE_KEY] = 'social_turing_test_agent'
            llm_config.setdefault("reviewer_backend", "gemini")
            llm_config.setdefault("reviewer_model", "gemini-2.5-flash")
            # Agent pipeline — dual-role investigator + multi-candidate generation + ranker
            # + strategic voting. Defaults mirror the release baseline;
            # advanced_llm_config (below) can still override any of these.
            llm_config.setdefault("multi_candidate_generation", True)
            llm_config.setdefault("num_candidates", 4)
            llm_config.setdefault("investigator", True)
            llm_config.setdefault("voting_strategy", "strategic")
            llm_config.setdefault("pipeline_timeout_s", 35.0)
            # Optional dedicated send/wait scheduler model (provider
            # auto-detected by the player). Empty = share the generator model.
            if scheduler_model:
                llm_config["scheduler_model"] = scheduler_model
            # Dedicated sub-agents model: all background workers — memory,
            # internet, persona, repetition etc. Provider auto-detected by the player.
            if sub_agent_model:
                llm_config["sub_agent_model"] = sub_agent_model

        llm_config["normalize_commas"] = normalize_commas

        # Merge advanced config overrides from schema-driven UI (3.2)
        advanced = data.get('advanced_llm_config', {})
        if not isinstance(advanced, dict):
            advanced = {}
        llm_config.update(advanced)

        # Pipeline integrity guard (post-merge). The advanced merge above can
        # silently clobber candidate ranking, investigation, or strategic voting
        # with stale schema defaults. The pipeline selection is the source of
        # truth, so re-assert its defining identity flags after the merge.
        # Only the genuinely tunable knobs (num_candidates, pipeline_timeout_s)
        # honor an explicit advanced override.
        _PIPELINE_ASYNC_TYPE = {'agent': 'social_turing_test_agent'}
        if pipeline in _PIPELINE_ASYNC_TYPE:
            llm_config[ASYNC_TYPE_KEY] = _PIPELINE_ASYNC_TYPE[pipeline]
        if pipeline == 'agent':
            # Identity flags — forced ON regardless of advanced (stale-frontend proof).
            llm_config["multi_candidate_generation"] = True
            llm_config["investigator"] = True
            llm_config["voting_strategy"] = "strategic"
            # Tunable knobs — keep defaults unless the admin set one explicitly.
            agent_tunables = {"num_candidates": 4, "pipeline_timeout_s": 35.0}
            for k, v in agent_tunables.items():
                if k not in advanced:
                    llm_config[k] = v
        llm_config['orchestration_mode'] = 'synchronous'
        llm_config['allow_active_authoritative'] = False
        llm_config['agent_code'] = 'current'

        imposter.llm_config = llm_config
        player_configs.append(imposter)

    # Build config dict — humans appended later at finalize_and_start.
    config = {
        PLAYERS_KEY_IN_CONFIG: [asdict(p) for p in player_configs],
        DAYTIME_MINUTES_KEY: daytime_minutes,
        "nighttime_minutes": 0,
        ANONYMOUS_VOTING_KEY: anonymous_voting,
        "is_test": is_test,
        "min_players": min_players,
        "publication_mode": False,
        "require_all_instructions_ack": False,
        "lobby_ai_codename": ai_codename,
    }
    if topic:
        config["enable_conversation_topic"] = True
        config["conversation_topic"] = topic

    # Save config
    config_dir = Path("configurations")
    config_dir.mkdir(exist_ok=True)
    config_name = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config_path = config_dir / config_name
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return jsonify({"config_path": str(config_path), "config": config})



@app.route('/api/admin/create_game', methods=['POST'])
def admin_create_game():
    """Create a game directory from an existing config.

    Open-lobby model: the config has only the AI (or no players, when
    has_llm=False). The PLAYER_NAMES_FILE / REAL_NAMES_FILE files are
    seeded empty/AI-only here; finalize_and_start fills them once the
    admin clicks START GAME.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    config_path = data.get('config_path')

    if not config_path or not Path(config_path).exists():
        return jsonify({"error": "Config file not found"}), 400

    game_id = _get_next_game_id()

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    config_error = _local_config_error(config)
    if config_error:
        return jsonify({"error": config_error}), 400

    # Route by the configured real/test game category.
    game_dir = target_dir_for_config(config, game_id)
    game_dir.mkdir(parents=True)
    ensure_game_subdirs(game_dir)

    config["config_original_path_when_game_created"] = config_path
    with open(game_dir / GAME_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    players = [PlayerConfig(**pc) for pc in config[PLAYERS_KEY_IN_CONFIG]]
    # Only AI codenames are known up-front in the open-lobby flow; humans
    # are appended at finalize_and_start.
    ai_names = [p.name for p in players if p.is_ai]
    (game_dir / PLAYER_NAMES_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (game_dir / REMAINING_PLAYERS_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (game_dir / AI_PLAYER_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (game_dir / REAL_NAMES_FILE).write_text("", encoding='utf-8')
    (game_dir / PHASE_STATUS_FILE).write_text("Waiting for Players", encoding='utf-8')
    (game_dir / PUBLIC_MANAGER_CHAT_FILE).touch()
    (game_dir / PUBLIC_DAYTIME_CHAT_FILE).touch()
    (game_dir / WHO_WINS_FILE).touch()
    (game_dir / GAME_START_TIME_FILE).touch()
    (game_dir / NOTES_FILE).touch()

    # Open-lobby roster file — admin sees this populate as players join.
    lobby = {
        "open": True,
        "ai_codename": config.get("lobby_ai_codename"),
        "min_players": 3,
        "publication_mode": False,
        "require_all_instructions_ack": False,
        "humans": [],
    }
    _save_lobby(game_dir, lobby)

    # Handle conversation topic
    topic = None
    if config.get("enable_conversation_topic"):
        topic = config.get("conversation_topic")
    if topic:
        (game_dir / CONVERSATION_TOPIC_FILE).write_text(topic, encoding='utf-8')
    else:
        (game_dir / CONVERSATION_TOPIC_FILE).touch()

    # Per-player files for known players (AI only at this stage). Human
    # files are touched lazily on /lobby_join.
    for p in players:
        (game_dir / PERSONAL_CHAT_FILE_FORMAT.format(p.name)).touch()
        (game_dir / PERSONAL_VOTE_FILE_FORMAT.format(p.name)).touch()
        (game_dir / PERSONAL_STATUS_FILE_FORMAT.format(p.name)).touch()
        if p.is_llm:
            (game_dir / LLM_LOG_FILE_FORMAT.format(p.name)).touch()
        else:
            (game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(p.name)).touch()

    # Pre-spawn the AI now (while the lobby is open) so it warms up and its
    # "ready" badge can turn green before the admin clicks START. The agent
    # waits (sub-agents gated) until the manager writes the game-start time.
    try:
        _launch_lobby_ai(game_id, game_dir)
    except Exception:
        pass

    resp = {"game_id": game_id, "game_dir": str(game_dir)}
    try:
        resp["staged_rollout"] = _config_active_preflight(config)
    except Exception as exc:
        resp["staged_rollout"] = {"label": "unknown", "warning": f"preflight check error: {exc}"}
    return jsonify(resp)


def _local_config_error(config):
    """Validate the supported agent configuration, including saved configurations."""
    try:
        reject_retired_config(config)
    except ConfigError as exc:
        return str(exc)
    for player in config.get('players', []):
        if not player.get('is_llm'):
            continue
        settings = player.get('llm_config') or {}
        async_type = settings.get('async_type')
        if (async_type not in {'social_turing_test_agent'}
                or settings.get('orchestration_mode', 'synchronous') != 'synchronous'
                or settings.get('agent_code', 'current') != 'current'
                or settings.get('allow_active_authoritative', False)):
            return 'Only the Social Turing Test agent configuration is supported'
    return None


def _config_active_preflight(config):
    """Return the supported agent configuration status for the admin UI."""
    players = config.get("players", [])
    llm = [p for p in players if p.get("is_llm")]
    has_agent = any(
        (p.get("llm_config") or {}).get("async_type") in {"social_turing_test_agent"}
        for p in llm
    )
    return {"active_requested": False,
            "label": "Social Turing Test Agent" if has_agent else "non-agent",
            "preflight_ok": None,
            "warning": None}



@app.route('/api/admin/start_game', methods=['POST'])
def admin_start_game():
    """Launch stt_main.py and llm_interface.py as subprocesses."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    game_id = data.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game directory not found"}), 404

    lobby = _load_lobby(game_dir)
    if lobby and lobby.get("open"):
        return admin_finalize_and_start()

    force = data.get('force', False)
    err = _launch_game_processes(game_id, game_dir, force=force)
    if err:
        code, body = err
        return jsonify(body), code

    return jsonify({
        "success": True,
        "game_id": game_id,
        "has_llm": running_games[game_id].get("llm") is not None,
    })


def _launch_game_processes(game_id: str, game_dir: Path, *, force: bool = False):
    """Spawn stt_main.py + llm_interface.py for a game. Returns None on
    success, or a (status_code, error_dict) tuple on failure.

    Mutates `running_games[game_id]` on success. Idempotent: if a previous
    process group exists for this id, returns 409 unless `force=True` is
    passed (in which case the previous procs are terminated first).

    Shared by `/api/admin/start_game`, `/api/admin/finalize_and_start`,
    and `/api/admin/recreate_game`.
    """
    # Verify a frozen agent before terminating an existing run or starting the
    # manager.  This guarantees a mislabeled worktree cannot leave an orphaned
    # manager process behind while the LLM spawn is rejected.
    config_file = game_dir / GAME_CONFIG_FILE
    config = None
    agent_spawn_preflight = None
    if not config_file.exists():
        return 400, {"error": "Game config is missing"}
    if config_file.exists():
        try:
            with open(config_file, encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, ValueError) as exc:
            return 400, {"error": f"Could not read game config: {exc}"}
        config_error = _local_config_error(config)
        if config_error:
            return 400, {"error": config_error}
        humans = [p for p in config.get('players', []) if not p.get('is_llm')]
        if len(humans) < 2:
            return 400, {"error": "Need at least two humans to start the game"}
        if any(p.get('is_llm') for p in config.get('players', [])):
            if not admin_lobby_state_for_start(game_id, game_dir):
                return 409, {"error": "Wait for the AI to be ready"}
            try:
                agent_spawn_preflight = _agent_spawn_preflight(game_dir)
            except AgentCheckoutVerificationError as exc:
                return 400, {
                    "error": str(exc),
                    "error_code": "agent_checkout_mismatch",
                }

    if game_id in running_games:
        if not force:
            return 409, {"error": "Game already running"}
        procs = running_games.pop(game_id)
        for key in ('manager', 'llm'):
            proc = procs.get(key)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for key in ('manager_log_f', 'llm_log_f'):
            fh = procs.get(key)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    python_exe = sys.executable
    proc_env = _get_env_with_api_keys()
    proc_env['PYTHONIOENCODING'] = 'utf-8'

    manager_log_path = game_file(game_dir, "manager_log.txt", for_write=True)
    manager_log_f = open(manager_log_path, 'w', encoding='utf-8', errors='replace')
    try:
        manager_proc = subprocess.Popen(
            [python_exe, 'stt_main.py', game_id],
            stdout=manager_log_f, stderr=subprocess.STDOUT,
            cwd=_BASE_DIR, env=proc_env,
            **_POPEN_FLAGS,
        )
    except Exception:
        manager_log_f.close()
        raise

    llm_proc = None
    llm_log_path = None
    llm_log_f = None
    agent_build = None
    if config_file.exists():
        if any(p.get('is_llm') for p in config.get('players', [])):
            # Reuse a pre-spawned lobby AI if it's still alive (open-lobby flow):
            # the agent has already been warming up while players joined, so we
            # promote that process into running_games instead of spawning a new
            # one. `terminate=False` detaches it without killing it.
            pre = _cleanup_lobby_ai(game_id, terminate=False)
            if pre and pre.get('llm') and pre['llm'].poll() is None:
                llm_proc = pre['llm']
                llm_log_path = Path(pre['llm_log']) if pre.get('llm_log') else None
                llm_log_f = pre.get('llm_log_f')
                agent_build = pre.get('agent_build')
            else:
                # No live pre-spawn (died / never created) — close any stale
                # handle and spawn a fresh subprocess.
                if pre and pre.get('llm_log_f'):
                    try:
                        pre['llm_log_f'].close()
                    except Exception:
                        pass
                llm_log_path = game_file(game_dir, "llm_interface_log.txt", for_write=True)
                llm_log_f = open(llm_log_path, 'w', encoding='utf-8', errors='replace')
                spawn_cwd, extra_env, _preflight_build = (
                    agent_spawn_preflight or _agent_spawn_preflight(game_dir)
                )
                proc_env.update(extra_env)
                try:
                    llm_proc = subprocess.Popen(
                        [python_exe, 'llm_interface.py', game_id],
                        stdout=llm_log_f, stderr=subprocess.STDOUT,
                        cwd=spawn_cwd, env=proc_env,
                        **_POPEN_FLAGS,
                    )
                except Exception:
                    llm_log_f.close()
                    raise
                agent_build = _record_agent_build_info(game_dir, spawn_cwd)

            # Lobby processes created before provenance recording was available
            # can still be promoted safely; backfill a runtime fingerprint now.
            if llm_proc is not None and agent_build is None:
                spawn_cwd, _ = _agent_spawn_params(game_dir)
                agent_build = _record_agent_build_info(game_dir, spawn_cwd)

    running_games[game_id] = {
        "manager": manager_proc,
        "llm": llm_proc,
        "started_at": datetime.now().isoformat(),
        "manager_log": str(manager_log_path),
        "llm_log": str(llm_log_path) if llm_log_path else None,
        "manager_log_f": manager_log_f,
        "llm_log_f": llm_log_f,
        "agent_build": agent_build,
    }
    return None


@app.route('/api/admin/finalize_and_start', methods=['POST'])
def admin_finalize_and_start():
    """Open-lobby finalize + START. Reads lobby_roster.json, materializes the
    final player list (PLAYER_NAMES_FILE / REMAINING_PLAYERS_FILE /
    REAL_NAMES_FILE / config.players), then spawns stt_main + llm_interface.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    game_id = data.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game directory not found"}), 404

    with _get_lobby_lock(game_id):
        lobby = _load_lobby(game_dir)
        if lobby is None:
            return jsonify({"error": "This game does not use the open lobby"}), 400
        if not lobby.get("open"):
            return jsonify({"error": "Lobby already finalized"}), 409

        config_file = game_dir / GAME_CONFIG_FILE
        with open(config_file, encoding='utf-8') as f:
            config = json.load(f)
        config_error = _local_config_error(config)
        if config_error:
            return jsonify({"error": config_error}), 400
        existing = list(config.get(PLAYERS_KEY_IN_CONFIG, []))
        has_llm = any(p.get('is_llm') for p in existing)

        humans = lobby.get("humans", [])
        if len(humans) < 2:
            return jsonify({
                "error": "Need at least two humans to start the game",
            }), 400
        if has_llm:
            readiness = admin_lobby_state_for_start(game_id, game_dir)
            if not readiness:
                return jsonify({"error": "Wait for the AI to be ready"}), 409

        # Build final player list: keep AI from config, append humans.
        final_players = list(existing)  # AI (or empty)
        for h in humans:
            final_players.append({
                "name": h["codename"],
                "real_name": h["real_name"],
                "is_ai": False,
                "is_llm": False,
                "llm_config": {},
            })

        # has_llm=False edge case: pick a random human as the AI imposter.
        if not has_llm and final_players:
            imposter_idx = random.randrange(len(final_players))
            final_players[imposter_idx]["is_ai"] = True

        config[PLAYERS_KEY_IN_CONFIG] = final_players
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

        # Materialize the final files.
        all_names = [p["name"] for p in final_players]
        (game_dir / PLAYER_NAMES_FILE).write_text("\n".join(all_names), encoding='utf-8')
        (game_dir / REMAINING_PLAYERS_FILE).write_text("\n".join(all_names), encoding='utf-8')
        ai_names = [p["name"] for p in final_players if p.get("is_ai")]
        (game_dir / AI_PLAYER_FILE).write_text("\n".join(ai_names), encoding='utf-8')
        real_mappings = [
            f"{p['real_name']}{REAL_NAME_CODENAME_DELIMITER}{p['name']}"
            for p in final_players if not p.get("is_llm")
        ]
        (game_dir / REAL_NAMES_FILE).write_text("\n".join(real_mappings), encoding='utf-8')

        # Touch any per-player files we may have missed (defensive — humans were
        # already touched at /lobby_join time).
        for p in final_players:
            (game_dir / PERSONAL_CHAT_FILE_FORMAT.format(p["name"])).touch()
            (game_dir / PERSONAL_VOTE_FILE_FORMAT.format(p["name"])).touch()
            sf = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(p["name"])
            if not sf.exists():
                sf.touch()
            if p.get("is_llm"):
                (game_dir / LLM_LOG_FILE_FORMAT.format(p["name"])).touch()
            else:
                (game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(p["name"])).touch()
            # Humans were marked JOINED at /lobby_join. The LLM marks itself JOINED
            # when llm_interface.py starts. Don't override here.

        # Close the lobby.
        lobby["open"] = False
        lobby["finalized_at"] = datetime.now().isoformat(timespec="seconds")
        _save_lobby(game_dir, lobby)

    # Launch the game processes (outside the lock — Popen can block briefly).
    err = _launch_game_processes(game_id, game_dir)
    if err:
        code, body = err
        return jsonify(body), code

    return jsonify({
        "success": True,
        "game_id": game_id,
        "has_llm": running_games[game_id].get("llm") is not None,
        "player_count": len(final_players),
    })


@app.route('/api/admin/lobby_kick', methods=['POST'])
def admin_lobby_kick():
    """Remove a human from the lobby before START (admin only)."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    game_id = data.get('game_id')
    codename = data.get('codename')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    if not codename:
        return jsonify({"error": "codename required"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    with _get_lobby_lock(game_id):
        lobby = _load_lobby(game_dir)
        if lobby is None or not lobby.get("open"):
            return jsonify({"error": "Lobby is closed"}), 409

        humans = lobby.get("humans", [])
        target = None
        for h in humans:
            if h.get("codename") == codename:
                target = h
                break
        if not target:
            return jsonify({"error": "player not in lobby"}), 404

        humans.remove(target)
        lobby["humans"] = humans
        _save_lobby(game_dir, lobby)

    # Drop their session + status file (so the codename is fully released).
    sid = target.get("session_id")
    if sid and sid in player_states:
        player_states.pop(sid, None)
        _persist_sessions()
    sf = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(codename)
    if sf.exists():
        sf.write_text("", encoding="utf-8")

    return jsonify({"success": True, "codename": codename})


@app.route('/api/admin/close_lobby', methods=['POST'])
def admin_close_lobby():
    """Close an unstarted local lobby without launching a game."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    game_id = data.get('game_id')
    reason = (data.get('reason') or 'Lobby closed by the researcher').strip()
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    if len(reason) > 300:
        return jsonify({"error": "reason too long"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    returned_to_standby = 0
    with _get_lobby_lock(game_id):
        lobby = _load_lobby(game_dir)
        if lobby is None:
            return jsonify({"error": "This game does not use the open lobby"}), 400
        if not _game_not_started(game_dir):
            return jsonify({"error": "Game already started"}), 409
        if not lobby.get("open"):
            return jsonify({
                "success": True,
                "game_id": game_id,
                "already_closed": True,
                "returned_to_standby": 0,
            })

        closed_at = _now_iso()
        lobby["open"] = False
        lobby["closed_without_start"] = True
        lobby["closed_at"] = closed_at
        lobby["close_reason"] = reason
        _save_lobby(game_dir, lobby)

    _cleanup_lobby_ai(game_id, terminate=True)
    return jsonify({
        "success": True,
        "game_id": game_id,
        "already_closed": False,
        "returned_to_standby": returned_to_standby,
    })


@app.route('/api/admin/open_lobbies')
def admin_open_lobbies():
    """List every game whose lobby is open and not started yet."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err

    lobbies = []
    try:
        for _cat, _ver, game_dir in iter_all_games(include_simulations=False):
            game_id = game_dir.name
            if not _valid_game_id(game_id):
                continue
            if not _is_canonical_game_dir(game_id, game_dir):
                continue
            lobby = _load_lobby(game_dir)
            if not lobby or not lobby.get("open"):
                continue
            if not _game_not_started(game_dir):
                continue
            humans = lobby.get("humans", [])
            lobbies.append({
                "game_id": game_id,
                "created_at": lobby.get("created_at"),
                "joined": _occupied_human_seat_count(lobby),
                "capacity": None,
                "acknowledged": sum(
                    1 for h in humans if h.get("instructions_ack")),
            })
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to scan lobbies: {exc}"}), 500

    lobbies.sort(key=lambda item: item["game_id"], reverse=True)
    return jsonify({
        "ok": True,
        "count": len(lobbies),
        # Same tie-break as the participant router: highest game_id wins.
        "active_game_id": lobbies[0]["game_id"] if lobbies else None,
        "lobbies": lobbies,
    })


@app.route('/api/admin/lobby_state')
def admin_lobby_state():
    """Return the open-lobby roster (real_name + codename per player) for the
    admin's lobby card. Includes inactive (last-seen-based) hint per session.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    lobby = _load_lobby(game_dir)
    if lobby is None:
        return jsonify({"error": "no lobby file"}), 404

    now = time.time()
    rows = []
    for h in lobby.get("humans", []):
        sid = h.get("session_id")
        last_seen = _last_seen_by_session.get(sid)
        seconds_since = (now - last_seen) if last_seen else None
        rows.append({
            "real_name": h.get("real_name"),
            "codename": h.get("codename"),
            "joined_at": h.get("joined_at"),
            "seconds_since_seen": seconds_since,
            "inactive": (seconds_since is not None and seconds_since > _INACTIVE_THRESHOLD_SECONDS),
            "instructions_ack": bool(h.get("instructions_ack")),
            "instructions_ack_at": h.get("instructions_ack_at"),
        })
    # AI readiness: the LLM subprocess writes an `agent_ready` event to
    # game_events*.jsonl once its wrapper is constructed (see llm_interface.py).
    # Surface that to the admin lobby so START can wait for the AI to be up.
    has_ai = bool(lobby.get("ai_codename"))
    ai_ready = False
    ai_ready_at = None
    # Model-health (from the agent_ready event, enriched by the preflight in
    # llm_interface.py). Lets the lobby warn "reviewer degraded" BEFORE START.
    generator_ok = None
    reviewer_ok = None
    reviewer_model = None
    model_errors = []
    if has_ai:
        try:
            for ev_file in _game_event_files(game_dir, "game_events*.jsonl"):
                try:
                    with open(ev_file, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or "agent_ready" not in line:
                                continue
                            try:
                                evt = json.loads(line)
                            except Exception:
                                continue
                            if evt.get("event_type") == "agent_ready":
                                ai_ready = True
                                ai_ready_at = evt.get("timestamp") or ai_ready_at
                                if "generator_ok" in evt:
                                    generator_ok = evt.get("generator_ok")
                                    reviewer_ok = evt.get("reviewer_ok")
                                    reviewer_model = evt.get("reviewer_model")
                                    model_errors = evt.get("model_errors") or []
                except Exception:
                    continue
        except Exception:
            pass

    # Cheap second signal: is the LLM subprocess process alive? (Process up but
    # not necessarily ready — used to show "initializing" before agent_ready.)
    # Check both the pre-spawned lobby AI and a promoted/running AI.
    ai_subprocess_alive = False
    try:
        entry = lobby_ai_procs.get(game_id) or running_games.get(game_id)
        if entry:
            llm_proc = entry.get("llm")
            ai_subprocess_alive = bool(llm_proc and llm_proc.poll() is None)
    except Exception:
        ai_subprocess_alive = False

    ack_count = sum(
        1 for h in lobby.get("humans", []) if h.get("instructions_ack")
    )

    payload = {
        "open": bool(lobby.get("open")),
        "ai_codename": lobby.get("ai_codename"),
        "min_players": 3,
        "max_humans": None,
        "publication_mode": False,
        "required_humans": None,
        "publication_ready": len(rows) >= 2,
        "humans": rows,
        "joined_count": len(rows),
        "instructions_ack_count": ack_count,
        "game_started": all_players_joined(game_dir),
        "has_ai": has_ai,
        "ai_ready": ai_ready,
        "ai_ready_at": ai_ready_at,
        "ai_subprocess_alive": ai_subprocess_alive,
        "generator_ok": generator_ok,
        "reviewer_ok": reviewer_ok,
        "reviewer_model": reviewer_model,
        "model_errors": model_errors,
    }
    response = jsonify(payload)
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def admin_lobby_state_for_start(game_id, game_dir):
    """Require both a live process and its agent_ready event, without request state."""
    entry = lobby_ai_procs.get(game_id) or running_games.get(game_id) or {}
    proc = entry.get('llm')
    if not proc or proc.poll() is not None:
        return False
    for path in _game_event_files(game_dir):
        try:
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    if json.loads(line).get('event_type') == 'agent_ready':
                        return True
                except ValueError:
                    continue
        except OSError:
            continue
    return False


@app.route('/api/admin/stop_game', methods=['POST'])
def admin_stop_game():
    """Stop a running game's processes."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    game_id = data.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    if game_id not in running_games:
        # The game may only have a pre-spawned lobby AI (never STARTed). If so,
        # stopping it is valid — tear it down. Otherwise there's nothing to stop.
        if game_id in lobby_ai_procs:
            _cleanup_lobby_ai(game_id, terminate=True)
            return jsonify({"success": True})
        return jsonify({"error": "Game not running"}), 404

    procs = running_games.pop(game_id)
    for key in ('manager', 'llm'):
        proc = procs.get(key)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    for key in ('manager_log_f', 'llm_log_f'):
        fh = procs.get(key)
        if fh:
            try:
                fh.close()
            except Exception:
                pass

    # Also drop any pre-spawned lobby AI that was never promoted to START.
    _cleanup_lobby_ai(game_id, terminate=True)

    return jsonify({"success": True})


@app.route('/api/admin/recreate_game', methods=['POST'])
def admin_recreate_game():
    """Create a NEW open-lobby game with the same AI/topic/timing settings as
    an existing game.

    Open-lobby model: humans are NOT carried over — recreate produces an
    empty lobby and the admin clicks START GAME after players rejoin. Returns
    {new_game_id, lobby_open: true} so the admin UI can switch to the new
    lobby card.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    data = request.json or {}
    old_game_id = data.get('game_id')
    if not _valid_game_id(old_game_id):
        return jsonify({"error": "invalid game_id"}), 400

    old_game_dir = resolve_game_dir(old_game_id)
    old_config_path = old_game_dir / GAME_CONFIG_FILE
    if not old_config_path.exists():
        return jsonify({"error": "Old game config.json not found"}), 404

    try:
        with open(old_config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read old config: {e}"}), 500

    config_error = _local_config_error(config)
    if config_error:
        return jsonify({"error": config_error}), 400
    for key in ('max_humans', 'required_humans', 'agent_preset_id', 'agent_preset_version'):
        config.pop(key, None)
    config['publication_mode'] = False
    config['require_all_instructions_ack'] = False

    # Stop the source game if still running.
    if old_game_id in running_games:
        procs = running_games.pop(old_game_id)
        for key in ('manager', 'llm'):
            proc = procs.get(key)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for key in ('manager_log_f', 'llm_log_f'):
            fh = procs.get(key)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
    # Tear down any pre-spawned lobby AI for the source game that never STARTed.
    _cleanup_lobby_ai(old_game_id, terminate=True)

    for stale_key in ("config_original_path_when_game_created",):
        config.pop(stale_key, None)

    # Strip humans from the copied config — only AI carries over. Re-pick a
    # fresh AI codename so it doesn't collide with what humans get from the
    # pool in the new lobby.
    old_players = list(config.get(PLAYERS_KEY_IN_CONFIG, []))
    ai_player_dicts = [p for p in old_players if p.get("is_ai") and p.get("is_llm")]
    new_ai_codename = None
    if ai_player_dicts:
        new_ai_codename = random.choice(OPTIONAL_CODE_NAMES)
        ai_player_dicts[0]["name"] = new_ai_codename
        config[PLAYERS_KEY_IN_CONFIG] = ai_player_dicts[:1]
    else:
        config[PLAYERS_KEY_IN_CONFIG] = []
    config["lobby_ai_codename"] = new_ai_codename
    config.setdefault("min_players", 3)

    # Save config + create new game dir.
    config_dir = Path("configurations")
    config_dir.mkdir(exist_ok=True)
    config_name = f"web_recreate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config_path = config_dir / config_name
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    new_game_id = _get_next_game_id()
    new_game_dir = target_dir_for_config(config, new_game_id)
    new_game_dir.mkdir(parents=True)
    ensure_game_subdirs(new_game_dir)

    config["config_original_path_when_game_created"] = str(config_path)
    with open(new_game_dir / GAME_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    # Seed files (AI-only, same as admin_create_game).
    players = [PlayerConfig(**pc) for pc in config[PLAYERS_KEY_IN_CONFIG]]
    ai_names = [p.name for p in players if p.is_ai]
    (new_game_dir / PLAYER_NAMES_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (new_game_dir / REMAINING_PLAYERS_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (new_game_dir / AI_PLAYER_FILE).write_text("\n".join(ai_names), encoding='utf-8')
    (new_game_dir / REAL_NAMES_FILE).write_text("", encoding='utf-8')
    (new_game_dir / PHASE_STATUS_FILE).write_text("Waiting for Players", encoding='utf-8')
    (new_game_dir / PUBLIC_MANAGER_CHAT_FILE).touch()
    (new_game_dir / PUBLIC_DAYTIME_CHAT_FILE).touch()
    (new_game_dir / WHO_WINS_FILE).touch()
    (new_game_dir / GAME_START_TIME_FILE).touch()
    (new_game_dir / NOTES_FILE).touch()

    _save_lobby(new_game_dir, {
        "open": True,
        "ai_codename": new_ai_codename,
        "min_players": int(config.get("min_players", 3)),
        "humans": [],
    })

    topic = None
    if config.get("enable_conversation_topic"):
        topic = config.get("conversation_topic")
    if topic:
        (new_game_dir / CONVERSATION_TOPIC_FILE).write_text(topic, encoding='utf-8')
    else:
        (new_game_dir / CONVERSATION_TOPIC_FILE).touch()

    for p in players:
        (new_game_dir / PERSONAL_CHAT_FILE_FORMAT.format(p.name)).touch()
        (new_game_dir / PERSONAL_VOTE_FILE_FORMAT.format(p.name)).touch()
        (new_game_dir / PERSONAL_STATUS_FILE_FORMAT.format(p.name)).touch()
        if p.is_llm:
            (new_game_dir / LLM_LOG_FILE_FORMAT.format(p.name)).touch()
        else:
            (new_game_dir / PERSONAL_SURVEY_FILE_FORMAT.format(p.name)).touch()

    # Open-lobby flow: do NOT launch processes here. The admin must click
    # START GAME, which goes through /api/admin/finalize_and_start.
    # Pre-spawn the AI now so its "ready" badge can turn green before START.
    try:
        _launch_lobby_ai(new_game_id, new_game_dir)
    except Exception:
        pass
    return jsonify({
        "success": True,
        "new_game_id": new_game_id,
        "old_game_id": old_game_id,
        "lobby_open": True,
        "min_players": int(config.get("min_players", 3)),
    })


@app.route('/api/admin/game_status')
def admin_game_status():
    """Return current game state: phase, joined players, process health, timer."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    # Phase
    phase_file = game_dir / PHASE_STATUS_FILE
    phase = phase_file.read_text(encoding='utf-8').strip() if phase_file.exists() else "unknown"

    # Game over
    winner_file = game_dir / WHO_WINS_FILE
    winner = winner_file.read_text(encoding='utf-8').strip() if winner_file.exists() else ""
    game_over_flag = bool(winner)

    # Players
    all_p = []
    pf = game_dir / PLAYER_NAMES_FILE
    if pf.exists():
        all_p = [n.strip() for n in pf.read_text(encoding='utf-8').splitlines() if n.strip()]

    joined = []
    for name in all_p:
        sf = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(name)
        if sf.exists() and sf.read_text(encoding='utf-8').strip():
            joined.append(name)

    remaining = []
    rf = game_dir / REMAINING_PLAYERS_FILE
    if rf.exists():
        remaining = [n.strip() for n in rf.read_text(encoding='utf-8').splitlines() if n.strip()]

    game_started = all_players_joined(game_dir)

    # Process health
    procs = running_games.get(game_id, {})
    manager_alive = bool(
        procs.get("manager") and procs["manager"].poll() is None
    )
    llm_alive = bool(procs.get("llm") and procs["llm"].poll() is None)

    # AI player name
    ai_name = get_llm_player_name(game_dir)

    # Game start time + discussion duration for timer
    game_start_time = None
    gst_file = game_dir / GAME_START_TIME_FILE
    if gst_file.exists():
        gst_text = gst_file.read_text(encoding='utf-8').strip()
        if gst_text:
            game_start_time = gst_text

    # Per-round anchor (preferred by admin timer to show round-based countdown
    # matching what players see). Both HH:MM:SS (display) and epoch ms
    # (authoritative, drift-free) are returned — client prefers epoch.
    discussion_start_time = None
    discussion_start_epoch_ms = None
    dst_file = game_dir / DISCUSSION_START_TIME_FILE
    if dst_file.exists():
        dst_text = dst_file.read_text(encoding='utf-8').strip()
        try:
            discussion_start_epoch_ms = int(dst_file.stat().st_mtime * 1000)
        except Exception:
            pass
        if dst_text:
            discussion_start_time = dst_text

    # Game start anchor as epoch ms (from file mtime) so admin timer never
    # wraps around midnight due to clock drift.
    game_start_epoch_ms = None
    if gst_file.exists():
        try:
            game_start_epoch_ms = int(gst_file.stat().st_mtime * 1000)
        except Exception:
            pass

    game_end_epoch_ms = None
    if game_over_flag and winner_file.exists():
        try:
            game_end_epoch_ms = int(winner_file.stat().st_mtime * 1000)
        except Exception:
            pass
    game_duration_seconds = None
    if game_start_epoch_ms is not None and game_end_epoch_ms is not None:
        game_duration_seconds = max(
            0, round((game_end_epoch_ms - game_start_epoch_ms) / 1000, 1)
        )

    def _process_state(key, alive):
        if alive:
            return "running"
        if game_over_flag:
            return "completed"
        return "stopped" if procs.get(key) else "not_tracked"

    daytime_minutes = None
    cfg_file = game_dir / GAME_CONFIG_FILE
    if cfg_file.exists():
        try:
            with open(cfg_file, encoding='utf-8') as f:
                cfg = json.load(f)
            daytime_minutes = cfg.get(DAYTIME_MINUTES_KEY)
        except Exception:
            pass

    # Per-player status table (alive / eliminated / inactive) for admin view.
    # Eliminated = status file equals VOTED_OUT exactly. Inactive = a JOINED
    # player whose session hasn't polled /messages within the threshold.
    from core.game_constants import VOTED_OUT
    eliminated_set = set()
    for name in all_p:
        sf = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(name)
        if sf.exists() and sf.read_text(encoding='utf-8').strip() == VOTED_OUT.strip():
            eliminated_set.add(name)

    # Map codename -> session_id (so we can check last-seen per player).
    code_to_sid = {}
    for sid, p in player_states.items():
        try:
            if p.game_dir == game_dir:
                code_to_sid[p.name] = sid
        except Exception:
            continue

    # Map codename -> real_name (preserve from config; AI shows "(AI)").
    code_to_real = {}
    try:
        cfg_players = cfg.get(PLAYERS_KEY_IN_CONFIG, []) if cfg_file.exists() else []
        for p in cfg_players:
            code_to_real[p.get("name")] = (
                "(AI)" if p.get("is_ai") else p.get("real_name")
            )
    except Exception:
        pass
    # Lobby is the source of truth for human real_names pre-finalize.
    lobby_obj = _load_lobby(game_dir)
    if lobby_obj is not None:
        for h in lobby_obj.get("humans", []):
            code_to_real[h.get("codename")] = h.get("real_name")
        if lobby_obj.get("ai_codename"):
            code_to_real.setdefault(lobby_obj["ai_codename"], "(AI)")

    now = time.time()
    player_status = []
    for name in all_p:
        sid = code_to_sid.get(name)
        last_seen = _last_seen_by_session.get(sid) if sid else None
        seconds_since = (now - last_seen) if last_seen else None
        if name in eliminated_set:
            status = "eliminated"
        elif name in joined and seconds_since is not None and seconds_since > _INACTIVE_THRESHOLD_SECONDS and not game_over_flag:
            status = "inactive"
        elif name in joined:
            status = "alive"
        else:
            status = "not_joined"
        player_status.append({
            "codename": name,
            "real_name": code_to_real.get(name),
            "status": status,
            "is_ai": (name == ai_name),
            "seconds_since_seen": seconds_since,
        })

    # Lobby block (open / closed + roster) for admin lobby card.
    lobby_block = None
    if lobby_obj is not None:
        lobby_block = {
            "open": bool(lobby_obj.get("open")),
            "ai_codename": lobby_obj.get("ai_codename"),
            "min_players": int(lobby_obj.get("min_players", 3)),
            "humans": [
                {
                    "real_name": h.get("real_name"),
                    "codename": h.get("codename"),
                    "joined_at": h.get("joined_at"),
                }
                for h in lobby_obj.get("humans", [])
            ],
            "joined_count": len(lobby_obj.get("humans", [])),
        }

    return jsonify({
        "phase": phase,
        "game_over": game_over_flag,
        "winner": winner,
        "all_players": all_p,
        "joined_players": joined,
        "remaining_players": remaining,
        "game_started": game_started,
        "manager_alive": manager_alive,
        "llm_alive": llm_alive,
        "manager_state": _process_state("manager", manager_alive),
        "llm_state": _process_state("llm", llm_alive),
        "ai_player": ai_name,
        "started_at": procs.get("started_at"),
        "game_start_time": game_start_time,
        "game_start_epoch_ms": game_start_epoch_ms,
        "game_end_epoch_ms": game_end_epoch_ms,
        "game_duration_seconds": game_duration_seconds,
        "discussion_start_time": discussion_start_time,
        "discussion_start_epoch_ms": discussion_start_epoch_ms,
        "daytime_minutes": daytime_minutes,
        # Round number = eliminations so far + 1 (first discussion = round 1).
        "round_number": (len(all_p) - len(remaining) + 1) if all_p and remaining else 1,
        "server_time_ms": int(time.time() * 1000),
        "stuck": _detect_stuck_game(
            game_dir,
            can_vote=("VOTING" in (phase or "").upper() or "Voting" in (phase or "")),
            remaining_players=remaining,
            dst_file=game_dir / DISCUSSION_START_TIME_FILE,
            game_over=game_over_flag,
            game_id=game_id,
        ),
        "lobby": lobby_block,
        "player_status": player_status,
    })


@app.route('/api/admin/votes')
def admin_votes():
    """Return per-player voting history for the given game.

    Reads each `{name}_vote.txt` file and returns all rounds. Each line in the
    file is one round's vote. Used by the admin Monitor tab (live who-voted-for-whom)
    and the Vote History accordion (post-game).
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    all_p = []
    pf = game_dir / PLAYER_NAMES_FILE
    if pf.exists():
        all_p = [n.strip() for n in pf.read_text(encoding='utf-8').splitlines() if n.strip()]

    ai_name = get_llm_player_name(game_dir)

    # Derive current round from remaining_players.
    remaining = []
    rf = game_dir / REMAINING_PLAYERS_FILE
    if rf.exists():
        remaining = [n.strip() for n in rf.read_text(encoding='utf-8').splitlines() if n.strip()]
    current_round = (len(all_p) - len(remaining) + 1) if all_p and remaining else 1

    per_player = {}
    max_rounds = 0
    for name in all_p:
        vf = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(name)
        lines = []
        if vf.exists():
            for ln in vf.read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        per_player[name] = lines
        max_rounds = max(max_rounds, len(lines))

    # latest_round: current voting status (one entry per player or "" if not yet voted).
    # The last line in the file belongs to whichever round it was cast in, so
    # showing it unconditionally made round 2 open with everyone already
    # "voted". Only a vote tagged with the current round token counts.
    latest_round = {
        name: (per_player[name][-1]
               if per_player[name] and has_voted_this_round(game_dir, name)
               else "")
        for name in all_p
    }

    return jsonify({
        "current_round": current_round,
        "total_rounds": max_rounds,
        "latest_round": latest_round,
        "history_by_player": per_player,
        "ai_player": ai_name,
    })


@app.route('/api/admin/force_tally', methods=['POST'])
def admin_force_tally():
    """Force-abstain any remaining players who have not cast a vote in the
    current round. Appends "<ABSTAIN>" to their {name}_vote.txt file so the
    stt_main voting loop unblocks. Also logs the forced abstain to
    invalid_votes.log for audit.

    Available at any point during the voting phase — the researcher can see the
    live vote table and does not have to wait out the 120s stuck heuristic
    before rescuing a round. Idempotent: players who already voted this round
    are never touched.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    payload = request.get_json(silent=True) or {}
    game_id = payload.get("game_id")
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    # Determine current phase + remaining players
    phase_file = game_dir / PHASE_STATUS_FILE
    phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.exists() else ""
    can_vote = ("VOTING" in phase.upper() or "Voting" in phase)
    if not can_vote:
        return jsonify({"error": "Not in voting phase", "phase": phase}), 400

    remaining = []
    rf = game_dir / REMAINING_PLAYERS_FILE
    if rf.exists():
        remaining = [n.strip() for n in rf.read_text(encoding="utf-8").splitlines() if n.strip()]

    # Use discussion_start mtime as the round anchor — any vote file modified
    # AFTER this time is considered a vote for the current round.
    dst_file = game_dir / DISCUSSION_START_TIME_FILE
    try:
        anchor = dst_file.stat().st_mtime if dst_file.exists() else 0.0
    except Exception:
        anchor = 0.0

    # Abstaining cannot help when the manager is dead and every vote is already
    # in — nobody is left to read them. Route to the tally endpoint instead.
    stuck = _detect_stuck_game(game_dir, can_vote, remaining, dst_file, game_over=False, game_id=game_id)
    if stuck.get("reason") == "manager_dead_all_votes_cast":
        return jsonify({"error": "All votes are cast but manager is dead — use force_complete_vote instead", "stuck": stuck}), 400

    abstained = []
    already_voted = []
    pending = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for name in remaining:
        if has_voted_this_round(game_dir, name):
            already_voted.append(name)
        else:
            pending.append(name)

    # Abstaining everyone when nobody has voted leaves a tally of all zeros, and
    # the manager then eliminates a player at random. That is the researcher
    # deciding the round by accident, so it has to be asked for explicitly.
    if not already_voted:
        if not (request.json or {}).get("confirm_no_votes"):
            return jsonify({
                "error": ("Nobody has voted in this round yet. Abstaining "
                          "everyone would leave every candidate tied at zero "
                          "and the game would eliminate someone at random. "
                          "Wait for at least one vote, or repeat the request "
                          "with confirm_no_votes to accept a random "
                          "elimination."),
                "error_code": "no_votes_cast",
                "total_remaining": len(remaining),
            }), 409

    for name in pending:
        vfile = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(name)
        # The same lock /vote uses, re-checking inside it: a player can be
        # tapping their vote at the moment the researcher presses this button,
        # and appending <ABSTAIN> after their real vote would throw it away.
        try:
            with _get_vote_lock(game_id, name):
                with vote_round_lock(game_dir, name):
                    if has_voted_this_round(game_dir, name):
                        already_voted.append(name)
                        continue
                    token = current_round_token(game_dir)
                    vfile.parent.mkdir(parents=True, exist_ok=True)
                    with vfile.open("a", encoding="utf-8") as f:
                        f.write("<ABSTAIN>\n")
                    record_vote_round(game_dir, name, token)
            abstained.append(name)
            # Audit log
            try:
                with (game_dir / "invalid_votes.log").open("a", encoding="utf-8") as lf:
                    lf.write(f"{ts} | {name} -> <ABSTAIN> (forced by admin)\n")
            except Exception:
                pass
        except Exception as e:
            print(f"[ForceTally] Failed to abstain {name}: {e}")

    return jsonify({
        "ok": True,
        "abstained": abstained,
        "already_voted": already_voted,
        "total_remaining": len(remaining),
    })


@app.route('/api/admin/drop_player', methods=['POST'])
def admin_drop_player():
    """Remove a human from a RUNNING game so play continues one player short.

    Use when a participant has clearly abandoned the session (inactive, never
    writes, blocks every vote). The manager process owns the roster, so this
    endpoint only records the codename in dropped_players.txt; stt_main applies
    it at the next safe point and announces a neutral "has left" message that
    reveals nothing about the player's role.

    The AI can never be dropped this way — removing it would silently decide
    the experiment. Before START use /api/admin/lobby_kick instead.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    payload = request.get_json(silent=True) or {}
    game_id = payload.get("game_id")
    codename = (payload.get("codename") or "").strip()
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    if not codename:
        return jsonify({"error": "codename required"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    gst_file = game_dir / GAME_START_TIME_FILE
    if not (gst_file.exists() and gst_file.read_text(encoding="utf-8").strip()):
        return jsonify({
            "error": "Game has not started — use lobby_kick to remove a player from the lobby"
        }), 400
    if (game_dir / WHO_WINS_FILE).exists():
        return jsonify({"error": "Game is already over"}), 400

    try:
        config = json.loads(
            (game_dir / GAME_CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"Cannot read config: {exc}"}), 500
    for player in config.get("players", []):
        if player.get("name") == codename and player.get("is_ai"):
            return jsonify({"error": "The AI player cannot be dropped"}), 400

    remaining_file = game_dir / REMAINING_PLAYERS_FILE
    remaining = []
    if remaining_file.exists():
        remaining = [
            n.strip() for n in
            remaining_file.read_text(encoding="utf-8").splitlines() if n.strip()
        ]
    if codename not in remaining:
        return jsonify({
            "error": f"'{codename}' is not an active player",
            "remaining": remaining,
        }), 404
    if len(remaining) <= 3:
        return jsonify({
            "error": "Only three players remain — dropping one would trigger the"
                     " AI-wins end condition, so the admin would be deciding"
                     " the game. Let the round play out instead.",
        }), 400

    dropped_file = game_dir / DROPPED_PLAYERS_FILE
    already = set()
    if dropped_file.exists():
        already = {
            n.strip() for n in
            dropped_file.read_text(encoding="utf-8").splitlines() if n.strip()
        }
    if codename not in already:
        with dropped_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{codename}\n")

    # Unblock a voting loop that is already waiting on this player. The manager
    # also drops them on its own, but an abstain here makes the rescue instant.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vote_file = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(codename)
    phase_file = game_dir / PHASE_STATUS_FILE
    phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.exists() else ""
    abstained_now = False
    if "VOTING" in phase.upper() or "Voting" in phase:
        try:
            # Same lock and re-check as /vote and force_tally: if they got a
            # vote in at the last second, keep it rather than appending an
            # abstain the manager would read instead.
            with _get_vote_lock(game_id, codename):
                with vote_round_lock(game_dir, codename):
                    if not has_voted_this_round(game_dir, codename):
                        token = current_round_token(game_dir)
                        vote_file.parent.mkdir(parents=True, exist_ok=True)
                        with vote_file.open("a", encoding="utf-8") as handle:
                            handle.write("<ABSTAIN>\n")
                        record_vote_round(game_dir, codename, token)
                        abstained_now = True
        except OSError as exc:
            print(f"[DropPlayer] Failed to abstain {codename}: {exc}")
    try:
        with (game_dir / "invalid_votes.log").open("a", encoding="utf-8") as log:
            log.write(f"{ts} | {codename} -> <DROPPED> (removed mid-game by admin)\n")
    except OSError:
        pass

    return jsonify({
        "ok": True,
        "dropped": codename,
        "abstained_now": abstained_now,
        "phase": phase,
        "remaining_before": remaining,
    })


@app.route('/api/admin/force_complete_vote', methods=['POST'])
def admin_force_complete_vote():
    """Manually tally existing votes and apply the elimination result when the
    manager process (stt_main.py) died after all players voted but before it
    could process the tally.

    Conditions:
    - Phase must be Voting
    - All remaining players must have a current-round vote file
    - Manager process must be dead (not alive in running_games)

    Writes: VOTED_OUT status, remaining_players.txt, public_manager_chat.txt,
    who_wins.txt (if game over), events/elimination_audit.jsonl.
    """
    from core.game_constants import (
        VOTED_OUT, AI_WINS_MESSAGE, HUMANS_WIN_MESSAGE,
        VOTED_OUT_MESSAGE_FORMAT, GAME_MANAGER_NAME, get_current_timestamp,
    )
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    payload = request.get_json(silent=True) or {}
    game_id = payload.get("game_id")
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    phase_file = game_dir / PHASE_STATUS_FILE
    phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.exists() else ""
    if not ("VOTING" in phase.upper() or "Voting" in phase):
        return jsonify({"error": "Not in voting phase", "phase": phase}), 400

    procs = running_games.get(game_id, {})
    manager_proc = procs.get("manager")
    manager_alive = manager_proc is not None and manager_proc.poll() is None
    if manager_alive:
        return jsonify({"error": "Manager is still running — use force_tally to abstain missing voters instead"}), 400

    remaining = []
    rf = game_dir / REMAINING_PLAYERS_FILE
    if rf.exists():
        remaining = [n.strip() for n in rf.read_text(encoding="utf-8").splitlines() if n.strip()]
    if not remaining:
        return jsonify({"error": "No remaining players found"}), 400

    dst_file = game_dir / DISCUSSION_START_TIME_FILE

    # Read current-round votes for each remaining player
    votes_tally = {name: 0 for name in remaining}
    vote_details = []
    missing_votes = []
    for name in remaining:
        vfile = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(name)
        # Round token, not mtime: otherwise a vote left over from the previous
        # round would be tallied as this round's, and the endpoint would decide
        # an elimination from votes nobody cast in this round.
        if not has_voted_this_round(game_dir, name) or not vfile.exists():
            missing_votes.append(name)
            continue
        lines = [l.strip() for l in vfile.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            missing_votes.append(name)
            continue
        voted_for = lines[-1]  # newest line = current round's vote
        if voted_for == "<ABSTAIN>":
            continue
        if voted_for in votes_tally:
            votes_tally[voted_for] += 1
            vote_details.append((name, voted_for))

    if missing_votes:
        return jsonify({
            "error": "Not all votes are in yet — use force_tally to abstain missing voters first",
            "missing_votes": missing_votes,
        }), 400

    # Tally with random tie-break
    max_v = max(votes_tally.values(), default=0)
    tied = [n for n, v in votes_tally.items() if v == max_v]
    voted_out_name = random.choice(tied)

    # Eliminate the player
    status_file = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(voted_out_name)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(VOTED_OUT, encoding="utf-8")

    # Update remaining_players.txt atomically, including retrying the brief
    # destination sharing locks that concurrent Windows readers can create.
    new_remaining = [n for n in remaining if n != voted_out_name]
    atomic_write_text(rf, "\n".join(new_remaining))

    # Determine AI status of eliminated player
    ai_names = []
    ai_file = game_dir / AI_PLAYER_FILE
    if ai_file.exists():
        ai_names = [n.strip() for n in ai_file.read_text(encoding="utf-8").splitlines() if n.strip()]
    is_ai = voted_out_name in ai_names

    # Announce elimination in manager chat
    role = get_role_display_string(is_ai)
    elim_msg = VOTED_OUT_MESSAGE_FORMAT.format(voted_out_name, role)
    manager_chat = game_dir / PUBLIC_MANAGER_CHAT_FILE
    manager_chat.parent.mkdir(parents=True, exist_ok=True)
    with open(manager_chat, "a", encoding="utf-8") as f:
        f.write(format_message(GAME_MANAGER_NAME, elim_msg.replace("\n", "\\n")))

    # Check win conditions
    winner = None
    if is_ai:
        winner = HUMANS_WIN_MESSAGE
    elif len(new_remaining) <= 2:
        winner = AI_WINS_MESSAGE

    if winner:
        (game_dir / WHO_WINS_FILE).write_text(winner, encoding="utf-8")
        with open(manager_chat, "a", encoding="utf-8") as f:
            f.write(format_message(GAME_MANAGER_NAME, winner))
        phase_file.write_text("Game_Over", encoding="utf-8")

    # Write audit trail
    try:
        audit = {
            "timestamp": get_current_timestamp(),
            "tally": votes_tally,
            "max_votes": max_v,
            "tied_candidates": tied,
            "voted_out": voted_out_name,
            "tie_break": "random" if len(tied) > 1 else "single_max",
            "source": "force_complete_vote",
        }
        audit_path = game_dir / "events" / "elimination_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit) + "\n")
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "voted_out": voted_out_name,
        "is_ai": is_ai,
        "tally": votes_tally,
        "tied_candidates": tied,
        "winner": winner,
        "new_remaining": new_remaining,
    })


@app.route('/api/admin/prompt_flow')
def admin_prompt_flow():
    """Return structured OPERATION blocks parsed from the AI player's llm_log_{name}.txt.

    Log format (from llm_players/logger.py):
        # NEW LOG
        ## TIME: HH:MM:SS
        ## OPERATION: <name>
        ## CONTENT: <multi-line content>
        <blank line>
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "Game not found"}), 404

    cfg = {}
    cfg_file = game_dir / GAME_CONFIG_FILE
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    ai_names = [p.get("name") for p in cfg.get("players", []) if p.get("is_llm") or p.get("is_ai")]
    if not ai_names:
        return jsonify({"operations": [], "error": "No AI player in this game"})

    ops = []
    for ai in ai_names:
        log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai)
        if not log_file.exists():
            continue
        try:
            text = log_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        # Split on "# NEW LOG" boundaries
        for block in text.split('# NEW LOG'):
            block = block.strip()
            if not block:
                continue
            time_match = re.search(r'##\s*TIME:\s*(.+)', block)
            op_match = re.search(r'##\s*OPERATION:\s*(.+)', block)
            content_match = re.search(r'##\s*CONTENT:\s*(.*)', block, re.DOTALL)
            if not op_match:
                continue
            ops.append({
                "player": ai,
                "time": time_match.group(1).strip() if time_match else "",
                "operation": op_match.group(1).strip(),
                "category": _classify_prompt_operation(op_match.group(1).strip()),
                "content": (content_match.group(1).strip() if content_match else "")[:8000],
            })
    return jsonify({"operations": ops, "count": len(ops)})


def _classify_prompt_operation(op_name: str) -> str:
    """Map raw operation names from logger.py to one of:
       prompt_out  = prompt sent TO agent
       response_in = text received FROM agent
       scheduler   = scheduler (send/wait) decision
       review      = reviewer-related (input, output, retry)
       other       = anything else
    The caller puts a small badge in the UI based on this category.
    """
    if not op_name:
        return "other"
    n = op_name.lower()
    if "scheduler" in n:
        return "scheduler"
    if "review" in n:
        return "review"
    # Prompt sent out has words like "prompt", "input", "send_to"
    if any(k in n for k in ["prompt", "input_to", "to_model", "send", "user_prompt", "system_prompt"]):
        return "prompt_out"
    # Response from model has words like "response", "output", "generated", "raw"
    if any(k in n for k in ["response", "output", "generated", "raw_response", "message_from", "reply"]):
        return "response_in"
    return "other"


@app.route('/api/admin/llm_log')
def admin_llm_log():
    """Return tail of the LLM log file for live monitoring."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    lines = max(1, min(int(request.args.get('lines', 50) or 50), 500))
    if not game_id:
        return jsonify({"error": "game_id required"}), 400

    game_dir = resolve_game_dir(game_id)
    ai_name = get_llm_player_name(game_dir)
    if not ai_name:
        return jsonify({"lines": [], "ai_name": None})

    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
    if not log_file.exists():
        return jsonify({"lines": [], "ai_name": ai_name})

    try:
        all_lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return jsonify({"lines": tail, "ai_name": ai_name, "total_lines": len(all_lines)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/manager_log')
def admin_manager_log():
    """Return tail of manager_log.txt for live debugging of the game manager process."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    lines = max(1, min(int(request.args.get('lines', 100) or 100), 500))
    if not game_id:
        return jsonify({"error": "game_id required"}), 400

    game_dir = resolve_game_dir(game_id)
    log_path = game_file(game_dir, "manager_log.txt")
    if not log_path.exists():
        return jsonify({"log": "", "total_lines": 0})

    try:
        all_lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return jsonify({"log": "\n".join(tail), "total_lines": len(all_lines)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/process_log')
def admin_process_log():
    """Return tail of llm_interface_log.txt (raw subprocess stdout/stderr of the LLM player).

    This is the unfiltered process output — includes import errors, library banners,
    deprecation warnings, and crash tracebacks that never make it into the structured log.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    lines = max(1, min(int(request.args.get('lines', 200) or 200), 500))

    game_dir = resolve_game_dir(game_id)
    log_path = game_file(game_dir, "llm_interface_log.txt")
    if not log_path.exists():
        return jsonify({"log": "", "total_lines": 0})

    try:
        all_lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return jsonify({"log": "\n".join(tail), "total_lines": len(all_lines)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/game_config')
def admin_game_config():
    """Return the raw config.json content for a given game."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    config_path = game_dir / GAME_CONFIG_FILE
    if not config_path.exists():
        return jsonify({"error": "config.json not found"}), 404

    try:
        with open(config_path, encoding='utf-8') as f:
            cfg = json.load(f)
        return jsonify({"config": cfg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── AI Metrics: parse log file into structured performance data ──

def _parse_log_entries(log_text):
    """Parse the raw log file into structured entries: [{time, operation, content}]."""
    import re as _re
    entries = []
    blocks = log_text.split("# NEW LOG\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = _re.match(
            r"## TIME:\s*(.+?)\n## OPERATION:\s*(.+?)\n## CONTENT:\s*(.*)",
            block, _re.DOTALL)
        if m:
            entries.append({
                "time": m.group(1).strip(),
                "operation": m.group(2).strip(),
                "content": m.group(3).strip(),
            })
    return entries


def _time_to_seconds(t):
    """Convert HH:MM:SS string to seconds since midnight."""
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def _read_discussion_start_hms(game_dir):
    """Read the discussion-start clock value, or None if absent or unreadable."""
    try:
        dst_file = game_dir / DISCUSSION_START_TIME_FILE
        if dst_file.exists():
            text = dst_file.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return None

def _read_ai_delivered_messages(game_dir: Path, ai_name: str) -> list[str]:
    """Read message bodies from the AI's append file in delivery order."""
    delivered = []
    path = Path(game_dir) / PERSONAL_CHAT_FILE_FORMAT.format(ai_name)
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\[\d\d:\d\d:\d\d\] [^:]+: (.*)$", line)
            if match:
                delivered.append(match.group(1))
    except OSError:
        pass
    return delivered


def _compute_ai_metrics(
    log_text, discussion_start_hms=None, delivered_messages=None
):
    """
    Extract structured AI performance metrics from the log file.

    discussion_start_hms — "HH:MM:SS" from discussion_start_time.txt. When
    given and usable, messages_per_minute uses the later of discussion start
    and the first log entry. Otherwise the first log entry is the anchor.
    Clock values are seconds since midnight, without date reconstruction.

    Returns dict with:
      messages  – per-message detail list
      aggregate – summary statistics
    """
    entries = _parse_log_entries(log_text)
    if not entries:
        return {"messages": [], "aggregate": {}}

    # Pre-scan AGENT_DROPPED_GENERATED_MESSAGE entries so we can mark messages
    # that were generated but never reached public_daytime_chat (typically
    # because voting started during generation or during the typing-delay
    # wait). Without this the per-message detail / dashboard misrepresents
    # them as sent. The log content looks like:
    #   "Dropped after typing-delay (voting started): 'bad topic man'"
    import re as _re_drop
    dropped_lookup = {}  # text → reason
    for _e in entries:
        if _e["operation"] == "AGENT_DROPPED_GENERATED_MESSAGE":
            m = _re_drop.match(r"(Dropped[^:]*):\s*'(.*)'\s*$", _e["content"].strip())
            if m:
                reason, text = m.group(1), m.group(2)
                dropped_lookup[text] = reason
        # Output gates: a selected message can still be suppressed AFTER the
        # pipeline completes (timeout / staleness). Mark these rows unsent;
        # completion and selection alone do not establish delivery.
        elif _e["operation"] in ("pipeline_timeout_drop", "stale_send_drop"):
            m = (_re_drop.search(r"suppressed:\s*'(.*)'\s*$", _e["content"].strip())
                 or _re_drop.search(r"'(.+?)'", _e["content"]))
            if m:
                dropped_lookup[m.group(1)] = _e["operation"].replace("_", " ")

    messages = []           # completed send events
    scheduler_calls = 0     # total scheduler invocations
    wait_decisions = 0
    send_decisions = 0
    # Every scheduler invocation (send + wait) with its API latency. Used to
    # produce a true "avg scheduler" across all decisions, plus a list of
    # wait-call durations to surface periods when the agent was blocked.
    all_scheduler_calls = []   # list of {time, latency_s, decision}
    long_wait_threshold_s = 10.0  # calls above this are flagged as stalls

    # Running state machine
    scheduler_start_time = None   # when current scheduler prompt was issued
    generator_start_time = None
    current_send_decision_time = None
    current_raw_message = None
    # Post-review text from LIVE_REVIEW_SUMMARY's FINAL line. Prefer it over
    # raw generator output, then reconcile delivery against the append file
    # when available; a final text record alone does not establish delivery.
    current_final_message = None
    # True when the reviewer dropped the turn (best score ≥ drop threshold).
    # Drop events generate a review_cycle entry but no chat message, so we
    # skip them entirely from the per-message table.
    current_dropped = False
    review_attempts = []
    review_final_score = None
    review_was_modified = False
    current_scheduler_latency = None
    current_generator_latency = None
    current_reviewer_latency = None
    # Real intentional typing-delay sleep, parsed from the `typing_delay`
    # log entry. None until a typing_delay log line lands; then 0.0 for
    # "no_sleep" / "skipped_short" or the actual sleep duration in seconds.
    current_typing_wait = None
    current_typing_elapsed = None    # natural pipeline time (from the typing_delay line)
    current_typing_ideal = None      # ideal budget (word_count*1.0*(1+jitter))
    current_budget_exceeded = False  # Did reviewer hit its wall-clock budget?
    current_review_degraded = False  # Did the reviewer FAIL (404/empty/parse)?
    # Candidate pools and explicit selections supply text and selection
    # provenance. The first candidate supplies fallback text without a
    # selection record; that fallback is not evidence of reviewer activity.
    current_candidates = None        # list[str] | None
    current_selected_text = None     # str | None (ranker's chosen candidate)
    current_selected_n = None        # pool size from the selection log line

    game_first_time = None
    game_last_time = None

    for i, e in enumerate(entries):
        t = e["time"]
        op = e["operation"]
        content = e["content"]
        ts = _time_to_seconds(t)

        if game_first_time is None:
            game_first_time = ts
        game_last_time = ts

        # ── Scheduler start ──
        # Logs may use either prompt label; treat both as scheduler-start markers.
        if op in ("prompt in should_generate_message", "scheduler_prompt"):
            scheduler_start_time = ts
            # Clear stale per-round state. LIVE_REVIEW_SUMMARY is logged
            # AFTER the round's UNLOCKED, so on the *next* round the
            # parser would otherwise read these values and tag the new
            # message with the previous round's final text/score. Reset
            # them here so each round starts clean.
            current_final_message = None
            review_attempts = []
            review_final_score = None
            review_was_modified = False
            current_dropped = False
            current_typing_wait = None
            current_typing_elapsed = None
            current_typing_ideal = None
            current_budget_exceeded = False
            current_candidates = None
            current_selected_text = None
            current_selected_n = None

        # ── Scheduler decision ──
        if op in ("decision in should_generate_message", "scheduler_decision"):
            scheduler_calls += 1
            if op == "scheduler_decision":
                # Match the decision token itself, not the SEND substring
                # inside a send_reason field on a WAIT line.
                is_send = re.search(r"decision\s*=\s*send", content, re.IGNORECASE)
                decision_tag = "<send>" if is_send else "<wait>"
            else:
                decision_tag = "<send>" if "<send>" in content else "<wait>"
            # Pair with the most recently logged scheduler_latency for this
            # call. The latency log entry comes immediately before the
            # decision, so current_scheduler_latency is set when we get here.
            if current_scheduler_latency is not None:
                all_scheduler_calls.append({
                    "time": t,
                    "latency_s": round(current_scheduler_latency, 2),
                    "decision": decision_tag,
                })
            if decision_tag == "<send>":
                send_decisions += 1
                current_send_decision_time = ts
            else:
                wait_decisions += 1
                scheduler_start_time = None
                # Reset so the next scheduler call records its own latency.
                current_scheduler_latency = None

        # ── Generator start ──
        if op in ("prompt in generate_message", "prompt in _generate_raw_message"):
            generator_start_time = ts

        # ── Generator output ──
        if op.startswith("final_output in generate") and generator_start_time is not None:
            current_raw_message = content

        # ── Candidate pool + ranker selection ──
        if op == "generator_candidates":
            try:
                import ast as _ast
                # The list repr is single-line; cost-telemetry lines appended
                # to the same log entry (everything until the next "# NEW LOG")
                # would break literal_eval on the full blob.
                _first = content.strip().splitlines()[0] if content.strip() else ""
                _pool = _ast.literal_eval(_first)
                current_candidates = ([str(c) for c in _pool]
                                      if isinstance(_pool, (list, tuple)) else None)
            except Exception:
                current_candidates = None

        if op == "reviewer_selected":
            _sel_m = re.match(r"^['\"](.*)['\"] from (\d+) candidates", content, re.DOTALL)
            if _sel_m:
                current_selected_text = _sel_m.group(1)
                current_selected_n = int(_sel_m.group(2))

        # ── Review attempt ──
        if op.startswith("review_attempt_"):
            import re as _re2
            score_m = _re2.search(r"Score:\s*(\d+)", content)
            if score_m:
                review_attempts.append(int(score_m.group(1)))

        # ── Precise latency from instrumented code ──
        if op == "scheduler_latency":
            import re as _re_sl
            m = _re_sl.match(r"([\d.]+)s", content)
            if m:
                current_scheduler_latency = float(m.group(1))

        if op == "generator_latency":
            import re as _re_gl
            m = _re_gl.match(r"([\d.]+)s", content)
            if m:
                current_generator_latency = float(m.group(1))

        if op == "reviewer_latency":
            import re as _re_rl
            m = _re_rl.match(r"([\d.]+)s", content)
            if m:
                current_reviewer_latency = float(m.group(1))

        # ── Intentional typing-delay sleep ──
        # Emitted by ReviewedPlayer._apply_typing_delay. Three forms:
        #   "sleep X.XXs (ideal Y.YYs, elapsed Z.ZZs)"  → X.XX (real sleep)
        #   "no_sleep: pipeline already took ..."        → 0.0
        #   "skipped_short (N words)"                    → 0.0
        if op == "typing_delay":
            import re as _re_td
            # The typing_delay line carries the FULL pipeline time, so we can compute
            # Total = natural pipeline latency (elapsed) + artificial wait (sleep)
            # without requiring a separate reviewer_latency entry. Forms:
            #   "sleep X.XXs (ideal Y.YYs, elapsed Z.ZZs)"  → wait=X, natural=Z, total=Z+X
            #   "no_sleep: pipeline already took Z.ZZs ≥ ideal Y.YYs" → wait=0, natural=Z
            #   "skipped_short (N words)"                    → wait=0, natural unknown
            m = _re_td.match(r"sleep\s+([\d.]+)s", content)
            if m:
                current_typing_wait = float(m.group(1))
                me = _re_td.search(r"elapsed\s+([\d.]+)s", content)
                mi = _re_td.search(r"ideal\s+([\d.]+)s", content)
                if me:
                    current_typing_elapsed = float(me.group(1))
                if mi:
                    current_typing_ideal = float(mi.group(1))
            elif content.startswith("no_sleep"):
                current_typing_wait = 0.0
                me = _re_td.search(r"took\s+([\d.]+)s", content)
                mi = _re_td.search(r"ideal\s+([\d.]+)s", content)
                if me:
                    current_typing_elapsed = float(me.group(1))
                if mi:
                    current_typing_ideal = float(mi.group(1))
            elif content.startswith("skipped_short"):
                current_typing_wait = 0.0
            # A typing_delay entry may follow its message row. Update only a
            # row lacking an instrumented typing value. Rows populated before
            # append carry _typing_from_log=True and must not receive another
            # turn's delay.
            if (messages and current_typing_wait is not None
                    and not messages[-1].get("_typing_from_log")):
                last_msg = messages[-1]
                last_msg["writing_wait_s"] = round(current_typing_wait, 1)
                if current_typing_ideal is not None:
                    last_msg["ideal_latency_s"] = round(current_typing_ideal, 2)
                    last_msg["target_latency_s"] = round(current_typing_ideal, 2)
                if current_typing_elapsed is not None:
                    # Authoritative: full wall-clock before send = natural + artificial.
                    last_msg["total_latency_s"] = round(current_typing_elapsed + current_typing_wait, 2)
                else:
                    # No elapsed (skipped_short): sum the known components instead.
                    comp = [c for c in (last_msg.get("scheduler_latency_s"),
                                        last_msg.get("generator_latency_s"),
                                        last_msg.get("reviewer_latency_s")) if c is not None]
                    last_msg["total_latency_s"] = round(sum(comp) + current_typing_wait, 2)
                # Recompute "Other" so Sched+Gen+Rev+Other+Wait still equals Total
                # after the total was patched above.
                _kl = [last_msg.get("scheduler_latency_s"), last_msg.get("generator_latency_s"),
                       last_msg.get("reviewer_latency_s")]
                if last_msg.get("total_latency_s") is not None and all(c is not None for c in _kl):
                    last_msg["other_latency_s"] = max(0.0, round(
                        last_msg["total_latency_s"] - last_msg["writing_wait_s"] - sum(_kl), 2))

        # Reviewer exceeded its wall-clock budget and had to short-circuit.
        # We track this to detect Vertex AI latency regressions.
        if op in ("reviewer_budget_exceeded", "reviewer_budget_truncated"):
            current_budget_exceeded = True

        # ── Live review summary (from llm_interface.py) ──
        if op == "LIVE_REVIEW_SUMMARY":
            import re as _re3
            score_m = _re3.search(r"Final score:\s*(\d+)/10", content)
            if score_m:
                review_final_score = int(score_m.group(1))
            review_was_modified = "(modified)" in content
            # Reviewer FAILED (API 404 / empty / parse) → the "5/10" is a
            # fail-open placeholder, NOT a real verdict. Flag it so the table
            # shows "degraded" instead of a misleading score.
            if ("API error" in content or "REVIEWER UNAVAILABLE" in content
                    or "REVIEWER PARSE FAILED" in content):
                current_review_degraded = True
            # Capture summary text in the form FINAL (<status>): 'TEXT'.
            # Delivery evidence is reconciled separately when supplied.
            final_m = _re3.search(r"FINAL \([^)]+\):\s*'(.*)'\s*$", content, _re3.MULTILINE)
            if final_m:
                current_final_message = final_m.group(1)
            # IMPORTANT: LIVE_REVIEW_SUMMARY is logged AFTER the matching
            # `generate_message UNLOCKED`, so by the time we get here the
            # message row has already been appended to `messages` using the
            # raw generator output (because current_final_message was None
            # for THIS round). Patch the last row in-place so it reflects
            # the post-review final text + score + modified flag.
            if messages and final_m:
                last_msg = messages[-1]
                final_text = final_m.group(1) or ""
                # Reviewer dropped the turn (empty FINAL) — the row that
                # UNLOCKED already appended is a phantom; remove it so the
                # table matches the public chat.
                if final_text == "":
                    messages.pop()
                else:
                    last_msg["text"] = final_text[:120]
                    last_msg["word_count"] = len(final_text.split())
                    if review_final_score is not None:
                        last_msg["reviewer_score"] = review_final_score
                    # Selection-mode rows keep reviewer_modified=False — the
                    # summary line says "(modified)" whenever the ranker
                    # picked any candidate other than the first draft, which
                    # is selection, not modification.
                    if not last_msg.get("selection_mode"):
                        last_msg["reviewer_modified"] = review_was_modified
                    last_msg["reviewer_degraded"] = current_review_degraded

        # ── Reviewer dropped the turn ──
        # An explicit drop-threshold record marks the current turn dropped.
        # An empty FINAL value also suppresses its message row below.
        if "drop threshold" in content and "skipping turn entirely" in content:
            current_dropped = True

        # ── Message generation complete → record full event ──
        if op == "generate_message" and "UNLOCKED" in content:
            scheduler_latency = None
            generator_latency = None
            total_latency = None

            if scheduler_start_time is not None and current_send_decision_time is not None:
                scheduler_latency = max(0, current_send_decision_time - scheduler_start_time)
            if generator_start_time is not None:
                generator_latency = max(0, ts - generator_start_time)
            if scheduler_start_time is not None:
                total_latency = max(0, ts - scheduler_start_time)

            # Prefer instrumented latency (sub-second) over timestamp-delta
            if current_scheduler_latency is not None:
                scheduler_latency = current_scheduler_latency
            if current_generator_latency is not None:
                generator_latency = current_generator_latency

            # Resolve text from summary FINAL, selection, raw generator output,
            # then the first candidate. These are text sources, not proof of
            # delivery; reconcile with the AI append file when supplied.
            msg_text = current_final_message
            if msg_text is None:
                msg_text = current_selected_text
            if msg_text is None:
                msg_text = current_raw_message
            if msg_text is None and current_candidates:
                msg_text = current_candidates[0]
            msg_text = msg_text or ""
            word_count = len(msg_text.split()) if msg_text else 0

            # Selection provenance: zero-based position of the selected text
            # in the candidate pool. A nonzero index indicates selection of
            # a candidate other than the first.
            selection_mode = (current_selected_text is not None
                              or current_candidates is not None)
            selected_idx = None
            if msg_text and current_candidates:
                try:
                    selected_idx = current_candidates.index(msg_text)
                except ValueError:
                    selected_idx = None
            num_candidates = (current_selected_n
                              if current_selected_n is not None
                              else (len(current_candidates) if current_candidates else None))

            # "Wait" = INTENTIONAL typing-delay sleep only. Prefer the value
            # parsed from the `typing_delay` log entry (precise, includes the
            # "no_sleep" / "skipped_short" branches as 0). Fall back to the
            # observed wallclock residual only when no instrumented value exists.
            if current_typing_wait is not None:
                writing_wait = current_typing_wait
            else:
                # Without an explicit wait, estimate the nonnegative residual.
                comp = (scheduler_latency or 0) + (generator_latency or 0) + (current_reviewer_latency or 0)
                if total_latency is not None:
                    writing_wait = max(0.0, total_latency - comp)
                else:
                    # No instrumented delay or measured total: use 0 as the
                    # display default, not as evidence of an observed sleep.
                    # Do not invent a word-count-based measurement.
                    writing_wait = 0.0

            # TOTAL = the full wall-clock before the message is sent = natural
            # pipeline latency + the artificial typing WAIT. Preferred source: the
            # typing_delay line's `elapsed` (natural) + `sleep` (artificial), which
            # covers the full in-turn pipeline. Otherwise sum available complete
            # phase measurements, then fall back to the clock-value delta.
            if current_typing_elapsed is not None:
                total_latency = round(current_typing_elapsed + writing_wait, 2)
            else:
                components = [scheduler_latency, generator_latency, current_reviewer_latency, writing_wait]
                if all(c is not None for c in components):
                    total_latency = round(sum(components), 2)
                elif total_latency is not None:
                    total_latency = round(total_latency, 2)

            # "Other" = in-pipeline overhead NOT attributed to the 3 LLM calls
            # (snapshot read, context builders, candidate parsing, humanizer).
            # Makes the row honest: Sched + Gen + Rev + Other + Wait = Total.
            _known_lats = [scheduler_latency, generator_latency, current_reviewer_latency]
            if total_latency is not None and all(c is not None for c in _known_lats):
                other_latency = max(0.0, round(total_latency - writing_wait - sum(_known_lats), 2))
            else:
                other_latency = None
            # Target human-typing budget (≈ word_count + 2s). The full chain should
            # be AT LEAST this, but ideally not much more ("too much" = slow pipeline).
            target_latency = (round(current_typing_ideal, 2)
                              if current_typing_ideal is not None
                              else round(word_count + 2.0, 2))

            msg_entry = {
                "msg_num": len(messages) + 1,
                "time": t,
                "text": msg_text[:120],
                "word_count": word_count,
                "scheduler_latency_s": round(scheduler_latency, 2) if scheduler_latency is not None else None,
                "generator_latency_s": round(generator_latency, 2) if generator_latency is not None else None,
                "writing_wait_s": round(writing_wait, 1),
                # True when this row already carries the precise typing value at
                # append time (typing_delay logged before UNLOCKED) — so the
                # post-append patch must skip it. Stripped before returning.
                "_typing_from_log": current_typing_wait is not None,
                "ideal_latency_s": round(current_typing_ideal, 2) if current_typing_ideal is not None else None,
                "total_latency_s": total_latency,
                # Overhead beyond the 3 LLM calls + wait, and the human-typing
                # target (≈ words+2). Together they make the row math transparent
                # and let the UI flag "too much" (total ≫ target).
                "other_latency_s": other_latency,
                "target_latency_s": target_latency,
                "reviewer_latency_s": round(current_reviewer_latency, 2) if current_reviewer_latency is not None else None,
                # Read the score from LIVE_REVIEW_SUMMARY independently of
                # whether individual review_attempt entries are present.
                "reviewer_score": review_final_score,
                "reviewer_attempts": len(review_attempts) if review_attempts else 0,
                # A verbatim candidate selection is not a text modification.
                # Only apply the summary's modification flag without selection.
                "reviewer_modified": review_was_modified and not selection_mode,
                "selection_mode": selection_mode,
                "selected_candidate_index": selected_idx,
                "num_candidates": num_candidates,
                "reviewer_budget_exceeded": current_budget_exceeded,
                "reviewer_degraded": current_review_degraded,
                # sent=False indicates the message was generated (and possibly
                # reviewed) but never written to the public chat — voting
                # started before send. discard_reason mirrors the log message.
                "sent": msg_text not in dropped_lookup,
                "discard_reason": dropped_lookup.get(msg_text),
            }
            # Reviewer-dropped turns produce a review_cycle but never reach
            # the public chat — omit from the table to keep it consistent
            # with the chat history. Detection priority:
            #   (a) explicit drop log seen in this cycle, OR
            #   (b) LIVE_REVIEW_SUMMARY's FINAL value is empty (the canonical
            #       "no message sent" signal — covers all drop reasons).
            if current_dropped or (current_final_message is not None and current_final_message == ""):
                pass  # skip — not a real chat message
            else:
                messages.append(msg_entry)

            # Reset per-message state
            scheduler_start_time = None
            generator_start_time = None
            current_send_decision_time = None
            current_raw_message = None
            current_final_message = None
            current_dropped = False
            review_attempts = []
            review_final_score = None
            review_was_modified = False
            current_scheduler_latency = None
            current_generator_latency = None
            current_reviewer_latency = None
            current_typing_wait = None
            current_budget_exceeded = False
            current_review_degraded = False
            current_candidates = None
            current_selected_text = None
            current_selected_n = None

    # ── Aggregate stats ──
    # Defensive post-filter: even with all the per-event guards above, a
    # stray cycle whose LIVE_REVIEW_SUMMARY regex didn't match can leave a
    # phantom row with empty `text`. Drop those before any aggregate uses
    # `messages`. (Renumber so msg_num stays contiguous in the UI.)
    messages = [m for m in messages if (m.get("text") or "").strip()]

    # ── Phantom send-tick de-dup ──
    # The main loop logs "generate_message UNLOCKED" on EVERY scheduler tick,
    # not only on a real send. A real send carries pipeline latencies
    # (scheduler+generator both present); the follow-up poll tick re-logs the
    # same text with all pipeline latencies None and only a typing wait.
    # Drop an adjacent repeated-text row when all three call latencies are
    # absent, carrying any measured wait onto the retained row.
    deduped = []
    for m in messages:
        no_pipeline = (m.get("scheduler_latency_s") is None
                       and m.get("generator_latency_s") is None
                       and m.get("reviewer_latency_s") is None)
        if (no_pipeline and deduped
                and (m.get("text") or "").strip() == (deduped[-1].get("text") or "").strip()):
            # Carry a real typing wait onto the kept row if it lacked one
            # (the send/typing happens on the poll tick AFTER generation, so
            # the real typing-delay often lands on this twin). Recompute Total
            # so Sched+Gen+Rev+Wait stays consistent.
            keep = deduped[-1]
            if not keep.get("writing_wait_s") and m.get("writing_wait_s"):
                keep["writing_wait_s"] = m["writing_wait_s"]
                comp = [keep.get("scheduler_latency_s"), keep.get("generator_latency_s"),
                        keep.get("reviewer_latency_s"), keep.get("writing_wait_s")]
                if all(c is not None for c in comp):
                    keep["total_latency_s"] = round(sum(comp), 2)
                    # Total is now exactly the 4 components → no extra overhead.
                    keep["other_latency_s"] = 0.0
            continue
        deduped.append(m)
    messages = deduped

    for _i, _m in enumerate(messages):
        _m["msg_num"] = _i + 1

    if delivered_messages is not None:
        def _delivery_norm(text):
            return re.sub(r"\s+", " ", (text or "").lower()).strip().rstrip("!.?")

        remaining_deliveries = Counter(
            _delivery_norm(text) for text in delivered_messages if text
        )
        for message in messages:
            key = _delivery_norm(message.get("text"))
            delivered = bool(key and remaining_deliveries[key] > 0)
            message["sent"] = delivered
            if delivered:
                remaining_deliveries[key] -= 1
                message["discard_reason"] = None
            elif not message.get("discard_reason"):
                message["discard_reason"] = "not present in AI chat append"

    # Only count messages that were actually sent (not dropped after
    # generation). This keeps total_sent / messages_per_minute / send_rate
    # consistent with the public chat history. ALL averages below use
    # `sent_messages` so the cards match what's visible in the chat.
    sent_messages = [m for m in messages if m.get("sent", True)]
    total_sent = len(sent_messages)
    # Rate window: from the moment discussion opened (discussion_start_time.txt)
    # to the last log entry, using the first entry when the discussion anchor
    # is unavailable or outside the usable clock-value range.
    rate_start = game_first_time
    if discussion_start_hms:
        ds = _time_to_seconds(discussion_start_hms.strip())
        if ds is not None and game_last_time is not None and ds <= game_last_time:
            rate_start = max(ds, game_first_time) if game_first_time is not None else ds
    elapsed_s = (game_last_time - rate_start) if rate_start is not None and game_last_time is not None else 0
    elapsed_min = elapsed_s / 60.0 if elapsed_s > 0 else 0

    def _safe_avg(lst):
        vals = [v for v in lst if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    # Count review evidence from a score, an identified candidate selection,
    # or measured reviewer latency; a numeric score is not required.
    def _was_reviewed(m):
        return (m["reviewer_score"] is not None
                or (m.get("selection_mode") and m.get("selected_candidate_index") is not None)
                or m.get("reviewer_latency_s") is not None)
    reviewed_msgs = [m for m in sent_messages if _was_reviewed(m)]
    unreviewed_msgs = [m for m in sent_messages if not _was_reviewed(m)]

    # Scheduler stats over ALL calls (send + wait), not just messages.
    # Wait-decision durations are real wall-clock time during which the
    # agent was blocked waiting for the scheduler API to return — they
    # belong in the scheduler average and we surface long ones (>10s)
    # as \"stalls\" so Vertex AI hangs become visible.
    all_sched_lats = [c["latency_s"] for c in all_scheduler_calls]
    wait_call_lats = [c["latency_s"] for c in all_scheduler_calls if c["decision"] == '<wait>']
    send_call_lats = [c["latency_s"] for c in all_scheduler_calls if c["decision"] == '<send>']
    long_wait_calls = [c for c in all_scheduler_calls if c["decision"] == '<wait>' and c["latency_s"] > long_wait_threshold_s]

    aggregate = {
        "total_messages_sent": total_sent,
        "total_scheduler_calls": scheduler_calls,
        "send_decisions": send_decisions,
        "wait_decisions": wait_decisions,
        "send_rate_pct": round(100 * send_decisions / scheduler_calls, 1) if scheduler_calls else 0,
        # Average over EVERY scheduler call (send + wait) — the honest number.
        "avg_scheduler_latency_s": _safe_avg(all_sched_lats),
        # Split by decision so the UI can show e.g. \"3.2s send / 7.4s wait\".
        "avg_scheduler_latency_send_s": _safe_avg(send_call_lats),
        "avg_scheduler_latency_wait_s": _safe_avg(wait_call_lats),
        "max_scheduler_latency_s": round(max(all_sched_lats), 2) if all_sched_lats else None,
        # Wait-call list (most recent first, capped) for the stall panel.
        "wait_call_durations": [
            {"time": c["time"], "latency_s": c["latency_s"]}
            for c in all_scheduler_calls if c["decision"] == '<wait>'
        ][-50:],
        "long_wait_call_count": len(long_wait_calls),
        "long_wait_threshold_s": long_wait_threshold_s,
        "avg_generator_latency_s": _safe_avg([m["generator_latency_s"] for m in sent_messages]),
        "avg_other_latency_s": _safe_avg([m.get("other_latency_s") for m in sent_messages]),
        "avg_total_latency_s": _safe_avg([m["total_latency_s"] for m in sent_messages]),
        "avg_writing_wait_s": _safe_avg([m["writing_wait_s"] for m in sent_messages]),
        "avg_words_per_message": _safe_avg([m["word_count"] for m in sent_messages]),
        "messages_per_minute": round(total_sent / elapsed_min, 2) if elapsed_min > 0 else 0,
        "elapsed_minutes": round(elapsed_min, 1),
        # Show reviewer columns when review evidence or reviewer latency exists.
        "reviewer_enabled": len(reviewed_msgs) > 0 or any(
            m.get("reviewer_latency_s") is not None for m in sent_messages),
        "avg_reviewer_score": _safe_avg([m["reviewer_score"] for m in reviewed_msgs]) if reviewed_msgs else None,
        "avg_reviewer_latency_s": _safe_avg([m["reviewer_latency_s"] for m in sent_messages if m.get("reviewer_latency_s") is not None]),
        # Preserve measured zero latency; return None when no latency is recorded.
        "max_reviewer_latency_s": round(max((m["reviewer_latency_s"] for m in sent_messages if m.get("reviewer_latency_s") is not None), default=0), 2) if any(m.get("reviewer_latency_s") is not None for m in sent_messages) else None,
        "messages_modified_by_reviewer": sum(1 for m in sent_messages if m["reviewer_modified"]),
        # Candidate-selection provenance, including how often the selected
        # candidate was not the first in the generator's pool.
        "selection_mode": any(m.get("selection_mode") for m in sent_messages),
        "messages_with_selection_index": sum(
            1 for m in sent_messages if m.get("selected_candidate_index") is not None),
        "non_first_candidate_selected": sum(
            1 for m in sent_messages
            if (m.get("selected_candidate_index") or 0) > 0),
        # Count how many messages had their reviewer loop truncated because
        # it ran past the wall-clock budget. Useful signal for Vertex latency
        # regressions or prompting bugs that make the reviewer keep retrying.
        "reviewer_budget_exceeded_count": sum(1 for m in sent_messages if m.get("reviewer_budget_exceeded")),
        # Descriptive latency split by recorded review evidence. This is
        # not a causal estimate of latency added by the reviewer.
        "messages_with_reviewer": len(reviewed_msgs),
        "messages_without_reviewer": len(unreviewed_msgs),
        "avg_total_latency_with_reviewer_s": _safe_avg([m["total_latency_s"] for m in reviewed_msgs]),
        "avg_total_latency_without_reviewer_s": _safe_avg([m["total_latency_s"] for m in unreviewed_msgs]),
        "avg_writing_wait_with_reviewer_s": _safe_avg([m["writing_wait_s"] for m in reviewed_msgs]),
        "avg_writing_wait_without_reviewer_s": _safe_avg([m["writing_wait_s"] for m in unreviewed_msgs]),
        "last_activity": entries[-1]["time"] if entries else None,
    }

    for _m in messages:                      # strip the internal append-ordering flag
        _m.pop("_typing_from_log", None)
    return {"messages": messages, "aggregate": aggregate}


# ── AI State / Cycles: structured "what is the AI doing now" view ──

def _compute_ai_cycles(log_text, max_cycles=20):
    """
    Parse the AI log into structured cycles + current state.

    A "cycle" starts at `prompt in should_generate_message` (scheduler invocation)
    and ends either when:
      - scheduler decision is <wait> (cycle completed with status=completed_wait), or
      - `generate_message` UNLOCKED appears (completed_send), or
      - the log ends mid-cycle (in_progress).

    Returns:
      {
        "cycles": [...last max_cycles completed cycles, newest first...],
        "current": {...in-progress cycle or None...},
        "current_state": one of idle / reading / scheduling / generating / reviewing / writing / sending,
        "current_state_since": "HH:MM:SS" or None,
        "total_cycles": int,
      }
    """
    entries = _parse_log_entries(log_text)
    if not entries:
        return {"cycles": [], "current": None, "current_state": "idle",
                "current_state_since": None, "total_cycles": 0}

    def _new_cycle(cycle_num, start_time, start_entry_idx):
        return {
            "cycle_num": cycle_num,
            "start_time": start_time,
            "end_time": None,
            "start_entry_idx": start_entry_idx,
            "scheduler_prompt_tail": None,
            "scheduler_decision": None,     # "send" | "wait" | None
            "scheduler_latency_s": None,
            "generator_prompt_tail": None,
            "generator_raw_output": None,
            "generator_latency_s": None,
            "reviewer_score": None,
            "reviewer_attempts": 0,
            "reviewer_modified": False,
            "reviewer_red_flags": [],
            "reviewer_latency_s": None,
            "reviewer_budget_exceeded": False,
            "final_message": None,
            "status": "in_progress",        # in_progress | completed_send | completed_wait
        }

    def _tail(s, n=600):
        if s is None:
            return None
        s = s.strip()
        return s if len(s) <= n else "…" + s[-n:]

    cycles = []           # completed cycles (chronological)
    current = None
    cycle_counter = 0

    for i, e in enumerate(entries):
        op = e["operation"]
        content = e["content"]
        t = e["time"]

        # Treat both scheduler-prompt labels as cycle starts so the AI-State
        # view populates for either supported log format.
        if op in ("prompt in should_generate_message", "scheduler_prompt"):
            # Close any previous in-progress cycle that never completed (safety)
            if current is not None and current["status"] == "in_progress":
                current["status"] = "completed_wait" if current["scheduler_decision"] == "wait" else "in_progress"
                current["end_time"] = current["end_time"] or t
                cycles.append(current)
            cycle_counter += 1
            current = _new_cycle(cycle_counter, t, i)
            current["scheduler_prompt_tail"] = _tail(content)
            continue

        if current is None:
            # Entry before any scheduler prompt (pre-game / voting phase) — ignore
            continue

        if op == "scheduler_latency":
            m = re.match(r"([\d.]+)s", content)
            if m:
                current["scheduler_latency_s"] = round(float(m.group(1)), 2)
        elif op == "decision in should_generate_message":
            if "<send>" in content:
                current["scheduler_decision"] = "send"
            else:
                current["scheduler_decision"] = "wait"
                # Wait-cycle ends right here
                current["status"] = "completed_wait"
                current["end_time"] = t
                cycles.append(current)
                current = None
        elif op == "scheduler_decision":
            # Scheduler decision: content is "decision=SEND, send_reason=…".
            # Match the decision token itself — NOT a bare "SEND", which also
            # appears in "send_reason=…" on a WAIT line.
            if re.search(r"decision\s*=\s*send", content, re.IGNORECASE):
                current["scheduler_decision"] = "send"
            else:
                current["scheduler_decision"] = "wait"
                current["status"] = "completed_wait"
                current["end_time"] = t
                cycles.append(current)
                current = None
        elif op in ("prompt in generate_message", "prompt in _generate_raw_message",
                    "generator_prompt"):
            current["generator_prompt_tail"] = _tail(content)
        elif op.startswith("final_output in generate") or op == "generator_candidates":
            current["generator_raw_output"] = content.strip()
            if op == "generator_candidates":
                try:
                    import ast as _ast
                    _pool = _ast.literal_eval(content.strip().splitlines()[0])
                    if isinstance(_pool, (list, tuple)):
                        current["_candidate_pool"] = [str(c) for c in _pool]
                except Exception:
                    pass
        elif op == "reviewer_selected":
            # Ranker: content is "'<selected text>' from N candidates (…)".
            m = re.match(r"^['\"](.*)['\"] from (\d+) candidates", content)
            if m:
                current["final_message"] = m.group(1)
                # Show candidate-selection provenance even without a numeric
                # reviewer score or per-attempt records.
                current["reviewer_mode"] = "ranker"
                current["num_candidates"] = int(m.group(2))
                pool = current.get("_candidate_pool") or []
                try:
                    current["selected_candidate_index"] = pool.index(m.group(1))
                except ValueError:
                    current["selected_candidate_index"] = None
        elif op == "reviewer_red_flags":
            current["reviewer_red_flags"] = [
                f.strip() for f in content.split(",") if f.strip()
            ]
        elif op == "generator_latency":
            m = re.match(r"([\d.]+)s", content)
            if m:
                current["generator_latency_s"] = round(float(m.group(1)), 2)
        elif op == "reviewer_latency":
            m = re.match(r"([\d.]+)s", content)
            if m:
                current["reviewer_latency_s"] = round(float(m.group(1)), 2)
        elif op in ("reviewer_budget_exceeded", "reviewer_budget_truncated"):
            current["reviewer_budget_exceeded"] = True
        elif op in ("pipeline_timeout_drop", "stale_send_drop"):
            # Record output-gate suppression separately from selection so a
            # SEND decision is not presented as confirmed delivery.
            current["suppressed"] = ("timeout" if op == "pipeline_timeout_drop"
                                     else "stale")
            current["suppressed_detail"] = _tail(content)
        elif op.startswith("review_attempt_"):
            current["reviewer_attempts"] += 1
        elif op == "LIVE_REVIEW_SUMMARY":
            score_m = re.search(r"Final score:\s*(\d+)/10", content)
            if score_m:
                current["reviewer_score"] = int(score_m.group(1))
            current["reviewer_modified"] = "(modified)" in content
            flags_m = re.search(r"red_flags:\s*\[([^\]]*)\]", content)
            if flags_m:
                current["reviewer_red_flags"] = [
                    f.strip().strip("'\"") for f in flags_m.group(1).split(",") if f.strip()
                ]
        elif op == "generate_message" and "UNLOCKED" in content:
            # Cycle complete. This fires every poll (the generate_message
            # wrapper's finally block); a WAIT cycle was already closed at the
            # decision, so an open cycle here means a SEND. Keep the reviewer's
            # selected final_message if we captured one, else the generator output.
            current["final_message"] = current["final_message"] or current["generator_raw_output"]
            current["status"] = "completed_send" if current["scheduler_decision"] == "send" else "completed_wait"
            current["end_time"] = t
            cycles.append(current)
            current = None

    # ── Derive current_state from tail of log / in-progress cycle ──
    last_op = entries[-1]["operation"] if entries else None
    last_time = entries[-1]["time"] if entries else None

    if current is not None:
        # A cycle is open. Determine sub-stage from what we've seen so far.
        if current["generator_raw_output"] is not None and current["reviewer_score"] is None \
                and last_op not in ("LIVE_REVIEW_SUMMARY",):
            # Generator finished, reviewer might be running or already done without summary
            if any(e["operation"].startswith(("review_", "GeminiReviewer", "evaluate_pending_fix"))
                   for e in entries[current["start_entry_idx"]:]):
                current_state = "reviewing"
            else:
                current_state = "sending"
        elif current["generator_prompt_tail"] is not None and current["generator_raw_output"] is None:
            current_state = "generating"
        elif current["scheduler_decision"] == "send" and current["generator_prompt_tail"] is None:
            current_state = "generating"  # about to call generator
        else:
            current_state = "scheduling"
        current_state_since = current["start_time"]
    else:
        # No open cycle — AI is idling between scheduler invocations.
        if last_op == "AGENT_HEARTBEAT":
            current_state = "idle"
        elif last_op == "AGENT_STATUS" and "Read " in (entries[-1]["content"] or ""):
            current_state = "reading"
        elif last_op == "generate_message":
            current_state = "idle"
        else:
            current_state = "idle"
        current_state_since = last_time

    # Truncate cycle fields that shouldn't be sent to frontend
    display_cycles = []
    for c in cycles[-max_cycles:][::-1]:  # newest first
        display_cycles.append({k: v for k, v in c.items()
                               if k not in ("start_entry_idx", "_candidate_pool")})
    display_current = None
    if current is not None:
        display_current = {k: v for k, v in current.items()
                           if k not in ("start_entry_idx", "_candidate_pool")}

    return {
        "cycles": display_cycles,
        "current": display_current,
        "current_state": current_state,
        "current_state_since": current_state_since,
        "total_cycles": cycle_counter,
    }


def _game_event_files(game_dir, pattern="game_events*.jsonl"):
    """Return matching event files from events/ and the game root.

    Search events/ first and de-duplicate by filename so its record takes
    precedence when a matching filename exists in both locations.
    """
    out, seen = [], set()
    for base in (game_dir / SUBDIR_EVENTS, game_dir):
        try:
            for p in sorted(base.glob(pattern)):
                if p.name not in seen:
                    seen.add(p.name)
                    out.append(p)
        except OSError:
            pass
    return out


@app.route('/api/admin/component_state')
def admin_component_state():
    """Return live worker activity and the latest observed shared snapshot."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)

    # Prefer game_events.jsonl. If only name-suffixed event files are present,
    # use the requested name or the first file without mixing state streams.
    events_file = game_file(game_dir, "game_events.jsonl")
    agents_available = []
    if not events_file.exists():
        per_agent = _game_event_files(game_dir, "game_events_*.jsonl")
        agents_available = [p.stem[len("game_events_"):] for p in per_agent]
        if not per_agent:
            return jsonify({
                "snapshot": None, "ic_ticks": [], "ps_events": [], "turns": [],
                "agents_available": [],
            })
        requested = request.args.get("agent")
        events_file = next(
            (p for p in per_agent if p.stem[len("game_events_"):] == requested),
            per_agent[0],
        )

    events = []
    try:
        for line in events_file.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return jsonify({"error": "could not read events file"}), 500

    # Latest snapshot view the graph received.
    snap = None
    for ev in reversed(events):
        if ev.get("event_type") == "snapshot_observed":
            snap = {
                "turn_index":      ev.get("turn_index"),
                "current_info":    ev.get("current_info"),
                "hedge_advice":    ev.get("hedge_advice"),
                "round_summaries": ev.get("round_summaries"),
                "self_claims":     ev.get("self_claims", []),
                "social_state":    ev.get("social_state", ""),
                "behavior_advice": ev.get("behavior_advice", ""),
                "entry_count":     ev.get("entry_count"),
                "timestamp":       ev.get("timestamp"),
            }
            break

    # Persona Monitor — latest extracted self-claims (fallback if no snapshot
    # event carried them yet).
    persona_claims = (snap or {}).get("self_claims") or []
    if not persona_claims:
        for ev in reversed(events):
            if ev.get("event_type") == "persona_claims_updated":
                persona_claims = ev.get("claims", [])
                break

    # Social State Tracker — latest standing (fallback to the update event).
    social_state = (snap or {}).get("social_state") or ""
    if not social_state:
        for ev in reversed(events):
            if ev.get("event_type") == "social_state_updated":
                bits = []
                sm_susp = ev.get("suspects_me") or []
                sm_point = ev.get("i_pointed_at") or []
                sm_out = ev.get("voted_out") or []
                if sm_susp:
                    bits.append(f"{', '.join(sm_susp)} suspected you")
                if sm_point:
                    bits.append(f"you pointed at {', '.join(sm_point)}")
                if sm_out:
                    bits.append(f"{', '.join(sm_out)} voted out")
                social_state = "; ".join(bits)
                break

    # Behavior Adapter — latest play-style directive (fallback to update event).
    behavior_advice = (snap or {}).get("behavior_advice") or ""
    if not behavior_advice:
        for ev in reversed(events):
            if ev.get("event_type") == "behavior_advice_updated":
                behavior_advice = ev.get("advice", "")
                break

    # Behavior diagnostics — latest participation assessment + any warnings.
    behavior_diag = {}
    for ev in reversed(events):
        if ev.get("event_type") == "behavior_tick":
            behavior_diag = {
                "participation_assessment": ev.get("participation_assessment", ""),
                "warnings": ev.get("warnings", []),
            }
            break

    # Agent decisions — the recent WHY-it-waited / acted trail so the admin can
    # see the agent's MEANINGFUL reasoning live: lull handling, anti-monologue,
    # awaiting-search waits, dedup drops, anti-filler skips. Two scheduler
    # reasons (history_unchanged, min_interval_floor) fire on essentially every
    # poll tick and would bury the interesting ones, so they're filtered out.
    _ROUTINE_WAIT_REASONS = {"history_unchanged", "min_interval_floor"}
    decisions = []
    for ev in events:
        et = ev.get("event_type")
        if et == "scheduler_blocked":
            reason = ev.get("reason")
            if reason in _ROUTINE_WAIT_REASONS:
                continue
            # lull_breaker_fired is a decision to SPEAK (break dead air), not a
            # wait — tag it distinctly so the panel shows it as an action.
            kind = "act" if reason == "lull_breaker_fired" else "wait"
            decisions.append({"ts": ev.get("timestamp"), "kind": kind,
                              "reason": reason})
        elif et == "dedup_blocked":
            decisions.append({"ts": ev.get("timestamp"), "kind": "dedup",
                              "reason": "duplicate response suppressed"})
        elif et == "generator_skip":
            decisions.append({"ts": ev.get("timestamp"), "kind": "skip",
                              "reason": ev.get("reason", "nothing specific to add")})
    decisions = decisions[-12:]

    # Internet Context — pair internet_decision_response + ticks
    ic_ticks = []
    for ev in events:
        et = ev.get("event_type")
        if et == "internet_decision_response":
            ic_ticks.append({
                "kind":     "decision",
                "ts":       ev.get("timestamp"),
                "parsed":   ev.get("parsed"),
                "latency":  ev.get("latency_s"),
            })
        elif et == "internet_search_response":
            ic_ticks.append({
                "kind":         "search",
                "ts":           ev.get("timestamp"),
                "query":        ev.get("query"),
                "n_results":    ev.get("n_results"),
                "summary_text": ev.get("summary_text"),
                "latency":      ev.get("latency_s"),
            })
        elif et == "internet_context_tick" and ev.get("dropped"):
            ic_ticks.append({
                "kind":     "dropped",
                "ts":       ev.get("timestamp"),
                "reason":   ev.get("reason"),
            })
    ic_ticks = ic_ticks[-8:]

    # Round Summarizer — last 3 emitted summaries.
    ps_events = []
    for ev in events:
        if ev.get("event_type") == "round_summary_tick":
            ps_events.append({
                "ts":           ev.get("timestamp"),
                "phase":        ev.get("phase"),
                "n_messages":   ev.get("n_messages"),
                "summary":      ev.get("summary"),
            })
    ps_events = ps_events[-3:]

    # Turn timings — last 3.
    turns = []
    for ev in events:
        if ev.get("event_type") == "generate_end":
            turns.append({
                "turn_index":     ev.get("turn_index"),
                "pipeline_s":     ev.get("pipeline_latency_s"),
                "final_chars":    ev.get("final_message_len"),
                "dropped":        ev.get("dropped"),
                "ts":             ev.get("timestamp"),
            })
    turns = turns[-3:]

    # Repetition Monitor — overused words + the LLM shape/approach note.
    overused_words = list((snap or {}).get("overused_words") or [])
    repetition_note = (snap or {}).get("repetition_note") or ""
    if not repetition_note:
        for ev in reversed(events):
            if ev.get("event_type") == "repetition_note_updated":
                repetition_note = ev.get("note", "")
                break
    if not overused_words:
        for ev in reversed(events):
            if ev.get("event_type") == "overused_words_updated":
                overused_words = ev.get("words", []) or []
                break

    # Cadence (Behavior Adapter) + room_norm (Social State Tracker) — the
    # scheduler-facing signals, so the admin can see WHY the agent held back.
    cadence = {}
    room_norm = ""
    for ev in reversed(events):
        if not cadence and ev.get("event_type") == "cadence_updated":
            cadence = {"level": ev.get("level"), "talk_share": ev.get("talk_share"),
                       "reason": ev.get("reason")}
        if not room_norm and ev.get("event_type") == "social_state_updated":
            room_norm = ev.get("room_norm", "") or ""
        if cadence and room_norm:
            break

    # Sub-agent activity summary — lets the admin confirm all SIX background
    # workers are alive at a glance (tick counts per sub-agent).
    def _count(et):
        return sum(1 for ev in events if ev.get("event_type") == et)
    subagent_status = {
        "internet":   _count("internet_tick_begin"),
        "phase":      _count("round_summary_tick_begin"),
        "persona":    _count("persona_monitor_tick"),
        "social":     _count("social_state_tick"),
        "behavior":   _count("behavior_tick"),
        "repetition": _count("repetition_monitor_tick"),
    }

    return jsonify({
        "snapshot":         snap,
        "persona_claims":   persona_claims,
        "social_state":     social_state,
        "behavior_advice":  behavior_advice,
        "behavior_diag":    behavior_diag,
        "overused_words":   overused_words,
        "repetition_note":  repetition_note,
        "cadence":          cadence,
        "room_norm":        room_norm,
        "subagent_status":  subagent_status,
        "decisions":        decisions,
        "ic_ticks":         ic_ticks,
        "ps_events":        ps_events,
        "turns":            turns,
        "agents_available": agents_available,
        "agent":            events_file.stem[len("game_events_"):]
                            if events_file.name.startswith("game_events_") else None,
    })


@app.route('/api/admin/agent_state')
def admin_agent_state():
    """Return agent snapshots, persona facts, and selection/delivery provenance."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)

    # Authoritative agent signal from the game config (works even before any
    # candidates have been generated), so the UI can show the agent tab promptly.
    is_agent = False
    try:
        cfg_path = game_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
            is_agent = detect_agent_identity(cfg) == "social_turing_test_agent"
    except Exception:
        is_agent = False

    # Build provenance is available even when the agent crashed before writing
    # its first event (the exact failure mode where it matters most).
    game_agent_code = _agent_code_from_game(game_dir)
    agent_build = _game_agent_build_info(game_dir)

    # As in component_state, prefer the unsuffixed event file, then select
    # one name-suffixed file without mixing state streams.
    events_file = game_file(game_dir, "game_events.jsonl")
    agents_available = []
    if not events_file.exists():
        per_agent = _game_event_files(game_dir, "game_events_*.jsonl")
        agents_available = [p.stem[len("game_events_"):] for p in per_agent]
        if not per_agent:
            return jsonify({
                "version": "agent", "is_agent": is_agent, "snapshot": None,
                "fixed_persona_facts": {}, "social_signals": {},
                "generations": [], "agents_available": [],
                "agent_build": agent_build,
                "game_agent_code": game_agent_code,
            })
        requested = request.args.get("agent")
        events_file = next(
            (p for p in per_agent if p.stem[len("game_events_"):] == requested),
            per_agent[0],
        )

    events = []
    try:
        for line in events_file.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return jsonify({"error": "could not read events file"}), 500

    # ── Shared-memory snapshot (same source the component panel uses) ──
    snap = None
    for ev in reversed(events):
        if ev.get("event_type") == "snapshot_observed":
            snap = {
                "turn_index":      ev.get("turn_index"),
                "current_info":    ev.get("current_info"),
                "round_summaries": ev.get("round_summaries"),
                "self_claims":     ev.get("self_claims", []),
                "social_state":    ev.get("social_state", ""),
                "behavior_advice": ev.get("behavior_advice", ""),
                "entry_count":     ev.get("entry_count"),
                # Per-entry provenance (origin: "seed" | "search") — lets the
                # panel separate background knowledge from verified lookups.
                "info_entries":    ev.get("info_entries", []),
                "timestamp":       ev.get("timestamp"),
                # Optional recorded shared-memory fields. Missing fields use
                # display defaults, not evidence of what the agent observed.
                "room_vibe":       ev.get("room_vibe", ""),
                "room_norm":       ev.get("room_norm", ""),
                "pressure_type":   ev.get("pressure_type", ""),
                "hedge_advice":    ev.get("hedge_advice", ""),
                "overused_words":  ev.get("overused_words", []),
                "repetition_note": ev.get("repetition_note", ""),
                "open_threads":    ev.get("open_threads", []),
                "awaiting_search": ev.get("awaiting_search"),
                "pending_search_queries": ev.get("pending_search_queries", []),
                "interrogation_pressure": ev.get("interrogation_pressure"),
                "attention_on_me": ev.get("attention_on_me"),
                # Optional recorded per-player communication-style map.
                "player_profiles": ev.get("player_profiles", {}) or {},
            }
            break

    # Immutable persona facts (location/timezone) + numeric social signals.
    # Read the latest social-state tick's parsed payload when available.
    fixed_persona_facts = {}
    social_signals = {}
    for ev in reversed(events):
        if ev.get("event_type") == "social_state_tick":
            parsed = ev.get("parsed") or {}
            if isinstance(parsed, dict):
                for k in ("interrogation_pressure", "attention_on_me"):
                    if parsed.get(k) is not None:
                        social_signals[k] = parsed.get(k)
            if social_signals:
                break
    for ev in reversed(events):
        # Read the facts dict from a fixed_persona_facts_set event's facts
        # field, or from a payload's top-level fixed_persona_facts field.
        ff = ev.get("facts") if ev.get("event_type") == "fixed_persona_facts_set" \
            else ev.get("fixed_persona_facts")
        if isinstance(ff, dict) and ff:
            fixed_persona_facts = ff
            break

    # ── Multi-candidate generation pipeline ──
    # Reconcile event rows with the AI append file and explicit text-log
    # phase-drop records when delivery outcomes are not present in events.
    ai_name = get_llm_player_name(game_dir)
    delivered_messages = []
    phase_dropped_messages = []
    if ai_name:
        delivered_messages = _read_ai_delivered_messages(game_dir, ai_name)
        llm_log = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
        try:
            for entry in _parse_log_entries(llm_log.read_text(encoding="utf-8", errors="replace")):
                if entry.get("operation") != "AGENT_DROPPED_GENERATED_MESSAGE":
                    continue
                match = re.search(r": '(.*)'\s*$", entry.get("content", ""))
                if match:
                    phase_dropped_messages.append(match.group(1))
        except OSError:
            pass
    generations = _agent_generation_rows(
        events,
        delivered_messages=delivered_messages,
        phase_dropped_messages=phase_dropped_messages,
    )

    # ── Per-turn fact usage + terminal outcome ──
    # Built ONLY from the structured pipeline events. A fact is reported "used"
    # solely from `verified_fact_ids_used` / `fact_usage_status` emitted by the
    # validator — NEVER inferred from `available_fact_ids`. Each row also carries
    # the distinct terminal outcome and the gate reason for a cancellation.
    fact_usage_turns = []
    for ev in events:
        if ev.get("event_type") != "turn_pipeline_complete":
            continue
        available = ev.get("available_fact_ids", []) or []
        used = ev.get("verified_fact_ids_used", []) or []
        fact_usage_turns.append({
            "turn_id":               ev.get("turn_id"),
            "ts":                    ev.get("timestamp"),
            "final_response_act":    ev.get("final_response_act"),
            "review_status":         ev.get("review_status"),
            "review_risk":           ev.get("review_risk", None),
            "available_fact_ids":    available,
            "verified_fact_ids_used": used,
            # Honest status straight from the event — not derived here.
            "fact_usage_status":     ev.get("fact_usage_status", "none"),
            "snapshot_revision":     ev.get("snapshot_revision"),
            "artificial_delay_s":    ev.get("artificial_delay_requested_s"),
            "pipeline_complete_at":  ev.get("pipeline_complete_at"),
        })
    fact_usage_turns = fact_usage_turns[-10:]

    # ── Orchestration operational status ──
    # Latest requested/allowed/operational status, including which
    # production adapters are present + the degraded reason when not operational.
    active_status = None
    for ev in reversed(events):
        if ev.get("event_type") in ("orchestration_mode", "active_not_operational"):
            st = ev.get("active_status") if ev.get("event_type") == "orchestration_mode" \
                else {k: ev.get(k) for k in (
                    "requested", "allowed", "operational", "scheduler_adapter_available",
                    "generator_adapter_available", "reviewer_adapter_available",
                    "fresh_state_provider_available", "degraded_reason")}
            if isinstance(st, dict) and st:
                active_status = st
                break

    # Distinct terminal outcomes (sent / scheduler wait / generator failure /
    # reviewer drop / pre-send cancel) for the timeline panel.
    _TERMINALS = {"turn_sent", "turn_scheduler_wait", "turn_generator_failure",
                  "turn_reviewer_drop", "turn_pre_send_cancel"}
    terminal_outcomes = [
        {"turn_id": ev.get("turn_id"), "ts": ev.get("timestamp"),
         "outcome": ev.get("event_type"), "is_send": ev.get("is_send", False),
         "gate_reason": ev.get("gate_reason"),
         "pipeline_complete_at": ev.get("pipeline_complete_at")}
        for ev in events if ev.get("event_type") in _TERMINALS
    ][-10:]

    return jsonify({
        "version":             "agent",
        "is_agent":            is_agent,
        # Prefer the game's persisted spawn fingerprint. If absent, label
        # checkout inference explicitly; it is not recorded process provenance.
        "agent_build":         agent_build,
        "game_agent_code":     game_agent_code,
        "snapshot":            snap,
        "fixed_persona_facts": fixed_persona_facts,
        "social_signals":      social_signals,
        "generations":         generations,
        "fact_usage_turns":    fact_usage_turns,
        "terminal_outcomes":   terminal_outcomes,
        "active_status":       active_status,
        # Recorded orchestration and timing provenance for applicable status cards.
        "orchestration":       _orchestration_provenance(game_dir),
        # Latest sub-agent activity and scheduler send/wait reasons.
        "subagents":           _agent_subagents_activity(events, generations),
        "scheduler_breakdown": _scheduler_reason_breakdown(events, generations),
        "agents_available":    agents_available,
        "agent":               events_file.stem[len("game_events_"):]
                               if events_file.name.startswith("game_events_") else None,
    })


_AGENT_BUILD_CACHE = {}

# Untracked runtime game/evaluation output is intentionally excluded.  These
# pathspecs cover files that can change imported agent code or its configuration
# despite not being represented by HEAD.
_UNTRACKED_AGENT_PATHSPECS = (
    ":(glob)**/*.py",
    ":(glob)**/*.pyi",
    ":(glob)**/*.toml",
    ":(glob)**/*.yaml",
    ":(glob)**/*.yml",
    ":(glob)**/*.ini",
    ":(glob)**/*.cfg",
    ":(glob)configurations/**/*.json",
    ":(glob)*.json",
    ":(exclude,glob)games/**",
)


def _run_git(checkout, *args):
    """Run a small read-only git query and return stdout, or raise on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        cwd=str(checkout),
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git failed").strip())
    return result.stdout


def _agent_source_fingerprint(checkout) -> dict:
    """Hash behavior-relevant source/config bytes without requiring git."""
    return relevant_source_fingerprint(Path(checkout).resolve())


def _agent_build_info(checkout=None, *, refresh=False):
    """Return a runtime-derived git fingerprint for an agent checkout.

    ``commit`` is the short hash displayed by the admin UI. Local changes use
    ``dirty`` plus a deterministic ``patch_marker`` derived from status+diff.
    """
    checkout = Path(checkout or _BASE_DIR).resolve()
    cache_key = str(checkout)
    if not refresh and cache_key in _AGENT_BUILD_CACHE:
        return dict(_AGENT_BUILD_CACHE[cache_key])

    source_fingerprint = _agent_source_fingerprint(checkout)
    try:
        commit_full = _run_git(checkout, "rev-parse", "HEAD").strip()
        commit = _run_git(checkout, "rev-parse", "--short", "HEAD").strip()
        subject = _run_git(checkout, "log", "-1", "--format=%s").strip()
        tracked_status = _run_git(
            # Runtime artifacts and game directories are intentionally
            # untracked. Fingerprint tracked patches independently from
            # unexpected untracked source/config below.
            checkout, "status", "--porcelain", "--untracked-files=no",
        ).strip()
        tracked_changes = [line for line in tracked_status.splitlines() if line.strip()]
        tracked_dirty = bool(tracked_changes)
        tracked_patch_marker = None
        if tracked_dirty:
            diff = _run_git(checkout, "diff", "--binary", "HEAD", "--")
            patch_material = f"{tracked_status}\n{diff}".encode("utf-8", errors="replace")
            tracked_patch_marker = hashlib.sha256(patch_material).hexdigest()[:12]

        untracked_raw = _run_git(
            checkout, "ls-files", "--others", "--exclude-standard", "--",
            *_UNTRACKED_AGENT_PATHSPECS,
        )
        untracked_source_config = sorted({
            line.strip().replace("\\", "/")
            for line in untracked_raw.splitlines() if line.strip()
        })
        untracked_marker = None
        if untracked_source_config:
            untracked_hash = hashlib.sha256()
            for relative in untracked_source_config:
                untracked_hash.update(relative.encode("utf-8", errors="replace"))
                untracked_hash.update(b"\0")
                path = (checkout / relative).resolve()
                try:
                    path.relative_to(checkout)
                    untracked_hash.update(path.read_bytes())
                except (OSError, ValueError):
                    untracked_hash.update(b"<unreadable>")
                untracked_hash.update(b"\0")
            untracked_marker = untracked_hash.hexdigest()[:12]

        dirty = tracked_dirty or bool(untracked_source_config)
        marker_parts = [commit]
        if tracked_dirty:
            marker_parts.append(f"dirty.{tracked_patch_marker}")
        if untracked_source_config:
            marker_parts.append(f"untracked.{untracked_marker}")
        build_marker = "+".join(marker_parts)
        info = {
            "commit": commit,
            "commit_full": commit_full,
            "last_commit_subject": subject[:100],
            "dirty": dirty,
            # This field identifies only the tracked patch, not untracked files.
            "patch_marker": tracked_patch_marker,
            "tracked_dirty": tracked_dirty,
            "tracked_changes": tracked_changes,
            "tracked_patch_marker": tracked_patch_marker,
            "untracked_source_config": untracked_source_config,
            "untracked_source_config_marker": untracked_marker,
            "build_marker": build_marker,
            "checkout_path": str(checkout),
            "source_fingerprint": source_fingerprint,
        }
    except Exception as exc:
        info = {
            "commit": "unknown",
            "commit_full": "unknown",
            "last_commit_subject": "",
            "dirty": None,
            "patch_marker": None,
            "tracked_dirty": None,
            "tracked_changes": [],
            "tracked_patch_marker": None,
            "untracked_source_config": [],
            "untracked_source_config_marker": None,
            "build_marker": "unknown",
            "checkout_path": str(checkout),
            # Production containers may intentionally omit .git.  The content
            # fingerprint still identifies the exact behavior-relevant bytes.
            "source_fingerprint": source_fingerprint,
            "error": f"{type(exc).__name__}: {exc}",
        }

    _AGENT_BUILD_CACHE[cache_key] = dict(info)
    return info


def _game_agent_build_info(game_dir):
    """Load a game's recorded spawn fingerprint, or label checkout inference.

    If the artifact is absent or unreadable, the selected checkout supplies
    an inference only, not evidence of the code that ran the game.
    """
    record_path = Path(game_dir) / _AGENT_BUILD_INFO_FILE
    try:
        recorded = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(recorded, dict) and recorded.get("commit"):
            return recorded
    except (OSError, ValueError, TypeError):
        pass

    agent_code = _agent_code_from_game(game_dir)
    checkout = _agent_worktree(agent_code) or Path(_BASE_DIR)
    inferred = dict(_agent_build_info(checkout))
    expected = _AGENT_CODE_EXPECTED_COMMITS.get(agent_code)
    full_commit = str(inferred.get("commit_full") or inferred.get("commit") or "")
    inferred.update({
        "agent_code": agent_code,
        "expected_commit": expected,
        "commit_matches_label": (
            full_commit.startswith(expected) if expected and full_commit != "unknown"
            else None
        ),
        "source": "checkout_inference",
        "recorded": False,
    })
    return inferred


def _agent_generation_rows(events, delivered_messages=None, phase_dropped_messages=None):
    """Build honest Generation → Review rows from structured turn events.

    A reviewer selection and a delivered chat message are different stages.
    Output gates can suppress selected text. Correlate by turn_id when present;
    otherwise group sequential generation and review records. Use supplied
    append-file evidence to reconcile rows without explicit delivery outcomes.
    """
    def _norm(text):
        return re.sub(r"\s+", " ", (text or "").lower()).strip().rstrip("!.?")

    rows = []
    by_turn = {}
    legacy_pending = None
    delivered = Counter(_norm(text) for text in (delivered_messages or []))
    phase_dropped = Counter(_norm(text) for text in (phase_dropped_messages or []))

    def _row_for(ev):
        turn_id = ev.get("turn_id")
        if turn_id and turn_id in by_turn:
            return by_turn[turn_id]
        return legacy_pending

    def _new_row(ev, *, candidates=None):
        return {
            "turn_id": ev.get("turn_id"),
            "ts": ev.get("timestamp"),
            "count": ev.get("count", len(candidates or [])),
            "candidates": list(candidates or []),
            "verdicts": [],
            "selected": "",
            "selected_index": -1,
            "review_risk": "none",
            "outcome": "pending",
            "outcome_reason": "",
            "new_messages": 0,
            "filter_events": [],
            "prefiltered_by_text": {},
            "parse_error": False,
        }

    for ev in events:
        et = ev.get("event_type")
        if et == "generator_candidates":
            row = _new_row(ev, candidates=ev.get("candidates") or [])
            rows.append(row)
            legacy_pending = row
            if ev.get("turn_id"):
                by_turn[ev["turn_id"]] = row
            continue

        if et == "generator_parse_drop":
            row = by_turn.get(ev.get("turn_id"))
            if row is None:
                row = _new_row(ev)
                rows.append(row)
                if ev.get("turn_id"):
                    by_turn[ev["turn_id"]] = row
            row["outcome"] = "parse_drop"
            row["outcome_reason"] = "generator returned malformed structured output"
            row["raw_preview"] = ev.get("raw_preview") or ""
            row["parse_error"] = True
            legacy_pending = row
            continue

        row = _row_for(ev)
        if row is None:
            continue

        if et in {
            "state_guard_filter",
            "reviewer_filter_response_required",
            "reviewer_filter_same_dodge_mode",
            "reviewer_filter_agreement_pileup",
            "reviewer_filter_time_answer",
            "reviewer_filter_known_accusation",
        }:
            reason = et.replace("reviewer_filter_", "").replace(
                "state_guard_filter", "state_guard"
            )
            removed = ev.get("removed") or []
            explicit = []
            for item in removed if isinstance(removed, list) else []:
                if isinstance(item, dict):
                    text = item.get("text") or ""
                    item_reason = item.get("reason") or reason
                else:
                    text = str(item)
                    item_reason = reason
                if text:
                    row["prefiltered_by_text"][_norm(text)] = item_reason
                    explicit.append(text)
            row["filter_events"].append({
                "type": et,
                "reason": reason,
                "removed": explicit,
                "removed_count": (
                    len(explicit) if explicit
                    else int(ev.get("removed") or 0)
                ),
            })
        elif et == "reviewer_analysis":
            row["verdicts"] = ev.get("verdicts") or []
            if ev.get("selected_index") is not None:
                row["selected_index"] = int(ev.get("selected_index"))
        elif et == "reviewer_selected":
            row["selected"] = ev.get("selected_text") or ""
            if ev.get("selected_index") is not None:
                row["selected_index"] = int(ev.get("selected_index"))
            row["review_status"] = ev.get("review_status", "ok")
        elif et == "reviewer_drop":
            row["outcome"] = "reviewer_drop"
            row["outcome_reason"] = ", ".join(ev.get("review_reason_codes") or [])
        elif et == "stale_send_drop":
            row["outcome"] = "suppressed_stale"
            row["outcome_reason"] = ev.get("reason") or "conversation moved on"
            row["new_messages"] = int(ev.get("new_messages") or 0)
            row["selected"] = row["selected"] or ev.get("suppressed_message") or ""
        elif et == "pipeline_timeout_drop":
            row["outcome"] = "suppressed_timeout"
            row["outcome_reason"] = (
                f"pipeline exceeded {ev.get('timeout_s')}s"
                if ev.get("timeout_s") is not None else "pipeline timeout"
            )
            row["selected"] = row["selected"] or ev.get("suppressed_message") or ""
        elif et == "delivery_sent":
            row["outcome"] = "sent"
            row["outcome_reason"] = "appended to personal chat"
            row["selected"] = row["selected"] or ev.get("message") or ""
        elif et == "delivery_phase_drop":
            row["outcome"] = "phase_drop"
            row["outcome_reason"] = ev.get("reason") or "voting started"
            row["outcome_stage"] = ev.get("stage")
            row["selected"] = row["selected"] or ev.get("message") or ""
        elif et == "delivery_dedup_drop":
            row["outcome"] = "dedup_suppressed"
            row["outcome_reason"] = "duplicate of previous delivered message"
            row["selected"] = row["selected"] or ev.get("message") or ""
        elif et == "pipeline_complete":
            final = ev.get("final") or ""
            row["review_risk"] = ev.get("review_risk", "none")
            if final:
                row["selected"] = row["selected"] or final
            elif row["outcome"] == "pending":
                row["outcome"] = "dropped_unknown"

    for row in rows:
        cands = row["candidates"]
        if not (0 <= row["selected_index"] < len(cands)) and row["selected"]:
            row["selected_index"] = next(
                (i for i, c in enumerate(cands)
                 if _norm(c) == _norm(row["selected"])), -1)
        if row["outcome"] == "pending":
            selected_norm = _norm(row["selected"])
            if selected_norm and phase_dropped[selected_norm] > 0:
                phase_dropped[selected_norm] -= 1
                row["outcome"] = "phase_drop"
                row["outcome_reason"] = "voting started (reconstructed from agent log)"
            elif selected_norm and delivered[selected_norm] > 0:
                delivered[selected_norm] -= 1
                row["outcome"] = "sent"
                row["outcome_reason"] = "confirmed in AI personal chat"
            else:
                row["outcome"] = "delivery_unconfirmed"
                row["outcome_reason"] = "reviewer selected; no terminal delivery evidence"

        verdict_indices = {
            verdict.get("index") for verdict in row["verdicts"]
            if isinstance(verdict.get("index"), int)
            and 0 <= verdict.get("index") < len(cands)
        }
        statuses = []
        unresolved = []
        for index, candidate in enumerate(cands):
            if index in verdict_indices:
                statuses.append({"status": "scored", "reason": None})
                continue
            reason = row["prefiltered_by_text"].get(_norm(candidate))
            statuses.append({
                "status": "prefiltered" if reason else "verdict_missing",
                "reason": reason,
            })
            if not reason:
                unresolved.append(index)

        # Count-only filter payloads lack candidate identities. Assign their
        # reported removals to unresolved positions as a display fallback;
        # this does not establish which candidate texts the filter removed.
        count_only_events = [
            event for event in row["filter_events"]
            if event["removed_count"] and not event["removed"]
        ]
        for event in count_only_events:
            for index in list(unresolved)[:event["removed_count"]]:
                statuses[index] = {
                    "status": "prefiltered",
                    "reason": event["reason"],
                }
                unresolved.remove(index)

        row["candidate_statuses"] = statuses
        row["reviewer_output_incomplete"] = bool(unresolved and row["verdicts"])
        row["dropped"] = row["outcome"] != "sent"
    return rows


def _agent_subagents_activity(events, generations=None):
    """Per-sub-agent latest activity for the admin 'Sub-agent activity' panel — built
    entirely from the existing event stream so the admin can see each sub-agent is
    working, what it produced, and when."""
    def _last(et, pred=None):
        for ev in reversed(events):
            if ev.get("event_type") == et and (pred is None or pred(ev)):
                return ev
        return None

    def _count(et):
        return sum(1 for ev in events if ev.get("event_type") == et)

    ic_dec = _last("internet_decision_response")
    ic_search = _last("internet_search_response") or _last("internet_search_trace")
    ic_seed = _last("internet_seed")
    sm = _last("social_state_tick") or _last("social_state_updated")
    be = _last("behavior_advice_updated")
    pm = _last("persona_claims_updated") or _last("persona_monitor_tick")
    rm = _last("repetition_note_updated") or _last("repetition_monitor_tick")
    ps = _last("round_summary_tick")

    def _ts(ev):
        return ev.get("timestamp") if ev else None

    return {
        "internet_context": {
            "ran": _count("internet_tick_end"),
            "searches": _count("internet_search_response"),
            "last_decision": (ic_dec or {}).get("parsed") if ic_dec else None,
            "last_search": ((ic_search or {}).get("query") if ic_search else None),
            "seed": (ic_seed or {}).get("summary") if ic_seed else None,
            "last_ts": _ts(ic_dec) or _ts(ic_seed),
            "healthy": bool(ic_dec or ic_seed),
            # Audit: full guard visibility — every query the deterministic
            # gate refused, with its stable reason code (trivial_test /
            # computable_locally / game_meta_query / seed_refusal / ...).
            "rejected_queries": [
                {"query": ev.get("query", ""), "reason": ev.get("reason", ""),
                 "ts": ev.get("timestamp")}
                for ev in events
                if ev.get("event_type") == "internet_search_rejected"
            ][-20:],
        },
        "social_state_tracker": {
            "ran": _count("social_state_tick"),
            "parsed": (sm or {}).get("parsed") if sm else None,
            "last_ts": _ts(sm), "healthy": bool(sm),
        },
        "behavior_adapter": {
            "ran": _count("behavior_tick"),
            "advice": (be or {}).get("advice") or (be or {}).get("behavior_advice") if be else None,
            "last_ts": _ts(be), "healthy": bool(be),
        },
        "persona_monitor": {
            "ran": _count("persona_monitor_tick"),
            "claims": (pm or {}).get("self_claims") or (pm or {}).get("claims") if pm else None,
            "last_ts": _ts(pm), "healthy": bool(pm),
        },
        "repetition_monitor": {
            "ran": _count("repetition_monitor_tick"),
            "overused": (rm or {}).get("overused_words") if rm else None,
            "note": (rm or {}).get("note") or (rm or {}).get("repetition_note") if rm else None,
            "last_ts": _ts(rm), "healthy": bool(rm),
        },
        "round_summarizer": {
            "ran": _count("round_summary_tick"),
            "summaries": (ps or {}).get("round_summaries") or (ps or {}).get("summary") if ps else None,
            "last_ts": _ts(ps), "healthy": bool(ps),
        },
        # Non-delivery outcomes, kept distinct from selected text and SEND
        # decisions. Selection does not establish an append to the chat.
        "suppressed_sends": [
            {
                "type": row.get("outcome"),
                "outcome": row.get("outcome"),
                "message": row.get("selected", ""),
                "reason": row.get("outcome_reason", ""),
                "ts": row.get("ts"),
                "selected_index": row.get("selected_index"),
            }
            for row in (generations or [])
            if row.get("outcome") not in ("sent", "delivery_unconfirmed")
        ],
    }


def _scheduler_reason_breakdown(events, generations=None):
    """Count scheduler decisions from boolean, token, or prose payload values.

    Prefer explicit decision events, then terminal turn events, then pipeline
    completion or selection counts. Keep blocked polls separate from model
    WAIT decisions and distinguish SEND decisions from delivered messages.
    """
    from collections import Counter
    decisions = Counter()
    blocked = Counter()
    pipeline_completes = 0
    reviewer_selected = 0
    suppressed_sends = 0
    outcome_breakdown = Counter()
    terminal_decisions = Counter()

    def _decision_value(ev):
        for key in ("decision", "scheduler_decision", "raw_decision"):
            value = ev.get(key)
            if isinstance(value, bool):
                return "send" if value else "wait"
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if normalized in ("send", "<send>", "use", "speak", "true"):
                return "send"
            if normalized in ("wait", "<wait>", "pass", "false"):
                return "wait"
            # Also accept a decision token embedded in prose, such as decision=SEND.
            match = re.search(r"(?:decision\s*=\s*)?<?(send|wait)>?", normalized)
            if match:
                return match.group(1)
        return None

    for ev in events:
        et = ev.get("event_type")
        if et in ("scheduler_decision", "scheduler_decision_legacy"):
            decision = _decision_value(ev)
            if decision:
                decisions[decision] += 1
        elif et == "scheduler_blocked":
            reason = ev.get("reason") or ev.get("reason_code") or ev.get("gate_reason") or "?"
            blocked[str(reason)] += 1
        elif et == "pipeline_complete":
            pipeline_completes += 1
        elif et == "reviewer_selected":
            reviewer_selected += 1
        elif et == "turn_sent":
            terminal_decisions["send"] += 1
        elif et == "turn_scheduler_wait":
            terminal_decisions["wait"] += 1
        elif et in ("pipeline_timeout_drop", "stale_send_drop"):
            # pipeline_complete fires BEFORE these output gates — a suppressed
            # turn is not confirmed delivery.
            suppressed_sends += 1

    if generations is not None:
        outcome_breakdown.update(
            row.get("outcome", "delivery_unconfirmed") for row in generations
        )
        suppressed_sends = sum(
            count for outcome, count in outcome_breakdown.items()
            if outcome not in ("sent", "delivery_unconfirmed")
        )

    # Logs without structured scheduler decisions use pipeline completion.
    # A candidate-selection cycle can emit both pipeline_complete and
    # reviewer_selected. Prefer completion counts, using selection counts only
    # when completion records are absent, to avoid counting a turn twice.
    pipeline_sends = max(0, (pipeline_completes or reviewer_selected) - suppressed_sends)
    if decisions:
        effective = decisions
        decision_source = "structured_scheduler"
    elif terminal_decisions:
        effective = terminal_decisions
        decision_source = "turn_terminal"
    else:
        effective = Counter({"send": pipeline_sends})
        decision_source = "pipeline_fallback"
    sends = effective.get("send", 0)
    # scheduler_blocked fires on every cheap guard poll (~1/s), not per
    # scheduler-model decision.
    # Expose it honestly as blocked polls, not as WAIT decisions.
    blocked_polls = sum(blocked.values())
    waits = effective.get("wait", 0)
    total = sends + waits
    return {
        "send": sends, "wait": waits or None,
        "blocked_polls": blocked_polls,
        "suppressed_sends": suppressed_sends,
        # SEND decisions are decisions; delivery is what reached the chat.
        # Output-gate suppression can prevent a SEND decision reaching the chat.
        "delivered": (
            outcome_breakdown.get("sent", 0)
            if generations is not None else max(0, sends - suppressed_sends)
        ),
        "send_rate_pct": round(100 * sends / total, 1) if total else None,
        "blocked_reasons": dict(blocked.most_common()),
        "outcome_breakdown": dict(outcome_breakdown),
        "decision_source": decision_source,
    }


@app.route('/api/admin/ai_state')
def admin_ai_state():
    """Return structured AI state + recent cycles for the live admin dashboard."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    ai_name = get_llm_player_name(game_dir)
    if not ai_name:
        return jsonify({"cycles": [], "current": None, "current_state": "idle",
                        "total_cycles": 0, "ai_name": None})

    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
    if not log_file.exists():
        return jsonify({"cycles": [], "current": None, "current_state": "idle",
                        "total_cycles": 0, "ai_name": ai_name})

    try:
        log_text = log_file.read_text(encoding='utf-8', errors='replace')
        result = _compute_ai_cycles(log_text)
        result["ai_name"] = ai_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/ai_metrics')
def admin_ai_metrics():
    """Return structured AI performance metrics parsed from the log file."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    game_dir = resolve_game_dir(game_id)
    ai_name = get_llm_player_name(game_dir)
    if not ai_name:
        return jsonify({"error": "No AI player found", "messages": [], "aggregate": {}})

    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
    if not log_file.exists():
        return jsonify({"messages": [], "aggregate": {}, "ai_name": ai_name})

    try:
        log_text = log_file.read_text(encoding='utf-8', errors='replace')
        result = _compute_ai_metrics(
            log_text,
            discussion_start_hms=_read_discussion_start_hms(game_dir),
            delivered_messages=_read_ai_delivered_messages(game_dir, ai_name),
        )
        result["ai_name"] = ai_name
        # Backfill missing text-log counts from game_events: pipeline_complete
        # and scheduler_blocked are the existing footer fallback sources.
        agg = result.get("aggregate", {})
        if agg and not agg.get("send_decisions") and not agg.get("wait_decisions"):
            ev_file = game_file(game_dir, "game_events.jsonl")
            if ev_file.exists():
                sends = waits = 0
                for ln in ev_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        et = json.loads(ln).get("event_type")
                    except json.JSONDecodeError:
                        continue
                    if et == "pipeline_complete":
                        sends += 1
                    elif et == "scheduler_blocked":
                        waits += 1
                if sends or waits:
                    agg["send_decisions"] = sends
                    agg["wait_decisions"] = waits
                    total = sends + waits
                    agg["send_rate_pct"] = round(100 * sends / total, 1) if total else 0

        # Report the source of timing measurements without asserting delivery.
        result["orchestration"] = _orchestration_provenance(game_dir)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _orchestration_provenance(game_dir):
    """Read pipeline timing metadata and explicit append observations."""
    info = {"effective_mode": None, "requested_mode": None, "is_active_run": False,
            "active_status_operational": None, "timing_source": "pipeline"}
    ev_file = game_file(game_dir, "game_events.jsonl")
    if not ev_file.exists():
        return info
    try:
        for ln in ev_file.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            et = e.get("event_type")
            if et == "orchestration_mode":
                info["effective_mode"] = e.get("effective_orchestration_mode")
                info["requested_mode"] = e.get("requested_orchestration_mode")
                st = e.get("active_status") or {}
                info["active_status_operational"] = st.get("operational")
            elif et == "turn_sent":
                info["is_active_run"] = True
    except Exception:
        return info
    info["timing_source"] = "active" if (info["is_active_run"]
                                            or info["effective_mode"] == "active") else "pipeline"
    return info


def _interface_log_issues(game_dir):
    """Read interface-log errors, including initialization failures that can
    occur before a player operation log exists, such as unavailable models,
    invalid configuration, and import errors."""
    issues = []
    try:
        iface_log = game_file(game_dir, "llm_interface_log.txt")
        if not iface_log.exists():
            return issues
        lines = iface_log.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        return issues
    tb_start = None
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            tb_start = i
    if tb_start is not None:
        block = "\n".join(lines[tb_start:tb_start + 25])
        issues.append({
            "severity": "critical",
            "title": "LLM interface crashed (traceback in llm_interface_log)",
            "detail": block[:1200],
            "time": "",
        })
    else:
        for line in lines[-200:]:
            if "AGENT CHECKOUT PRE-FLIGHT FAILED" in line:
                issues.append({
                    "severity": "critical",
                    "title": "Frozen agent checkout verification failed",
                    "detail": line.strip()[:600],
                    "time": "",
                })
                return issues
        # No traceback — still surface hard API failures (404 model, auth).
        for line in lines[-200:]:
            low = line.lower()
            if ("was not found" in low or "not_found" in low or "404" in low
                    or "permission denied" in low or "unauthenticated" in low):
                issues.append({
                    "severity": "critical",
                    "title": "LLM interface API failure",
                    "detail": line.strip()[:400],
                    "time": "",
                })
                break
    return issues


@app.route('/api/admin/ai_issues')
def admin_ai_issues():
    """Detect live issues from the AI log: errors, crashes, stuck states."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"issues": [], "summary": {"critical": 0, "warning": 0, "info": 0}})

    game_dir = resolve_game_dir(game_id)
    ai_name = get_llm_player_name(game_dir)
    if not ai_name:
        return jsonify({"issues": [], "summary": {"critical": 0, "warning": 0, "info": 0}})

    def _respond(issue_list):
        summary = {
            "critical": sum(1 for i in issue_list if i["severity"] == "critical"),
            "warning": sum(1 for i in issue_list if i["severity"] == "warning"),
            "info": sum(1 for i in issue_list if i["severity"] == "info"),
        }
        return jsonify({"issues": issue_list, "summary": summary})

    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
    if not log_file.exists():
        # Init-time crashes never create the player log — check the
        # interface log before declaring "no issues".
        return _respond(_interface_log_issues(game_dir))

    try:
        log_text = log_file.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return _respond(_interface_log_issues(game_dir))

    entries = _parse_log_entries(log_text)
    issues = []

    last_heartbeat_ts = None
    activity_since_heartbeat = False
    for e in entries:
        op = e["operation"]
        ts = _time_to_seconds(e["time"])

        if op == "AGENT_FATAL_ERROR":
            issues.append({"severity": "critical", "title": "Agent crashed", "detail": e["content"][:300], "time": e["time"]})
        elif op == "AGENT_ERROR":
            if e["content"].startswith("RATE_LIMIT:"):
                issues.append({"severity": "critical", "title": "Rate limited by API", "detail": e["content"][:300], "time": e["time"]})
            else:
                issues.append({"severity": "warning", "title": "Agent error", "detail": e["content"][:200], "time": e["time"]})
        elif "error generating" in op.lower():
            issues.append({"severity": "warning", "title": f"API error: {op}", "detail": e["content"][:200], "time": e["time"]})
        elif op == "DEDUP_BLOCKED":
            issues.append({"severity": "info", "title": "Duplicate message blocked", "detail": e["content"][:120], "time": e["time"]})
        elif op == "AGENT_HEARTBEAT":
            if (last_heartbeat_ts is not None and (ts - last_heartbeat_ts) > 30
                    and not activity_since_heartbeat):
                gap = ts - last_heartbeat_ts
                issues.append({"severity": "warning", "title": f"Agent unresponsive for {gap:.0f}s", "detail": f"Heartbeat gap between {e['time']} and previous", "time": e["time"]})
            last_heartbeat_ts = ts
            activity_since_heartbeat = False
        else:
            activity_since_heartbeat = True

    # Check if LLM process actually died UNEXPECTEDLY.
    # If the game already ended (who_wins.txt written), the LLM is *supposed* to
    # exit with code 0 — flagging that as "critical" is a false alarm and was
    # confusing the Issues panel after every normal game.
    procs = running_games.get(game_id, {})
    llm_proc = procs.get("llm")
    if llm_proc and llm_proc.poll() is not None:
        exit_code = llm_proc.poll()
        game_already_over = is_game_over(game_dir)
        # Read crash log tail so the admin sees the actual traceback
        crash_detail = "The AI player process has terminated unexpectedly."
        crash_log_path = game_file(game_dir, "llm_interface_log.txt")
        if crash_log_path.exists():
            try:
                crash_lines = crash_log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                tail = crash_lines[-30:] if len(crash_lines) > 30 else crash_lines
                crash_detail = "\n".join(tail)
            except Exception:
                pass
        if not game_already_over:
            issues.append({"severity": "critical", "title": f"LLM process exited (code {exit_code})", "detail": crash_detail, "time": ""})
        elif exit_code != 0:
            # Game ended but AI exited with non-zero — still worth flagging.
            issues.append({"severity": "warning", "title": f"LLM exited with code {exit_code})", "detail": crash_detail, "time": ""})

    # Interface-log tracebacks are critical even when the player log exists
    # (e.g. a crash after partial init). Skip if the process check above
    # already surfaced the same tail.
    if not any("LLM process exited" in i["title"] for i in issues):
        issues.extend(_interface_log_issues(game_dir))

    return _respond(issues)


@app.route('/api/admin/reviewer_detail')
def admin_reviewer_detail():
    """Return per-message reviewer before/after data from LIVE_REVIEW_SUMMARY log entries."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"reviews": []})

    game_dir = resolve_game_dir(game_id)
    ai_name = get_llm_player_name(game_dir)
    if not ai_name:
        return jsonify({"reviews": []})

    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai_name)
    if not log_file.exists():
        return jsonify({"reviews": []})

    try:
        log_text = log_file.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return jsonify({"reviews": []})

    entries = _parse_log_entries(log_text)
    # Build set of AI's actually-sent messages from public chat to detect
    # cycles where the reviewer accepted a message but it never got sent
    # (e.g. phase ended before the agent could submit).
    sent_msgs = set()
    chat_file = game_dir / PUBLIC_DAYTIME_CHAT_FILE
    if chat_file.exists():
        for line in chat_file.read_text(encoding='utf-8', errors='replace').splitlines():
            # Format: "[HH:MM:SS] Name: text"
            m = re.match(r'^\[\d{2}:\d{2}:\d{2}\]\s*([^:]+):\s*(.*)$', line)
            if m and m.group(1).strip() == ai_name:
                sent_msgs.add(m.group(2).strip())

    reviews = []
    for e in entries:
        if e["operation"] != "LIVE_REVIEW_SUMMARY":
            continue
        content = e["content"]
        # Use line-anchored greedy patterns with MULTILINE so embedded apostrophes
        # (e.g. "i'm winter") don't truncate the match at the first single quote.
        raw_m = re.search(r"^RAW:\s*'(.*)'\s*$", content, re.MULTILINE)
        final_m = re.search(r"^FINAL \((modified|unchanged)\):\s*'(.*)'\s*$", content, re.MULTILINE)
        score_m = re.search(r"Final score:\s*(\d+)/10", content)
        flags_m = re.search(r"red_flags:\s*\[([^\]]*)\]", content)
        # Parse per-step lines: "  Step N: score=X, action=Y, msg='...'"
        steps = []
        for sm in re.finditer(
            r"^\s*Step\s+(\d+):\s*score=(-?\d+),\s*action=([^,]+?)(?:\([^)]*\))?\s*(?::\s*'([^']*)')?,?\s*msg='(.*)'\s*$",
            content, re.MULTILINE,
        ):
            steps.append({
                "step": int(sm.group(1)),
                "score": int(sm.group(2)),
                "action": sm.group(3).strip(),
                "fix": sm.group(4) or None,
                "message": sm.group(5),
            })
        final_text = final_m.group(2) if final_m else ""
        # Empty final_text means the reviewer dropped the turn — nothing
        # was written to the public chat. Mark sent=False explicitly so
        # the "Approved as-is" / Reviewer Activity rendering doesn't
        # claim a phantom message reached players.
        sent = bool(final_text and final_text.strip() in sent_msgs)
        reviews.append({
            "time": e["time"],
            "raw_message": raw_m.group(1) if raw_m else "",
            "final_message": final_text,
            "was_modified": final_m.group(1) == "modified" if final_m else False,
            "score": int(score_m.group(1)) if score_m else None,
            "red_flags": [f.strip().strip("'\"") for f in flags_m.group(1).split(",") if f.strip()] if flags_m else [],
            "sent": sent,
            "discard_reason": None if sent else "phase ended before send",
            "steps": steps,
        })
    return jsonify({"reviews": reviews})


# ── Per-message Journey: full pipeline trace for a single sent message ──

def _hms_to_sec(ts):
    """Parse 'HH:MM:SS' → seconds since midnight (None on bad input)."""
    if not ts:
        return None
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None


@app.route('/api/admin/message_journey')
def admin_message_journey():
    """Return the full pipeline trace for a single sent AI message.

    Joins three sources keyed by `(player, time)`:
      • llm_log_<player>.txt — scheduler/generator prompts + decisions
        (the most recent block of each kind ≤ 60s before `time`).
      • game_events.jsonl — `review_cycle` event with full per-attempt
        candidates, situation analysis, expected human response.
      • _compute_ai_metrics row — latency breakdown + word count.

    Query params:
      game_id : 4-digit id (required)
      player  : AI codename (defaults to the first AI in config)
      time    : HH:MM:SS of the sent message (required)

    Returns one consolidated JSON so the admin Journey modal can render
    every stage in a single fetch. All timestamps are best-effort matches
    within ±60s; if no source is found for a stage, that stage is null
    and the UI hides it.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    target_time = (request.args.get('time') or '').strip()
    player_q = (request.args.get('player') or '').strip()
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    target_sec = _hms_to_sec(target_time)
    if target_sec is None:
        return jsonify({"error": "invalid time (expected HH:MM:SS)"}), 400

    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "game not found"}), 404

    cfg = {}
    cfg_file = game_dir / GAME_CONFIG_FILE
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    ai_names = [p.get("name") for p in cfg.get("players", []) if p.get("is_llm") or p.get("is_ai")]
    player = player_q if player_q in ai_names else (ai_names[0] if ai_names else None)
    if not player:
        return jsonify({"error": "no AI player in this game"}), 404

    # ── 1. llm_log_<player>.txt: find latest scheduler + generator blocks ──
    scheduler_block = None      # {time, prompt, decision, latency_s}
    generator_block = None      # {time, prompt, raw_output, latency_s}
    log_file = game_dir / LLM_LOG_FILE_FORMAT.format(player)
    if log_file.exists():
        try:
            text = log_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            text = ""
        # Walk events newest→oldest and stop when both are filled.
        events = []
        for blk in text.split("# NEW LOG"):
            blk = blk.strip()
            if not blk:
                continue
            tm = re.search(r"##\s*TIME:\s*(.+)", blk)
            om = re.search(r"##\s*OPERATION:\s*(.+)", blk)
            cm = re.search(r"##\s*CONTENT:\s*(.*)", blk, re.DOTALL)
            if not tm or not om:
                continue
            events.append({
                "time": tm.group(1).strip(),
                "op": om.group(1).strip(),
                "content": (cm.group(1).strip() if cm else "")[:6000],
            })
        # Sort chronologically; we want the latest occurrence of each op
        # whose timestamp is ≤ target_time + 5s (the UNLOCKED log can fire
        # a couple of seconds *after* the chat write timestamp).
        sched_prompt = sched_decision = sched_lat = None
        sched_time = None
        gen_prompt = gen_raw = gen_lat = None
        gen_time = None
        for e in events:
            esec = _hms_to_sec(e["time"])
            if esec is None or esec > target_sec + 5 or esec < target_sec - 120:
                continue
            op = e["op"]
            if op == "prompt in should_generate_message":
                sched_prompt = e["content"]; sched_time = e["time"]
            elif op == "decision in should_generate_message":
                sched_decision = e["content"]
            elif op == "scheduler_latency":
                m = re.search(r"([0-9.]+)", e["content"])
                if m:
                    try: sched_lat = float(m.group(1))
                    except ValueError: pass
            elif op in ("prompt in generate_message", "prompt in _generate_raw_message"):
                gen_prompt = e["content"]; gen_time = e["time"]
            elif op == "generator_latency":
                m = re.search(r"([0-9.]+)", e["content"])
                if m:
                    try: gen_lat = float(m.group(1))
                    except ValueError: pass
            elif op == "generated_raw":
                gen_raw = e["content"]
        if sched_prompt or sched_decision is not None:
            scheduler_block = {
                "time": sched_time,
                "prompt": sched_prompt,
                "decision": sched_decision,
                "latency_s": sched_lat,
            }
        if gen_prompt or gen_raw:
            generator_block = {
                "time": gen_time,
                "prompt": gen_prompt,
                "raw_output": gen_raw,
                "latency_s": gen_lat,
            }

    # ── 2. game_events.jsonl: matching review_cycle event ──
    review_cycle = None
    events_file = game_file(game_dir, "game_events.jsonl")
    if events_file.exists():
        best_dt = None
        try:
            for ln in events_file.read_text(encoding='utf-8', errors='replace').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if ev.get("event_type") != "review_cycle":
                    continue
                if ev.get("player") and ev["player"] != player:
                    continue
                esec = _hms_to_sec(ev.get("timestamp"))
                if esec is None:
                    continue
                dt = abs(esec - target_sec)
                if dt > 30:  # widen a bit — review_cycle is logged at end
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    review_cycle = ev
        except Exception:
            pass

    # ── 3. Per-message metrics row (latency breakdown) ──
    metrics_row = None
    if log_file.exists():
        try:
            log_text_for_metrics = log_file.read_text(encoding='utf-8', errors='replace')
            metrics = _compute_ai_metrics(log_text_for_metrics)
            for m in metrics.get("messages", []):
                if m.get("time") == target_time:
                    metrics_row = m
                    break
            # Fall back to closest by time if exact missed.
            if metrics_row is None:
                best = None
                for m in metrics.get("messages", []):
                    msec = _hms_to_sec(m.get("time"))
                    if msec is None:
                        continue
                    dt = abs(msec - target_sec)
                    if best is None or dt < best[0]:
                        best = (dt, m)
                if best and best[0] <= 5:
                    metrics_row = best[1]
        except Exception:
            pass

    # ── 4. Recent chat context (10 lines before target) ──
    context_chat = []
    chat_file = game_file(game_dir, "chat_displayed_full.txt")
    if not chat_file.exists():
        chat_file = game_file(game_dir, "public_daytime_chat.txt")
    if chat_file.exists():
        try:
            lines = chat_file.read_text(encoding='utf-8', errors='replace').splitlines()
            for ln in lines:
                m = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", ln)
                if not m:
                    continue
                lsec = _hms_to_sec(m.group(1))
                if lsec is None or lsec > target_sec:
                    continue
                context_chat.append(ln)
            context_chat = context_chat[-10:]
        except Exception:
            pass

    # ── 5. Enrichment from game_events.jsonl ──
    # Rebuild missing text-log detail from structured pipeline_complete,
    # generator_input, reviewer_trace, turn_context, and snapshot_observed events.
    # Join the send
    # by timestamp (±5s) and the shared-memory snapshot by turn_index.
    shared_memory = None
    turn_context = None
    if events_file.exists():
        try:
            evs = []
            for ln in events_file.read_text(encoding='utf-8', errors='replace').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if ev.get("player") and ev["player"] != player:
                    continue
                evs.append(ev)

            def _closest(etype, max_dt=6):
                best, best_dt = None, None
                for ev in evs:
                    if ev.get("event_type") != etype:
                        continue
                    esec = _hms_to_sec(ev.get("timestamp"))
                    if esec is None:
                        continue
                    dt = abs(esec - target_sec)
                    if dt > max_dt:
                        continue
                    if best_dt is None or dt < best_dt:
                        best, best_dt = ev, dt
                return best

            pipe = _closest("pipeline_complete")
            gen_in = _closest("generator_input")
            rtrace = _closest("reviewer_trace")
            tctx = _closest("turn_context")

            # Scheduler — only if the llm_log path didn't already find one.
            if scheduler_block is None and pipe is not None:
                dec = (pipe.get("scheduler_decision") or "").upper()
                reason = pipe.get("send_reason") or ""
                why = pipe.get("why_speaking") or ""
                scheduler_block = {
                    "time": pipe.get("timestamp"),
                    "prompt": (f"why_speaking: {why}\nstyle_notes: {pipe.get('style_notes','')}"
                               if why else None),
                    "decision": f"{dec}" + (f" ({reason})" if reason else ""),
                    "latency_s": None,
                }
            # Generator
            if generator_block is None and (pipe is not None or gen_in is not None):
                generator_block = {
                    "time": (gen_in or pipe).get("timestamp"),
                    "prompt": (gen_in or {}).get("task_prompt")
                              or (gen_in or {}).get("history_tail"),
                    "raw_output": (pipe or {}).get("draft"),
                    "latency_s": None,
                }
            # Reviewer — map reviewer_trace.attempts → candidates.
            if review_cycle is None and rtrace is not None:
                attempts = rtrace.get("attempts") or []
                cands = [{
                    "message": a.get("message"),
                    "score": a.get("score"),
                    "red_flags": a.get("flags") or [],
                    "fix": a.get("fix"),
                    "action": ("fix" if a.get("fix") else "keep"),
                } for a in attempts]
                action = rtrace.get("action") or ""
                degraded = any("API error" in str(f) or "UNAVAILABLE" in str(f)
                               or "PARSE FAILED" in str(f)
                               for a in attempts for f in (a.get("flags") or []))
                review_cycle = {
                    "final_score": rtrace.get("final_score"),
                    "stop_reason": rtrace.get("stop_reason"),
                    "was_modified": action == "fixed",
                    "dropped": action == "dropped",
                    "degraded": degraded,
                    "candidates": cands,
                    "reviewer_attempts": len(cands),
                }
            # Shared memory at send-time: join snapshot_read_full by turn_index.
            if tctx is not None:
                turn_context = {
                    "turn_index": tctx.get("turn_index"),
                    "current_info": tctx.get("current_info"),
                    "self_claims": tctx.get("self_claims"),
                    "social_state": tctx.get("social_state"),
                    "behavior_advice": tctx.get("behavior_advice"),
                    "pressure_type": tctx.get("pressure_type"),
                    "overused_words": tctx.get("overused_words"),
                }
                ti = tctx.get("turn_index")
                for ev in evs:
                    if (ev.get("event_type") == "snapshot_observed"
                            and ev.get("turn_index") == ti):
                        shared_memory = {k: ev.get(k) for k in (
                            "turn_index", "current_info", "hedge_advice",
                            "round_summaries", "self_claims", "social_state",
                            "behavior_advice", "entry_count")}
                        break
        except Exception:
            pass

    return jsonify({
        "game_id": game_id,
        "player": player,
        "time": target_time,
        "metrics": metrics_row,
        "scheduler": scheduler_block,
        "generator": generator_block,
        "review_cycle": review_cycle,
        "context_chat": context_chat,
        "shared_memory": shared_memory,
        "turn_context": turn_context,
    })


@app.route('/api/admin/list_games')
def admin_list_games():
    """List all game directories with basic metadata."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    parent = Path(DIRS_PREFIX)
    if not parent.exists():
        return jsonify([])

    games = []
    for _cat, _ver, d in iter_all_games(include_simulations=False):
        if not d.name.isdigit():
            continue
        cfg_file = d / GAME_CONFIG_FILE
        winner_file = d / WHO_WINS_FILE
        winner = winner_file.read_text(encoding='utf-8').strip() if winner_file.exists() else ""
        num_players = 0
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding='utf-8') as f:
                    c = json.load(f)
                num_players = len(c.get('players', []))
            except Exception:
                pass
        is_running = d.name in running_games
        games.append({
            "game_id": d.name,
            "num_players": num_players,
            "game_over": bool(winner),
            "winner": winner,
            "is_running": is_running,
        })

    return jsonify(games)


@app.route('/api/admin/survey_results')
def admin_survey_results():
    """Return parsed survey responses for a game."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id', '')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    game_dir = resolve_game_dir(game_id)
    if not game_dir.is_dir():
        return jsonify({"error": "Game not found"}), 404

    ai_player = get_llm_player_name(game_dir)
    responses = []
    # Find all survey files (surveys/ subfolder, with flat fallback)
    _survey_dir = game_dir / "surveys"
    _survey_files = sorted(_survey_dir.glob("*_survey.txt")) if _survey_dir.is_dir() else sorted(game_dir.glob("*_survey.txt"))
    for sf in _survey_files:
        player_name = sf.name.replace("_survey.txt", "")
        try:
            text = sf.read_text(encoding="utf-8").strip()
            if not text:
                continue
            lines = text.splitlines()
            entry = {"player": player_name}
            reading_comments = False
            comment_lines = []
            for line in lines:
                if reading_comments:
                    comment_lines.append(line)
                    continue
                if line.strip() == SURVEY_COMMENTS_TITLE or line.strip() == "Comments:":
                    reading_comments = True
                    continue
                if METRIC_NAME_AND_SCORE_DELIMITER in line:
                    parts = line.split(METRIC_NAME_AND_SCORE_DELIMITER, 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key == "TIME":
                        entry["time"] = val
                    elif key == HUMAN_SIMILARITY:
                        entry["human_similarity"] = int(val)
                    elif key == TIMING:
                        entry["timing"] = int(val)
                    elif key == RELEVANCE:
                        entry["relevance"] = int(val)
                    elif key == ENJOYMENT:
                        entry["enjoyment"] = int(val)
                    elif key == "LLM_CONFIDENCE":
                        entry["llm_confidence"] = int(val)
            if comment_lines:
                entry["comments"] = "\n".join(comment_lines).strip()
            # Only include if at least one rating was parsed
            if any(k in entry for k in ("human_similarity", "timing", "relevance")):
                responses.append(entry)
        except Exception:
            continue

    summary = None
    if responses:
        hs = [r["human_similarity"] for r in responses if "human_similarity" in r]
        tm = [r["timing"] for r in responses if "timing" in r]
        rv = [r["relevance"] for r in responses if "relevance" in r]
        en = [r["enjoyment"] for r in responses if "enjoyment" in r]
        all_scores = hs + tm + rv
        # Count total human players from config
        total_humans = 0
        cfg_file = game_dir / GAME_CONFIG_FILE
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding='utf-8') as f:
                    c = json.load(f)
                total_humans = sum(1 for p in c.get('players', []) if not p.get('is_llm', False))
            except Exception:
                pass
        summary = {
            "total_responses": len(responses),
            "total_human_players": total_humans,
            "avg_human_similarity": round(sum(hs) / len(hs), 1) if hs else None,
            "avg_timing": round(sum(tm) / len(tm), 1) if tm else None,
            "avg_relevance": round(sum(rv) / len(rv), 1) if rv else None,
            "avg_enjoyment": round(sum(en) / len(en), 1) if en else None,
            "overall_avg": round(sum(all_scores) / len(all_scores), 1) if all_scores else None,
        }

    return jsonify({"responses": responses, "summary": summary, "ai_player": ai_player})




@app.route('/spectator')
def spectator():
    """Read-only live view of a game — no auth required.

    Shows: chat messages, player code names, phase, timer.
    Does NOT expose: real names, who is AI, vote tallies, LLM logs.
    """
    game_id = request.args.get('game_id', '').strip()
    if not _valid_game_id(game_id):
        return "<h2>Missing or invalid game_id — use /spectator?game_id=XXXX</h2>", 400
    game_dir = resolve_game_dir(game_id)
    if not game_dir.is_dir():
        return f"<h2>Game {game_id} not found</h2>", 404
    return render_template('spectator.html', game_id=game_id)


@app.route('/api/spectator/status')
def spectator_status():
    """Public polling endpoint for the spectator view.

    Returns: phase, code-name player list, remaining players, chat messages,
    game-over status. Deliberately omits real names, who is AI, vote tallies.
    """
    game_id = request.args.get('game_id', '').strip()
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    game_dir = resolve_game_dir(game_id)
    if not game_dir.is_dir():
        return jsonify({"error": "game not found"}), 404

    # Phase
    phase_file = game_dir / PHASE_STATUS_FILE
    phase = phase_file.read_text(encoding='utf-8').strip() if phase_file.exists() else "unknown"

    # All players (code names only — no real names)
    all_players = []
    pf = game_dir / PLAYER_NAMES_FILE
    if pf.exists():
        all_players = [n.strip() for n in pf.read_text(encoding='utf-8').splitlines() if n.strip()]

    # Remaining players
    remaining = list(all_players)
    rf = game_dir / REMAINING_PLAYERS_FILE
    if rf.exists():
        remaining = [n.strip() for n in rf.read_text(encoding='utf-8').splitlines() if n.strip()]

    # Game over / winner
    wf = game_dir / WHO_WINS_FILE
    winner = wf.read_text(encoding='utf-8').strip() if wf.exists() else ""

    # Public chat messages (latest 200)
    chat_file = game_dir / PUBLIC_DAYTIME_CHAT_FILE
    messages = []
    if chat_file.exists():
        lines = chat_file.read_text(encoding='utf-8').splitlines()
        messages = [l for l in lines if l.strip()][-200:]

    # Discussion start time (for timer)
    dst_file = game_dir / DISCUSSION_START_TIME_FILE
    discussion_start = None
    if dst_file.exists():
        try:
            discussion_start = float(dst_file.read_text(encoding='utf-8').strip())
        except (ValueError, OSError):
            pass

    # Conversation topic
    topic_file = game_dir / CONVERSATION_TOPIC_FILE
    topic = topic_file.read_text(encoding='utf-8').strip() if topic_file.exists() else ""

    return jsonify({
        "game_id": game_id,
        "phase": phase,
        "all_players": all_players,
        "remaining_players": remaining,
        "winner": winner,
        "game_over": bool(winner),
        "messages": messages,
        "discussion_start": discussion_start,
        "topic": topic,
        # CB9: server clock for offset-correction, so spectator timer matches
        # the game UI timer across clients with skewed local clocks.
        "server_time_ms": int(time.time() * 1000),
    })




@app.route('/api/admin/config_schema')
def admin_config_schema():
    """Return grouped LLM config schema for admin UI rendering.

    Schema is public metadata (field names, types, descriptions) — no auth required.
    Surfaces import errors as JSON so the UI can show the real cause instead of a
    silent 500 (e.g. ModuleNotFoundError when running inside Docker).
    """
    try:
        # Defensive: ensure the directory holding web_player.py is on sys.path so
        # `core.config_schema` imports cleanly even when gunicorn's
        # cwd does not match /app or sys.path is minimal.
        import sys, os
        _proj_root = os.path.dirname(os.path.abspath(__file__))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)
        from core.config_schema import schema_by_group
        return jsonify(schema_by_group())
    except Exception as e:
        import traceback, os, sys
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            files_here = sorted(f for f in os.listdir(here) if f.endswith('.py'))
        except Exception:
            here, files_here = '?', []
        app.logger.error("config_schema endpoint failed: %s\n%s", e, traceback.format_exc())
        return jsonify({
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc().splitlines()[-6:],
            "diagnostic": {
                "cwd": os.getcwd(),
                "web_player_dir": here,
                "sys_path_head": sys.path[:5],
                "config_schema_present": os.path.isfile(os.path.join(here, 'core', 'config_schema.py')),
                "py_files_in_web_player_dir_count": len(files_here),
                "hint": ("core/config_schema.py is MISSING from the application. "
                         "Restore the complete local release before starting the server."
                         if not os.path.isfile(os.path.join(here, 'core', 'config_schema.py')) else
                         "core/config_schema.py is present — investigate sys.path / import errors above."),
            },
        }), 500


@app.route('/api/admin/game_health')
def admin_game_health():
    """Return per-process health for a running game (CPU, RSS, last log line, idle seconds)."""
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400

    if game_id not in running_games:
        return jsonify({"error": "game not running"}), 404

    procs = running_games[game_id]
    result = {}
    for key in ('manager', 'llm'):
        proc = procs.get(key)
        log_path = procs.get(f'{key}_log')
        info = {"pid": None, "alive": False, "cpu_pct": None, "rss_mb": None,
                "last_log_line": None, "seconds_since_last_message": None}
        if proc:
            info["pid"] = proc.pid
            info["alive"] = proc.poll() is None
            try:
                import psutil
                ps = psutil.Process(proc.pid)
                info["cpu_pct"] = ps.cpu_percent(interval=0.1)
                info["rss_mb"] = round(ps.memory_info().rss / 1024 / 1024, 1)
            except Exception:
                pass
        if log_path and Path(log_path).exists():
            try:
                lines = Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines()
                info["last_log_line"] = lines[-1] if lines else ""
            except Exception:
                pass
        # Time since last chat message (from public daytime chat file)
        try:
            chat_file = resolve_game_dir(game_id) / PUBLIC_DAYTIME_CHAT_FILE
            if chat_file.exists():
                mtime = chat_file.stat().st_mtime
                info["seconds_since_last_message"] = int(time.time() - mtime)
        except Exception:
            pass
        result[key] = info
    result["started_at"] = procs.get("started_at")
    # Running cost from cost_log.jsonl
    try:
        cost_file = game_file(resolve_game_dir(game_id), "cost_log.jsonl")
        total_cost = 0.0
        if cost_file.exists():
            for line in cost_file.read_text(encoding='utf-8').splitlines():
                try:
                    entry = json.loads(line)
                    total_cost += entry.get("estimated_cost_usd", 0.0)
                except Exception:
                    pass
        result["running_cost_usd"] = round(total_cost, 5)
    except Exception:
        result["running_cost_usd"] = None
    return jsonify(result)


# Filenames (or glob suffixes) that any player may read — game state only, no internals.
_PLAYER_READABLE_FILES = {
    PUBLIC_DAYTIME_CHAT_FILE,       # public_daytime_chat.txt
    PUBLIC_MANAGER_CHAT_FILE,       # public_manager_chat.txt
    GAME_START_TIME_FILE,           # game_start_time.txt
    DISCUSSION_START_TIME_FILE,     # discussion_start_time.txt
    PHASE_STATUS_FILE,              # phase_status.txt
    CONVERSATION_TOPIC_FILE,        # conversation_topic.txt
    "game_events.jsonl",            # unified reviewer/scheduler event log (public, game-state)
    # Public game-state files available in the Raw Files drawer.
    "all_messages.txt",             # merged chat transcript
    "player_names.txt",             # display names (already seen in chat)
    "remaining_players.txt",        # current survivors
    "ai_elimination_info.txt",      # post-game stat
    "who_wins.txt",                 # post-game outcome
    "notes.txt",                    # reserved notes file
    "topic.txt",                    # alt topic file name
}
# Suffix patterns that are player-readable (personalised files are OK for the owner,
# but we can't verify ownership here — expose only clearly non-sensitive ones).
_PLAYER_READABLE_SUFFIXES = (
    "_chat.txt",    # {name}_chat.txt — personal chat history
    "_status.txt",  # {name}_status.txt — personal voted-out status
    "_vote.txt",    # {name}_vote.txt — voting history (public after round ends)
)


def _is_player_readable(filename: str) -> bool:
    """Return True if this filename is safe to serve to unauthenticated players."""
    return filename in _PLAYER_READABLE_FILES or any(
        filename.endswith(sfx) for sfx in _PLAYER_READABLE_SUFFIXES
    )


@app.route('/api/file')
def serve_file():
    """Serve game files. Player-readable files are public; sensitive files require admin auth."""
    path = request.args.get('path', '')
    # Resolve games/<id>/<file> requests through the game's runtime directory.
    parts = path.replace('\\', '/').split('/')
    if len(parts) >= 3 and parts[0] == 'games' and _valid_game_id(parts[1]):
        gid = parts[1]
        rest = parts[2:]
        gd = resolve_game_dir(gid)
        # Resolve a bare filename through game_file's category subdirectory
        # lookup and direct-path fallback. Explicit subpaths remain relative
        # to the resolved game directory.
        if len(rest) == 1:
            abs_path = os.path.abspath(str(game_file(gd, rest[0])))
        else:
            abs_path = os.path.abspath(str(gd / Path(*rest)))
    else:
        abs_path = os.path.abspath(os.path.join(_BASE_DIR, path))
    allowed_base = os.path.abspath(os.path.join(_BASE_DIR, 'games'))
    if not abs_path.startswith(allowed_base + os.sep) and abs_path != allowed_base:
        return "Forbidden", 403
    if not os.path.isfile(abs_path):
        return "Not found", 404

    filename = os.path.basename(abs_path)
    if not _is_player_readable(filename):
        auth_err = _check_admin_auth()
        if auth_err:
            return auth_err

    try:
        with open(abs_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return f"Error reading file: {e}", 500


@app.route('/api/list_files')
def list_files():
    """List files inside a game directory so the dashboard can build a raw-files browser.

    Only games/<id>/ is allowed. Filenames are filtered through the same
    player-readable allowlist used by /api/file — admin auth is required to
    list the rest.
    """
    path = request.args.get('path', '')
    parts = path.replace('\\', '/').split('/')
    if len(parts) >= 2 and parts[0] == 'games' and _valid_game_id(parts[1]):
        gid = parts[1]
        rest = parts[2:]
        gd = resolve_game_dir(gid)
        abs_path = os.path.abspath(str(gd / Path(*rest))) if rest else os.path.abspath(str(gd))
    else:
        abs_path = os.path.abspath(os.path.join(_BASE_DIR, path))
    allowed_base = os.path.abspath(os.path.join(_BASE_DIR, 'games'))
    if not abs_path.startswith(allowed_base + os.sep) and abs_path != allowed_base:
        return jsonify({"error": "forbidden"}), 403
    if not os.path.isdir(abs_path):
        return jsonify({"error": "not_found"}), 404

    try:
        all_entries = sorted(os.listdir(abs_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    files = [f for f in all_entries if os.path.isfile(os.path.join(abs_path, f))]
    # Filter by player-readable rules first; if caller is admin, return everything.
    is_admin = _check_admin_auth() is None
    if not is_admin:
        files = [f for f in files if _is_player_readable(f)]
    return jsonify({"files": files, "is_admin": is_admin})


@app.route('/api/admin/list_game_files')
def admin_list_game_files():
    """Admin-only listing of all files in a game dir with size + mtime.

    Used by the Raw Files drawer to display every file under games/<id>/
    (including sensitive logs that /api/list_files would hide from players).
    Files are still fetched via /api/file, which gates each individual read.
    """
    auth_err = _check_admin_auth()
    if auth_err:
        return auth_err
    game_id = request.args.get('game_id', '')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    gdir = resolve_game_dir(game_id)
    if not gdir.is_dir():
        return jsonify({"error": "not_found"}), 404
    files = []
    try:
        for entry in sorted(gdir.iterdir()):
            if not entry.is_file():
                continue
            try:
                st = entry.stat()
                files.append({
                    "name": entry.name,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                })
            except Exception:
                continue
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"game_id": game_id, "files": files})


@app.route('/api/games_list')
def api_games_list():
    """List runtime games for dashboard cards, newest first.

    Each entry includes id, outcome, duration, AI player, player count, and date.
    """
    games_root = Path(DIRS_PREFIX)
    if not games_root.exists():
        return jsonify({"games": []})
    items = []
    # Enumerate runtime games directly under the real/test category folders.
    all_dirs = [(c, identity, d) for (c, identity, d) in iter_all_games(include_simulations=False)
                if d.name.isdigit()]
    # Sort newest first by id.
    all_dirs.sort(key=lambda t: t[2].name, reverse=True)
    for cat, identity, gdir in all_dirs:
        gid = gdir.name
        entry = {"id": gid, "category": cat, "agent_identity": identity}
        # Config → players, AI, daytime
        try:
            cfg_path = gdir / GAME_CONFIG_FILE
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                entry["daytime_minutes"] = cfg.get("daytime_minutes")
                entry["anonymous_voting"] = cfg.get("anonymous_voting", False)
                players = cfg.get("players", [])
                entry["players"] = [p.get("name") for p in players]
                ai = [p.get("name") for p in players if p.get("is_ai") or p.get("is_llm")]
                entry["ai_player"] = ai[0] if ai else None
                entry["player_count"] = len(players)
        except Exception:
            pass
        # Outcome
        try:
            wf = gdir / WHO_WINS_FILE
            if wf.exists():
                entry["outcome"] = wf.read_text(encoding='utf-8').strip() or None
        except Exception:
            pass
        # Start time
        try:
            gst = gdir / GAME_START_TIME_FILE
            if gst.exists():
                entry["start_time"] = gst.read_text(encoding='utf-8').strip()
                entry["start_epoch_ms"] = int(gst.stat().st_mtime * 1000)
        except Exception:
            pass
        # AI elimination
        try:
            ai_elim = gdir / "ai_elimination_info.txt"
            if ai_elim.exists():
                text = ai_elim.read_text(encoding='utf-8')
                for line in text.splitlines():
                    if 'AI_ELIMINATED_ROUND' in line:
                        entry["ai_eliminated_round"] = line.split('-', 1)[-1].strip()
                    if 'TOTAL_ROUNDS' in line:
                        entry["total_rounds"] = line.split('-', 1)[-1].strip()
        except Exception:
            pass
        items.append(entry)
    return jsonify({"games": items, "count": len(items)})


@app.route('/api/game_detail')
def api_game_detail():
    """Return EVERYTHING we know about a single game for the Games-tab drill view.

    Response shape (all fields best-effort — missing sections are omitted):
        {
          id, config, players, ai_player, outcome, duration_info,
          votes_by_round: [ {round, votes: {player: voted_for}} ],
          chat: [ {time, sender, text, is_ai} ],
          manager_chat: [...],
          reviewer: {modified_count, total, outcomes: [...]},
          prompt_flow: [...],           (only if admin)
          costs: {total_usd, entries},
          survey: {player: {...}},
          taps: [...]
        }
    """
    game_id = request.args.get('game_id', '')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    gdir = resolve_game_dir(game_id)
    if not gdir.exists():
        return jsonify({"error": "not_found"}), 404

    is_admin = _check_admin_auth() is None
    out = {"id": game_id, "is_admin": is_admin}

    # --- Config
    try:
        cfg_path = gdir / GAME_CONFIG_FILE
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            out["config"] = cfg
            players = cfg.get("players", [])
            out["players"] = [p.get("name") for p in players]
            ai = [p.get("name") for p in players if p.get("is_ai") or p.get("is_llm")]
            out["ai_player"] = ai[0] if ai else None
            out["daytime_minutes"] = cfg.get("daytime_minutes")
    except Exception as e:
        out["config_error"] = str(e)

    # --- Outcome
    try:
        wf = gdir / WHO_WINS_FILE
        if wf.exists():
            out["outcome"] = wf.read_text(encoding='utf-8').strip() or None
        ai_elim = gdir / "ai_elimination_info.txt"
        if ai_elim.exists():
            out["ai_elimination"] = ai_elim.read_text(encoding='utf-8').strip()
    except Exception:
        pass

    # --- Timing
    for fname, key in [(GAME_START_TIME_FILE, "game_start_time"),
                        (DISCUSSION_START_TIME_FILE, "discussion_start_time")]:
        try:
            f = gdir / fname
            if f.exists():
                out[key] = f.read_text(encoding='utf-8').strip()
                out[key + "_epoch_ms"] = int(f.stat().st_mtime * 1000)
        except Exception:
            pass

    # --- Remaining players
    try:
        rp = gdir / REMAINING_PLAYERS_FILE
        if rp.exists():
            out["remaining_players"] = [n.strip() for n in rp.read_text(encoding='utf-8').splitlines() if n.strip()]
    except Exception:
        pass

    # --- Votes: read {player}_vote.txt for each player.
    try:
        votes_by_player = {}
        max_rounds = 0
        for pname in out.get("players", []):
            vf = gdir / f"{pname}_vote.txt"
            if vf.exists():
                lines = [l.strip() for l in vf.read_text(encoding='utf-8').splitlines() if l.strip()]
                votes_by_player[pname] = lines
                if len(lines) > max_rounds:
                    max_rounds = len(lines)
        rounds = []
        for r in range(max_rounds):
            round_votes = {}
            for pname, lines in votes_by_player.items():
                if r < len(lines):
                    round_votes[pname] = lines[r]
                else:
                    round_votes[pname] = None
            rounds.append({"round": r + 1, "votes": round_votes})
        out["votes_by_round"] = rounds
        out["votes_by_player"] = votes_by_player
    except Exception as e:
        out["votes_error"] = str(e)

    # --- Eliminations (ground truth from manager chat)
    # Vote-tally heuristics break under ties / hidden AI votes / anonymity, so
    # parse the manager's authoritative announcements instead.
    try:
        mcf = gdir / "public_manager_chat.txt"
        if mcf.exists():
            elims = []
            for line in mcf.read_text(encoding='utf-8').splitlines():
                em = re.match(
                    r'^\[(\d{2}:\d{2}:\d{2})\]\s+Game-Manager:\s+(.+?)\s+was eliminated\.\s*They were:\s*(AI|Human)',
                    line.strip(), re.IGNORECASE)
                if em:
                    elims.append({
                        "round": len(elims) + 1,
                        "time": em.group(1),
                        "name": em.group(2).strip(),
                        "role": em.group(3).strip().capitalize(),
                    })
            if elims:
                out["eliminations_actual"] = elims
    except Exception:
        pass

    # --- Chat
    try:
        cf = gdir / PUBLIC_DAYTIME_CHAT_FILE
        if cf.exists():
            chat = []
            for line in cf.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^\[([^\]]+)\]\s+([^:]+):\s?(.*)$', line)
                if m:
                    sender = m.group(2).strip()
                    chat.append({
                        "time": m.group(1),
                        "sender": sender,
                        "text": m.group(3),
                        "is_ai": sender == out.get("ai_player"),
                        "type": "chat",
                    })
                else:
                    chat.append({"time": "", "sender": "?", "text": line, "is_ai": False, "type": "chat"})
            # Merge elimination announcements from manager chat so the dashboard
            # can render round dividers between discussion phases. Only the
            # concise "X was eliminated. They were: AI/Human" lines are added —
            # all other manager noise (rules, phase announcements) is ignored.
            ai_name = out.get("ai_player")
            mcf = gdir / "public_manager_chat.txt"
            if mcf.exists():
                for line in mcf.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    em = re.match(
                        r'^\[(\d{2}:\d{2}:\d{2})\]\s+Game-Manager:\s+(.+?)\s+was eliminated\.\s*They were:\s*(AI|Human)',
                        line, re.IGNORECASE)
                    if not em:
                        continue
                    elim_name = em.group(2).strip()
                    role = em.group(3).strip().capitalize()
                    chat.append({
                        "time": em.group(1),
                        "sender": "__elimination__",
                        "text": f"{elim_name} eliminated ({role})",
                        "is_ai": (role == "AI") or (ai_name and elim_name == ai_name),
                        "type": "elimination",
                        "eliminated_name": elim_name,
                        "eliminated_role": role,
                    })
            # Stable sort by timestamp keeps elimination dividers between rounds.
            def _ts_key(m):
                t = m.get("time") or ""
                parts = t.split(":")
                try:
                    return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                except (ValueError, IndexError):
                    return 0
            chat.sort(key=_ts_key)
            out["chat"] = chat
    except Exception:
        pass

    # --- Manager announcements
    try:
        mcf = gdir / "public_manager_chat.txt"
        if mcf.exists():
            manager = []
            for line in mcf.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    manager.append(line)
            out["manager_chat"] = manager
    except Exception:
        pass

    # --- Reviewer outcomes (reuse existing parser via internal call)
    try:
        ai_name = out.get("ai_player")
        if ai_name:
            log_file = gdir / LLM_LOG_FILE_FORMAT.format(ai_name)
            if log_file.exists():
                modified, total = 0, 0
                outcomes = _parse_review_outcomes_from_log(log_file)  # may not exist
                if outcomes is not None:
                    # Enrich outcomes from review_cycle events in file order.
                    # Without event records, retain the parsed log outcomes.
                    events_file = game_file(gdir, "game_events.jsonl")
                    if events_file.exists():
                        review_events = []
                        for line in events_file.read_text(encoding='utf-8', errors='replace').splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if ev.get("event") == "review_cycle" and not ev.get("skipped_short"):
                                review_events.append(ev)
                        # Pair by position. Skip-short events are filtered out on
                        # both sides since _parse_review_outcomes only sees real reviews.
                        for i, o in enumerate(outcomes):
                            if i < len(review_events):
                                ev = review_events[i]
                                o["candidates"] = ev.get("candidates") or []
                                o["situation_analysis"] = ev.get("situation_analysis") or ""
                                o["expected_human_response"] = ev.get("expected_human_response") or ""
                                o["reviewer_attempts"] = ev.get("reviewer_attempts")
                                o["reviewer_ms"] = ev.get("reviewer_ms")
                    # For outcomes lacking candidates, read attempt details
                    # directly from the AI log and pair cycles by position.
                    needs_legacy = any(not o.get("candidates") for o in outcomes)
                    if needs_legacy:
                        legacy_cycles = _parse_review_attempts_from_log(log_file)
                        if legacy_cycles:
                            for i, o in enumerate(outcomes):
                                if o.get("candidates"):
                                    continue
                                if i < len(legacy_cycles):
                                    lc = legacy_cycles[i]
                                    o["candidates"] = lc.get("candidates") or []
                                    o["situation_analysis"] = lc.get("situation_analysis") or ""
                                    o["expected_human_response"] = lc.get("expected_human_response") or ""
                                    o["reviewer_attempts"] = len(lc.get("candidates") or []) or None
                    total = len(outcomes)
                    modified = sum(1 for o in outcomes if o.get("was_modified"))
                    # Mark cycles whose final_message never landed in the public
                    # chat as "discarded" (e.g. phase ended before send). The UI
                    # uses this to filter Per-Message Detail / Reviewer Activity
                    # so they reflect what actually happened in chat.
                    sent_msgs = set()
                    for m in out.get("chat") or []:
                        if m.get("speaker") == ai_name:
                            t = (m.get("text") or "").strip()
                            if t:
                                sent_msgs.add(t)
                    discarded = 0
                    for o in outcomes:
                        fm = (o.get("final_message") or "").strip()
                        if fm and fm not in sent_msgs:
                            o["sent"] = False
                            o["discard_reason"] = "phase ended before send"
                            discarded += 1
                        else:
                            o["sent"] = True
                    out["reviewer"] = {
                        "modified": modified,
                        "total": total,
                        "discarded": discarded,
                        "outcomes": outcomes[:200],
                    }
    except Exception:
        pass

    # --- Costs
    try:
        cl = game_file(gdir, "cost_log.jsonl")
        if cl.exists():
            entries = []
            total = 0.0
            for line in cl.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    entries.append(e)
                    total += float(e.get("cost_usd", 0) or 0)
                except Exception:
                    continue
            out["costs"] = {"total_usd": round(total, 5), "entry_count": len(entries), "entries": entries[-50:]}
    except Exception:
        pass

    # --- Survey
    try:
        survey = {}
        _sdir = gdir / "surveys"
        _sfiles = _sdir.glob("*_survey.txt") if _sdir.is_dir() else gdir.glob("*_survey.txt")
        for f in _sfiles:
            name = f.stem.replace("_survey", "")
            try:
                survey[name] = f.read_text(encoding='utf-8').strip()
            except Exception:
                continue
        if survey:
            out["survey"] = survey
    except Exception:
        pass

    # --- Taps
    try:
        tf = gdir / "in_game_taps.jsonl"
        if tf.exists():
            taps = []
            for line in tf.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    try:
                        taps.append(json.loads(line))
                    except Exception:
                        pass
            out["taps"] = taps
    except Exception:
        pass

    return jsonify(out)


def _parse_review_outcomes_from_log(log_file: Path):
    """Lightweight parser — extract review_outcome dicts from AI log.

    Format emitted by llm_players/reviewed_player.py _log_review_result():
        ## OPERATION: review_outcome
        ## CONTENT: {'original_message': '...', 'final_message': '...',
                     'was_modified': bool, 'final_score': int,
                     'final_suspicious': bool, 'red_flags': [...]}
    """
    try:
        text = log_file.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None
    outcomes = []
    for block in text.split('# NEW LOG'):
        if 'review_outcome' not in block:
            continue
        content_match = re.search(r'##\s*CONTENT:\s*(.*)', block, re.DOTALL)
        if not content_match:
            continue
        try:
            # Content is a Python-repr dict; use ast.literal_eval for safety.
            import ast
            data = ast.literal_eval(content_match.group(1).strip())
            if isinstance(data, dict):
                # Accept both original_message and original payload keys.
                orig = data.get("original_message", data.get("original", ""))
                final = data.get("final_message", data.get("final", ""))
                score = data.get("final_score", data.get("score"))
                red_flags = data.get("red_flags") or []
                if isinstance(red_flags, (list, tuple)):
                    red_flags = [str(f)[:300] for f in red_flags][:10]
                else:
                    red_flags = []
                outcomes.append({
                    "original": str(orig),
                    "final": str(final),
                    "was_modified": bool(data.get("was_modified", False)),
                    "score": score,
                    "suspicious": bool(data.get("final_suspicious", False)),
                    "red_flags": red_flags,
                })
        except Exception:
            continue
    return outcomes


def _parse_review_attempts_from_log(log_file: Path):
    """Extract per-attempt review details from the AI's text log.

    Walks `# NEW LOG` blocks and groups them per review cycle (terminated by a
    `review_outcome` block). Within each cycle, pairs the i-th
    `GeminiReviewer: review_response` JSON with the i-th `review_attempt_N`
    line to build a candidates list compatible with the dashboard's
    drill-down UI.

    Returns a list parallel to `_parse_review_outcomes_from_log()`:
        [
          {
            "candidates": [
              {"message": "...", "score": 5, "red_flags": [...],
               "suggested_fix": "...", "action": "rewritten" | "passed" |
               "reverted" | "no_fix"},
              ...
            ],
            "situation_analysis": "...",
            "expected_human_response": "...",
            "was_modified": bool,
          },
          ...
        ]
    Best-effort: silently skips malformed cycles.
    """
    try:
        text = log_file.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None

    cycles = []
    review_responses = []   # list of dict (parsed JSON) for current cycle
    attempt_msgs = []       # list of (idx, message_str) for current cycle
    attempt_actions = []    # actions logged by the agent per attempt: "passed"|"failed"|"good_enough"
    situation = None
    expected = None

    attempt_re = re.compile(
        r"Score:\s*(-?\d+)\s*,\s*Suspicious:\s*(True|False)\s*,\s*Message:\s*'(.*)'\s*$",
        re.DOTALL,
    )

    for block in text.split('# NEW LOG'):
        # Extract OPERATION line
        op_m = re.search(r'##\s*OPERATION:\s*(.+)', block)
        if not op_m:
            continue
        op = op_m.group(1).strip()
        content_m = re.search(r'##\s*CONTENT:\s*(.*)', block, re.DOTALL)
        content = content_m.group(1).strip() if content_m else ""

        if op.endswith("review_response"):
            # Parse JSON; tolerate trailing junk
            try:
                # Trim to outermost {..}
                start = content.find('{')
                end = content.rfind('}')
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end + 1])
                else:
                    parsed = None
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                review_responses.append(parsed)
                # First response in cycle carries situation/expected
                if situation is None:
                    s = parsed.get("situation")
                    if isinstance(s, str) and s and not s.startswith("<"):
                        situation = s
                if expected is None:
                    e = parsed.get("expected_human_response")
                    if isinstance(e, str) and e and not e.startswith("<"):
                        expected = e

        elif op.startswith("review_attempt_"):
            try:
                idx = int(op.rsplit("_", 1)[1])
            except Exception:
                idx = len(attempt_msgs) + 1
            am = attempt_re.search(content)
            if am:
                msg = am.group(3)
                # The message string was Python-repr; turn \\' into '
                msg = msg.replace("\\'", "'").replace('\\\\', '\\')
                attempt_msgs.append((idx, msg))

        elif op == "review_passed":
            attempt_actions.append("passed")
        elif op == "review_good_enough":
            attempt_actions.append("good_enough")
        elif op == "review_failed":
            attempt_actions.append("failed")

        elif op == "review_outcome":
            # Close cycle. Parse outcome dict for was_modified.
            was_modified = False
            try:
                import ast
                data = ast.literal_eval(content)
                if isinstance(data, dict):
                    was_modified = bool(data.get("was_modified", False))
            except Exception:
                pass

            # Build candidates by pairing attempts with their review_response
            # by index. attempt_msgs is in order; review_responses likewise.
            n = max(len(attempt_msgs), len(review_responses))
            candidates = []
            for i in range(n):
                msg = attempt_msgs[i][1] if i < len(attempt_msgs) else ""
                resp = review_responses[i] if i < len(review_responses) else {}
                score = resp.get("suspicion_score")
                is_susp = bool(resp.get("is_suspicious"))
                red_flags = resp.get("red_flags") or []
                if not isinstance(red_flags, list):
                    red_flags = []
                red_flags = [str(f)[:300] for f in red_flags][:10]
                fix = resp.get("suggested_fix")
                if not isinstance(fix, str) or not fix.strip():
                    fix = None
                is_last = (i == n - 1)
                if is_last:
                    if was_modified:
                        action = "rewritten"
                    elif n > 1:
                        action = "reverted"
                    else:
                        action = "passed"
                elif is_susp:
                    action = "rewritten" if fix else "no_fix"
                else:
                    action = "passed"
                candidates.append({
                    "message": msg,
                    "score": score,
                    "red_flags": red_flags,
                    "suggested_fix": fix,
                    "action": action,
                })
            cycles.append({
                "candidates": candidates,
                "situation_analysis": situation or "",
                "expected_human_response": expected or "",
                "was_modified": was_modified,
            })
            # Reset cycle state
            review_responses = []
            attempt_msgs = []
            situation = None
            expected = None

    return cycles


@app.route('/api/review_outcomes')
def review_outcomes():
    """Return sanitized reviewer outcomes for a game — original vs final message.

    Safe for the dashboard's "Show without reviewer" counterfactual toggle.
    Reads the AI player's llm_log_{name}.txt and extracts ONLY the
    `review_outcome` records (original_message, final_message, was_modified,
    final_score). Prompts, reviewer red flags, and everything else are
    discarded. Public (no auth) since the conversation is already public.
    """
    game_id = request.args.get('game_id', '')
    if not _valid_game_id(game_id):
        return jsonify({"error": "invalid game_id"}), 400
    game_dir = resolve_game_dir(game_id)
    if not game_dir.exists():
        return jsonify({"error": "not_found"}), 404

    cfg = {}
    cfg_file = game_dir / GAME_CONFIG_FILE
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    ai_names = [p.get("name") for p in cfg.get("players", []) if p.get("is_llm") or p.get("is_ai")]
    outcomes = []
    for ai in ai_names:
        log_file = game_dir / LLM_LOG_FILE_FORMAT.format(ai)
        if not log_file.exists():
            continue
        try:
            text = log_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        for block in text.split('# NEW LOG'):
            if 'OPERATION: review_outcome' not in block:
                continue
            # Content uses Python-repr dict (single quotes) — use regex.
            m_orig = re.search(r"'original_message'\s*:\s*'((?:\\'|[^'])*)'", block)
            m_final = re.search(r"'final_message'\s*:\s*'((?:\\'|[^'])*)'", block)
            if not (m_orig and m_final):
                continue
            orig = m_orig.group(1).replace("\\'", "'")
            final = m_final.group(1).replace("\\'", "'")
            was_modified = bool(re.search(r"'was_modified'\s*:\s*True", block))
            outcomes.append({
                "ai": ai,
                "original": orig,
                "final": final,
                "was_modified": was_modified,
            })
    return jsonify({"outcomes": outcomes})


if __name__ == "__main__":
    print("Starting Social Turing Test Web Interface...")
    if not ADMIN_PASSWORD:
        print("No admin password configured. Set STT_ADMIN_PASSWORD.")
    _load_persisted_sessions()
    print("Open your browser to: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
else:
    # When imported under gunicorn / docker, still rehydrate sessions from disk
    # so a restart doesn't invalidate in-flight games.
    try:
        _load_persisted_sessions()
    except Exception as _e:
        print(f"[Sessions] Import-time load failed: {_e}")
