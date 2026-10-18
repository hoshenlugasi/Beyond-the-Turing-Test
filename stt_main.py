import json
import os
from core.game_constants import *  # incl. argparse, time, Path (from pathlib), colored (from termcolor), random
from core.atomic_io import atomic_write_text

# global variable for the game dir
game_dir = Path()  # will be updated only if __name__ == __main__ (prevents new ones in imports)

class Player:

    def __init__(self, name, is_ai, **kwargs):
        self.name = name
        self.is_ai = is_ai
        self.personal_chat_file = game_dir / PERSONAL_CHAT_FILE_FORMAT.format(self.name)
        self.personal_chat_file_lines_read = 0
        self.personal_vote_file = game_dir / PERSONAL_VOTE_FILE_FORMAT.format(self.name)
        self.personal_vote_file_lines_read = 0
        # status is whether the player has joined and then whether was voted out
        self.personal_status_file = game_dir / PERSONAL_STATUS_FILE_FORMAT.format(self.name)

    def get_new_messages(self):
        with open(self.personal_chat_file, "r", encoding="utf-8", errors="ignore") as f:
            # the readlines method includes the "\n"
            lines = f.readlines()[self.personal_chat_file_lines_read:]
        self.personal_chat_file_lines_read += len(lines)
        return lines

    def get_voted_player(self):
        all_votes = self.personal_vote_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_votes = all_votes[self.personal_vote_file_lines_read:]
        if new_votes:
            self.personal_vote_file_lines_read += len(new_votes)  # should be 1 if works correctly
            return new_votes[-1].strip()
        else:
            return None

    def eliminate(self):
        self.personal_status_file.write_text(VOTED_OUT, encoding="utf-8")  # 1.8 — explicit utf-8

def get_config():
    with open(game_dir / GAME_CONFIG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        config = json.load(f)
    return config

def get_players(config):
    return [Player(**player_config) for player_config in config[PLAYERS_KEY_IN_CONFIG]]

def is_game_over(players):
    """
    Social Turing Test win conditions:
    - Humans win: AI is eliminated (0 mafia players remaining)
    - AI wins: Only 2 players remain (AI + 1 human)
    """
    ai_players = [player for player in players if player.is_ai]
    human_players = [player for player in players if not player.is_ai]
    
    # Humans win if AI is eliminated
    if len(ai_players) == 0:
        (game_dir / WHO_WINS_FILE).write_text(HUMANS_WIN_MESSAGE)
        return True
    
    # AI wins once it can no longer be out-voted — only the AI plus at most one
    # human remain (the canonical "2 players left" end state). Using <= (not ==)
    # is defensive: if the roster ever drops BELOW 2 for any reason — a game that
    # only ever had the AI, or every human disconnecting/leaving mid-game — the
    # AI is the last one standing and the game must still END, not loop forever
    # (game 9073 hung exactly this way: only the AI remained, who_wins never
    # written).
    if len(players) <= 2 and len(ai_players) >= 1:
        (game_dir / WHO_WINS_FILE).write_text(AI_WINS_MESSAGE)
        return True
    
    return False

def run_chat_round_between_players(players, chat_room):
    from core.chat_logging import append_chat_displayed
    for player in players:
        lines = player.get_new_messages()
        if not lines:
            continue
        with open(chat_room, "a", encoding="utf-8") as f:
            f.writelines(lines)  # lines already include "\n"
        # Mirror what the user sees in the browser (frontend lowercases the body).
        for line in lines:
            append_chat_displayed(game_dir, line)

def notify_players_about_voting_time(phase_name, public_chat_file):
    # BUG FIX #2: Write to manager chat file, not daytime chat
    phase_end_message = DAYTIME_VOTING_TIME_MESSAGE
    game_manager_announcement(phase_end_message)
    voting_phase_name = DAYTIME_VOTING_TIME
    (game_dir / PHASE_STATUS_FILE).write_text(voting_phase_name)

def _dropped_player_names():
    """Codenames the researcher removed mid-game via /api/admin/drop_player.

    The admin process cannot touch the manager's in-memory roster, so it only
    records the request in dropped_players.txt and the manager applies it.
    """
    dropped_file = game_dir / DROPPED_PLAYERS_FILE
    if not dropped_file.exists():
        return set()
    try:
        return {
            name.strip()
            for name in dropped_file.read_text(
                encoding="utf-8", errors="ignore").splitlines()
            if name.strip()
        }
    except OSError:
        return set()


def get_voted_out_name(optional_votes_players, public_chat_file, voting_players, anonymous_voting=False,
                       voting_timeout_minutes=5.0):
    votes = {player.name: 0 for player in optional_votes_players}
    vote_details = []  # Collect who voted for whom

    # Bounded wait: this loop used to spin forever when a player process died
    # mid-vote (automated games 9124/9125 sat stuck in Daytime_Voting). After
    # the timeout, missing voters are auto-ABSTAINed — the same semantics the
    # admin force-tally already uses — and the tally proceeds.
    voting_deadline = time.time() + max(0.5, float(voting_timeout_minutes)) * 60

    while voting_players:
        if time.time() > voting_deadline:
            for player in list(voting_players):
                _log_invalid_vote(player.name, "<ABSTAIN>", reason="voting_timeout_auto_abstain")
            print(f"Voting timeout ({voting_timeout_minutes} min) — "
                  f"auto-abstaining: {', '.join(p.name for p in voting_players)}")
            voting_players.clear()
            break
        # A player dropped mid-round is no longer expected to vote; without
        # this the loop would block on them for the full timeout.
        dropped = _dropped_player_names()
        if dropped:
            for player in [p for p in voting_players if p.name in dropped]:
                _log_invalid_vote(player.name, "<ABSTAIN>", reason="admin_dropped_mid_game")
                voting_players.remove(player)
        voted_players = []
        for player in voting_players:
            voted_for = player.get_voted_player()
            if not voted_for:
                continue
            voted_players.append(player)
            # Admin force-tally writes "<ABSTAIN>" into the vote file to unblock
            # the voting loop when a player has disconnected. Count them as
            # voted (so they're removed from voting_players) but don't tally
            # their vote toward any candidate.
            if voted_for == "<ABSTAIN>":
                _log_invalid_vote(player.name, voted_for, reason="forced_abstain")
                continue
            if voted_for in votes:
                vote_details.append((player.name, voted_for))
                votes[voted_for] += 1
            else:
                # 1.7 — log invalid votes for audit trail
                _log_invalid_vote(player.name, voted_for)
        for player in voted_players:
            voting_players.remove(player)
        time.sleep(0.1)  # 1.3 — prevent busy-wait CPU spin

    # A player dropped mid-round is excluded as both candidate AND voter.  The
    # manager may already have consumed their vote before the admin's drop
    # marker appeared, so filtering only targets left a removed/cheating
    # player's vote in the tally depending on millisecond timing.
    dropped = _dropped_player_names()
    if dropped:
        active_names = [name for name in votes if name not in dropped]
        vote_details = [
            (voter, target) for voter, target in vote_details
            if voter not in dropped and target not in dropped
        ]
        votes = {name: 0 for name in active_names}
        for _voter, target in vote_details:
            votes[target] += 1
    if not votes:
        # Everyone left. The caller's is_game_over() ends the game.
        return None

    # 1.6 — true random tie-break (max() was alphabetical)
    max_v = max(votes.values())
    tied = [n for n, v in votes.items() if v == max_v]
    voted_out_name = random.choice(tied)
    # Mirror vote outcome into chat_displayed_full.txt for the unified log.
    try:
        from core.chat_logging import append_vote_tally
        append_vote_tally(game_dir, vote_details, votes, voted_out_name, anonymous_voting)
    except Exception:
        pass
    # Audit: persist tally + tie-break decision so the admin UI can explain
    # why an unexpected player was eliminated (issue from game 0084).
    try:
        audit = {
            "timestamp": get_current_timestamp(),
            "tally": votes,
            "max_votes": max_v,
            "tied_candidates": tied,
            "voted_out": voted_out_name,
            "tie_break": "random" if len(tied) > 1 else "single_max",
        }
        audit_path = game_dir / "events" / "elimination_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit) + "\n")
    except Exception:
        pass  # audit failure must never crash the game
    return voted_out_name


def _log_invalid_vote(voter_name: str, target_name: str, reason: str = "invalid target"):
    """Append invalid vote to audit log (1.7)."""
    try:
        from core.game_constants import get_current_timestamp
        line = f"{get_current_timestamp()} | {voter_name} -> {target_name} ({reason})\n"
        with open(game_dir / "invalid_votes.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # audit log failure must never crash the game

def drop_players_removed_by_admin(players):
    """Apply /api/admin/drop_player requests to the live roster.

    A dropped player LEFT — they were not eliminated by the group — so no
    role-revealing elimination announcement is made and the AI-elimination
    round is not affected. Returns the players actually removed.
    """
    dropped = _dropped_player_names()
    if not dropped:
        return []
    removed = [player for player in players if player.name in dropped]
    if not removed:
        return []
    for player in removed:
        player.eliminate()
        players.remove(player)
    remaining_file = game_dir / REMAINING_PLAYERS_FILE
    try:
        remaining = [
            name.strip()
            for name in remaining_file.read_text(
                encoding="utf-8", errors="ignore").splitlines()
            if name.strip() and name.strip() not in dropped
        ]
        atomic_write_text(remaining_file, "\n".join(remaining))
    except OSError as exc:
        print(f"[DropPlayer] Failed to rewrite remaining players: {exc}")
    for player in removed:
        print(f"{player.name} was removed from the game by the researcher.")
        game_manager_announcement(PLAYER_LEFT_MESSAGE_FORMAT.format(player.name))
    return removed


def voting_sub_phase(phase_name, voting_players, optional_votes_players, public_chat_file, players, anonymous_voting=False):
    notify_players_about_voting_time(phase_name, public_chat_file)
    # AI votes ARE counted in the tally (alongside human votes).
    voted_out_name = get_voted_out_name(optional_votes_players, public_chat_file, voting_players[:], anonymous_voting)
    if voted_out_name is None:
        # Every candidate was dropped by the admin — nobody to eliminate.
        drop_players_removed_by_admin(players)
        return None
    # update info file of remaining players
    remaining_players = [n.strip() for n in (game_dir / REMAINING_PLAYERS_FILE).read_text(encoding="utf-8", errors="ignore").splitlines() if n.strip()]
    if voted_out_name not in remaining_players:
        print(f"[WARNING] Voted out '{voted_out_name}' not in remaining players {remaining_players}. Picking random.")
        voted_out_name = remaining_players[0] if remaining_players else voted_out_name
    remaining_players.remove(voted_out_name)
    # Atomic replacement prevents partial reads. On Windows the admin/UI can
    # briefly hold the destination without delete sharing, so the shared helper
    # also retries that transient lock and uses a unique temporary file.
    atomic_write_text(
        game_dir / REMAINING_PLAYERS_FILE,
        "\n".join(remaining_players),
    )
    # update player object status
    voted_out_player = {player.name: player for player in optional_votes_players}[voted_out_name]
    voted_out_player.eliminate()
    players.remove(voted_out_player)
    announce_voted_out_player(voted_out_player)
    return voted_out_name

def game_manager_announcement(message):
    """
    Write a Game Manager announcement to the manager chat file.
    Multi-line messages are escaped (\n → \\n) so they appear as ONE bubble in the UI.
    """
    # Mirror the announcement into chat_displayed_full.txt so the unified
    # "what users saw" log includes voting prompts, eliminations, win banner.
    try:
        from core.chat_logging import append_manager_announcement
        append_manager_announcement(game_dir, message)
    except Exception:
        pass
    with open(game_dir / PUBLIC_MANAGER_CHAT_FILE, "a", encoding="utf-8") as f:
        # Escape newlines so the entire message stays in one file line
        final_message = format_message(GAME_MANAGER_NAME, message.replace("\n", "\\n"))
        f.write(final_message)

def announce_voted_out_player(voted_out_player):
    role = get_role_display_string(voted_out_player.is_ai)
    voted_out_message = VOTED_OUT_MESSAGE_FORMAT.format(voted_out_player.name, role)
    game_manager_announcement(voted_out_message)

def run_phase(players, voting_players, optional_votes_players, public_chat_file,
              time_limit_seconds, phase_name, anonymous_voting=False):
    if len(voting_players) > 1:
        start_time = time.time()
        while time.time() - start_time < time_limit_seconds:
            run_chat_round_between_players(voting_players, public_chat_file)
            # A player dropped during discussion leaves the room immediately,
            # so nobody wastes a vote on someone who is no longer there.
            for player in drop_players_removed_by_admin(players):
                for roster in (voting_players, optional_votes_players):
                    if player in roster:
                        roster.remove(player)
            if len(voting_players) <= 1:
                break
    else:
        game_manager_announcement(CUTTING_TO_VOTE_MESSAGE)
    print("Now voting starts...")
    eliminated = voting_sub_phase(phase_name, voting_players, optional_votes_players, public_chat_file, players, anonymous_voting)
    return eliminated

def run_daytime(players, daytime_minutes, anonymous_voting=False):
    """Run discussion phase where all players can communicate and vote"""
    (game_dir / PHASE_STATUS_FILE).write_text(DAYTIME)
    # Per-round anchor so the client timer resyncs each discussion round
    # (GAME_START_TIME_FILE is written once at game start and used elsewhere).
    (game_dir / DISCUSSION_START_TIME_FILE).write_text(get_current_timestamp())

    # Build daytime start message
    message = DAYTIME_START_MESSAGE_FORMAT.format(daytime_minutes)
    
    # Add conversation topic if available (playful format)
    topic_file = game_dir / CONVERSATION_TOPIC_FILE
    if topic_file.exists():
        topic = topic_file.read_text(encoding="utf-8", errors="ignore").strip()
        if topic:
            message += f"\n\nFeel free to chat about: {topic}\n(or whatever else comes up!)"
    
    print(colored(message.replace("\\n", "\n"), DAYTIME_COLOR))
    game_manager_announcement(message)
    eliminated = run_phase(players, players, players, game_dir / PUBLIC_DAYTIME_CHAT_FILE,
              minutes_to_seconds(daytime_minutes), DAYTIME, anonymous_voting)
    return eliminated

def wait_for_players(players):
    havent_joined_yet = [player for player in players]
    print("Waiting for all players to connect and start running their programs to join:")
    print(",  ".join([player.name for player in havent_joined_yet]))
    while havent_joined_yet:
        joined = []
        for player in havent_joined_yet:
            if bool(player.personal_status_file.read_text(encoding="utf-8", errors="ignore")):  # file isn't empty once joined
                joined.append(player)
                print(f"{player.name} has joined!")
        for player in joined:
            havent_joined_yet.remove(player)
        if havent_joined_yet:
            time.sleep(0.5)
    (game_dir / GAME_START_TIME_FILE).write_text(get_current_timestamp())
    print("Game is now running! Its content is displayed to players.")

def get_all_player_out_of_voting_time():
    current_phase = (game_dir / PHASE_STATUS_FILE).read_text(encoding="utf-8", errors="ignore")
    (game_dir / PHASE_STATUS_FILE).write_text(current_phase.replace(VOTING_TIME, ""))

def write_ai_elimination_info(ai_eliminated_round, total_rounds):
    """Write AI elimination tracking data for research analysis"""
    info_file = game_dir / AI_ELIMINATION_INFO_FILE
    
    if ai_eliminated_round is None:
        elimination_status = "survived"
    else:
        elimination_status = str(ai_eliminated_round)
    
    content = f"AI_ELIMINATED_ROUND - {elimination_status}\nTOTAL_ROUNDS - {total_rounds}\n"
    info_file.write_text(content, encoding="utf-8")

def end_game():
    get_all_player_out_of_voting_time()
    print("Game has finished.")


def main():
    global game_dir
    game_dir = get_game_dir_from_argv()
    config = get_config()
    players = get_players(config)
    anonymous_voting = config.get(ANONYMOUS_VOTING_KEY, False)
    wait_for_players(players)
    
    # Track rounds and AI elimination for research
    round_counter = 0
    ai_eliminated_round = None
    
    # Get AI player name for tracking
    ai_players = [p for p in players if p.is_ai]
    ai_player_name = ai_players[0].name if ai_players else None
    
    # Social Turing Test: Only day phases (discussion + voting), no night phase
    while not is_game_over(players):
        # Apply any drop requested between rounds before announcing the round.
        if drop_players_removed_by_admin(players) and is_game_over(players):
            break
        round_counter += 1
        eliminated_name = run_daytime(players, config[DAYTIME_MINUTES_KEY], anonymous_voting)
        
        # Check if AI was eliminated this round
        if eliminated_name == ai_player_name and ai_eliminated_round is None:
            ai_eliminated_round = round_counter
    
    # A drop requested during the final round never reached the loop head,
    # because is_game_over() ended the game first. Applying it here keeps
    # remaining_players.txt and the announcements honest: a player the
    # researcher removed must never appear alive in the final roster.
    drop_players_removed_by_admin(players)

    # Write elimination tracking data
    write_ai_elimination_info(ai_eliminated_round, round_counter)
    
    end_game()

if __name__ == '__main__':
    main()
