import json
import random
from social_turing_test.core.game_constants import *  # incl. argparse, time, Path (from pathlib), colored (from termcolor)
from social_turing_test.core.game_status_checks import is_game_over, is_voted_out, is_time_to_vote, \
    all_players_joined, get_is_ai
from social_turing_test.core.vote_rounds import current_round_token, record_vote_round, vote_round_lock
from social_turing_test.llm_players.factory import llm_player_factory
from social_turing_test.llm_players.llm_constants import GAME_DIR_KEY, VOTING_WAITING_TIME, MAX_TIME_TO_WAIT


OPERATOR_COLOR = "yellow"  # the person running this file is the "operator" of the model
LLM_PLAYER_LOADED_MESSAGE = "The LLM PLayer was loaded successfully, " \
                            "now waiting for all other players to join..."
ALL_PLAYERS_JOINED_MESSAGE = "All players have joined, now the game can start."
LLM_VOTE_MESSAGE_FORMAT = "The LLM player has voted for: {}"
GAME_ENDED_MESSAGE = "Game has ended, without being voted out!"
GET_LLM_PLAYER_NAME_MESSAGE = "This game has multiple LLM players, which one you want to run now?"
ELIMINATED_MESSAGE = "This LLM player was eliminated from the game..."


# global variable
game_dir = Path()  # will be updated in get_llm_player


def get_llm_player():
    global game_dir
    game_dir = get_game_dir_from_argv()
    with open(game_dir / GAME_CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    llm_players_configs = [player for player in config[PLAYERS_KEY_IN_CONFIG] if player["is_llm"]]
    if not llm_players_configs:
        raise ValueError("No LLM player configured in this game")
    elif len(llm_players_configs) == 1:
        player_config = llm_players_configs[0]
    else:
        # Multi-agent runs pass --player <name> so each process controls a
        # specific agent non-interactively. Fall back to the interactive
        # prompt only when no --player arg was supplied.
        requested_player = get_player_from_argv()
        if requested_player:
            matches = [p for p in llm_players_configs if p["name"] == requested_player]
            if not matches:
                available = ", ".join(p["name"] for p in llm_players_configs)
                raise ValueError(
                    f"--player {requested_player!r} not found among LLM players: {available}")
            player_config = matches[0]
        else:
            player_name = get_player_name_from_user([player["name"] for player in llm_players_configs],
                                                    GET_LLM_PLAYER_NAME_MESSAGE, OPERATOR_COLOR)
            player_config = [player for player in llm_players_configs
                             if player["name"] == player_name][0]
    player_config[GAME_DIR_KEY] = game_dir
    llm_player = llm_player_factory(player_config)
    (game_dir / PERSONAL_STATUS_FILE_FORMAT.format(llm_player.name)).write_text(JOINED, encoding="utf-8")
    return llm_player


def _strip_timestamp(line: str) -> str:
    """Strip a leading [HH:MM:SS] (or similar) timestamp from a chat line.

    Returns the 'Name: text' portion, lowercased and whitespace-collapsed,
    so it can be compared across timestamps written by different writers
    (LLM-side vs game-manager-side may differ by a second or two).
    """
    import re as _re
    s = _re.sub(r'^\s*\[[^\]]*\]\s*', '', line or '')
    return _re.sub(r'\s+', ' ', s).strip().lower()


def read_messages_from_file(message_history, file_name, num_read_lines, skip_self_line=None):
    """Read new lines from the chat file, advance cursor, append to history.

    If `skip_self_line` is provided, any incoming line whose timestamp-stripped
    body matches it is consumed (cursor advances) but NOT appended to history.
    This prevents the agent's own message — written first by us locally and
    later by the game manager (with a different timestamp) — from being
    duplicated in the prompt history seen by the scheduler/generator/reviewer.

    Returns a tuple (consumed_count, skipped_self) where consumed_count is the
    total number of file lines processed (for cursor bookkeeping) and
    skipped_self is True iff we matched & dropped the pending self line.
    """
    with open(game_dir / file_name, "r", encoding='utf-8') as f:
        lines = f.readlines()[num_read_lines:]
    skipped_self = False
    target = _strip_timestamp(skip_self_line) if skip_self_line else None
    for line in lines:
        if target and not skipped_self and _strip_timestamp(line) == target:
            skipped_self = True  # consume but don't append - it's our echo
            continue
        message_history.append(line)
    return len(lines), skipped_self


def wait_writing_time(player, message, pipeline_elapsed: float = 0.0):
    """Adaptive typing-delay sleep, used when the player itself didn't apply one.

    Mirrors ReviewedPlayer._apply_typing_delay: subtracts the time the
    pipeline (scheduler + generator + reviewer) already burned, so a slow
    reviewer never adds extra latency on top of its own cost. Emits a
    `typing_delay` log entry so the metrics parser doesn't have to fall
    back on "total - components" residual estimates.
    """
    if player.num_words_per_second_to_wait <= 0 or not message:
        try: player.logger.log("typing_delay", f"skipped_short (0 words)")
        except Exception: pass
        return
    num_words = len(message.split())
    # Use post-reviewer message word count (the function receives the FINAL
    # text, not the generator's raw output). 1 second per word, capped.
    ideal = min(num_words * 1.0, MAX_TIME_TO_WAIT)
    remaining = max(0.0, ideal - pipeline_elapsed)
    try:
        if remaining > 0:
            player.logger.log(
                "typing_delay",
                f"sleep {remaining:.2f}s (ideal {ideal:.2f}s, elapsed {pipeline_elapsed:.2f}s)",
            )
        else:
            player.logger.log(
                "typing_delay",
                f"no_sleep: pipeline already took {pipeline_elapsed:.2f}s ≥ ideal {ideal:.2f}s",
            )
    except Exception:
        pass
    if remaining > 0:
        time.sleep(remaining)


def eliminate(player):
    # currently doesn't use player, but maybe in the future we can use player.logger for example
    print(colored(ELIMINATED_MESSAGE, OPERATOR_COLOR))


def _strategic_vote(player, candidate_vote_names: list, message_history: list) -> str:
    """Vote strategically — remove whoever most threatens the agent.

    Prefer the social-state slice (deterministic, no LLM) when the player
    carries a snapshot — vote the accuser who's actually suspicious of us, else
    the room's prime suspect. This makes the vote consistent with the same
    social read the generator/behaviour agents already use. Falls back to the
    LLM read and finally to random.
    """
    # Social-state-driven choice (only when a snapshot is present).
    snapshot = getattr(player, "snapshot", None)
    if snapshot is not None:
        try:
            mem = snapshot.get_social_state()
            suspects = [s for s in (mem.get("suspects_me") or []) if s in candidate_vote_names]
            if suspects:
                player.logger.log("strategic_vote", f"Chose accuser: {suspects[0]} (suspects_me)")
                return suspects[0]
            prime = (mem.get("prime_suspect") or "").strip()
            if prime and prime in candidate_vote_names:
                player.logger.log("strategic_vote", f"Chose prime_suspect: {prime}")
                return prime
        except Exception as e:
            player.logger.log("strategic_vote_error", f"snapshot read failed: {e}")
    try:
        names_list = ", ".join(candidate_vote_names)
        history_text = "".join(message_history[-30:])
        prompt = (
            f"You are an AI player in a Social Turing Test. Humans are trying to identify you.\n"
            f"Based on this conversation, which player is MOST suspicious of you or most likely to vote against you?\n"
            f"Conversation:\n{history_text}\n"
            f"Players you can vote for: {names_list}\n"
            f"Answer with ONLY one of these exact names (no explanation): {names_list}"
        )
        response = player.llm.generate(prompt, False, "").strip()
        for name in candidate_vote_names:
            if name.lower() in response.lower():
                player.logger.log("strategic_vote", f"Chose: {name} (response: {response!r})")
                return name
    except Exception as e:
        player.logger.log("strategic_vote_error", str(e))
    return random.choice(candidate_vote_names)


def get_vote_from_llm(player, message_history):
    """AI vote is random by default; strategic if voting_strategy='strategic' in config."""
    candidate_vote_names = (game_dir / REMAINING_PLAYERS_FILE).read_text(encoding="utf-8").splitlines()
    candidate_vote_names = [n for n in candidate_vote_names if n != player.name]

    voting_strategy = player.llm_config.get("voting_strategy", "random")

    if voting_strategy == "strategic":
        vote = _strategic_vote(player, candidate_vote_names, message_history)
        player.logger.log("strategic_vote_cast", vote)
        print(colored(f"AI voting strategically (silent): {vote}", OPERATOR_COLOR))
    else:
        vote = random.choice(candidate_vote_names)
        player.logger.log(MODEL_RANDOMLY_VOTED_LOG, vote)
        print(colored(f"AI voting randomly (silent): {vote}", OPERATOR_COLOR))
    update_vote(vote, player)


def update_vote(voted_name, player):
    time.sleep(VOTING_WAITING_TIME)
    # The AI is a separate process from Gunicorn.  Keep the append and token
    # write under the same cross-process lock used by /vote and Force Tally so
    # the admin cannot insert <ABSTAIN> between them.
    with vote_round_lock(game_dir, player.name):
        token = current_round_token(game_dir)
        with open(game_dir / PERSONAL_VOTE_FILE_FORMAT.format(player.name), "a", encoding="utf-8") as f:
            f.write(voted_name + "\n")
        record_vote_round(game_dir, player.name, token)
    print(colored(LLM_VOTE_MESSAGE_FORMAT.format(voted_name), OPERATOR_COLOR))


def _record_delivery_event(player, event_type, message, **fields):
    """Best-effort outer-loop outcome event correlated to the completed turn."""
    if not hasattr(player, "write_game_event"):
        return
    turn_id = getattr(player, "_last_completed_turn_id", "")
    turn_index = getattr(player, "_last_completed_turn_index", None)
    if turn_id:
        fields["turn_id"] = turn_id
    if turn_index is not None:
        fields["turn_index"] = turn_index
    fields["message"] = message[:200]
    try:
        player.write_game_event(event_type, **fields)
    except Exception as exc:
        player.logger.log("DELIVERY_EVENT_ERROR", f"{event_type}: {exc}")


def add_message_to_game(player, message_history):
    # Social Turing Test: no nighttime restrictions, AI can always chat during discussion phase
    # FIX: Check BEFORE generating to prevent race condition with voting phase
    if is_time_to_vote(game_dir):
        return None  # Voting has started, don't generate message
    
    # Use review-aware generation if available (ReviewedPlayer) for live reviewer logging
    # Track wall-clock from before the generate call so the typing delay
    # can subtract the pipeline cost (avoids double-waiting after a slow
    # reviewer).
    pipeline_start = time.time()
    # Whether the player's own generate_* method already applied a
    # `_apply_typing_delay` internally. Used to avoid double-waiting (and
    # double-logging `typing_delay`) below.
    inner_typing_done = False
    if hasattr(player, 'generate_message_with_review_info') and hasattr(player, 'reviewer') and player.reviewer:
        review_info = player.generate_message_with_review_info(message_history)
        message = review_info.get("message", "").strip()
        # LangGraph players apply the typing delay inside the graph's
        # output_node, so they set `typing_already_applied` to stop us waiting
        # twice. Flat ReviewedPlayers do NOT, so we still owe the outer wait.
        inner_typing_done = review_info.get("typing_already_applied", False)
        # Log reviewer details for live game visibility
        if review_info.get("was_reviewed"):
            raw = review_info.get("raw_message", "")
            modified = review_info.get("was_modified", False)
            steps = review_info.get("revision_steps", [])
            review_result = review_info.get("review_result", {})
            log_lines = [f"RAW: '{raw}'"]
            for step in steps:
                log_lines.append(
                    f"  Step {step.get('step_number')}: score={step.get('score')}, "
                    f"action={step.get('action')}, msg='{step.get('message', '')[:80]}'")
            if modified:
                log_lines.append(f"FINAL (modified): '{message}'")
            else:
                log_lines.append(f"FINAL (unchanged): '{message}'")
            if review_result:
                log_lines.append(f"Final score: {review_result.get('suspicion_score', '?')}/10, "
                                 f"red_flags: {review_result.get('red_flags', [])}")
            player.logger.log("LIVE_REVIEW_SUMMARY", "\n".join(log_lines))
    else:
        message = player.generate_message(message_history).strip()
        # ScheduleThenGeneratePlayer.generate_message and
        # ReviewedPlayer.generate_message both call _apply_typing_delay
        # internally before returning. Skip the outer wait_writing_time so
        # we don't double-sleep / double-log.
        inner_typing_done = hasattr(player, '_apply_typing_delay')
    
    # Dedup guard: prevent sending the exact same message twice in a row
    if message and hasattr(player, '_last_sent_message') and message == player._last_sent_message:
        player.logger.log("DEDUP_BLOCKED", f"Blocked duplicate message: '{message}'")
        _record_delivery_event(
            player, "delivery_dedup_drop", message,
            outcome="dedup_suppressed",
        )
        return None
    
    # Check again after generation (in case voting started during generation)
    if is_time_to_vote(game_dir):
        # Log explicitly so dashboards / per-message-detail can tell apart
        # "scheduler chose <wait>" (no message ever generated) from
        # "message generated but dropped because voting started".
        if message:
            player.logger.log(
                "AGENT_DROPPED_GENERATED_MESSAGE",
                f"Dropped after generation (voting started): '{message}'")
            _record_delivery_event(
                player, "delivery_phase_drop", message,
                outcome="phase_drop", stage="after_generation",
                reason="voting_started",
            )
        return None  # Voting started while generating, drop message
    
    if message:
        # artificially making the model taking time to write the message.
        # Pass the elapsed pipeline time so the typing wait subtracts it
        # (no extra sleep when generator+reviewer already exceeded budget).
        # Skip when the player already applied its own typing delay.
        if not inner_typing_done:
            wait_writing_time(player, message, pipeline_elapsed=time.time() - pipeline_start)
        # Final check after waiting
        if is_time_to_vote(game_dir):
            player.logger.log(
                "AGENT_DROPPED_GENERATED_MESSAGE",
                f"Dropped after typing-delay (voting started): '{message}'")
            _record_delivery_event(
                player, "delivery_phase_drop", message,
                outcome="phase_drop", stage="after_typing_delay",
                reason="voting_started",
            )
            return None  # Voting started while waiting, drop message
        formatted_message = format_message(player.name, message)
        with open(game_dir / PERSONAL_CHAT_FILE_FORMAT.format(player.name), "a", encoding="utf-8") as f:
            f.write(formatted_message)
        _record_delivery_event(
            player, "delivery_sent", message,
            outcome="sent", append_target="personal_chat",
        )
        # The authoritative personal-chat append just happened.
        # Notify the player hook; the synchronous agent has no pending acknowledgement.
        if hasattr(player, "record_public_append"):
            try:
                player.record_public_append(formatted_message)
            except Exception as exc:
                player.logger.log("record_public_append_error", str(exc))
        # Track last sent message for dedup guard
        player._last_sent_message = message
        print(colored(MODEL_CHOSE_TO_USE_TURN_LOG, OPERATOR_COLOR))
        # Return the formatted message so caller can add it to history immediately
        # This prevents the double-response bug where the scheduler re-triggers
        # before the game manager propagates our message to the public chat file
        return formatted_message
    else:
        print(colored(MODEL_CHOSE_TO_PASS_TURN_LOG, OPERATOR_COLOR))
        return None  # no message sent


def end_game(eliminated: bool):
    if not eliminated:
        print(colored(GAME_ENDED_MESSAGE, OPERATOR_COLOR))


def _preflight_validate_models(player):
    """Probe the generator + reviewer models BEFORE the game starts.

    Returns a dict {generator_ok, reviewer_ok, generator_model, reviewer_model,
    errors:[...]}. A misconfigured reviewer model (the 9056 failure) is caught
    here and surfaced to the admin lobby instead of silently degrading every
    message to a fake score. Never raises.
    """
    status = {
        "generator_ok": True,
        "reviewer_ok": None,   # None = no reviewer configured
        "generator_model": (player.llm_config or {}).get("model_name", "unknown"),
        "reviewer_model": None,
        "errors": [],
    }
    # Generator probe (tiny, one-time).
    try:
        out = player.llm.generate("Reply with the single word: ok", False, "",
                                   max_output_tokens=5)
        if not (out and str(out).strip()):
            status["generator_ok"] = False
            status["errors"].append("generator returned empty on preflight")
    except Exception as e:  # noqa: BLE001
        status["generator_ok"] = False
        status["errors"].append(f"generator: {str(e)[:200]}")
    # Reviewer probe (only if one is configured/enabled).
    reviewer = getattr(player, "reviewer", None)
    if reviewer is not None:
        status["reviewer_model"] = getattr(reviewer, "model_name", None)
        try:
            ok, err = reviewer.validate()
            status["reviewer_ok"] = bool(ok)
            if not ok:
                status["errors"].append(f"reviewer ({status['reviewer_model']}): {err}")
        except Exception as e:  # noqa: BLE001
            status["reviewer_ok"] = False
            status["errors"].append(f"reviewer ({status['reviewer_model']}): {str(e)[:200]}")
    return status


def main():
    player = get_llm_player()
    print(colored(LLM_PLAYER_LOADED_MESSAGE, OPERATOR_COLOR))
    # Pre-flight validate the models so the lobby can show generator/reviewer
    # health and START isn't pressed against a dead reviewer (warn + degrade,
    # not block - per product decision).
    model_status = _preflight_validate_models(player)
    if model_status["reviewer_ok"] is False:
        print(colored(
            f"[PREFLIGHT] Reviewer model '{model_status['reviewer_model']}' FAILED validation — "
            "messages will be sent UNREVIEWED (degraded). Fix the reviewer model to enable review.",
            "yellow"), flush=True)
    if not model_status["generator_ok"]:
        print(colored(
            f"[PREFLIGHT] Generator model '{model_status['generator_model']}' FAILED validation: "
            f"{model_status['errors']}", "red"), flush=True)
    # Signal readiness so the admin lobby can show an "AI ready" badge and
    # only enable START once the agent's wrapper is constructed. Best-effort.
    try:
        player.write_game_event(
            "agent_ready",
            model=model_status["generator_model"],
            generator_ok=model_status["generator_ok"],
            reviewer_ok=model_status["reviewer_ok"],
            reviewer_model=model_status["reviewer_model"],
            model_errors=model_status["errors"],
        )
        player.write_game_event("agent_models_status", **model_status)
    except Exception:
        pass
    player.logger.log("AGENT_LIFECYCLE", "Agent initialized, waiting for players to join")
    
    while not all_players_joined(game_dir):
        time.sleep(0.5)
    
    print(colored(ALL_PLAYERS_JOINED_MESSAGE, OPERATOR_COLOR))
    player.logger.log("AGENT_LIFECYCLE", "All players joined, game starting")
    
    message_history = []
    num_read_lines_manager = num_read_lines_daytime = 0
    eliminated = False
    iteration_count = 0
    voting_wait_start = None
    VOTING_TIMEOUT = 300  # 5 minutes max wait for voting
    
    try:
        while not is_game_over(game_dir):
            iteration_count += 1
            # Diluted: every tick (~1/s) flooded the log (part of the 5-7K
            # lines/min bloat). Every 15th keeps the healthy-gap signal well
            # under the Issues tab's 30s unresponsive threshold.
            if iteration_count == 1 or iteration_count % 15 == 0:
                player.logger.log("AGENT_HEARTBEAT", f"Loop iteration {iteration_count} at {get_current_timestamp()}")
            
            # Read new messages. If we just sent a message, the
            # `_pending_self_line` skip prevents the manager-written copy of
            # our own message (with a slightly different timestamp than the
            # locally-appended one) from being ingested twice into the prompt
            # history seen by the reviewer/scheduler/generator.
            self_skip = getattr(player, '_pending_self_line', None)
            new_manager_lines, _ = read_messages_from_file(
                message_history, PUBLIC_MANAGER_CHAT_FILE, num_read_lines_manager)
            new_daytime_lines, skipped_self = read_messages_from_file(
                message_history, PUBLIC_DAYTIME_CHAT_FILE, num_read_lines_daytime,
                skip_self_line=self_skip)
            num_read_lines_manager += new_manager_lines
            num_read_lines_daytime += new_daytime_lines
            if skipped_self:
                player._pending_self_line = None
            
            if new_manager_lines > 0 or new_daytime_lines > 0:
                player.logger.log("AGENT_STATUS", 
                    f"Read {new_manager_lines} manager messages, {new_daytime_lines} daytime messages. "
                    f"Total history: {len(message_history)} lines")
            
            # Check elimination
            if is_voted_out(player.name, game_dir):
                player.logger.log("AGENT_LIFECYCLE", "Agent was voted out")
                eliminate(player)
                eliminated = True
                break
            
            # Handle voting phase
            if is_time_to_vote(game_dir):
                if voting_wait_start is None:
                    voting_wait_start = time.time()
                    player.logger.log("AGENT_STATUS", "Entering voting wait loop")
                    
                    # Cast vote when first entering voting phase
                    player.logger.log("AGENT_STATUS", "Attempting to cast vote")
                    try:
                        get_vote_from_llm(player, message_history)
                    except Exception as e:
                        player.logger.log("AGENT_ERROR", f"Failed to cast vote: {str(e)}")
                        print(colored(f"ERROR casting vote: {str(e)}", "red"))
                
                # Check timeout
                if time.time() - voting_wait_start > VOTING_TIMEOUT:
                    player.logger.log("AGENT_ERROR", 
                        f"Voting timeout exceeded ({VOTING_TIMEOUT}s) - breaking out")
                    break
                
                # Wait for voting to end
                time.sleep(0.5)
                continue
            else:
                # Reset voting wait tracker when not in voting phase
                if voting_wait_start is not None:
                    player.logger.log("AGENT_STATUS", "Exited voting phase")
                    voting_wait_start = None
                
                # Try to generate and send message
                player.logger.log("AGENT_STATUS", "Attempting to generate message")
                try:
                    sent_formatted = add_message_to_game(player, message_history)
                    
                    if sent_formatted:
                        # Add our message to history IMMEDIATELY so the scheduler
                        # sees we already responded (prevents double-response bug
                        # when game manager hasn't yet propagated to public chat).
                        # Track the pending self-line so the next public-file read
                        # consumes the manager-written echo without duplicating it.
                        message_history.append(sent_formatted)
                        player._pending_self_line = sent_formatted
                        player.logger.log("AGENT_STATUS", "Message sent successfully, waiting for propagation")
                        # Cooldown: wait for game manager to propagate message to public chat
                        time.sleep(2)
                        # Refresh history after sending
                        new_manager, _ = read_messages_from_file(
                            message_history, PUBLIC_MANAGER_CHAT_FILE, num_read_lines_manager)
                        new_daytime, skipped_self = read_messages_from_file(
                            message_history, PUBLIC_DAYTIME_CHAT_FILE, num_read_lines_daytime,
                            skip_self_line=player._pending_self_line)
                        num_read_lines_manager += new_manager
                        num_read_lines_daytime += new_daytime
                        if skipped_self:
                            player._pending_self_line = None
                        player.logger.log("AGENT_STATUS",
                            f"After refresh: {new_manager} manager, {new_daytime} daytime messages")
                        continue
                    else:
                        player.logger.log("AGENT_STATUS", "No message sent (decided to wait)")
                        time.sleep(0.5)
                        
                except Exception as e:
                    player.logger.log("AGENT_ERROR", f"Exception in add_message_to_game: {str(e)}")
                    print(colored(f"ERROR: {str(e)}", "red"))
                    time.sleep(1)  # Brief pause before retrying
        
        player.logger.log("AGENT_LIFECYCLE", f"Exited main loop. Game over: {is_game_over(game_dir)}")
        
    except Exception as e:
        player.logger.log("AGENT_FATAL_ERROR", f"Fatal exception in main loop: {str(e)}")
        print(colored(f"FATAL ERROR: {str(e)}", "red"))
        import traceback
        player.logger.log("AGENT_FATAL_ERROR", f"Traceback: {traceback.format_exc()}")
        raise
    
    # Final check: if game ended but we haven't detected elimination yet, check now
    if not eliminated and is_voted_out(player.name, game_dir):
        eliminated = True
        eliminate(player)
        player.logger.log("AGENT_LIFECYCLE", "Detected elimination in final check")
    
    player.logger.log("AGENT_LIFECYCLE", f"Game ended. Eliminated: {eliminated}")
    end_game(eliminated)


if __name__ == '__main__':
    main()
