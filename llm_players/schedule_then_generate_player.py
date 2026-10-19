import re
import time
import random
from core.game_constants import REMAINING_PLAYERS_FILE, GAME_MANAGER_NAME, MESSAGE_PARSING_PATTERN
from llm_players.llm_constants import turn_task_into_prompt, SCHEDULE_THEN_GENERATE_TYPE, \
    make_more_human_like, TALKATIVE_PROMPT, QUIETER_PROMPT, STRONGLY_QUIETER_PROMPT, \
    BALANCED_TALK_PROMPT, INDIFFERENCE_NUDGE
from llm_players.llm_player import LLMPlayer
from llm_players.llm_wrapper import LLMWrapper


def no_one_has_talked_yet_in_current_phase(message_history):
    if not message_history:
        return True
    matcher = re.match(MESSAGE_PARSING_PATTERN, message_history[-1])
    if not matcher:
        return True
    name = matcher.group(4)  # depends on MESSAGE_PARSING_PATTERN
    return name == GAME_MANAGER_NAME


class ScheduleThenGeneratePlayer(LLMPlayer):

    TYPE_NAME = SCHEDULE_THEN_GENERATE_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = self.llm  # using the same one for generation...
        self.message_in_progress = False  # Prevent duplicate messages during generation

        # Adaptive typing-rhythm configuration and memory window.
        llm_config = kwargs.get('llm_config', {})
        self._typing_enabled = llm_config.get("typing_delay_enabled", True)
        self._typing_sec_per_word = llm_config.get("typing_sec_per_word", 1.0)
        self._typing_wps = llm_config.get("typing_wps", 1.6)  # unused by seconds-per-word timing
        self._typing_jitter = llm_config.get("typing_jitter_pct", 0.25)
        self._typing_min = llm_config.get("typing_min_delay_sec", 0.8)
        self._typing_max = llm_config.get("typing_max_delay_sec", 20.0)
        # Typing-delay skip only: very short messages have no realistic typing
        # pause. Reviewer-skip is handled separately in ReviewedPlayer.
        self._typing_skip_words = llm_config.get("typing_min_words", 2)
        self._memory_window_size = llm_config.get("memory_window_size", None)
        self._memory_decay_mode = llm_config.get("memory_decay_mode", "none")

    def should_generate_message(self, message_history):
        # Prevent duplicate messages: if already generating a message, return False
        if self.message_in_progress:
            self.logger.log("should_generate_message", "BLOCKED: message already in progress")
            return False
        
        if no_one_has_talked_yet_in_current_phase(message_history):
            return False
        prompt = self.create_scheduling_prompt(message_history)
        self.logger.log("prompt in should_generate_message", prompt)
        sched_start = time.time()
        decision = self.scheduler.generate(
            prompt, True, self.get_system_info_message(only_special_tokens=True))
        self._log_cost("scheduler")
        sched_duration = time.time() - sched_start
        self.logger.log("scheduler_latency", f"{sched_duration:.2f}s")
        self.logger.log("decision in should_generate_message", decision)
        will_send = self.interpret_scheduling_decision(decision)
        # Mirror the scheduler call into game_events.jsonl so the admin
        # Journey modal can show the decision for *every* turn (send or
        # wait) without parsing the text log. Capture the last 5 chat
        # lines as `recent_chat` to make wait-decisions intelligible
        # post-hoc.
        try:
            recent_chat = list(message_history[-5:]) if message_history else []
            self.write_game_event(
                "scheduler_decision",
                decision=("send" if will_send else "wait"),
                raw_decision=str(decision)[:200],
                latency_ms=int(sched_duration * 1000),
                prompt_chars=len(prompt) if prompt else 0,
                recent_chat=recent_chat,
            )
        except Exception:
            pass
        return will_send

    def _apply_typing_delay(self, message: str, pipeline_elapsed: float):
        """Sleep whatever remains of the ideal human-typing budget."""
        if not self._typing_enabled or not message:
            return
        word_count = len(message.split())
        # ~1s/word floor (see ReviewedPlayer._apply_typing_delay),
        # plus a constant base so even short replies aren't instant. The cap must
        # never drop below the floor (long messages under-waited).
        sec_per_word = getattr(self, "_typing_sec_per_word", 1.0)
        base = getattr(self, "_typing_base_delay", 2.0)
        floor = float(word_count) * sec_per_word + base
        ideal = max(floor, self._typing_min)
        ideal = min(ideal, max(self._typing_max, floor))
        # No jitter: exact word_count+2 seconds total.
        remaining = ideal - pipeline_elapsed
        if remaining > 0:
            self.logger.log("typing_delay", f"sleep {remaining:.2f}s (ideal {ideal:.2f}s, elapsed {pipeline_elapsed:.2f}s, base {base:.1f}s)")
            time.sleep(remaining)
        else:
            self.logger.log("typing_delay", f"no_sleep: pipeline already took {pipeline_elapsed:.2f}s ≥ ideal {ideal:.2f}s")

    def generate_message(self, message_history):
        if self.should_generate_message(message_history):
            # Lock to prevent duplicate message generation
            self.message_in_progress = True
            typing_flag = self.game_dir / f"{self.name}_typing.txt"
            typing_flag.touch()
            pipeline_start = time.time()
            try:
                prompt = self.create_generation_prompt(message_history)
                self.logger.log("prompt in generate_message", prompt)
                gen_start = time.time()
                message = self.llm.generate(
                    prompt, False, self.get_system_info_message(attention_to_not_repeat=True))
                self._log_cost("generator")
                gen_duration = time.time() - gen_start
                self.logger.log("generator_latency", f"{gen_duration:.2f}s")
                # Provenance: literal generator output (pre-trim, pre-humanizer).
                try:
                    from core.chat_logging import append_ai_generator_raw
                    append_ai_generator_raw(self.game_dir, self.name, message)
                except Exception:
                    pass
                # Silent backstop: trim if too long (20 words max)
                message = self.trim_message_if_too_long(message, max_words=20)
                # Pass message_history for smart capitalization matching.
                # Comma normalization opt-in via llm_config (admin UI toggle).
                message = make_more_human_like(
                    message,
                    message_history,
                    normalize_commas=self.llm_config.get("normalize_commas", False),
                )
                # Provenance: generator output after our forced edits.
                # (No reviewer in this player class — this is the final text.)
                try:
                    from core.chat_logging import append_ai_generator_processed
                    append_ai_generator_processed(self.game_dir, self.name, message)
                except Exception:
                    pass
                self._apply_typing_delay(message, time.time() - pipeline_start)
                return message
            finally:
                # Always unlock, even if an error occurs
                self.message_in_progress = False
                typing_flag.unlink(missing_ok=True)
                self.logger.log("generate_message", "UNLOCKED: message generation complete")
        else:
            return ""

    def talkative_scheduling_prompt_modifier(self, message_history):
        if not message_history:
            return TALKATIVE_PROMPT
        
        all_players = (self.game_dir / REMAINING_PLAYERS_FILE).read_text().splitlines()
        players_counts = {player: 0 for player in all_players}
        
        phases_ended_count = 0
        for message in message_history[::-1]:
            if f"] {GAME_MANAGER_NAME}: " in message:
                if "voted for" in message:
                    continue
                if "has ended" in message:
                    phases_ended_count += 1
                    if phases_ended_count >= 2:
                        break
                    continue
            
            matcher = re.match(MESSAGE_PARSING_PATTERN, message)
            if matcher:
                speaker = matcher.group(4)
                if speaker in players_counts:
                    players_counts[speaker] += 1
        
        all_player_messages = sum(players_counts.values())
        n_players = max(len([p for p in all_players if p.strip()]), 1)
        if not all_player_messages:
            return TALKATIVE_PROMPT

        # Ratio of THIS agent's recent messages to the per-player average.
        #   ratio == 1.0  -> exactly average
        #   ratio  > 1.0  -> talking MORE than its fair share (graduated brake)
        #   ratio  < 1.0  -> quieter than average (encourage joining in)
        avg_per_player = all_player_messages / n_players
        ai_count = players_counts.get(self.name, 0)
        ratio = (ai_count / avg_per_player) if avg_per_player > 0 else 0.0
        stats = (f"(You've sent {ai_count} of the last {all_player_messages} messages; the "
                 f"per-player average is {avg_per_player:.1f}, so you're at about {ratio:.1f}x "
                 f"the average.) ")

        # Graduated: the further above average, the firmer the brake.
        if ratio < 0.75:
            return TALKATIVE_PROMPT
        if ratio <= 1.25:
            return BALANCED_TALK_PROMPT
        if ratio <= 1.75:
            return stats + QUIETER_PROMPT
        return stats + STRONGLY_QUIETER_PROMPT

    def _cadence_scheduling_nudge(self, directed_at_agent: bool) -> str:
        """Cadence guidance for the scheduler prompt from shared memory.

        Uses Behavior Adapter cadence and Social State Tracker room norms.
        Returns an empty string without a snapshot or applicable guidance.
        """
        snap = getattr(self, "snapshot", None)
        if snap is None:
            return ""
        try:
            cadence = snap.get_cadence() if hasattr(snap, "get_cadence") else {}
            social = snap.get_social_state() if hasattr(snap, "get_social_state") else {}
        except Exception:
            return ""
        level = (cadence or {}).get("level", "balanced")
        room_norm = (social or {}).get("room_norm", "")
        answers_everything = "everyone" in (room_norm or "")
        if level == "over" and not directed_at_agent and not answers_everything:
            return ("Note: you've been talking noticeably MORE than the others, this "
                    "wasn't aimed at you, and people here don't answer every message — "
                    "strongly prefer to WAIT and let others talk. ")
        if level == "over" and not directed_at_agent:
            return ("Note: you've been talking more than others — only jump in if you "
                    "really add something, otherwise wait. ")
        if level == "under":
            return ("Note: you've been quieter than the others — it's fine to jump in "
                    "a bit more. ")
        return ""

    @staticmethod
    def _merge_talk_and_indifference(talk_signal: str, mentioned: bool):
        """Merge encouragement and room-wide indifference into one instruction.

        Returns (talk_text, skip_indifference) so the scheduler does not receive
        contradictory participation guidance.
        """
        encourage = talk_signal.strip() == TALKATIVE_PROMPT.strip()
        if encourage and not mentioned:
            return ("Note: you've been quieter than the others. This particular "
                    "message wasn't aimed at you, so you don't HAVE to answer it — "
                    "but do look for a natural opening soon: react to something "
                    "specific someone said, or start a light topic of your own. ",
                    True)
        return talk_signal.strip() + " ", False

    def create_scheduling_prompt(self, message_history, extra_context=""):
        mentioned = False
        asked_question = False
        
        if message_history:
            # Track if AI was recently active in conversation (for follow-up detection)
            ai_was_recently_active = False
            for message in message_history[-3:]:  # Check last 3 messages
                matcher = re.match(MESSAGE_PARSING_PATTERN, message)
                if matcher:
                    speaker = matcher.group(4)
                    if speaker == self.name:
                        ai_was_recently_active = True
                        break
            
            # Check last 1-2 messages for direct mentions
            recent_messages = message_history[-2:]
            
            for message in recent_messages:
                message_lower = message.lower()
                
                # Parse and filter by speaker
                matcher = re.match(MESSAGE_PARSING_PATTERN, message)
                if matcher:
                    speaker = matcher.group(4)
                    # Skip messages from Game Manager or self
                    if speaker in {GAME_MANAGER_NAME, self.name}:
                        continue
                
                # Direct mention by name with question words
                if self.name.lower() in message_lower:
                    mentioned = True
                    # Check for question words if mentioned
                    question_words = ["where", "what", "who", "how", "why", "when", "which"]
                    for qword in question_words:
                        if qword in message_lower:
                            asked_question = True
                            break
            
            # Follow-up question detection: only when AI was recently active
            if ai_was_recently_active and recent_messages:
                last_message = recent_messages[-1]
                last_message_lower = last_message.lower()
                
                # Parse speaker
                matcher = re.match(MESSAGE_PARSING_PATTERN, last_message)
                if matcher:
                    speaker = matcher.group(4)
                    # Only check if it's from another player (not GM or self)
                    if speaker not in {GAME_MANAGER_NAME, self.name}:
                        # Check if it's a question (with or without "?")
                        is_question = (
                            "?" in last_message_lower or
                            any(last_message_lower.strip().startswith(qw) for qw in 
                                ["where", "what", "when", "why", "who", "how", "which"]) or
                            last_message_lower.strip() in ["really", "really?", "why", "how come"]
                        )
                        
                        # Check for follow-up patterns
                        is_follow_up = any(
                            last_message_lower.strip().startswith(p) for p in 
                            ["and ", "so ", "but ", "then ", "really", "why", "how come"]
                        )
                        
                        # If it's a question or follow-up after AI spoke, treat as directed to AI
                        if is_question or is_follow_up:
                            asked_question = True
                            self.logger.log("follow_up_detected", 
                                f"Detected follow-up: ai_active={ai_was_recently_active}, "
                                f"is_question={is_question}, is_follow_up={is_follow_up}")
        
        message_history = self._apply_memory_window(message_history)

        task = f"Do you want to send a message to the group chat now, or do you prefer to wait " \
               f"for now and see what messages others will send? " \
               f"Remember to choose to send a message only if your contribution to the " \
               f"discussion in the current time will be meaningful enough. " \

        if mentioned:
            task += f"Note: Your name was mentioned recently. Consider responding if it feels natural. "
        else:
            if asked_question:
                task += f"Note: Someone may be asking something. Consider responding if it feels natural. "
        # Always inject talk-share signal so the LLM knows if it's over-talking,
        # even when it's being directly addressed (being asked doesn't disable the
        # signal). Coordinate with the indifference nudge below so the
        # prompt never carries "be talkative!" and "prefer to WAIT" together.
        talk_signal = self.talkative_scheduling_prompt_modifier(message_history)
        talk_text, skip_indifference = self._merge_talk_and_indifference(talk_signal, mentioned)
        task += talk_text

        # Cadence brake (applies EVEN when mentioned). Behavior Adapter
        # writes how much the agent is talking vs others; Social State Tracker writes
        # whether the room answers every question. If the agent is over-talking,
        # this question isn't aimed at it, and the room is selective, strongly
        # prefer waiting — humans don't answer every single message. The LLM
        # still makes the final call; this only shapes its input.
        # Skipped when the merged quiet+room-wide line already covered cadence.
        if not skip_indifference:
            task += self._cadence_scheduling_nudge(mentioned or asked_question)

        # When the recent message did NOT name the agent, it's a room-wide
        # question/comment. Nudge the agent to stay a bit detached and not feel it
        # must answer every one — humans field only some open questions. Directly
        # addressed (name mentioned) skips this so the agent still answers those.
        if not mentioned and not skip_indifference:
            task += INDIFFERENCE_NUDGE
        if extra_context:
            task += "\n" + extra_context.strip() + "\n"

        task += f"Reply only with `{self.use_turn_token}` if you want to send a message now, " \
               f"or only with `{self.pass_turn_token}` if you want to wait for now, " \
               f"based on your decision! "
        
        return turn_task_into_prompt(task, message_history)

    def create_generation_prompt(self, message_history):
        message_history = self._apply_memory_window(message_history)
        task = f"Add a very short message to the game's chat. " \
               f"Be specific and keep it relevant to the current situation, " \
               f"according to the last messages and the game's status. " \
               f"Your message should only be one short sentence! " \
               f"Don't add a message that you've already added (in the chat history)! " \
               f"It is very important that you don't repeat yourself! " \
               f"Match your style of message to the other player's message style, " \
               f"with more emphasis on more recent messages.\n"
        
        return turn_task_into_prompt(task, message_history)
