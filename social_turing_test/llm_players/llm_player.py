import json
import re
import threading
import time
from abc import ABC, abstractmethod
from social_turing_test.core.game_constants import get_role_string, GAME_START_TIME_FILE, PERSONAL_CHAT_FILE_FORMAT, \
    MESSAGE_PARSING_PATTERN, SCHEDULING_DECISION_LOG, MODEL_CHOSE_TO_USE_TURN_LOG, MODEL_CHOSE_TO_PASS_TURN_LOG
from social_turing_test.llm_players.llm_constants import turn_task_into_prompt, GENERAL_SYSTEM_INFO, \
    PASS_TURN_TOKEN_KEY, USE_TURN_TOKEN_KEY, WORDS_PER_SECOND_WAITING_KEY, PASS_TURN_TOKEN_OPTIONS, \
    PERSONA_KEY, build_system_prompt
from social_turing_test.llm_players.llm_wrapper import LLMWrapper
from social_turing_test.llm_players.logger import Logger

# Module-level lock for game_events.jsonl / cost_log.jsonl appends. Guards
# against interleaved writes when multiple players run in the same process
# File.write() on CPython 3 is thread-safe for single short writes,
# but we acquire a lock for extra safety (partial-line corruption would
# break JSONL parsers downstream).
_EVENT_LOG_LOCK = threading.Lock()


class LLMPlayer(ABC):

    TYPE_NAME = None

    def __init__(self, name, is_ai, llm_config, game_dir, **kwargs):
        self.name = name
        self.is_ai = is_ai
        self.role = get_role_string(is_ai)
        self.game_dir = game_dir
        self.llm_config = llm_config  # stored for voting strategy and other per-player config lookups
        self.logger = Logger(name, game_dir)
        # Multi-agent (agent-vs-agent) games run N players in N separate
        # processes that would otherwise clobber the shared game_events.jsonl /
        # cost_log.jsonl / memory_trace.jsonl. When `multi_agent_logs` is set
        # we namespace those files per player. Single-AI games use shared
        # filenames for the dashboard and readers.
        self.multi_agent_logs = bool(llm_config.get("multi_agent_logs", False))
        self._log_suffix = f"_{name}" if self.multi_agent_logs else ""
        self.pass_turn_token = llm_config[PASS_TURN_TOKEN_KEY]
        self.use_turn_token = llm_config[USE_TURN_TOKEN_KEY]
        self.num_words_per_second_to_wait = llm_config[WORDS_PER_SECOND_WAITING_KEY]
        # Get persona from config, or use None for default
        self.persona_id = llm_config.get(PERSONA_KEY, None)
        # Experimental toggle - affects system prompt + reviewer
        # prompts + max_tokens override. Disabled by default.
        self.experimental = bool(llm_config.get("experimental_agent", False))
        # Build personalized system prompt
        self.system_prompt = build_system_prompt(self.persona_id, experimental=self.experimental)
        # Experimental override: bump max_tokens / max_new_tokens before the
        # wrapper is constructed. 0 means "keep original".
        if self.experimental:
            exp_mt = llm_config.get("experimental_max_tokens", 80)
            if isinstance(exp_mt, int) and exp_mt > 0:
                llm_config = dict(llm_config)
                if "max_tokens" in llm_config:
                    llm_config["max_tokens"] = exp_mt
                if "max_new_tokens" in llm_config:
                    llm_config["max_new_tokens"] = exp_mt
        self.llm = LLMWrapper(self.logger, **llm_config)

    def get_system_info_message(self, attention_to_not_repeat=False, only_special_tokens=False):
        system_info = f"Your name is {self.name}. {self.system_prompt}\n"
        chat_room_open_time = (self.game_dir / GAME_START_TIME_FILE).read_text(encoding='utf-8').strip()
        if chat_room_open_time:
            system_info += f"The game's chat room was open at [{chat_room_open_time}].\n"
        
        from social_turing_test.core.game_constants import CONVERSATION_TOPIC_FILE
        topic_file = self.game_dir / CONVERSATION_TOPIC_FILE
        if topic_file.exists():
            topic = topic_file.read_text(encoding='utf-8').strip()
            if topic:
                system_info += f"\nThe current conversation topic is: {topic}. "
                system_info += "You don't need to stick to it strictly — people may drift naturally.\n"
        
        if attention_to_not_repeat:
            try:
                previous_messages = (self.game_dir / PERSONAL_CHAT_FILE_FORMAT.format(self.name)
                                     ).read_text(encoding='utf-8').splitlines()
            except UnicodeDecodeError:
                # Fallback to cp1252 for Windows compatibility
                previous_messages = (self.game_dir / PERSONAL_CHAT_FILE_FORMAT.format(self.name)
                                     ).read_text(encoding='cp1252').splitlines()
            if previous_messages:
                system_info += f"\nFor reference, here are a few of your recent messages:\n"
                for message in previous_messages[-3:]:
                    matcher = re.match(MESSAGE_PARSING_PATTERN, message)
                    if not matcher:
                        continue
                    message_content = matcher.group(5)
                    system_info += f"  • \"{message_content}\"\n"
                
                system_info += "\nWhen deciding what to say next, avoid repeating the same wording or reaction.\n"
        
        if only_special_tokens:
            system_info += f"You can ONLY respond with one of two possible outputs:\n" \
                           f"{self.pass_turn_token} - indicating your character in the game " \
                           f"should wait and not send a message in the current timing;\n" \
                           f"{self.use_turn_token} - indicating your character in the game should " \
                           f"send a message to the public chat now.\n\n" \
                           f"You must NEVER output any other text, explanations, or variations " \
                           f"of these tokens. Only these exact tokens are allowed: " \
                           f"{self.pass_turn_token} or {self.use_turn_token}.\n"
        return system_info

    @abstractmethod
    def should_generate_message(self, context):
        raise NotImplementedError()

    @abstractmethod
    def generate_message(self, message_history):
        raise NotImplementedError()
    
    def _apply_memory_window(self, message_history: list) -> list:
        """Return message_history trimmed to the configured memory window.

        memory_window_size=None (default): full history.
        memory_decay_mode='drop_older': keep last N.
        memory_decay_mode='summarize_older': keep last N verbatim; older N/2
            messages are concatenated into one synthetic summary line.
        """
        size = getattr(self, '_memory_window_size', None)
        if size is None or not message_history or len(message_history) <= size:
            return message_history
        mode = getattr(self, '_memory_decay_mode', 'none')
        if mode == 'summarize_older':
            older = message_history[:-size]
            half = max(1, len(older) // 2)
            summary_msgs = older[-half:]
            summary_text = " | ".join(m.strip() for m in summary_msgs)
            summary_line = f"[earlier conversation summary: {summary_text}]\n"
            return [summary_line] + message_history[-size:]
        # 'drop_older' or 'none' with size set: drop older
        return message_history[-size:]

    def trim_message_if_too_long(self, message, max_words=20):
        """Silent backstop: trim message if it exceeds max_words"""
        words = message.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return message

    def interpret_scheduling_decision(self, decision):
        if not decision:
            generate = False
        elif self.pass_turn_token in decision:
            generate = False
        elif self.use_turn_token in decision:
            generate = True
        # for more robustness:
        elif any([option in decision for option in PASS_TURN_TOKEN_OPTIONS]):
            generate = False
        else:
            generate = True
        if generate:
            self.logger.log(SCHEDULING_DECISION_LOG, MODEL_CHOSE_TO_USE_TURN_LOG)
        else:
            self.logger.log(SCHEDULING_DECISION_LOG, MODEL_CHOSE_TO_PASS_TURN_LOG)
        return generate

    def _log_cost(self, component: str, **extra):
        """Write LLM usage from the last generate() call to cost_log.jsonl.

        ``extra`` fields (e.g. latency_s) are merged into the usage entry."""
        usage = self.llm.pop_usage()
        if not usage:
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "component": component,
            "model": self.llm.model_name,
            **usage,
            **extra,
        }
        cost_file = self.game_dir / "events" / f"cost_log{self._log_suffix}.jsonl"
        # Lock-protected append to avoid interleaved partial writes.
        with _EVENT_LOG_LOCK:
            cost_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cost_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")

    def write_game_event(self, event_type: str, **fields):
        """Append a structured event to games/<id>/game_events.jsonl.

        Used for real-time observability (admin Reviewer Log tab, dashboard,
        counterfactual viewer). Writes are best-effort — failures are silently
        swallowed so logging never breaks the game loop.

        Serialized through a module-level lock to prevent partial-line
        corruption when multiple players in the same process log concurrently.
        """
        try:
            from social_turing_test.core.event_clock import append_event

            entry = {
                "event_type": event_type,
                "player": self.name,
            }
            entry.update(fields)
            out = self.game_dir / "events" / f"game_events{self._log_suffix}.jsonl"
            with _EVENT_LOG_LOCK:
                append_event(out, entry)
        except Exception as e:
            # Never let logging kill the game.
            try:
                self.logger.log("game_events_error", f"{type(e).__name__}: {e}")
            except Exception:
                pass

    def get_vote(self, message_history, candidate_vote_names):
        import random
        
        if not candidate_vote_names:
            self.logger.log("get_vote error", "No candidates to vote for")
            return None
        
        vote = random.choice(candidate_vote_names)
        self.logger.log("random vote selected", 
                       f"Voting for: {vote} (randomly chosen from {len(candidate_vote_names)} candidates)")
        
        return vote
