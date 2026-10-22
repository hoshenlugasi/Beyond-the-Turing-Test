"""
Shared review and typing helpers for the Social Turing Test agent pipeline.

Provides reviewer configuration, review-result handling, provenance logging and
human-typing delays inherited by the LangGraph pipeline base.
"""

from social_turing_test.llm_players.schedule_then_generate_player import ScheduleThenGeneratePlayer
from social_turing_test.llm_players.reviewer_base import ReviewResult
from social_turing_test.llm_players.reviewer_factory import create_reviewer
from social_turing_test.llm_players.llm_constants import make_more_human_like
import time
import random

# Config keys for reviewer settings
REVIEWER_ENABLED_KEY = "reviewer_enabled"
REVIEWER_THRESHOLD_KEY = "reviewer_threshold"
REVIEWER_MAX_ATTEMPTS_KEY = "reviewer_max_attempts"
REVIEWER_MODEL_KEY = "reviewer_model"
REVIEWER_BACKEND_KEY = "reviewer_backend"  # "gemini" (default) or "together"

class ReviewedPlayer(ScheduleThenGeneratePlayer):
    """
    Shared reviewer setup, review helpers and typing-delay implementation.
    
    Supplies review and typing utilities to the pipeline through inheritance.
    
    Config options (add to llm_config):
        reviewer_enabled: bool - Enable/disable reviewer (default: True)
        reviewer_threshold: int - Suspicion score threshold 1-10 (default: 4)
        reviewer_max_attempts: int - Max fix attempts (default: 3)
        reviewer_model: str - Optional reviewer model override
    """
    
    TYPE_NAME = "reviewed_schedule_then_generate"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get reviewer config from llm_config
        llm_config = kwargs.get('llm_config', {})
        
        self.reviewer_enabled = llm_config.get(REVIEWER_ENABLED_KEY, True)
        self.reviewer_threshold = llm_config.get(REVIEWER_THRESHOLD_KEY, 4)
        # Early-exit ceiling: after attempt 2, accept a message whose score is
        # <= this value instead of burning a 3rd reviewer call. Scores of 5
        # remain borderline-suspicious, so those messages still get one
        # more fix attempt rather than being silently accepted.
        self.reviewer_good_enough_score = llm_config.get("reviewer_good_enough_score", 4)
        # Allow up to 3 attempts. The pipeline budget below caps
        # wall-clock time so a 3rd attempt only runs if the first two were fast.
        self.reviewer_max_attempts = llm_config.get(REVIEWER_MAX_ATTEMPTS_KEY, 3)
        # Hard wall-clock budget for the entire pipeline (gen + review + fix
        # loop), measured from the scheduler's send-decision moment. If
        # exceeded, SELECTED_BEST picks the best of attempts so far. The only
        # reason a message is dropped is reviewer_drop_score below.
        # Default None = no budget cap. Set a float in
        # llm_config / admin UI to enforce a max wall-clock latency.
        self.reviewer_budget_sec = llm_config.get("reviewer_budget_sec", None)
        # Drop score: skip the turn (stay SILENT) if even the best attempt is
        # still ≥ N. Default 8: a message that is still suspicious after
        # the fix loop is suppressed rather than sent. A threshold of 10
        # suppresses only messages at the maximum suspicion score.
        self.reviewer_drop_score = llm_config.get("reviewer_drop_score", 8)
        # Per-call reviewer API timeout (seconds). Caps how long ONE reviewer
        # call can hang before we abort it; on timeout, review_message catches
        # the TimeoutError and returns a pass-through ReviewResult so the
        # agent stays alive and continues to the voting phase. Default 25s
        # protects against indefinite SDK hangs that could prevent voting.
        # 25s leaves headroom for multi-attempt Pro reviews (4
        # attempts can reach ~19s). Hardened: any null/0/invalid value falls
        # back to 25.0 so accidental config like
        # `"reviewer_api_timeout_sec": null` cannot disable the protection.
        _v = llm_config.get("reviewer_api_timeout_sec", 25.0)
        try:
            _v = float(_v) if _v is not None else 25.0
            if _v <= 0:
                _v = 25.0
        except (TypeError, ValueError):
            _v = 25.0
        self.reviewer_api_timeout_sec = _v
        # Inter-attempt backoff: if a reviewer attempt took >5s, sleep
        # 0.5s × attempt# before retrying so a slow backend can recover.
        # Default False = no backoff.
        self.reviewer_backoff_enabled = llm_config.get("reviewer_backoff_enabled", False)
        self.reviewer_model = llm_config.get(REVIEWER_MODEL_KEY, None)
        self.reviewer_backend = llm_config.get(REVIEWER_BACKEND_KEY, "gemini")
        self.reviewer_self_consistency = llm_config.get("reviewer_self_consistency", True)
        self._lite_reviewer = None  # lazy-created for self-consistency re-scoring

        # Adaptive typing-rhythm config: ~1s/word floor so the
        # total pre-send time ≥ word_count (realistic human typing).
        # The 20s cap never reduces the per-word floor for longer messages.
        self._typing_enabled = llm_config.get("typing_delay_enabled", True)
        self._typing_sec_per_word = llm_config.get("typing_sec_per_word", 1.0)
        self._typing_wps = llm_config.get("typing_wps", 1.6)  # unused by the 1s/word path
        self._typing_jitter = llm_config.get("typing_jitter_pct", 0.25)
        self._typing_min = llm_config.get("typing_min_delay_sec", 0.8)
        self._typing_max = llm_config.get("typing_max_delay_sec", 20.0)
        # A constant "reading + thinking" base added ON TOP of ~1s/word,
        # applied to ALL messages (incl. short) so the agent doesn't reply
        # instantly. Total floor ≈ word_count + base seconds.
        self._typing_base_delay = llm_config.get("typing_base_delay_sec", 2.0)
        # Typing-delay short-message skip: how many words are considered "too
        # short to bother with a realistic typing pause". Separate from the
        # reviewer skip below so that short msgs are still reviewed by default.
        self._typing_skip_words = llm_config.get("typing_min_words", 2)
        # Reviewer short-message skip: messages with <= this many words bypass
        # the reviewer entirely. Default 0 = review every message (the 40s
        # budget protects against runaway latency).
        self._reviewer_skip_words = llm_config.get("reviewer_skip_short_words", 0)
        
        # Initialize reviewer if enabled
        self.reviewer = None
        if self.reviewer_enabled:
            try:
                model = self.reviewer_model or None
                self.reviewer = create_reviewer(
                    backend=self.reviewer_backend,
                    model_name=model,
                    threshold=self.reviewer_threshold,
                    logger=self.logger,
                    api_timeout_sec=self.reviewer_api_timeout_sec,
                    experimental=self.experimental,
                )
                self.logger.log("ReviewedPlayer init",
                               f"Reviewer enabled: backend={self.reviewer_backend}, "
                               f"model={self.reviewer.model_name}, "
                               f"threshold={self.reviewer_threshold}, "
                               f"max_attempts={self.reviewer_max_attempts}")
            except Exception as e:
                self.logger.log("ReviewedPlayer init error", 
                               f"Failed to initialize reviewer: {e}. Continuing without review.")
                self.reviewer = None
    
    def _apply_typing_delay(self, message: str, pipeline_elapsed: float):
        """Sleep whatever remains of the ideal human-typing budget.

        If generation + review already took longer than the ideal delay, we
        send immediately. This way a slow reviewer never adds extra latency on
        top of its own cost.
        """
        if not self._typing_enabled or not message:
            return
        word_count = len(message.split())
        # Human typing ≈ 1s/word, plus a constant base so even a
        # short reply isn't instant. The TOTAL time before the final message is
        # sent must be AT LEAST word_count*sec_per_word + base: if the pipeline
        # (sched+gen+rev) already took that long, add nothing; otherwise fill the
        # gap. Jitter is upward-only so we never drop below the floor.
        base = getattr(self, "_typing_base_delay", 2.0)
        floor = float(word_count) * self._typing_sec_per_word + base
        ideal = max(floor, self._typing_min)
        # The cap must NEVER reduce the delay below the floor (otherwise a long
        # message under-waits: 25 words with cap 20 → 20s < 27s). Keep the cap only
        # as an upper bound ABOVE the floor.
        ideal = min(ideal, max(self._typing_max, floor))
        # No jitter: preserve the word-count-plus-base total delay.
        remaining = ideal - pipeline_elapsed
        if remaining > 0:
            self.logger.log("typing_delay", f"sleep {remaining:.2f}s (ideal {ideal:.2f}s, elapsed {pipeline_elapsed:.2f}s, base {base:.1f}s)")
            time.sleep(remaining)
        else:
            self.logger.log("typing_delay", f"no_sleep: pipeline already took {pipeline_elapsed:.2f}s ≥ ideal {ideal:.2f}s")

    def generate_message(self, message_history):
        """
        Generate a message with AI suspicion review.

        Flow:
        1. Check if should generate (from parent)
        2. Generate initial message (from parent)
        3. Review message with GeminiReviewer
        4. If suspicious, try to fix or regenerate
        5. Apply adaptive typing delay (budget = elapsed so far)
        6. Return final message
        """
        # First, check if we should generate at all
        if not self.should_generate_message(message_history):
            return ""

        # Lock to prevent duplicate message generation
        self.message_in_progress = True
        pipeline_start = time.time()

        # Show "AI is typing" indicator while we generate + review. The parent
        # ScheduleThenGeneratePlayer.generate_message creates this flag, but
        # this override replaces that path entirely - so we must create it
        # ourselves. Re-touched at each return site so the freshness window
        # in web_player covers long reviewer attempts.
        typing_flag = self.game_dir / f"{self.name}_typing.txt"
        try:
            typing_flag.touch()
        except Exception:
            pass

        # Instrumentation accumulators for the unified game_events.jsonl log.
        gen_ms = 0
        review_ms = 0
        reviewer_attempts = 0
        review_result = None
        final_message = ""
        initial_message = ""
        was_skipped_short = False

        try:
            # Generate initial message using parent's logic
            gen_start = time.time()
            initial_message = self._generate_raw_message(message_history)
            gen_ms = int((time.time() - gen_start) * 1000)

            if not initial_message:
                return ""

            # If reviewer not available, apply typing delay and return
            if not self.reviewer:
                self.logger.log("generate_message", "Reviewer not available, returning raw message")
                final_message = initial_message
                try: typing_flag.touch()
                except Exception: pass
                self._apply_typing_delay(initial_message, time.time() - pipeline_start)
                return initial_message

            # Short-message shortcut: skip reviewer for very short messages.
            if self._reviewer_skip_words > 0 and len(initial_message.split()) <= self._reviewer_skip_words:
                self.logger.log("reviewer_skipped_short", f"{len(initial_message.split())} words ≤ {self._reviewer_skip_words}")
                was_skipped_short = True
                final_message = initial_message
                try: typing_flag.touch()
                except Exception: pass
                self._apply_typing_delay(initial_message, time.time() - pipeline_start)
                return initial_message

            # Review and potentially fix the message. Pass pipeline_start so
            # the budget covers the WHOLE pipeline (gen + review), not just
            # the review loop in isolation.
            rev_start = time.time()
            final_message, review_result = self._review_and_fix_message(
                initial_message, message_history, pipeline_start=pipeline_start
            )
            review_ms = int((time.time() - rev_start) * 1000)
            reviewer_attempts = getattr(review_result, "attempts_completed", None) \
                or (getattr(review_result, "attempts", 1) if review_result else 1)

            # Capture the reviewer's output (final text, or "" on drop)
            # for the per-game ai_post_reviewer_messages.txt log.
            try:
                from social_turing_test.core.chat_logging import append_ai_post_reviewer
                append_ai_post_reviewer(self.game_dir, self.name, final_message)
            except Exception:
                pass

            # Log the review outcome
            self._log_review_result(initial_message, final_message, review_result)

            # If the reviewer determined this message
            # should not be sent (best score ≥ reviewer_drop_score), skip the
            # turn. final_message is "" - caller treats that as "no send".
            if not final_message:
                self.logger.log("turn_skipped",
                               f"Reviewer dropped message — stop_reason="
                               f"{getattr(review_result, 'stop_reason', '?')}")
                return ""

            # Adaptive typing delay - only sleep what's left of the budget
            try: typing_flag.touch()
            except Exception: pass
            self._apply_typing_delay(final_message, time.time() - pipeline_start)

            return final_message

        finally:
            # Emit structured review_cycle event (always, even on skip / no-reviewer paths).
            total_ms = int((time.time() - pipeline_start) * 1000)
            try:
                self.write_game_event(
                    "review_cycle",
                    original=initial_message,
                    final=final_message,
                    was_modified=(bool(initial_message) and initial_message != final_message),
                    generator_ms=gen_ms,
                    reviewer_ms=review_ms,
                    reviewer_attempts=reviewer_attempts,
                    total_ms=total_ms,
                    skipped_short=was_skipped_short,
                    final_score=(review_result.suspicion_score if review_result else None),
                    red_flags=(review_result.red_flags if review_result else []),
                    # Telemetry: WHY did the review loop end? Lets the user
                    # filter the admin Reviewer Log by stop_reason and decide
                    # whether to switch reviewer models. Values:
                    #   passed | good_enough | exhausted_attempts |
                    #   budget_exceeded | api_timeout | fix_identical |
                    #   no_fix | dropped_high_suspicion
                    stop_reason=getattr(review_result, "stop_reason", None),
                    budget_exceeded=bool(getattr(review_result, "budget_exceeded", False)),
                    timeout_hit=bool(getattr(review_result, "timeout_hit", False)),
                    dropped=bool(getattr(review_result, "dropped", False)),
                    attempts_completed=getattr(review_result, "attempts_completed", None),
                    attempts_planned=getattr(review_result, "attempts_planned", None),
                    review_loop_duration_sec=getattr(review_result, "review_loop_duration_sec", None),
                    # Drill-down: full per-attempt history + situation/expected
                    # analysis captured from the first reviewer call. Empty when
                    # no reviewer ran (short-skip / disabled) so the dashboard
                    # can distinguish "no review" from "review had 0 attempts".
                    candidates=list(getattr(review_result, "attempts", []) or []),
                    situation_analysis=getattr(review_result, "situation_analysis", None),
                    expected_human_response=getattr(review_result, "expected_human_response", None),
                )
            except Exception:
                pass
            # Always unlock, even if an error occurs
            self.message_in_progress = False
            try:
                typing_flag.unlink(missing_ok=True)
            except Exception:
                pass
            self.logger.log("generate_message", "UNLOCKED: message generation complete")
    
    def _generate_raw_message(self, message_history) -> str:
        """
        Generate a raw message using parent's generation logic.
        This is the same as ScheduleThenGeneratePlayer.generate_message() internals.
        """
        prompt = self.create_generation_prompt(message_history)
        self.logger.log("prompt in _generate_raw_message", prompt)

        gen_start = time.time()
        message = self.llm.generate(
            prompt, False, self.get_system_info_message(attention_to_not_repeat=True)
        )
        self._log_cost("generator")
        gen_duration = time.time() - gen_start
        # Log pure generator latency separately from reviewer time.
        self.logger.log("generator_latency", f"{gen_duration:.2f}s")

        # Provenance: literal generator output (pre-trim, pre-humanizer).
        try:
            from social_turing_test.core.chat_logging import append_ai_generator_raw
            append_ai_generator_raw(self.game_dir, self.name, message)
        except Exception:
            pass

        # Apply standard processing
        message = self.trim_message_if_too_long(message, max_words=20)
        message = make_more_human_like(
            message,
            message_history,
            normalize_commas=self.llm_config.get("normalize_commas", False),
        )

        # Provenance: generator output after our forced edits, just before
        # the reviewer is invoked (or before the message is sent if no
        # reviewer is configured).
        try:
            from social_turing_test.core.chat_logging import append_ai_generator_processed
            append_ai_generator_processed(self.game_dir, self.name, message)
        except Exception:
            pass

        return message

    def _persona_claims_for_review(self) -> list:
        """Durable self-claims to feed the reviewer's self-contradiction check.

        The agent carries a SharedSnapshot whose Persona Monitor populates
        ``self_claims``; this extends the reviewer's contradiction check beyond
        the recent-message window. Returns [] when no snapshot is available.
        """
        snap = getattr(self, "snapshot", None)
        if snap is None:
            return []
        try:
            return list(snap.read().get("self_claims", []) or [])
        except Exception:
            return []

    def _current_info_for_review(self) -> str:
        """Internet Context facts available to the reviewer.

        Returns an empty string when no snapshot is available.
        """
        snap = getattr(self, "snapshot", None)
        if snap is None:
            return ""
        try:
            return snap.read().get("current_info", "") or ""
        except Exception:
            return ""

    def _overused_words_for_review(self) -> list:
        """Words the Repetition Monitor flagged as overused so the
        reviewer flags reuse and avoids adding them. [] without a snapshot."""
        snap = getattr(self, "snapshot", None)
        if snap is None:
            return []
        try:
            return list(snap.read().get("overused_words", []) or [])
        except Exception:
            return []

    def _persona_for_review(self) -> str:
        """The agent's persona id so the reviewer treats in-character knowledge
        as expected. Returns an empty string when unknown."""
        return (getattr(self, "persona_id", None) or "") or ""

    # Give the reviewer the same shared context as the generator.
    # These helpers expose social state, behavior advice and repetition notes.
    # All return empty values when no snapshot is available.
    def _social_memory_for_review(self) -> str:
        snap = getattr(self, "snapshot", None)
        try:
            return (snap.read().get("social_memory", "") or "") if snap else ""
        except Exception:
            return ""

    def _behavior_advice_for_review(self) -> str:
        snap = getattr(self, "snapshot", None)
        try:
            return (snap.read().get("behavior_advice", "") or "") if snap else ""
        except Exception:
            return ""

    def _repetition_note_for_review(self) -> str:
        snap = getattr(self, "snapshot", None)
        try:
            return (snap.read().get("repetition_note", "") or "") if snap else ""
        except Exception:
            return ""

    def _avoid_phrases_for_review(self) -> list:
        """Persona-static phrases this persona doesn't use (e.g. israeli_student
        avoids 'lol') — so the reviewer never INTRODUCES them in a fix."""
        try:
            from social_turing_test.llm_players.llm_constants import get_persona
            persona = get_persona(getattr(self, "persona_id", None))
            return list((persona.get("communication_style", {}) or {}).get("avoid_phrases", []) or [])
        except Exception:
            return []

    def _review_and_fix_message(self, message: str, message_history: list, collect_steps: bool = False, pipeline_start: float = None) -> tuple:
        """
        Review message and attempt to fix if suspicious using ONLY reviewer fixes.
        
        Strategy:
        1. Review with Gemini
        2. If suspicious and has suggested_fix -> use it
        3. Repeat until max_attempts or message passes
        4. If all attempts fail, return the message with the BEST (lowest) score
        
        NO REGENERATION - only Reviewer fixes!
        
        Returns:
            If collect_steps=False: Tuple of (final_message, last_review_result)
            If collect_steps=True: Tuple of (final_message, last_review_result, revision_steps)
        """
        current_message = message
        review_start = time.time()
        # Budget is measured from pipeline_start (= scheduler decision) when
        # caller provides it; otherwise falls back to review_start (test paths).
        budget_anchor = pipeline_start if pipeline_start is not None else review_start

        # Track all attempts to pick the best one if all fail
        all_attempts = []  # List of (message, result, step_dict)
        accumulated_flags = []  # red flags from all failed attempts so far
        situation = None         # captured from the first call
        expected_human = None    # captured from the first call

        # Collect revision steps for detailed logging
        revision_steps = [] if collect_steps else None

        # Emit ONE consolidated reviewer trace at whatever exit we take
        # (passed / good_enough / fixed / reverted / dropped), so the reviewer's
        # full draft→attempts→final is reconstructable from a single event.
        def _emit_reviewer_trace(final_msg, final_score, action):
            try:
                self.write_game_event(
                    "reviewer_trace",
                    draft_in=(message or "")[:200],
                    attempts=[{
                        "message": (m or "")[:160],
                        "score": getattr(r, "suspicion_score", None),
                        "flags": list(getattr(r, "red_flags", []) or []),
                        "fix": ((getattr(r, "suggested_fix", None) or "")[:160]) or None,
                    } for (m, r, _s) in all_attempts],
                    final=(final_msg or "")[:200],
                    final_score=final_score,
                    action=action,
                    stop_reason=stop_reason,
                )
            except Exception:
                pass

        budget_exceeded = False
        timeout_hit = False
        # stop_reason: WHY the loop ended (passed | good_enough |
        # exhausted_attempts | budget_exceeded | api_timeout | fix_identical |
        # no_fix | dropped_high_suspicion). Surfaced in admin + dashboard.
        stop_reason = "in_progress"
        for attempt in range(self.reviewer_max_attempts):
            # Wall-clock budget guard: if we've already exceeded the per-message
            # review budget, stop calling the API and fall through to
            # SELECTED_BEST below. This prevents a slow provider (e.g. Vertex
            # AI latency spikes of 30-50s) from (a) burning a whole discussion
            # round on one message and (b) causing the message to be dropped
            # entirely by the downstream `is_time_to_vote` guard.
            elapsed_so_far = time.time() - budget_anchor
            if (self.reviewer_budget_sec is not None
                    and elapsed_so_far >= self.reviewer_budget_sec):
                budget_exceeded = True
                stop_reason = "budget_exceeded"
                self.logger.log("reviewer_budget_exceeded",
                               f"{elapsed_so_far:.1f}s (from pipeline start) ≥ {self.reviewer_budget_sec}s "
                               f"— stopped at attempt {attempt+1}/{self.reviewer_max_attempts} "
                               f"with {len(all_attempts)} scored attempts")
                if collect_steps:
                    revision_steps.append({
                        "step_number": attempt + 1,
                        "message": current_message,
                        "score": None,
                        "is_suspicious": None,
                        "red_flags": [],
                        "action": f"BUDGET_EXCEEDED ({elapsed_so_far:.1f}s ≥ {self.reviewer_budget_sec}s)",
                        "suggested_fix": None,
                    })
                break

            # Review current message. Catch TimeoutError from a hung
            # Gemini API call so a single slow attempt can't block the pipeline
            # for the full 40s budget. Treat timeouts as a "skip" that falls
            # through to SELECTED_BEST (or the original message if no prior
            # attempt succeeded).
            attempt_start = time.time()
            try:
                result = self.reviewer.review_message(
                    current_message,
                    self.name,
                    message_history,
                    previous_attempts=accumulated_flags if accumulated_flags else None,
                    situation_context=situation,
                    expected_human_response=expected_human,
                    best_score=min((e[1].suspicion_score for e in all_attempts), default=None),
                    persona_claims=self._persona_claims_for_review(),
                    current_info=self._current_info_for_review(),
                    persona=self._persona_for_review(),
                    overused_words=self._overused_words_for_review(),
                    social_memory=self._social_memory_for_review(),
                    behavior_advice=self._behavior_advice_for_review(),
                    repetition_note=self._repetition_note_for_review(),
                    avoid_phrases=self._avoid_phrases_for_review(),
                )
            except TimeoutError as e:
                attempt_duration = time.time() - attempt_start
                timeout_hit = True
                stop_reason = "api_timeout"
                self.logger.log("reviewer_timeout",
                               f"{attempt_duration:.2f}s | attempt: {attempt+1} | {e}")
                if collect_steps:
                    revision_steps.append({
                        "step_number": attempt + 1,
                        "message": current_message,
                        "score": None,
                        "is_suspicious": None,
                        "red_flags": [],
                        "action": f"TIMEOUT ({attempt_duration:.1f}s)",
                        "suggested_fix": None,
                    })
                # Stop retrying: if the backend is hung, further attempts will
                # also time out. Fall through to SELECTED_BEST below.
                break
            attempt_duration = time.time() - attempt_start
            self.logger.log(f"reviewer_attempt_latency", f"{attempt_duration:.2f}s | attempt: {attempt+1}")
            
            # Capture situation from first call
            if attempt == 0:
                situation = result.situation_analysis
                expected_human = result.expected_human_response
            
            self.logger.log(f"review_attempt_{attempt+1}", 
                           f"Score: {result.suspicion_score}, "
                           f"Suspicious: {result.is_suspicious}, "
                           f"Message: '{current_message}'")
            
            # Record step if collecting
            step = None
            if collect_steps:
                step = {
                    "step_number": attempt + 1,
                    "message": current_message,
                    "score": result.suspicion_score,
                    "is_suspicious": result.is_suspicious,
                    "red_flags": result.red_flags,
                    "action": None,  # Will be set below
                    "suggested_fix": result.suggested_fix
                }
            
            # Track this attempt
            all_attempts.append((current_message, result, step))
            
            # If message passed, we're done
            if not result.is_suspicious:
                stop_reason = "passed"
                review_duration = time.time() - review_start
                self.logger.log("reviewer_latency", f"{review_duration:.2f}s | attempts: {attempt+1} | final_score: {result.suspicion_score}")
                self.logger.log("review_passed",
                               f"Message passed review with score {result.suspicion_score}")
                try:
                    result.stop_reason = stop_reason
                    result.timeout_hit = timeout_hit
                    result.budget_exceeded = budget_exceeded
                    result.attempts_completed = len(all_attempts)
                    result.attempts_planned = self.reviewer_max_attempts
                    result.review_loop_duration_sec = round(review_duration, 2)
                except Exception:
                    pass
                _emit_reviewer_trace(current_message, result.suspicion_score,
                                     "fixed" if current_message != message else "kept_original")
                if collect_steps:
                    step["action"] = "PASSED"
                    revision_steps.append(step)
                    return current_message, result, revision_steps
                return current_message, result

            # Early-exit on "good enough" - after attempt 2, accept a
            # message scoring <= reviewer_good_enough_score (default 4) to avoid
            # burning a 3rd reviewer call when the fix is already reasonable.
            if attempt + 1 >= 2 and result.suspicion_score <= self.reviewer_good_enough_score:
                stop_reason = "good_enough"
                review_duration = time.time() - review_start
                self.logger.log("reviewer_latency", f"{review_duration:.2f}s | attempts: {attempt+1} | final_score: {result.suspicion_score}")
                self.logger.log("review_good_enough",
                               f"Accepting message at attempt {attempt+1} with score {result.suspicion_score} (\u2264 {self.reviewer_good_enough_score})")
                try:
                    result.stop_reason = stop_reason
                    result.timeout_hit = timeout_hit
                    result.budget_exceeded = budget_exceeded
                    result.attempts_completed = len(all_attempts)
                    result.attempts_planned = self.reviewer_max_attempts
                    result.review_loop_duration_sec = round(review_duration, 2)
                except Exception:
                    pass
                _emit_reviewer_trace(current_message, result.suspicion_score,
                                     "fixed" if current_message != message else "kept_original")
                if collect_steps:
                    step["action"] = f"ACCEPTED_GOOD_ENOUGH (score {result.suspicion_score}/10)"
                    revision_steps.append(step)
                    return current_message, result, revision_steps
                return current_message, result

            # Message is suspicious - try to fix using Reviewer's suggestion ONLY
            self.logger.log("review_failed", 
                           f"Message failed with score {result.suspicion_score}. "
                           f"Red flags: {result.red_flags}")
            
            # Record this failed attempt with score so next iteration knows what was already tried
            accumulated_flags.append({
                "message": current_message,
                "red_flags": result.red_flags,
                "score": result.suspicion_score,
            })
            
            # Use suggested fix if available (NO regeneration!)
            if result.suggested_fix and result.suggested_fix.strip():
                new_fix = result.suggested_fix.strip()
                # Make sure the fix is actually different from current message
                if new_fix.lower() != current_message.lower():
                    if collect_steps:
                        step["action"] = f"REVIEWER_FIX: '{new_fix}'"
                        revision_steps.append(step)
                    current_message = new_fix
                    self.logger.log(f"using_suggested_fix_{attempt+1}", current_message)
                    # Optional backoff (config-gated via reviewer_backoff_enabled).
                    # If the just-finished attempt took >5s, the backend is likely
                    # under load - sleep briefly so the next attempt has a chance
                    # at a faster response.
                    if (self.reviewer_backoff_enabled
                            and attempt_duration > 5.0
                            and attempt + 1 < self.reviewer_max_attempts):
                        backoff = 0.5 * (attempt + 1)
                        self.logger.log("reviewer_backoff",
                                       f"attempt {attempt+1} took {attempt_duration:.1f}s, "
                                       f"sleeping {backoff:.1f}s before retry")
                        time.sleep(backoff)
                    continue  # Review the fix
                else:
                    stop_reason = "fix_identical"
                    self.logger.log(f"fix_same_as_original_{attempt+1}",
                                   "Suggested fix is same as current message, stopping")
                    if collect_steps:
                        step["action"] = "FIX_IDENTICAL_TO_CURRENT"
                        revision_steps.append(step)
                    break
            else:
                # No fix available from reviewer
                stop_reason = "no_fix"
                self.logger.log(f"no_fix_available_{attempt+1}",
                               "No suggested fix from reviewer, stopping")
                if collect_steps:
                    step["action"] = "NO_FIX_AVAILABLE"
                    revision_steps.append(step)
                break
        
        # Self-consistency: if the loop exited with an unscored fix in
        # current_message, re-score it with flash-lite (cheap, fast).
        # If it fails too, discard it - SELECTED_BEST below will pick the
        # best of the already-scored attempts.
        # Skip self-consistency entirely if we already blew the budget - an
        # extra API call would make latency even worse.
        scored_messages = {m for m, _, _ in all_attempts}
        if (not budget_exceeded
                and self.reviewer_self_consistency
                and self.reviewer is not None
                and current_message not in scored_messages
                and current_message != message):
            try:
                if self._lite_reviewer is None:
                    self._lite_reviewer = create_reviewer(
                        backend="gemini",
                        model_name="gemini-2.5-flash-lite",
                        threshold=self.reviewer_threshold,
                        logger=self.logger,
                    )
                sc_start = time.time()
                sc_result = self._lite_reviewer.review_message(
                    current_message, self.name, message_history)
                self.logger.log("self_consistency",
                                f"flash-lite rescored last fix: {sc_result.suspicion_score}/10 "
                                f"({time.time()-sc_start:.2f}s)")
                all_attempts.append((current_message, sc_result, None))
            except Exception as _e:
                self.logger.log("self_consistency_error", str(_e))

        # If the loop fell through without setting stop_reason (i.e. exhausted
        # all max_attempts without passing or breaking), record that explicitly
        # so admin/dashboard distinguishes "ran out of attempts" from
        # "hit budget" or "API timeout".
        if stop_reason == "in_progress":
            stop_reason = "exhausted_attempts"

        # ── SELECTED_BEST guardrails ──
        review_duration = time.time() - review_start
        if all_attempts:
            # Filter out score=10 (identical / catastrophic)
            candidates = [(m, r, s) for m, r, s in all_attempts if r.suspicion_score < 10]
            if not candidates:
                candidates = all_attempts  # fallback if everything is 10

            best_message, best_result, best_step = min(candidates, key=lambda x: x[1].suspicion_score)
            best_score = best_result.suspicion_score

            # Drop the message if even the best attempt is still
            # suspicion ≥ reviewer_drop_score (default 8). This is the ONLY way
            # a message gets dropped - never on time alone.
            dropped = best_score >= self.reviewer_drop_score
            if dropped:
                stop_reason = "dropped_high_suspicion"
                self.logger.log("review_dropped",
                               f"Best score {best_score}/10 ≥ drop threshold "
                               f"{self.reviewer_drop_score}; skipping turn entirely "
                               f"(stop_reason={stop_reason}, attempts={len(all_attempts)}, "
                               f"budget_exceeded={budget_exceeded}, timeout={timeout_hit}, "
                               f"loop_duration={review_duration:.1f}s)")
                best_message = ""  # empty = caller skips this turn
            elif best_score >= 7:
                # Soft fallback: normally send the ORIGINAL generator output
                # instead of the reviewer's awkward fix. BUT if the ORIGINAL was
                # itself flagged evasive/defensive, reverting keeps that tell;
                # prefer the best ENGAGED fix instead.
                orig_flags = [str(f).lower() for f in (all_attempts[0][1].red_flags or [])] \
                    if all_attempts else []
                orig_was_evasive = any(
                    ("evasiv" in f or "defensiv" in f or "dodge" in f) for f in orig_flags)
                alts = [(m, r, s) for (m, r, s) in candidates if m != message]
                if orig_was_evasive and alts:
                    best_message, best_result, best_step = min(
                        alts, key=lambda x: x[1].suspicion_score)
                    best_score = best_result.suspicion_score
                    self.logger.log("revert_skipped_evasive_original",
                                   f"original flagged evasive/defensive ({orig_flags}); using best "
                                   f"engaged fix '{best_message}' ({best_score}/10) over reverting")
                else:
                    self.logger.log("revert_to_original",
                                   f"Best score {best_score}/10 in [7, {self.reviewer_drop_score-1}] — "
                                   f"reverting to original generator message '{message}'")
                    best_message = message

            if not dropped and best_score < 7:
                self.logger.log("using_best_attempt",
                               f"All attempts suspicious. Using message with lowest score: "
                               f"{best_score}/10 - '{best_message}'")
            self.logger.log("reviewer_latency",
                           f"{review_duration:.2f}s | attempts: {len(all_attempts)} | "
                           f"final_score: {best_score} | stop_reason: {stop_reason}")
            # Stamp telemetry on the result for the admin/dashboard event log.
            try:
                best_result.budget_exceeded = budget_exceeded
                best_result.timeout_hit = timeout_hit
                best_result.stop_reason = stop_reason
                best_result.dropped = dropped
                best_result.attempts_completed = len(all_attempts)
                best_result.attempts_planned = self.reviewer_max_attempts
                best_result.review_loop_duration_sec = round(review_duration, 2)
            except Exception:
                pass

            # Consolidated reviewer trace for the post-loop exit paths
            # (dropped / reverted / selected-best).
            _emit_reviewer_trace(
                best_message, best_score,
                "dropped" if dropped else ("fixed" if best_message != message else "kept_original"))

            if budget_exceeded:
                self.logger.log("reviewer_budget_truncated",
                               f"Review loop cut short after {review_duration:.1f}s "
                               f"(budget {self.reviewer_budget_sec}s); used best of "
                               f"{len(all_attempts)} attempts, final_score={best_score}")

            if collect_steps:
                action = f"SELECTED_BEST (score {best_score}/10)"
                if dropped:
                    action = f"DROPPED (score {best_score}/10 ≥ {self.reviewer_drop_score})"
                elif best_score >= 7:
                    action += " — REVERTED to original"
                final_step = {
                    "step_number": len(revision_steps) + 1,
                    "message": best_message,
                    "score": best_score,
                    "is_suspicious": best_result.is_suspicious,
                    "red_flags": list(best_result.red_flags) if best_result.red_flags else [],
                    "action": action,
                    "suggested_fix": None
                }
                revision_steps.append(final_step)
                return best_message, best_result, revision_steps
            return best_message, best_result

        # Fallback (no scored attempts at all - happens only when first attempt timed out)
        last_result = locals().get('result', None)
        try:
            if last_result is not None:
                last_result.stop_reason = stop_reason
                last_result.timeout_hit = timeout_hit
                last_result.budget_exceeded = budget_exceeded
                last_result.dropped = False
        except Exception:
            pass
        if collect_steps:
            return current_message, last_result, revision_steps
        return current_message, last_result
    
    def _log_review_result(self, original: str, final: str, result: ReviewResult):
        """Log detailed review outcome."""
        changed = original != final
        
        log_data = {
            "original_message": original,
            "final_message": final,
            "was_modified": changed,
            "final_score": result.suspicion_score if result else "N/A",
            "final_suspicious": result.is_suspicious if result else "N/A",
            "red_flags": result.red_flags if result else []
        }
        
        self.logger.log("review_outcome", str(log_data))
        
        if changed:
            self.logger.log("message_modified", 
                           f"Original: '{original}' -> Final: '{final}'")
    
    def generate_message_with_review_info(self, message_history) -> dict:
        """
        Generate a message and return detailed review information.
        
        Captures the review process and its per-attempt provenance.
        
        Returns:
            dict with:
                - message: The final generated message
                - raw_message: The initial message before review
                - was_reviewed: Whether review was applied
                - review_result: Review details (if reviewed)
                - was_modified: Whether the message was changed by review
                - revision_steps: List of all revision attempts with scores
        """
        # First, check if we should generate at all
        if not self.should_generate_message(message_history):
            return {
                "message": "",
                "raw_message": "",
                "was_reviewed": False,
                "review_result": None,
                "was_modified": False,
                "revision_steps": []
            }
        
        # Lock to prevent duplicate message generation
        self.message_in_progress = True
        pipeline_start = time.time()

        try:
            # Generate initial message using parent's logic
            initial_message = self._generate_raw_message(message_history)
            
            if not initial_message:
                return {
                    "message": "",
                    "raw_message": "",
                    "was_reviewed": False,
                    "review_result": None,
                    "was_modified": False,
                    "revision_steps": []
                }

            # If reviewer not available, return as-is
            if not self.reviewer:
                self.logger.log("generate_message", "Reviewer not available, returning raw message")
                return {
                    "message": initial_message,
                    "raw_message": initial_message,
                    "was_reviewed": False,
                    "review_result": None,
                    "was_modified": False,
                    "revision_steps": []
                }
            
            # Review and potentially fix the message - collect all steps!
            final_message, review_result, revision_steps = self._review_and_fix_message(
                initial_message, message_history, collect_steps=True,
                pipeline_start=pipeline_start
            )

            # Capture post-reviewer (final text or "" on drop) provenance.
            try:
                from social_turing_test.core.chat_logging import append_ai_post_reviewer
                append_ai_post_reviewer(self.game_dir, self.name, final_message)
            except Exception:
                pass
            
            # Log the review outcome
            self._log_review_result(initial_message, final_message, review_result)
            
            # Build review info dict
            review_info = None
            if review_result:
                review_info = {
                    "suspicion_score": review_result.suspicion_score,
                    "is_suspicious": review_result.is_suspicious,
                    "red_flags": review_result.red_flags,
                    "suggested_fix": review_result.suggested_fix,
                    "budget_exceeded": bool(getattr(review_result, "budget_exceeded", False)),
                }
            
            return {
                "message": final_message,
                "raw_message": initial_message,
                "was_reviewed": True,
                "review_result": review_info,
                "was_modified": initial_message != final_message,
                "revision_steps": revision_steps
            }
            
        finally:
            # Always unlock, even if an error occurs
            self.message_in_progress = False
            self.logger.log("generate_message", "UNLOCKED: message generation complete")


