import re
import time

from social_turing_test.core.game_constants import get_current_date
from social_turing_test.llm_players.llm_constants import turn_task_into_prompt, make_more_human_like
from social_turing_test.llm_players.pipeline_base.agent_state import AgentState


def _game_relative_date(player) -> str:
    """Return the game's recorded date (game_date.txt) if present, else
    today's wall-clock date. Mirrors the same logic in InternetContextAgent
    so the generator and the sub-agent agree on what "today" means.
    """
    try:
        game_date_file = player.game_dir / "game_date.txt"
        if game_date_file.exists():
            iso = game_date_file.read_text(encoding="utf-8", errors="replace").strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
                return iso
    except (AttributeError, OSError):
        pass
    return get_current_date()


# Vague deflection markers - used to detect when the agent has ALREADY been
# evasive recently, so a second flat "idk" would read as a robotic tell.
_DEFLECTION_MARKERS = (
    "idk", "no clue", "not sure", "i forget", "no idea", "dunno",
    "have to think", "pass", "beats me", "who knows",
)


def _recent_deflection_count(memory_notes: str) -> int:
    """Count how many of the agent's recent messages (memory_notes string)
    look like vague deflections. Used to escalate away from repeated 'idk'."""
    if not memory_notes:
        return 0
    low = memory_notes.lower()
    return sum(1 for m in _DEFLECTION_MARKERS if m in low)


# Bare low-content agreement that several players pile onto in a row
# ("yeah", "true that", "me too", "definitely") is a mild AI tell when it
# clusters. Detect a pile-up so the generator adds substance or stays quiet.
_AGREEMENT_MARKERS = (
    "yeah", "true", "same", "me too", "definitely", "for sure", "fair enough",
    "fair", "right", "agreed", "totally", "exactly", "for real", "facts",
)


def _is_bare_agreement(content: str) -> bool:
    c = content.strip().lower().rstrip("!.").strip()
    if len(c.split()) > 4:
        return False
    return any(c == m or c.startswith(m + " ") or c == m + "," for m in _AGREEMENT_MARKERS)


def _recent_agreement_pileup(history: list, player_name: str, n: int = 3) -> bool:
    """True if the last n non-Game-Manager messages are all bare agreement —
    the conversation is in an echo loop the agent shouldn't extend."""
    import re as _re
    from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    recent = []
    for line in reversed(history):
        m = _re.match(MESSAGE_PARSING_PATTERN, line)
        if not m or m.group(4) == GAME_MANAGER_NAME:
            continue
        recent.append(m.group(5))
        if len(recent) >= n:
            break
    return len(recent) >= n and all(_is_bare_agreement(c) for c in recent)


def make_generator_node(player):
    """
    LLM call: generates draft_message.

    The task prompt is enriched with
    why_speaking, style_notes, and memory_notes (all pre-computed by
    context_builder_gen). current_info is also injected here.

    Reuses:
        player._apply_memory_window()
        player.llm.generate()
        player._log_cost()
        player.trim_message_if_too_long()
        make_more_human_like()
        turn_task_into_prompt()
    """
    def _node(state: AgentState) -> dict:
        history = player._apply_memory_window(state["message_history"])
        why     = state.get("why_speaking", "you want to contribute")
        style   = state.get("style_notes", "")
        memory  = state.get("memory_notes", "")
        info    = state.get("current_info", "")

        phases_dbg = state.get("round_summaries", [])
        player.logger.log(
            "generator_context_received",
            f"current_info={'<empty>' if not info else info!r} | "
            f"phase_summaries={len(phases_dbg)} item(s)",
        )

        task = (
            f"Today's date is {_game_relative_date(player)}. "
            f"Add a very short message to the game's chat. "
            f"Reason you're speaking now: {why}. "
            f"Be specific and keep it relevant to the current discussion. "
            f"One short sentence only! "
            # Anti-anachronism: training memory may misrepresent a current
            # person or event as past. Treat NOW as now.
            f"This conversation is happening NOW (today's date above) — don't brush off "
            f"recent topics as 'years ago' or call a current person 'out of office' from "
            f"old memory; if the chat treats something as current, it is. If you're unsure "
            f"of a current detail, stay vague ('not sure where that landed') rather than "
            f"claiming it's old or over. "
        )
        # Apply knowledge-recency framing only when the current question needs
        # a missing fact, keeping ordinary turns concise.
        phases = state.get("round_summaries", [])
        if phases:
            task += f"Earlier in this game: {' | '.join(phases[-3:])}. "
        # Fact-awareness without monotone deflection.
        # Hard facts you don't have → don't invent. But for opinions / future /
        # "what do you think" questions, a human SPECULATES casually rather than
        # repeating "idk". Flat deflection used twice in a row is itself a tell.
        info_status = state.get("current_info_status", "irrelevant")
        if info_status == "missing_for_question":
            recent_deflections = _recent_deflection_count(memory)
            # A soft recency reminder, only here, so the agent
            # knows old big events but treats recent news as "only if I caught it".
            try:
                from social_turing_test.llm_players.knowledge_cutoff import get_knowledge_cutoff
                _model = (getattr(player, "llm_config", {}) or {}).get("model_name") or getattr(player, "model_name", "")
                _cut = get_knowledge_cutoff(_model, (getattr(player, "llm_config", {}) or {}).get("knowledge_cutoff"))
            except Exception:
                _cut = ""
            task += (
                f"(You'd know big past events{(' up to ~'+_cut) if _cut else ''}; "
                "very recent news only if you actually caught it.) "
            )
            # Natural mix: a real person knows some things and not others. Let
            # the model route by HOW common the fact is, rather than always
            # deflecting (reads as evasive on obvious questions) or always
            # answering (superhuman recall of niche/current trivia is a tell).
            persona_id = (getattr(player, "persona_id", None) or "").replace("_", " ")
            persona_local = (
                f"- This is the kind of thing YOUR background ({persona_id}) would "
                "plausibly know (local geography, the school calendar / when breaks "
                "start, daily-life stuff, well-known local events) — give a casual "
                "plausible answer, do NOT say 'no idea'. Being a local who can't "
                "answer basic local questions is the #1 way to get caught.\n"
                if persona_id else ""
            )
            task += (
                "Someone asked something fact-ish and you don't have a researched "
                "answer in front of you. React like a real person would:\n"
                "- If it's COMMON KNOWLEDGE almost anyone would just know (basic "
                "geography, famous people, obvious everyday stuff), answer it "
                "naturally and correctly — don't play dumb.\n"
                + persona_local +
                "- If it's a SPECIFIC recent/niche thing not everyone follows "
                "(yesterday's exact score, breaking news from today, an obscure "
                "detail), it's totally normal NOT to know: drop a casual guess "
                "('bulgaria maybe?', 'sweden again lol') or a brief honest shrug "
                "('no clue tbh', 'havent kept up'). Pick ONE, keep it short and in "
                "character. Do not invent specific dates/names/numbers as if "
                "certain; don't apologise, don't explain.\n"
            )
            if recent_deflections >= 1:
                task += (
                    "You ALREADY gave a vague answer recently — repeating 'idk' is a "
                    "robotic tell, so lean toward a casual guess, or just answer if "
                    "it's something you'd plausibly know.\n"
                )
        elif info:
            task += (
                "Recent facts you happen to know (use ONLY if directly relevant to "
                "what's being discussed — never dump them unprompted, and hedge when "
                "told to):\n" + info + "\n"
            )
        if memory:
            task += f"Do NOT repeat yourself — you recently said: {memory}. "

        # Give the generator the Repetition Monitor's overused words so it can
        # avoid producing them before review.
        overused = state.get("overused_words", []) or []
        if overused:
            task += ("You've leaned on these words too much — DON'T use them again: "
                     f"{', '.join(overused[:6])}. ")
        # Repetition Monitor LLM pass: vary message SHAPE/approach, not just words.
        repetition_note = state.get("repetition_note", "") or ""
        if repetition_note:
            task += f"{repetition_note} "

        # Terse guard against the clearest tell: refusing/hostile non-
        # answers ("i don't have to tell you", "google it"). Kept short to avoid
        # over-prompting.
        task += (
            "Never refuse to answer or get hostile/defensive (no 'i don't have to "
            "tell you', no 'google it') — just give an easy real reply or guess. "
        )

        # Persona self-consistency. Never contradict prior self-claims.
        self_claims = state.get("self_claims", [])
        if self_claims:
            task += (
                "Things you've ALREADY said about yourself this game — you MUST "
                "stay consistent and NEVER contradict them: "
                + "; ".join(self_claims)
                + ". If asked something touching these, answer in line with what "
                "you already said.\n"
            )

        # Social State Tracker. The agent's own social standing (who suspected
        # it, who it pointed at, who's out). React naturally to accusers instead
        # of acting oblivious, and stay consistent with the stance already taken.
        social_memory = state.get("social_state", "")
        if social_memory:
            task += (
                "Social context you should act on (" + social_memory + "). React "
                "naturally to it — don't act oblivious if someone suspects you, and "
                "stay consistent with any stance you already took. If there's an "
                "unanswered question aimed at you and it fits the moment, you can "
                "circle back and answer it casually. Never announce that you're "
                "tracking any of this.\n"
            )

        # Comprehension of obvious typos (don't echo them back).
        task += (
            "If a player clearly misspelled something obvious (e.g. 'aurussiion' "
            "for 'Eurovision'), understand what they meant from context — don't "
            "ask them to spell it out.\n"
        )

        # Anti-echo: if the last few messages are all bare agreement,
        # don't pile on another "yeah/true/me too". Add a specific new point or
        # stay quiet. Skipped on a revive turn (revive's whole job is to talk).
        revive = state.get("revive_conversation", False)
        if not revive and _recent_agreement_pileup(state["message_history"], player.name):
            task += (
                "The last few messages are all just agreement ('yeah', 'true', "
                "'me too'). Do NOT add another bare agreement — it reads as robotic "
                "pile-on. Either say something SPECIFIC and new, or output exactly "
                "<skip> to stay quiet.\n"
            )

        # Lull-breaker revive turn. The conversation stalled; produce a
        # genuine re-engagement, NOT filler and NOT a <skip>.
        if revive:
            task += (
                "The conversation has gone quiet. Restart it naturally like a real "
                "person would: ask the group a specific question, react to "
                "something said earlier, or bring up the topic. Keep it short and "
                "casual. Do NOT comment on the silence itself ('it's quiet', "
                "'anyone there?') — just naturally move things forward.\n"
            )

        # Anti-filler. On a purely proactive turn with no real hook,
        # forbid vacuous filler and allow the agent to stay silent instead.
        # Skipped on a revive turn - there the whole point is to break the lull.
        send_reason = state.get("send_reason", "proactive")
        if send_reason == "proactive" and not revive:
            task += (
                "Only speak if you have something SPECIFIC to add to what was "
                "just said. Vague filler like 'lets go', 'the game', 'ready when "
                "you are' is forbidden. If you have nothing specific, output "
                "exactly <skip> and nothing else.\n"
            )

        if style:
            task += f"Room style: {style}. "

        # Behavior Adapter. One play-style directive for this stage of
        # the game (how talkative / cautious to be). Guides delivery, not content.
        behavior_advice = state.get("behavior_advice", "")
        if behavior_advice:
            task += f"Play-style note for this stage of the game: {behavior_advice}.\n"

        task += "Match the other players' message style exactly.\n"

        prompt = turn_task_into_prompt(task, history)
        player.logger.log("generator_prompt", prompt)

        # Only emit the structured input event when current_info is non-empty
        # so turns without retrieved context keep a lean event log.
        if info:
            try:
                player.write_game_event(
                    "generator_input",
                    history_lines=len(history),
                    history_tail=[h.strip() for h in history[-3:]],
                    current_info_chars=len(info),
                    task_prompt=prompt[:3000],
                )
            except Exception:
                pass

        gen_start = time.time()
        message = player.llm.generate(
            prompt, False, player.get_system_info_message(attention_to_not_repeat=True)
        )
        player._log_cost("generator")
        player.logger.log("generator_latency", f"{time.time() - gen_start:.2f}s")

        # Anti-filler skip sentinel. If the LLM decided it has nothing
        # specific to add, it returns "<skip>" → emit an empty draft so the
        # turn is dropped (the agent stays quiet rather than posting filler).
        if "<skip>" in message.lower():
            player.logger.log("generator_skip", f"raw={message!r}")
            try:
                player.write_game_event("generator_skip", reason="no_substantive_content")
            except Exception:
                pass
            return {"draft_message": ""}

        # Provenance: literal generator output
        try:
            from social_turing_test.core.chat_logging import append_ai_generator_raw
            append_ai_generator_raw(player.game_dir, player.name, message)
        except Exception:
            pass

        message = player.trim_message_if_too_long(message, max_words=20)
        message = make_more_human_like(
            message, history,
            normalize_commas=player.llm_config.get("normalize_commas", False),
        )

        # Provenance: post-humaniser output
        try:
            from social_turing_test.core.chat_logging import append_ai_generator_processed
            append_ai_generator_processed(player.game_dir, player.name, message)
        except Exception:
            pass

        return {"draft_message": message}

    return _node
