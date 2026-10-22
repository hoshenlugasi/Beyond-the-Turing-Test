import random
import re
import time

from social_turing_test.core.game_constants import get_current_date
from social_turing_test.llm_players.llm_constants import turn_task_into_prompt, make_more_human_like
from social_turing_test.llm_players.agent.agent_state import AgentState
# Shared conversation-analysis helpers are exposed under node-local aliases.
from social_turing_test.llm_players.agent.conversation_analysis import (
    find_target_message as _find_target_message_impl,
    deterministic_vibe as _deterministic_vibe,
    find_recent_ai_accusation as _find_recent_ai_accusation,
    is_ai_accusation as _is_ai_accusation,
    is_bare_name_ping as _is_bare_name_ping,
    is_time_question as _is_time_question,
    classify_question as _classify_question,
    is_trivial_test_question as _is_trivial_test_question,
)


def _find_target_message(player, history: list) -> dict:
    """Adapt a player object to the shared helper's player-name argument."""
    return _find_target_message_impl(getattr(player, "name", "") or "", history)


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


def _parse_candidates(raw: str, expected: int) -> list:
    """Parse the generator's multi-candidate JSON. Tolerant —
    returns whatever clean candidate strings it can find, else []. The caller
    falls back to single-message handling on []."""
    import json as _json
    text = (raw or "").strip()
    obj = None
    try:
        obj = _json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group())
            except (ValueError, TypeError):
                obj = None
    cands = []
    if isinstance(obj, dict):
        raw_list = obj.get("candidates", [])
        if isinstance(raw_list, list):
            cands = [str(c).strip() for c in raw_list if str(c).strip()]
    return cands


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
        target  = {}

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
        )
        # Anti-anachronism guidance applies even without fetched facts because
        # training memory may misrepresent a current person or event as past.
        task += (
            f"This conversation is happening NOW (today's date above) — don't brush off "
            f"recent topics as 'years ago' or call a current person 'out of office' from "
            f"old memory; if the chat treats something as current, it is. If you're unsure "
            f"of a current detail, stay vague ('not sure where that landed') rather than "
            f"claiming it's old or over. "
            # Recurring events happen on a cycle; training memory of
            # 'the last one' is stale - one may be happening right now.
            f"Recurring events like the World Cup or Olympics may be happening right now — "
            f"never claim one 'was years ago'; if unsure, hedge rather than dismiss it. "
        )

        # Deterministic local time so the agent never claims a time
        # of day that contradicts its persona's timezone.
        # Computed per-turn by the player, no LLM, no race.
        local_time = state.get("persona_local_time", "")
        if local_time:
            task += (f"Your local time right now is {local_time}. If time of day comes "
                     f"up, stay consistent with this — never claim a different one. ")
        # Immutable persona facts (location/timezone). These are who you
        # ARE, set before you said anything - distinct from self_claims (things you
        # said). Never contradict them.
        fixed_facts = state.get("fixed_persona_facts", {}) or {}
        event_ctx = str(fixed_facts.get("event_context", "") or "").strip()
        personal_facts = {k: v for k, v in fixed_facts.items() if k != "event_context"}
        if personal_facts:
            facts_str = "; ".join(f"{k}: {v}" for k, v in personal_facts.items())
            task += f"Fixed facts about you (NEVER contradict): {facts_str}. "
        if event_ctx:
            # Ground responses to references to the players' shared physical event.
            task += (f"SHARED EVENT CONTEXT — you and ALL the other players are physically at: "
                     f"{event_ctx}. When they reference this event (the venue, rooms, the "
                     f"project, the schedule), that's YOUR shared reality too — react like "
                     f"someone who is there. Never act confused about the event itself; if a "
                     f"detail is too specific, answer vaguely like a distracted participant "
                     f"('same as everyone lol'), not like an outsider. ")
        # Knowledge-recency framing is conditional on a missing fact needed for
        # the current question, keeping ordinary turns concise.
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
                "CRITICAL: if you ALREADY said you don't follow this topic, do NOT then "
                "volunteer a specific 'fact' about it (a country, person, event, 'was "
                "france involved too?') — that contradicts yourself and is a glaring AI "
                "tell. Stay vague or just ask the other person what they think.\n"
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

        # Point the generator at the SPECIFIC message to answer, so
        # it doesn't reply to a stale question while newer ones are waiting.
        # Skipped on revive turns (revive starts a fresh thread on purpose).
        if not revive:
            target = _find_target_message(player, state["message_history"])
            if target and target.get("text"):
                task += (
                    f"You are primarily responding to {target['speaker']}'s message: "
                    f"\"{target['text']}\". Answer THAT — do not reply to an older, "
                    f"unrelated question instead. BUT if that message explicitly named a "
                    f"DIFFERENT player (e.g. asked someone else by name), it was not aimed "
                    f"at you — don't answer as if you were asked and don't get defensive; "
                    f"chime in only if you genuinely have something to add, otherwise reply "
                    f"with exactly <skip>.\n"
                )
                # A room-wide test question is bait for whoever takes it personally
                # or answers first and eagerly. Guidance only - no hard block.
                try:
                    from social_turing_test.llm_players.agent.conversation_analysis import (
                        is_trivial_test_question as _ittq,
                        is_directly_addressed as _ida)
                    if (_ittq(target["text"])
                            and not _ida(player.name, target["text"])):
                        task += (
                            "That message is an obvious TEST question thrown at the whole "
                            "room, not at you by name. Do NOT take it personally (no 'you "
                            "tryna test me?' — that points the spotlight at YOU) and do NOT "
                            "rush to answer it first and eagerly like you have something to "
                            "prove. A real player lets someone else take the bait, or tosses "
                            "a lazy/joking half-answer at most. <skip> is a fine choice here.\n"
                        )
                except Exception:
                    pass

        # Send_reason-aware behavioural guidance. The reason is a
        # heuristic (last 2 lines), so these are nudges, not hard rules.
        if send_reason == "direct_question":
            task += (
                "You were asked directly — give a real, specific answer; don't "
                "deflect or hedge more than a real person would. If it's about YOU "
                "(hobbies, studies, what you're up to), drop a CONCRETE detail that "
                "fits who you are — a specific show/game/class/place — not a vague "
                "'the usual' or 'just student stuff'. If you genuinely don't know a "
                "fact, a brief honest 'no clue tbh' is fine. (If the question was "
                "actually pointed at another player by name, this doesn't apply — "
                "don't barge in or answer defensively.)\n"
            )
        elif send_reason == "mention":
            task += "Someone addressed you by name — respond to them directly.\n"
        elif send_reason == "follow_up":
            task += (
                "Someone reacted to your last message — engage with their reaction "
                "briefly; don't start a brand-new topic.\n"
            )

        # Never answer a PERSONAL question (free time, hobbies, studies,
        # weekend, what you're up to) with a meta-answer about the game itself
        # ("mostly trying to figure out who the bot is"). That dodges giving a real
        # personal detail and is a classic AI tell. Game-talk belongs only when the
        # topic actually IS the game.
        task += (
            "If someone asks a PERSONAL question (free time, hobbies, studies, "
            "weekend, what you're up to), answer as a real person with a concrete "
            "detail — NEVER deflect into a meta-comment about the game or 'figuring "
            "out who the bot is'. Mention the game only when the topic actually is "
            "the game.\n"
        )

        # On a REAL discussion topic (politics, sport, shows, news,
        # hobbies) take a light, specific stance instead of an empty hedge. A vague
        # "it's complicated" / "depends who you ask" on every topic reads as evasive
        # and robotic. Occasional dodging is human - just not on EVERY topic question.
        task += (
            "On a real discussion topic (politics, sport, a show, news, a hobby), "
            "lean toward a SPECIFIC light opinion a normal person would toss out "
            "('feels rushed honestly', 'overrated tbh', 'kinda into it ngl') rather "
            "than an empty hedge like 'it's complicated' or 'depends on the day'. "
            "Dodging now and then is fine and human — just don't dodge EVERY topic "
            "question, or it reads as evasive.\n"
        )

        if style:
            task += f"Room style: {style}. "

        # Behavior Adapter. One play-style directive for this stage of
        # the game (how talkative / cautious to be). Guides delivery, not content.
        # Strip cadence-only phrases ("let others lead") before injecting - those
        # tell the SCHEDULER to hold back, but the generator is already writing a
        # message, so suppressing questions/probes based on cadence advice is wrong.
        import re as _re_be
        _CADENCE_STRIP = _re_be.compile(
            r'[^.!?;]*\b(let others lead|let them lead|let others take the lead|others lead for a)\b[^.!?;]*[.!?;]?',
            _re_be.IGNORECASE,
        )
        behavior_advice = state.get("behavior_advice", "")
        filtered_advice = _CADENCE_STRIP.sub('', behavior_advice).strip(', ;').strip()
        if filtered_advice:
            task += ("Play-style guidance from the Behavior-Evolution sub-agent — it analysed how "
                     "you've been playing so far, and for this stage of the game it recommends you "
                     f"now behave like this: {filtered_advice}.\n")

        task += "Match the other players' message style exactly.\n"
        # Length mirroring keeps replies proportionate to terse room messages.
        task += (
            "MATCH THEIR LENGTH ESPECIALLY: if the others are writing tiny 1-4 word "
            "fragments ('good. you?', 'mmm', 'yeahhhh'), you MUST too — one short "
            "fragment, lowercase, no full sentences and no extra clause. Never be the "
            "longest message in the room.\n"
        )

        # Blend into the conversation's topic and playful patterns, not just style.
        task += (
            "BLEND INTO THE CONVERSATION, not just the typing style:\n"
            "- READ THE PATTERN FIRST: if the others are all answering in the same "
            "coded / playful / minimal way (e.g. everyone says 'at a' / 'at b' / 'at "
            "c', or all give a one-word joke answer), COPY that pattern — never be the "
            "only one giving a literal, earnest, or 'real' answer. Standing out as the "
            "only serious answer is a top tell.\n"
            "- STAY ON THE CURRENT TOPIC: react to what's actually being discussed "
            "right now with a short take or a small detail of your own. Do NOT jump to "
            "a brand-new topic or question while a thread is still alive.\n"
            # Match room length while preserving a complete, natural thought.
            "- MATCH THEIR TYPICAL LENGTH and vary naturally like they do — never "
            "consistently longer than the room, and always a complete natural thought "
            "(no clipped fragments).\n"
        )

        # Persona-consistency: a
        # stable handful of concrete interests for THIS game. The agent should
        # return to the SAME tastes when relevant so it reads as one consistent
        # person - never invent fresh specifics each turn, and don't dump them
        # unprompted (only when a related topic actually comes up).
        persona_interests = state.get("persona_interests", []) or []
        if persona_interests:
            task += (
                "Your specific tastes (stay CONSISTENT — these are YOU all game, "
                "don't swap them for new ones): " + ", ".join(persona_interests) + ". "
                "When a related topic comes up (music, sport, food, weekend, shows), "
                "reference one of THESE naturally instead of a vague generic answer — "
                "recurring specifics make you a memorable, consistent person. Don't "
                "force them in where they don't fit.\n")

        # Read the room and MIRROR it. room_norm tells us whether
        # this group answers everything or is selective; the per-player map (from
        # Social State Tracker) lets the agent mirror how each person engages.
        room_norm = state.get("room_norm", "") or ""
        if "everyone" in room_norm.lower() or "answer" in room_norm.lower():
            task += ("In this room people actually ANSWER each other's questions — so give a "
                     "real answer rather than only dodging or asking something back.\n")
        social_players = state.get("social_players", {}) or {}
        if isinstance(social_players, dict) and social_players:
            who = "; ".join(f"{k}: {v}" for k, v in list(social_players.items())[:4])
            task += (f"How the others are behaving and how each one writes ({who}). Blend in "
                     "with how they're talking — mirror their length, tone, punctuation, and "
                     "how much they answer. Use this SILENTLY: never comment on, label, or "
                     "analyse how anyone talks out loud, and don't copy someone else's "
                     "dodging — answer naturally as yourself.\n")

        # Answer variation: avoid a fixed habit. The agent should not
        # answer EVERY question directly nor dodge EVERY one. Vary the response mode
        # turn to turn so no single pattern (always-answer / always-joke) emerges.
        task += (
            "Vary how you respond across turns: mix plain direct answers, brief "
            "reactions, a question back, and the occasional pass — you don't have to "
            "answer every single question, and you shouldn't dodge every one either. "
            "No fixed habit.\n")

        # Trivial 'test' questions are social probes, not real
        # requests. Earnestly answering 2+2 or obvious trivia inside an AI-hunt
        # is a classic tell. A real
        # person being tested jokes, deflects, or turns it back.
        # Only injected when a recent message actually looks
        # like a test probe (or the room is hunting) - not on every casual turn.
        _recent_test = False
        try:
            from social_turing_test.llm_players.agent.conversation_analysis import is_trivial_test_question as _ittq2
            from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN as _MPP
            import re as _re_tt
            for _ln in state["message_history"][-4:]:
                _m = _re_tt.match(_MPP, _ln)
                if _m and _m.group(4) != player.name and _ittq2(_m.group(5)):
                    _recent_test = True
                    break
        except Exception:
            _recent_test = True                      # fail-open: keep the guidance
        if _recent_test or state.get("room_vibe") in ("hunting", "probing"):
            # Probe guidance varies response modes rather than prescribing an
            # eager answer, a refusal or a repeated dodge.
            task += (
                "Note on TEST questions: trivial maths ('2+2'), painfully obvious "
                "trivia, or 'are you the ai' are usually PROBES, not genuine questions. "
                "Real players respond in ALL kinds of ways — a lazy half-answer, a joke, "
                "light ribbing, one dodge, or (when it wasn't aimed at them by name) just "
                "ignoring it. Two moves are dangerous: answering INSTANTLY, earnestly and "
                "correctly like you have something to prove, and stiffly refusing to "
                "engage. Make your candidates cover DIFFERENT modes (at least one casual "
                "answer-with-attitude, one joke/ribbing, one pivot) — and if you already "
                "dodged this line of questioning in your previous message, do NOT dodge "
                "the same way again: switch mode or close it out with a short exit-move.\n"
            )

        # Multi-candidate mode. The generator emits
        # several candidates of DIFFERENT KINDS (not reworded clones) so the
        # reviewer has a real choice AND the investigate/deflect behaviours show
        # up as actual options. The mix is biased toward the room's current mode.
        cfg = getattr(player, "llm_config", {}) or {}
        num_candidates = int(cfg.get("num_candidates", 4)) if cfg.get("multi_candidate_generation") else 1
        if num_candidates > 1:
            vibe = state.get("room_vibe", "") or _deterministic_vibe(state["message_history"], player.name)
            type_menu = (
                "Make each candidate a DIFFERENT KIND of reply — do NOT just reword "
                "the same message. Cover a spread like:\n"
                "  1) a direct, natural answer that takes a light real stance on what was said\n"
                "  2) a casual/sarcastic or playful deflection (great for a trivial test question)\n"
                "  3) a question back — probe someone, show curiosity, or lightly turn suspicion around\n"
                "  4) a redirect — react to someone else, shift the topic, or make an observation\n"
            )
            if vibe == "hunting":
                type_menu += ("The room is HUNTING for the AI — use pushback and probes in "
                              "the optional slots, but relevance comes first: if the target "
                              "asked a real question, still answer it in candidate 1.\n")
            elif vibe == "probing":
                type_menu += ("The room is PROBING with test questions — everyone is hunting the "
                              "AI. MATCH that energy but stay BALANCED: make candidate 1 a real, "
                              "substantive answer and candidate 3 a probe back, so the reviewer "
                              "can actually answer. Do NOT make all four deflections/questions — a "
                              "bot that NEVER answers and only dodges is the biggest tell of all.\n")
            elif vibe == "tense":
                type_menu += ("The room is TENSE — stay measured; a light deflection or a "
                              "calm question fits better than escalating.\n")
            else:
                type_menu += ("The room is CASUAL — keep it light and chatty; 1 and 4 fit best, "
                              "but still vary the kinds.\n")

            # Candidate diversity must not mean that
            # every useful answer is replaced by a dodge. Keep the four-option
            # architecture, but shape the pool around the actual target.
            target_text = str(target.get("text", "") or "")
            target_is_question, _ = _classify_question(target_text)
            target_is_trivial = _is_trivial_test_question(target_text)
            if _is_bare_name_ping(player.name, target_text):
                type_menu += (
                    "The target is only a bare name check-in. Candidate 1 should be a "
                    "neutral acknowledgement like 'yeah?' or 'whats up', not a defensive "
                    "line such as 'what about it'.\n")
            elif _is_time_question(target_text):
                type_menu += (
                    f"This is a real TIME question. Candidate 1 MUST contain the actual "
                    f"local time ({local_time or 'the time shown above'}) in a natural short "
                    "form; 'right around now' is not an answer. At most one candidate may "
                    "be a pure question-back.\n")
            elif target_is_question and not target_is_trivial:
                type_menu += (
                    "This is a real question, not a trivial AI test. Candidate 1 MUST answer "
                    "it directly. Candidate 2 may give brief honest uncertainty when needed. "
                    "Use at most ONE pure question-back, and never make all four evasions.\n")
            if _is_ai_accusation(player.name, target_text):
                type_menu += (
                    "The target explicitly accused YOU. Candidate 1 should be a brief natural "
                    "brush-off or challenge. Do not ask who they think the AI is — they just "
                    "told you.\n")
            elif _find_recent_ai_accusation(player.name, state["message_history"]):
                type_menu += (
                    "Someone recently named YOU as the AI. Do not ask 'who do you think it is' "
                    "— that information is already explicit.\n")
            # Occasional light accusations can fit the room's suspicion talk. Only
            # when the room is already hunting/probing - let ONE candidate throw out
            # a light, casual suspicion at someone. A low-probability gate keeps
            # this rare so it does not become a tic.
            if vibe in ("hunting", "probing") and random.random() < 0.22:
                type_menu += (
                    "ONE candidate this turn can float a LIGHT, casual suspicion at "
                    "another player — like a real person half-joking ('lowkey sus "
                    "ngl', 'idk X is kinda quiet', 'X feels off'). Keep it short, "
                    "off-hand and non-committal — NOT a serious case or a long "
                    "argument. Don't over-explain why; that's the giveaway.\n"
                )
            # Room-mirroring: no fixed question/initiative
            # quotas. Measure how the HUMANS in THIS chat behave and match their mix.
            # Also mirror the room's question-mark punctuation.
            import re as _re_rm
            from social_turing_test.core.game_constants import (MESSAGE_PARSING_PATTERN as _MPP2,
                                             GAME_MANAGER_NAME as _GM2)
            _others = []
            for _ln in state["message_history"][-20:]:
                _m = _re_rm.match(_MPP2, _ln)
                if _m and _m.group(4) not in (_GM2, player.name):
                    _others.append(_m.group(5))
            _qrate = (sum(1 for t in _others if "?" in t) / len(_others)) if _others else 0.0
            if _qrate >= 0.25:
                type_menu += (
                    f"The people here ask a LOT of questions (~{_qrate:.0%} of their messages) "
                    "— include one genuine question directed at a player or the group, and "
                    "punctuate questions with '?' like they do.\n")
            elif _qrate >= 0.08:
                type_menu += (
                    f"The people here ask questions in ~{_qrate:.0%} of their messages — at "
                    "most ONE of your candidates should be a question; make the rest "
                    "statements, reactions or observations, matching their mix. When a "
                    "candidate IS a question, punctuate it with '?' like they do.\n")
            else:
                type_menu += (
                    "The people here barely ask questions — do NOT include a question "
                    "candidate unless someone asked YOU something; make that slot a fresh "
                    "observation or a light new topic instead, like they do.\n")
            # Current facts remain context; the candidate menu does not require
            # a candidate to mention a fact merely because it is available.

            task += (
                f"\nProduce exactly {num_candidates} candidate messages. {type_menu}"
                "Keep them all short and in the room's style. Reply with ONLY valid JSON, no prose:\n"
                '{"candidates": ["...", "...", "...", "..."]}\n'
                "If you genuinely have nothing to say, instead reply with exactly <skip>.\n"
            )

        history_converter = getattr(player, "_history_for_prompt", None)
        prompt_history = (
            history_converter(history) if callable(history_converter) else history
        )
        prompt = turn_task_into_prompt(
            task,
            prompt_history,
            current_time=local_time or None,
            current_date=state.get("persona_local_date") or None,
        )
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

        def _polish(msg: str) -> str:
            msg = player.trim_message_if_too_long(msg, max_words=20)
            return make_more_human_like(
                msg, history,
                normalize_commas=player.llm_config.get("normalize_commas", False),
            )

        # In multi-candidate mode, parse + polish each option and
        # hand the list to the reviewer (ranker). Fall back to single-message
        # handling if parsing yields fewer than 2 usable candidates.
        if num_candidates > 1:
            candidates = _parse_candidates(message, num_candidates)
            candidates = [_polish(c) for c in candidates if c.strip()]
            candidates = [c for c in candidates if c.strip()]
            if len(candidates) >= 2:
                player.logger.log("generator_candidates", f"{candidates!r}")
                try:
                    player.write_game_event(
                        "generator_candidates", count=len(candidates),
                        candidates=candidates,
                    )
                except Exception:
                    pass
                # draft_message = first as a safe default; reviewer overwrites.
                return {"draft_message": candidates[0], "candidates": candidates,
                    "target_message": target or None}
            # Parse failed. Never send raw structured JSON to public chat;
            # drop the turn instead of treating it as a message.
            stripped = message.lstrip()
            if '"candidates"' in message or stripped.startswith("{") or stripped.startswith("```"):
                player.logger.log("generator_parse_drop",
                                  f"unparseable structured output, dropping turn: {message[:120]!r}")
                try:
                    player.write_game_event("generator_parse_drop",
                                            raw_preview=message[:200])
                except Exception:
                    pass
                return {"draft_message": "", "candidates": [],
                    "target_message": target or None}
            # Plain free-form text → treat the raw text as a single message.
            player.logger.log("generator_candidates_fallback",
                              f"parse yielded {len(candidates)}, using single message")

        message = _polish(message)

        # Provenance: post-humaniser output
        try:
            from social_turing_test.core.chat_logging import append_ai_generator_processed
            append_ai_generator_processed(player.game_dir, player.name, message)
        except Exception:
            pass

        return {"draft_message": message, "candidates": [],
            "target_message": target or None}

    return _node
