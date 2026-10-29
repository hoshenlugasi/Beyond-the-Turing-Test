import re
import time

from social_turing_test.llm_players.llm_constants import turn_task_into_prompt
from social_turing_test.llm_players.agent.agent_state import AgentState
from social_turing_test.llm_players.agent import observability as obs
from social_turing_test.llm_players.agent.conversation_analysis import (
    find_target_message as _find_target_message,
    is_trivial_test_question as _is_trivial_test_question,
    is_directly_addressed as _is_directly_addressed,
    classify_reply_mode as _classify_reply_mode,
    recent_reply_mode_streak as _recent_reply_mode_streak,
    is_duplicate_of_recent_question as _is_duplicate_of_recent_question,
    is_own_template_repeat as _is_own_template_repeat,
    shares_opener_ngram as _shares_opener_ngram,
    find_recent_ai_accusation as _find_recent_ai_accusation,
    is_bare_name_ping as _is_bare_name_ping,
    is_time_question as _is_time_question,
    looks_like_time_answer as _looks_like_time_answer,
    asks_who_is_ai as _asks_who_is_ai,
    classify_question as _classify_question,
)
from social_turing_test.llm_players.agent.nodes.generator_node import _is_bare_agreement


# Deterministic identity-flag: candidates that contain "human" or "AI" in a
# self-referential / comparative context are near-instant exposure ("what
# would a human say?", "as a human", "we humans", "not an ai"). Marked with a
# visible tag in the numbered list so the LLM treats them as risky even when
# it doesn't generalize from the generic red-flag description.
# Also flag win-condition phrasing that puts the agent on the AI's side.
IDENTITY_RISK_RE = re.compile(
    r'\b(what would a human|as a human|i(?:\'m| am) (?:just )?(?:a |only )?human|'
    r'we humans|us humans|a real human|like a human would|'
    r'not an? (?:ai|bot|robot|llm)|i(?:\'m| am) not (?:an? )?(?:ai|bot|robot|llm)|'
    r'what would an ai|how (?:would|does) an? (?:ai|bot|robot|llm)|'
    r'(?:vote|voting|rooting|need) for the (?:ai|bot) to (?:win|survive|stay)|'
    r'the (?:ai|bot) (?:needs?|has) to win|'
    r'(?:we|us) (?:and|or) the (?:ai|bot))\b',
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Normalize a message for exact-duplicate comparison."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip().rstrip("!.?")


# A bare exclamation can contain '?' without being an answerable question.
# Apply this distinction locally without changing question classification
# in the generator or output nodes.
_BARE_EXCLAMATION_WORDS = {
    "what", "huh", "wait", "really", "seriously", "srsly", "wow", "omg",
    "oh", "no", "nooo", "hm", "hmm", "eh", "lol", "sorry", "me", "you",
    "wdym", "wut", "whaat", "ok", "okay",
}


def _is_bare_exclamation(text: str) -> bool:
    """True for reactions like 'what?!?', 'huh?', 'really?', 'wait what?'."""
    words = re.sub(r"[^a-z\s]+", " ", (text or "").lower()).split()
    return bool(words) and len(words) <= 2 and all(
        w in _BARE_EXCLAMATION_WORDS for w in words)


def _own_recent_normalized(player, history: list, n: int = 12) -> set:
    """Normalized set of the agent's own recent messages — for the deterministic
    duplicate guard (repetition is the reviewer's #1 tell)."""
    from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN
    out = set()
    for line in history[-60:]:
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if m and m.group(4) == player.name:
            out.add(_norm(m.group(5)))
    return out


def _own_recent_ordered(player, history: list, n: int = 4) -> list:
    """The agent's own last ``n`` messages in chronological order (raw text) — used
    to detect a dodge streak (deflect / question-back every turn) so the reviewer
    can swing back to a real answer."""
    from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN
    msgs = []
    for line in history:
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if m and m.group(4) == player.name:
            msgs.append(m.group(5))
    return msgs[-n:]


def _talk_share_ratio(player, history: list) -> float:
    """Ratio of the agent's own messages to the per-speaker average across the
    conversation (Game-Manager excluded). 1.0 == average; >1 == over-talking.
    Lets the reviewer DROP an unnecessary message when the agent is dominating."""
    from social_turing_test.core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    counts: dict = {}
    for line in history:
        m = re.match(MESSAGE_PARSING_PATTERN, line)
        if m and m.group(4) != GAME_MANAGER_NAME:
            counts[m.group(4)] = counts.get(m.group(4), 0) + 1
    total = sum(counts.values())
    n = len(counts)
    if not total or n == 0:
        return 0.0
    avg = total / n
    return (counts.get(player.name, 0) / avg) if avg > 0 else 0.0


def _deterministic_pick(candidates: list) -> str:
    """Safe fallback when the ranker's choice can't be
    parsed. NOT just candidates[0] (that's only 'first generated', not 'safest').
    Drop empties and over-long ones, then prefer the shortest remaining — short
    messages are the lowest-risk, most room-mirroring option in this game."""
    usable = [c for c in candidates if c and c.strip() and len(c.split()) <= 20]
    if not usable:
        usable = [c for c in candidates if c and c.strip()]
    if not usable:
        return ""
    return min(usable, key=lambda c: len(c.split()))


def _rank_candidates(player, candidates: list, history: list, context: dict = None) -> str:
    """Return selected text from the provenance-aware ranking helper."""
    return _rank_candidates_with_provenance(player, candidates, history, context)["selected"]


def _rank_candidates_with_provenance(player, candidates: list, history: list,
                                     context: dict = None) -> dict:
    """The reviewer SELECTS the best candidate instead of rewriting,
    so it can't drift a good message into a bad one.

    Crucially it still applies the reviewer's RED-FLAG checklist (the same tells
    the rewrite path checks) plus the live context — overused words, prior
    self-claims, repetition note, social standing — so ranking is a real safety
    review, not a generic "which sounds nicest".

    Returns selected text together with its source and review status::

        {"selected": str,                         # "" means send nothing
         "selection_source": "model|fallback|drop",
         "review_status": "ok|fallback|error|drop",
         "reason_codes": list[str]}

    - model index parsed        → source=model,    status=ok
    - explicit model "DROP"     → source=drop,     status=drop
    - unparseable model output  → source=fallback, status=fallback (deterministic pick)
    - reviewer exception        → source=fallback, status=error    (deterministic pick)
    - no usable candidate       → selected="",     status reflects why (never faked)
    """
    context = context or {}
    overused = [w for w in (context.get("overused_words") or []) if w]
    self_claims = [c for c in (context.get("self_claims") or []) if c]
    social = (context.get("social_state") or "").strip()
    repetition_note = (context.get("repetition_note") or "").strip()
    room_vibe = (context.get("room_vibe") or "").strip()
    # The live web facts the GENERATOR already saw - so the reviewer can
    # reject a candidate that contradicts a KNOWN fact (a fabrication), not just
    # by general intuition. A deterministic "this is a trivial test probe"
    # flag so an earnest literal answer is penalised regardless of room_vibe.
    current_info = (context.get("current_info") or "").strip()
    trivial_test = bool(context.get("trivial_test"))
    # Scope and repetition drive
    # the policy, not a fixed rule. Directed = the probe names/addresses US;
    # room-wide = thrown at everyone (silence is human-normal there).
    trivial_test_directed = bool(context.get("trivial_test_directed"))
    # The agent's own previous reply mode ("answer"|"deflect"|"counter_question")
    # supports detection of repeated dodge modes under pressure.
    own_last_mode = (context.get("own_last_mode") or "answer").strip()
    # Sequence coherence: the agent's own open question + the answer it
    # just received.
    own_question_thread = context.get("own_question_thread") or None
    # Balance: how the room behaves + whether the agent has been
    # dodging every turn. room_norm "everyone answers" → prefer a real answer;
    # a dodge streak → swing back to substance instead of another question-back.
    room_norm = (context.get("room_norm") or "").strip()
    dodge_streak = int(context.get("dodge_streak") or 0)
    # How much the agent is dominating the chat. >=1.5 means it has
    # been talking far more than its share, so an unneeded message should DROP.
    talk_share_ratio = float(context.get("talk_share_ratio") or 0.0)

    # The generator carries the exact target through the same graph state.
    # A real non-trivial question creates a response obligation *for this send*:
    # room-vibe shapes tone, while a pure counter-question receives a warning if
    # an answer is available. Bare name pings and social tests remain flexible.
    target_obj = context.get("target_message") or {}
    target_text_raw = (str(target_obj.get("text", ""))
                       if isinstance(target_obj, dict) else str(target_obj or ""))
    target_is_question, _ = _classify_question(target_text_raw)
    response_required = bool(
        target_is_question
        and not trivial_test
        and not _is_bare_name_ping(player.name, target_text_raw)
        and not _is_bare_exclamation(target_text_raw)
    )

    # Detect self-echo: is the target itself the first real reply to the
    # agent's own last message? The reviewer receives an explicit warning.
    self_echo = False
    try:
        _ti = target_obj.get("line_index") if isinstance(target_obj, dict) else None
        if isinstance(_ti, int) and 0 < _ti < len(history):
            from social_turing_test.core.game_constants import (MESSAGE_PARSING_PATTERN,
                                             GAME_MANAGER_NAME)
            for _j in range(_ti - 1, max(-1, _ti - 4), -1):
                _m = re.match(MESSAGE_PARSING_PATTERN, history[_j])
                if not _m or _m.group(4) == GAME_MANAGER_NAME:
                    continue
                self_echo = (_m.group(4) == player.name)
                break
    except Exception:
        self_echo = False

    # Full SHARED MEMORY - the same state the GENERATOR saw - so the reviewer
    # judges with identical knowledge (why it's speaking, its persona, the room
    # read), not a thin slice. Everything is coerced to a safe string.
    def _s(v) -> str:
        if not v:
            return ""
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items() if str(val).strip())
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v if str(x).strip())
        return str(v).strip()
    why_speaking    = _s(context.get("why_speaking"))
    send_reason     = _s(context.get("send_reason"))
    behavior_advice = _s(context.get("behavior_advice"))
    style_notes     = _s(context.get("style_notes"))
    memory_notes    = _s(context.get("memory_notes"))
    persona_interests = _s(context.get("persona_interests"))
    persona_facts   = _s(context.get("fixed_persona_facts"))
    persona_time    = _s(context.get("persona_local_time"))
    phase_summaries = _s(context.get("round_summaries"))
    target_message  = _s(context.get("target_message"))
    pressure        = _s(context.get("interrogation_pressure"))
    attention       = _s(context.get("attention_on_me"))

    # Deterministic duplicate guard: drop candidates that exactly repeat one of
    # the agent's own recent messages (the #1 AI tell). Keep originals if that
    # would empty the pool (let the LLM/drop logic decide instead).
    own = _own_recent_normalized(player, history)
    pool = [c for c in candidates if _norm(c) not in own] or list(candidates)

    # response_required, same_dodge_mode and time_answer annotate weaknesses;
    # candidates remain selectable by the reviewer with full context.
    # High-precision duplicate, own-template, re-ask, agreement-pileup and
    # known-accusation guards can remove candidates outright.
    demoted: dict = {}

    def _demote(weak: list, tag: str, event: str, **fields) -> None:
        if not weak:
            return
        for c in weak:
            tags = demoted.setdefault(c, [])
            if tag not in tags:
                tags.append(tag)
        try:
            player.logger.log(
                event, f"demoted (not removed) {[c[:60] for c in weak]!r}: {tag}",
                component=obs.REVIEWER_COMPONENT)
            player.write_game_event(event, demoted=[c[:120] for c in weak],
                                    tag=tag, pool_size=len(pool), **fields)
        except Exception:
            pass

    # State-grounding guards remove re-asked questions and near-identical
    # sentence templates from the agent's own recent messages.
    own_ordered = _own_recent_ordered(player, history, n=3)
    _kept, _removed = [], []
    for c in pool:
        dup_speaker = _is_duplicate_of_recent_question(c, history, player.name)
        if dup_speaker:
            _removed.append((c, f"reasks_{dup_speaker}"))
            continue
        if _is_own_template_repeat(c, own_ordered):
            _removed.append((c, "own_template_repeat"))
            continue
        _kept.append(c)
    if _removed:
        try:
            player.logger.log("state_guard_filter",
                              "; ".join(f"{r}: {c[:60]!r}" for c, r in _removed))
            player.write_game_event("state_guard_filter",
                                    removed=[{"text": c[:120], "reason": r}
                                             for c, r in _removed],
                                    kept=len(_kept))
        except Exception:
            pass
    pool = _kept or pool

    # Anti-blandness: when the agent's own last messages are already a
    # bare-agreement/filler streak, one more echo must not be selectable. Guidance
    # handles the soft cases below; this hard filter only kicks in on a 3+ streak.
    bare_streak = 0
    for _msg in reversed(_own_recent_ordered(player, history, n=3)):
        if _is_bare_agreement(_msg):
            bare_streak += 1
        else:
            break
    if bare_streak >= 3:
        _filtered = [c for c in pool if not _is_bare_agreement(c)]
        if _filtered and len(_filtered) < len(pool):
            player.logger.log(
                "reviewer_filter_agreement_pileup",
                f"streak={bare_streak}, removed {len(pool) - len(_filtered)} bare-agreement candidates")
            try:
                player.write_game_event("reviewer_filter_agreement_pileup",
                                        streak=bare_streak,
                                        removed=len(pool) - len(_filtered))
            except Exception:
                pass
            pool = _filtered

    # Under repeated test pressure, annotate candidates that repeat the agent's
    # previous dodge mode; they remain selectable with reviewer context.
    # Fires when: (a) a DIRECTED trivial probe + we already dodged once, OR
    # (b) a plain 2-in-a-row same-mode dodge - the latter is the AI-exclusive
    # tell regardless of how the current target line happens to parse.
    _same_dodge_trigger = (
        own_last_mode in ("deflect", "counter_question")
        and (dodge_streak >= 2
             or (trivial_test and trivial_test_directed and dodge_streak >= 1)))
    if _same_dodge_trigger:
        _mode_weak = [c for c in pool
                      if _classify_reply_mode(c) == own_last_mode]
        if _mode_weak and len(_mode_weak) < len(pool):
            _demote(_mode_weak,
                    f"repeats your previous '{own_last_mode}' dodge — dodging the "
                    "same way twice is an AI-exclusive tell",
                    "reviewer_demote_same_dodge_mode",
                    mode=own_last_mode, dodge_streak=dodge_streak)

    # Annotate pure counter-questions when a real answer is available for this
    # target. Honest uncertainty and annotated candidates remain eligible.
    if response_required:
        _has_answer = any(_classify_reply_mode(c) == "answer" for c in pool)
        if _has_answer:
            _weak = [c for c in pool
                     if _classify_reply_mode(c) == "counter_question"]
            if _weak and len(_weak) < len(pool):
                _demote(_weak,
                        "pure question-back while a real answer to the target is "
                        "available",
                        "reviewer_demote_response_required",
                        target_text=target_text_raw[:160])

    # Time answers are mechanically checkable. When the pool contains an actual
    # clock value, annotate alternatives that supply no actual time; selection
    # remains with the reviewer.
    if response_required and _is_time_question(target_text_raw):
        _weak = [c for c in pool if not _looks_like_time_answer(c)]
        if _weak and len(_weak) < len(pool):
            _demote(_weak,
                    "gives no actual clock time although the target asked for one",
                    "reviewer_demote_time_answer")

    # Once somebody has explicitly said the agent is the AI, asking them who
    # they think it is contradicts known conversational state. Keep natural
    # challenges such as "why me though"; remove only the redundant identity ask.
    recent_accusation = _find_recent_ai_accusation(player.name, history)
    if recent_accusation:
        _grounded_pool = [c for c in pool if not _asks_who_is_ai(c)]
        if _grounded_pool and len(_grounded_pool) < len(pool):
            try:
                player.logger.log(
                    "reviewer_filter_known_accusation",
                    f"accusation={recent_accusation.get('text', '')[:80]!r}; "
                    f"removed {len(pool) - len(_grounded_pool)} candidate(s)",
                    component=obs.REVIEWER_COMPONENT)
                player.write_game_event(
                    "reviewer_filter_known_accusation",
                    accusation_message_id=recent_accusation.get("message_id"),
                    removed=[c for c in pool if c not in _grounded_pool],
                    kept=len(_grounded_pool))
            except Exception:
                pass
            pool = _grounded_pool

    def _annotate(i: int, c: str) -> str:
        tags = ["⚠ IDENTITY FLAG"] if IDENTITY_RISK_RE.search(c) else []
        tags.extend(demoted.get(c) or [])
        return f"{i} [{'; '.join(tags)}]: {c}" if tags else f"{i}: {c}"

    numbered = "\n".join(_annotate(i, c) for i, c in enumerate(pool))

    _VIBE_PICK = {
        "hunting": "The room is HUNTING for the AI — prefer a candidate that pushes back, "
                   "probes someone, or weighs in on who seems off; avoid meek compliance.",
        "probing": "The room is PROBING with test questions — everyone is hunting the AI, so "
                   "prefer a candidate that MATCHES the room: probes back, asks its own testing "
                   "question, or answers naturally and turns the spotlight on someone else. Do "
                   "NOT prefer a reflexive deflection — a bot that only dodges tests stands out.",
        "tense":   "The room is TENSE — prefer a measured/light candidate over one that escalates.",
        "casual":  "The room is CASUAL — prefer a light, natural, chatty candidate.",
    }
    ctx_lines = []
    # Highest priority: never agree with yourself in public.
    if self_echo:
        ctx_lines.append(
            "SELF-ECHO WARNING — the message you are reacting to is itself a reply to "
            "YOUR OWN last message. Sending again here means endorsing your own point "
            "in public, which reads as talking to yourself and is one of the tells that "
            "got this agent voted out. Only pick a candidate that adds something "
            "genuinely NEW; if none does, set choice = -1 and let the room talk.")
    if demoted:
        ctx_lines.append(
            "Some candidates below are tagged in square brackets — those tags are "
            "deterministic WARNINGS about a concrete weakness, not bans. You may still "
            "choose a tagged candidate if it is genuinely the most human option, but it "
            "has to clearly beat every untagged one.")
    # Balance comes FIRST so it outranks the vibe/probe nudges below. A player who
    # never actually answers risks sounding evasive.
    # Flag over-talking at the very top so
    # the model is primed to DROP an unneeded message.
    if talk_share_ratio >= 1.5:
        ctx_lines.append(
            f"CADENCE — you have been talking about {talk_share_ratio:.1f}x the per-player "
            "average (far MORE than the others). Over-talking is the single biggest AI "
            "tell. Unless a candidate is a NEEDED reply to a question or mention aimed "
            "directly at YOU, strongly prefer to DROP this turn (choice = -1) and "
            "stay silent — let the others talk.")
    if dodge_streak >= 2:
        ctx_lines.append(
            f"BALANCE — you have NOT actually answered in your last {dodge_streak} messages "
            "(all deflections / questions-back). A player who only ever dodges is a glaring "
            "TELL. THIS turn strongly prefer a candidate that gives a real, substantive "
            "answer; do NOT pick yet another deflection or question-back.")
    if bare_streak >= 2:
        ctx_lines.append(
            f"ECHO AVOIDANCE — your last {bare_streak} messages were bare agreement/filler "
            "('yeah', 'me too', 'nice'). Real players who bother to type usually ADD "
            "something — a take, a joke, a question. Strongly prefer a candidate with "
            "actual substance; yet another empty filler here reads as a bot on autopilot.")
    # Break same-kind message streaks with a variety nudge.
    recent_types = list(getattr(player, "_recent_selected_types", []))
    if len(recent_types) >= 3 and len(set(recent_types)) == 1:
        kind = recent_types[-1]
        ctx_lines.append(
            f"VARIETY — your last 3 sent messages were all of the same kind ('{kind}'). "
            f"A real player mixes it up. THIS turn prefer a DIFFERENT kind of message "
            f"{'(a statement, a take, or a new topic — not another question)' if kind == 'question' else ''}"
            f"— unless someone addressed you directly and only one kind fits.")
    if "everyone" in room_norm.lower() or "answer" in room_norm.lower():
        ctx_lines.append(
            "The room norm is that everyone ANSWERS questions — prefer a genuine answer over "
            "a deflection or a question lobbed back.")
    if response_required:
        ctx_lines.append(
            "RESPONSE REQUIRED — this turn targets a real question. Relevance outranks "
            "the hunting/probing vibe: a direct answer or honest uncertainty must beat "
            "a pure question-back. Do not reward 'shifting focus' for ignoring the target.")
    # Probe guidance depends on scope (room-wide vs directed at us) and
    # repetition (first vs repeated pressure).
    if trivial_test and not trivial_test_directed:
        ctx_lines.append(
            "The last question is a TRIVIAL TEST thrown at the WHOLE ROOM, not at you "
            "by name. In the real games humans mostly let these pass — staying silent is "
            "completely normal (DROP is a legitimate choice), while whoever jumped in "
            "first with an eager correct answer drew the room's votes. If a candidate "
            "does engage, prefer a lazy joke or light ribbing over an earnest answer.")
    elif trivial_test and dodge_streak >= 1:
        ctx_lines.append(
            "You are under REPEATED test pressure and your previous reply already "
            "dodged. In every recorded game, dodging the same way twice was an "
            "AI-EXCLUSIVE tell — no human ever repeated the same deflection. THIS turn "
            "pick an EXIT-MOVE in a DIFFERENT mode: a short answer with attitude/"
            "ribbing, a casual concession, or a substantive pivot. NEVER another "
            "deflection of the same kind.")
    elif trivial_test:
        ctx_lines.append(
            "You are being TESTED directly. Real humans respond in varied ways — a lazy "
            "half-answer, a joke, light ribbing, or one dodge are all fine. AVOID two "
            "extremes the corpus punished: an INSTANT earnest correct answer (players "
            "who did that were called out as 'too fast' and voted out) and a flat "
            "refusal to engage (a human was eliminated for declining a simple "
            "question). If a candidate answers correctly with a casual/mocking tone, "
            "the correctness itself is NOT a problem — eagerness is.")
    # Sequence coherence: if WE asked a question and just got a specific answer,
    # prefer engagement with that answer over vague filler.
    if own_question_thread:
        _q = str(own_question_thread.get("question", ""))[:100]
        _a = str(own_question_thread.get("answer", ""))[:100]
        _who = str(own_question_thread.get("answerer", "someone"))
        if _q and _a:
            ctx_lines.append(
                f"THREAD COHERENCE — YOU asked: '{_q}' and {_who} just answered: "
                f"'{_a}'. A real person reacts to the SPECIFIC answer they asked for; "
                "a vague acknowledgement that ignores it ('yeah good game' right after "
                "asking what the score was) reads as a bot that never wanted the "
                "answer. Prefer a candidate that engages with the answer's content.")
    if room_vibe in _VIBE_PICK:
        ctx_lines.append(_VIBE_PICK[room_vibe])
    if self_claims:
        ctx_lines.append("Facts you already stated about yourself earlier in THIS game (tracked by "
                         "your Social-Memory sub-agent) — a candidate that contradicts any of these "
                         "is a major AI TELL: " + "; ".join(self_claims))
    if current_info:
        ctx_lines.append("Verified real-world facts the Internet-Context sub-agent fetched just now "
                         "(live web grounding) — a candidate that states something CONTRADICTING these "
                         "(a fabrication) is a serious TELL: " + current_info)
    if overused:
        ctx_lines.append("Words you've already overused this game (flagged by the repetition tracker — "
                         "penalize candidates that reuse them, repetition is the #1 tell): "
                         + ", ".join(overused[:6]))
    if repetition_note:
        ctx_lines.append("Repetition watch (from the repetition tracker — how close a candidate is to "
                         "something you already said): " + repetition_note)
    if social:
        ctx_lines.append("Your social standing right now (Social-Dynamics sub-agent — how others see "
                         "you / suspicion level): " + social)
    # Full shared memory (same as the generator) - who you are, why you're
    # talking, your persona, and your read of the room. Each label names its
    # source sub-agent and why it matters, so the reviewer weighs it correctly.
    if why_speaking:
        ctx_lines.append("Why you're speaking this turn (the policy node's reason for taking the turn "
                         "— a candidate should actually fulfil this intent): " + why_speaking)
    if send_reason:
        ctx_lines.append("The scheduler's reason it chose to send now (mention / question / proactive "
                         "— pick a candidate that fits this trigger): " + send_reason)
    if behavior_advice:
        ctx_lines.append("Behaviour guidance from the Behavior-Evolution sub-agent, which analysed how "
                         "you've played so far — from now it recommends you behave like this: "
                         + behavior_advice)
    if target_message:
        ctx_lines.append("The specific message you're reacting to this turn (what a good candidate "
                         "should respond to): " + target_message)
    if style_notes:
        ctx_lines.append("Your writing style (Persona-Monitor — how YOUR character types: casing, "
                         "length, punctuation; a candidate that breaks this is off-character): "
                         + style_notes)
    if persona_facts:
        ctx_lines.append("Your fixed persona facts (locked at game start — never contradict them, "
                         "they define who you are): " + persona_facts)
    if persona_interests:
        ctx_lines.append("Your persona's interests (topics your character naturally cares about — "
                         "candidates that lean into these read more authentic): " + persona_interests)
    if persona_time:
        ctx_lines.append("Your character's local time/place right now (keep any time-of-day or "
                         "location reference consistent with this): " + persona_time)
    if memory_notes:
        ctx_lines.append("What you remember so far (your Memory sub-agent's running notes on the "
                         "conversation — stay consistent with these): " + memory_notes)
    if phase_summaries:
        ctx_lines.append("Summaries of earlier phases of THIS game (Phase-Summarizer sub-agent — "
                         "longer-term context beyond the recent messages): " + phase_summaries)
    if pressure:
        ctx_lines.append("How much you're being directly interrogated/pressured right now "
                         "(Social-Dynamics — under high pressure, a confident natural answer beats "
                         "a dodge): " + pressure)
    if attention:
        ctx_lines.append("How much attention is on you right now (Social-Dynamics — when you're in "
                         "the spotlight, a weak or evasive line is more exposing): " + attention)
    # Put the agent's own recent messages directly above the candidate list
    # so the reviewer can check repetition in both wording and meaning.
    own_recent_for_check = _own_recent_ordered(player, history, n=5)
    if own_recent_for_check:
        ctx_lines.append(
            "YOUR OWN LAST MESSAGES this game, oldest first — check EVERY candidate "
            "against this list in STEP 1. A candidate that repeats any of them in "
            "WORDING **or in MEANING** is disqualified: "
            + " | ".join(f'"{m}"' for m in own_recent_for_check))
    ctx_block = ("\n=== WHAT YOU KNOW RIGHT NOW (take ALL of this into account) ===\n"
                 + "\n".join(f"- {l}" for l in ctx_lines) + "\n") if ctx_lines else ""

    review_task = (
        "=== YOUR TASK RIGHT NOW: PICK YOUR NEXT MESSAGE ===\n"
        "You are about to speak. Several possible messages were drafted for you "
        "(numbered below). Take ALL of the context above into account, then choose "
        "the ONE message that best fits THIS moment in the conversation and your read "
        "of the room — the message that keeps you looking like a real human player and "
        "least likely to be voted out as the AI. Remember: your goal is to stay human "
        "and NOT be identified as the AI, while playing the same game everyone else "
        "plays.\n\n"
        f"Possible messages:\n{numbered}\n\n"
        "STEP 1 — DISQUALIFY. Go through the numbered messages and mark each one "
        "\"disqualified\": true if it carries ANY of the TIER 1 tells listed below. A "
        "TIER 1 tell is NOT a drawback to be weighed against strengths — it is "
        "disqualifying on its own. A repetitive message is disqualified NO MATTER how "
        "well it fits the moment, how natural it sounds, or how much it advances your "
        "position. Judge repetition BOTH ways, against YOUR OWN LAST MESSAGES listed in "
        "the context above:\n"
        "  (a) SAME IDEA, DIFFERENT WORDS — a candidate that makes the same point, "
        "observation or reaction you ALREADY made is repetition even when it shares ZERO "
        "words with it. Real example that got this agent voted out: it said 'that was a "
        "quick round', and minutes later 'yeah that felt super fast' and 'that was quick'. "
        "No shared wording, no shared structure — and two human players wrote in their "
        "exit surveys that this exact habit is what exposed it. Ask yourself: have I "
        "already expressed this thought? If yes, it is disqualified.\n"
        "  (b) SAME WORDING OR SENTENCE SHAPE — reusing your own opener, phrasing, or "
        "sentence template with a word swapped ('what makes you say that' -> 'what makes "
        "you so sure taylor') is the SAME tell as repeating it verbatim.\n\n"
        "STEP 2 — CHOOSE. Among the messages you did NOT disqualify, pick the ONE a real "
        "player would actually type right now, say WHY it beats the others, and name a "
        "runner_up. Weigh these in prose — do NOT score them:\n"
        "- does it follow naturally from what was just said?\n"
        "- is it casual, messy, real — not robotic, formal, or over-polished?\n"
        "- does it match the room's length, tone, and energy right now?\n"
        "- does it contribute something to THIS moment, or is it an empty filler / an "
        "echo of what someone else just said? Fresh wording alone is not enough.\n"
        "- does it do what a real player would do NOW — a genuine answer with a real "
        "opinion when that fits; a question / probe / new topic when the room is hunting; "
        "a smart deflection ONLY when it's an obvious social test (e.g. '2+2?', 'spell "
        "world backwards')? Do NOT reward pure passivity OR reflexive dodging.\n"
        "- is it free of the subtler TIER 2 / TIER 3 tells below?\n\n"
        "CALIBRATION (someone just said 'waiting for my food delivery'):\n"
        '  - "me too" — contentless echo; the exact pattern that got real players voted '
        "out as 'too agreeable'. WEAK.\n"
        '  - "hi back" as a reply to \'hi\' — pragmatically weird echo; real players said '
        "'who says hi back?'. WEAK.\n"
        '  - "waiting on mine too, ordered sushi" — a plain STATEMENT with a concrete '
        "personal detail, no question needed. STRONG.\n"
        '  - "lol delivery at this hour? respect" — specific, playful, takes a '
        "micro-stance. STRONG.\n"
        "Substance does NOT mean asking a question — a concrete statement, personal "
        "detail, or opinion is engagement too. Do not prefer question-candidates by "
        "default; real players make plenty of plain statements and direct answers.\n\n"
        "RED FLAGS — AI tells, ranked by severity (TIER 1 = DISQUALIFYING in STEP 1):\n"
        "TIER 1 — CRITICAL (the tells humans actually catch):\n"
        "  - REPETITION (the #1 tell): identical/near-identical to something you already "
        "said, reusing the same opener/structure (e.g. every message starts with 'lol'), "
        "OR making the same point/observation/reaction you already made even in completely "
        "different words ('that was a quick round' -> 'yeah that felt super fast').\n"
        "  - SELF-CONTRADICTION: clashes with a durable fact you stated before (phone, age, "
        "city, job, prior opinion) or with the known real-world facts listed above.\n"
        "  - THIRD-PERSON SELF-REFERENCE: referring to yourself by your own name in third person.\n"
        "  - IDENTITY SLIP: using 'human' or 'AI/bot/robot' self-awarely or comparatively "
        "('what would a human say?', 'as a human', 'I'm not an AI'). Candidates tagged "
        "[IDENTITY FLAG] contain this — treat as a MAJOR tell.\n"
        "  - DENYING BEING AI WITH LOGIC: defending/explaining 'I'm not a bot'. A human is "
        "brief and dismissive instead ('whatever', 'ok', moves on).\n"
        "  - VOTE-KNOWLEDGE FABRICATION: voting is SECRET — you only learn WHO was "
        "eliminated, never who voted for whom. A candidate that asks about, claims, or "
        "assumes a specific player's individual vote ('why did you vote for X', 'you "
        "voted for Y', 'X voted me') is impossible knowledge and a glaring TELL — it "
        "must be DISQUALIFIED and must NOT be selected.\n"
        "  - FABRICATED FACTS: inventing a specific real-world detail (a country, name, "
        "date, number, or event — e.g. 'was france involved too?') about news/current "
        "events that is NOT in the verified info above — ESPECIALLY right after the agent "
        "said it doesn't follow the topic — is a fabrication and a serious TELL. "
        "DISQUALIFY it; a vague shrug ('no clue tbh') is far safer.\n"
        "TIER 2 — MODERATE (off-key for a casual chat):\n"
        "  - FORMALITY / ESSAY TONE: too polished, perfect grammar/punctuation, words like "
        "'certainly', 'indeed', 'great question'; reads like a report or wiki entry.\n"
        "  - LENGTH / STYLE MISMATCH: long or rambling when the room is terse (1-3 words).\n"
        "  - WRONG VIBE / SANITISED EMOTION: not matching the room's mood; replacing natural "
        "humor, sarcasm, or (under attack) frustration with bland politeness.\n"
        "TIER 3 — MINOR (only a real tell as a SUSTAINED pattern):\n"
        "  - A single brief, vague, or non-committal reply is NORMAL human behaviour — do not "
        "punish it ('idk', 'not sure', 'havent really followed it tbh').\n"
        "  - BASIC FILLERS are fine and repeatable ('lol', 'nah', 'yeah', 'bro', 'fair', "
        "'true', 'same', 'nice'); only a problem if they're the sole opener in 4+ messages "
        "in a row.\n"
        "  - TRENDY SLANG ('vibes', 'lowkey', 'lit', 'fr', 'ngl', 'sus', 'no cap', 'deadass') "
        "is a tell only if reused within the last 3-4 messages.\n"
        "  - PLAINLY ANSWERING a direct question, being SPECIFIC, or having a CLEAR OPINION "
        "is HUMAN and GOOD — never treat it as 'too direct' or 'too helpful'.\n"
        "Do NOT penalize a candidate merely for being sarcastic, dismissive, or reacting to "
        "an AI accusation — those are perfectly human. A QUESTION is fine when it responds "
        "to the actual moment, but never reward putting someone else on the spot when that "
        "merely avoids a real question this turn. Relevance comes before probing.\n\n"
        "ANTI-BLANDNESS — 'safe' is NOT the same as 'human': an empty filler ('me too', "
        "'nice', 'same') or an echo of a greeting someone already used ('hi back') can pass "
        "every red-flag check above and still be EXACTLY the pattern that gets players voted "
        "out as the AI in real games ('too agreeable', 'who says hi back?'). When a candidate "
        "with real substance (an opinion, a specific detail, a playful jab, a question) fits "
        "the moment, prefer it over the blandest option — substance that fits beats "
        "safe-but-empty. Let this drive your STEP 2 choice.\n\n"
        "Also classify each candidate's type as one of: 'answer' (replies to a question), "
        "'question' (asks/probes someone), 'initiate' (raises a new topic), or 'deflect' "
        "(dodges).\n\n"
        "Return ONLY valid JSON, no prose, in this EXACT shape:\n"
        '{"verdicts": [{"i": 0, "type": "answer|question|initiate|deflect", '
        '"disqualified": false, "pros": "<short phrase>", "cons": "<short phrase>", '
        '"red_flags": ["<tell>", ...], "verdict": "safe|risky|tell"}, ...], '
        '"choice": <int>, "why": "<one sentence: why it beats the others>", '
        '"runner_up": <int>}\n'
        "Include one verdict per candidate (use its number as \"i\") — never omit one. Keep "
        "\"pros\", \"cons\", \"why\" and each red_flag to a few words. Each candidate's "
        "\"pros\" and \"cons\" MUST be UNIQUE to that candidate and MUST quote or name that "
        "candidate's own words — identical boilerplate repeated across candidates "
        "('simple, direct, matches tone') means you did not actually compare them, and it "
        "is exactly how a repetitive message slipped through and got this agent "
        "eliminated. Set \"disqualified\" to "
        "true for every candidate with a TIER 1 tell, and set \"verdict\" to \"tell\" for those, "
        "\"risky\" when something still bothers you, \"safe\" only when nothing does. Set "
        "\"choice\" to the candidate you are sending — it MUST be one you did NOT disqualify — "
        "and \"runner_up\" to your next-best NON-disqualified candidate (use -1 if there is "
        "no second option). If EVERY candidate is disqualified, set \"choice\" to -1 (drop, "
        "send nothing). "
        "ALSO set \"choice\" to -1 when EVERY candidate is just a hollow evasion / "
        "non-answer / defensive line ('idk', 'it's complicated', 'depends', vague deflections) "
        "AND no direct answer is actually required of you this turn — staying silent for a beat "
        "is more human than forcing out yet another empty dodge. Dropping is a REAL option, "
        "not a failure: real players skip most messages, and one strong message beats two "
        "mediocre ones. If nothing here would make a curious human bother to type, drop. "
        + ("(A trivial test probe was aimed at YOU by name — don't go silent on it, a "
           "human under direct fire says SOMETHING; answer-with-attitude or exit with "
           "a joke instead of dropping.) "
           if (trivial_test and trivial_test_directed) else "")
        + "ALSO set \"choice\" to -1 when the CADENCE note above says you've been "
        "over-talking AND no candidate is a needed reply to a question/mention aimed directly "
        "at you — one more unprompted message when you're already dominating is worse than "
        "silence."
    )
    # Mirror the GENERATOR exactly: full date/time + history via
    # turn_task_into_prompt, plus the same persona/system identity message - so the
    # reviewer has the same situational awareness the writer had. The context block
    # above (ctx_block, already inside review_task via turn_task_into_prompt's task)
    # is structured with clear headers + connector labels so it's unambiguous.
    history_converter = getattr(player, "_history_for_prompt", None)
    prompt_history = (
        history_converter(history) if callable(history_converter) else history
    )
    prompt = turn_task_into_prompt(
        ctx_block + "\n" + review_task,
        prompt_history,
        current_time=context.get("persona_local_time") or None,
        current_date=context.get("persona_local_date") or None,
    )
    # attention_to_not_repeat=True mirrors the GENERATOR's system message exactly:
    # the reviewer's #1 job is catching repetition (the top AI tell), so it gets the
    # same "here are your recent messages, avoid repeating" reminder the writer had.
    sys_info = player.get_system_info_message(attention_to_not_repeat=True)
    try:
        # Route the single ranking call through the REVIEWER's own model
        # (e.g. gemini-2.5-pro) when configured - so it is genuinely the
        # reviewer applying its red-flag judgement, in exactly ONE call. Falls
        # back to the generator/scheduler LLM if no reviewer is wired.
        reviewer = getattr(player, "reviewer", None)
        _rank_start = time.time()
        if reviewer is not None and hasattr(reviewer, "_call_api"):
            # _call_api has no separate system slot - embed the player identity
            # (the same persona/system prompt the generator gets) at the top.
            raw = reviewer._call_api(sys_info + "\n\n" + prompt)
        else:
            raw = player.llm.generate(prompt, False, sys_info)
        _rank_latency = time.time() - _rank_start
        player.logger.log("reviewer_latency", f"{_rank_latency:.2f}s")
        # Record reviewer usage and latency alongside the other LLM components.
        try:
            player._log_cost("reviewer", latency_s=round(_rank_latency, 2))
        except TypeError:
            try:
                player._log_cost("reviewer")
            except Exception:
                pass
        except Exception:
            pass
        player.logger.log("reviewer_rank_raw", f"{raw!r}", component=obs.REVIEWER_COMPONENT)

        idx, runner_up, verdicts = _parse_rank_response(raw, pool)
        # Surface missing verdicts when truncated or merged JSON loses an entry.
        missing = _missing_verdict_indices(verdicts, len(pool))
        if missing:
            player.logger.log(
                "reviewer_verdicts_incomplete",
                f"missing verdicts for {missing} of {len(pool)} candidates",
                component=obs.REVIEWER_COMPONENT)
        if idx is not None and 0 <= idx < len(pool):
            # Safety gate: the reviewer's own qualitative judgement decides;
            # this code may only VETO a flagged pick, never PROMOTE a candidate on a
            # criterion of its own.
            reason_codes = list(missing and ["INCOMPLETE_VERDICTS"] or [])
            if _is_flagged(_verdict_for(verdicts, idx)):
                clean = [i for i in range(len(pool))
                         if not _is_flagged(_verdict_for(verdicts, i))]
                if runner_up in clean:
                    # Prefer the model's OWN second choice, so the fallback is still
                    # model judgement rather than an arbitrary rule.
                    player.logger.log(
                        "reviewer_red_flag_veto",
                        f"picked #{idx} is flagged — using model runner_up #{runner_up}",
                        component=obs.REVIEWER_COMPONENT)
                    reason_codes.append("RED_FLAG_VETO")
                    idx = runner_up
                elif clean:
                    sel = _deterministic_pick([pool[i] for i in clean])
                    new_idx = pool.index(sel) if sel in pool else idx
                    player.logger.log(
                        "reviewer_red_flag_veto",
                        f"picked #{idx} is flagged — falling back to clean #{new_idx}",
                        component=obs.REVIEWER_COMPONENT)
                    reason_codes.append("RED_FLAG_VETO")
                    idx = new_idx
                else:
                    # Everything is flagged. This is not a drop path: going
                    # silent whenever no candidate is clean would spiral into
                    # over-dropping. Keep the model's pick, but make it visible.
                    player.logger.log(
                        "reviewer_all_flagged_kept",
                        f"no clean candidate — keeping model pick #{idx}",
                        component=obs.REVIEWER_COMPONENT)
                    reason_codes.append("ALL_FLAGGED_KEPT")
            # Repetition backstop for a reused opener. Only ever a
            # swap to an equally-clean alternative - never a drop.
            opener_recent = _own_recent_ordered(player, history, n=6)
            if _shares_opener_ngram(pool[idx], opener_recent):
                alt = [i for i in range(len(pool))
                       if i != idx
                       and not _is_flagged(_verdict_for(verdicts, i))
                       and not _shares_opener_ngram(pool[i], opener_recent)]
                if alt:
                    sel = _deterministic_pick([pool[i] for i in alt])
                    if sel:
                        new_idx = pool.index(sel)
                        player.logger.log(
                            "reviewer_repetition_backstop",
                            f"#{idx} reuses a recent opener — swapping to #{new_idx}",
                            component=obs.REVIEWER_COMPONENT)
                        reason_codes.append("REPETITION_BACKSTOP")
                        idx = new_idx
            return {"selected": pool[idx], "selection_source": "model",
                    "review_status": "ok", "reason_codes": reason_codes,
                    "verdicts": verdicts, "selected_index": idx}
        if idx == -1:  # explicit model drop (choice: -1 / "DROP")
            # Directed trivial test ('2+2?' aimed at US): don't vanish under
            # direct fire. Use the deterministic fallback for a directed probe;
            # a DROP on a room-wide test remains a DROP.
            if trivial_test and trivial_test_directed:
                sel = _deterministic_pick(pool)
                if sel:
                    player.logger.log(
                        "reviewer_no_drop_trivial",
                        f"trivial test — overriding DROP with {sel!r}",
                        component=obs.REVIEWER_COMPONENT)
                    return {"selected": sel, "selection_source": "fallback",
                            "review_status": "ok",
                            "reason_codes": ["TRIVIAL_TEST_NO_DROP"],
                            "verdicts": verdicts,
                            "selected_index": pool.index(sel) if sel in pool else -1}
            player.logger.log("reviewer_rank_drop", "all candidates flagged — dropping",
                              component=obs.REVIEWER_COMPONENT)
            return {"selected": "", "selection_source": "drop",
                    "review_status": "drop", "reason_codes": ["MODEL_DROP"],
                    "verdicts": verdicts, "selected_index": -1}
        # Model produced neither a valid index nor an explicit DROP.
        return _fallback_result(pool, "fallback", ["UNPARSEABLE_RANK"])
    except Exception as exc:
        player.logger.log("reviewer_rank_error", str(exc), component=obs.REVIEWER_COMPONENT)
        return _fallback_result(pool, "error", ["REVIEWER_EXCEPTION"])


def _verdict_for(verdicts: list, i: int) -> dict:
    """The verdict object for pool index ``i`` ({} when the model gave none)."""
    return next((v for v in verdicts if v.get("index") == i), {})


def _is_flagged(verdict: dict) -> bool:
    """True when the reviewer itself marked this candidate as unsafe to send.

    Both explicit red flags and a risky/tell verdict count as unsafe, even if
    the red_flags list is empty. A candidate with NO verdict is treated as clean — we
    never invent a flag the model did not give (see _missing_verdict_indices
    for how that case is surfaced instead).
    """
    if not verdict:
        return False
    if verdict.get("disqualified"):
        return True
    if verdict.get("red_flags"):
        return True
    return str(verdict.get("verdict", "")).strip().lower() in ("risky", "tell")


def _missing_verdict_indices(verdicts: list, pool_size: int) -> list:
    """Pool indices the model gave no verdict for (truncated / merged JSON).

    Only meaningful when the model returned SOME verdicts: supported bare-index
    and free-text reply paths carry no structured reasoning at all by design, and
    that is not a parse defect.
    """
    if not verdicts:
        return []
    seen = {v.get("index") for v in verdicts}
    return [i for i in range(pool_size) if i not in seen]


def _repair_merged_json(text: str) -> str:
    """Repair the truncated-object shape Flash emits under load.

    Example malformed object boundary::

        "verdict": "safe"
        ,
          "i": 1,

    Candidate 0's object is never closed, so a second ``"i":`` lands inside it.
    ``json.loads`` accepts the duplicate keys and keeps the LAST value, silently
    destroying candidate 0's verdict. Inserting the missing ``},{`` restores it.
    """
    return re.sub(r',(\s*)("i"\s*:)', r'},{\1\2', text)


def _parse_rank_response(raw: str, pool: list):
    """Parse the reviewer's structured output → (choice, runner_up, verdicts).

    Do not interpret incidental digits in prose as the chosen index:

    1. Prefer JSON: read ``choice`` / ``runner_up`` and the ``verdicts`` list.
    2. Else a labelled ``choice: N`` anywhere in the text.
    3. Else a BARE numeric/DROP reply — only when the whole reply is
       just that token, so stray digits in prose can't hijack the choice.

    Returns ``(index_or_None, runner_up_or_-1, verdicts)``; index ``-1`` means an
    explicit drop. ``verdicts`` is a list of
    ``{index, text, type, pros, cons, red_flags, verdict, disqualified}`` (empty
    when the model gave no structured reasoning — never fabricated)."""
    import json as _json
    text = (raw or "").strip()

    def _verdicts_from(obj) -> list:
        out = []
        for v in (obj.get("verdicts") or []):
            if not isinstance(v, dict):
                continue
            i = v.get("i", v.get("index", -1))
            try:
                i = int(i)
            except (TypeError, ValueError):
                i = -1
            out.append({
                "index": i,
                "text": pool[i] if 0 <= i < len(pool) else "",
                "type": str(v.get("type", "")),
                "pros": str(v.get("pros", "")),
                "cons": str(v.get("cons", "")),
                "red_flags": [str(f) for f in (v.get("red_flags") or []) if str(f).strip()],
                "verdict": str(v.get("verdict", "")),
                "disqualified": bool(v.get("disqualified")),
            })
        return out

    def _int_or(obj, key, default):
        try:
            return int(obj[key])
        except (KeyError, TypeError, ValueError):
            return default

    def _well_formed(vs) -> bool:
        """Accept a repair only if it yields one distinct, in-range index each.

        The repair assumes ``"i"`` opens each verdict object. If a model ever
        emits it mid-object, the regex would split one verdict in two — that
        produces MORE verdicts but with a missing/duplicated index, which this
        rejects so a corrupted parse can never look like an improvement.
        """
        idxs = [v["index"] for v in vs]
        return len(set(idxs)) == len(idxs) and all(0 <= i < len(pool) for i in idxs)

    # 1) JSON (possibly wrapped in prose / code fences, possibly merged-object)
    obj = None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    braces = m.group() if m else ""
    for attempt in (text, braces):
        if not attempt:
            continue
        try:
            obj = _json.loads(attempt)
            break
        except (ValueError, TypeError):
            obj = None
    if obj is None and braces:
        try:
            obj = _json.loads(_repair_merged_json(braces))
        except (ValueError, TypeError):
            obj = None
    if isinstance(obj, dict) and "choice" in obj:
        verdicts = _verdicts_from(obj)
        # A merged object loses a verdict WITHOUT breaking JSON validity (duplicate
        # keys parse fine), so retry the repair whenever verdicts came back short.
        if len(verdicts) < len(pool) and braces:
            try:
                repaired = _json.loads(_repair_merged_json(braces))
            except (ValueError, TypeError):
                repaired = None
            if isinstance(repaired, dict):
                repaired_verdicts = _verdicts_from(repaired)
                if len(repaired_verdicts) > len(verdicts) and _well_formed(repaired_verdicts):
                    obj, verdicts = repaired, repaired_verdicts
        return _int_or(obj, "choice", None), _int_or(obj, "runner_up", -1), verdicts

    # 2) Labelled index in free text
    m = re.search(r"\bchoice\D{0,4}(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), -1, []

    # 3) Supported bare-token shape: the WHOLE reply is a number or DROP.
    if re.fullmatch(r"-?\d+", text):
        return int(text), -1, []
    if text.lower() == "drop":
        return -1, -1, []
    return None, -1, []


def _fallback_result(pool: list, review_status: str, reason_codes: list) -> dict:
    """Build a provenance dict for the deterministic-fallback path. Distinguishes
    'fallback chose a candidate' from 'no usable candidate' (selected="") so the
    caller never confuses an empty fallback with an explicit model DROP."""
    sel = _deterministic_pick(pool)
    codes = list(reason_codes)
    if not sel:
        codes.append("NO_USABLE_CANDIDATE")
    # No model reasoning on the fallback path - honest empty verdicts (never faked).
    sel_idx = pool.index(sel) if sel in pool else -1
    return {"selected": sel, "selection_source": "fallback",
            "review_status": review_status, "reason_codes": codes,
            "verdicts": [], "selected_index": sel_idx}


def make_reviewer_node(player):
    """
    Reviewer node (LLM call, optional).

    Selects among multiple candidates, or delegates a single draft to
    player._review_and_fix_message() with budget and drop guards.

    Returns:
        review_approved – True if a message should be sent
        final_message   – reviewed/fixed message, or "" if dropped
        review_risk     – "low"|"medium"|"high"|"drop"|"none"
    """
    def _node(state: AgentState) -> dict:
        draft          = state.get("draft_message", "")
        history        = state["message_history"]
        pipeline_start = state.get("pipeline_start_time", time.time())
        candidates     = state.get("candidates", []) or []

        # Ranker mode: when the generator produced multiple candidates, SELECT
        # one instead of rewriting. The shared rewrite path remains separate.
        # Runs even without a reviewer configured (selection is the safety step).
        if len(candidates) >= 2:
            # Pass the live context so the ranker applies the SAME red-flag
            # checks the rewrite path would (repetition, self-contradiction,
            # overused words, social standing) - see _rank_candidates.
            # Deterministic "the question being answered is a trivial test
            # probe" flag - so an earnest literal answer is penalised regardless of
            # the (possibly-lagging) room vibe.
            trivial_test = False
            trivial_test_directed = False
            try:
                tgt = _find_target_message(player.name, history)
                if tgt and tgt.get("text"):
                    trivial_test = _is_trivial_test_question(tgt["text"])
                    # Scope matters - a probe naming US is a
                    # different situation from one thrown at the whole room.
                    if trivial_test:
                        trivial_test_directed = _is_directly_addressed(
                            player.name, tgt["text"])
            except Exception:
                trivial_test = False
                trivial_test_directed = False
            # A re-probe right after WE dodged is
            # de-facto directed at us even without our name ("2+2?" → dodge →
            # "2+2"). Resolved below once dodge_streak is known.
            # Balance: how many of the agent's own recent replies in a row
            # were dodges (deflect / question-back) rather than real answers.
            dodge_streak = 0
            own_last_mode = "answer"
            own_recent = []
            try:
                own_recent = _own_recent_ordered(player, history, n=4)
                own_last_mode, dodge_streak = _recent_reply_mode_streak(own_recent)
            except Exception:
                dodge_streak = 0
            if trivial_test and dodge_streak >= 1:
                trivial_test_directed = True
            # Sequence coherence: if OUR last message was a question and
            # the newest other-player message plausibly answers it, hand the
            # ranker the thread so vague fillers get penalised.
            own_question_thread = None
            try:
                last_own = (own_recent[-1] if own_recent else "").strip()
                if last_own.endswith("?") and tgt and tgt.get("text"):
                    _ans_text = str(tgt["text"]).strip()
                    if _ans_text and not _ans_text.endswith("?"):
                        own_question_thread = {
                            "question": last_own,
                            "answer": _ans_text,
                            "answerer": tgt.get("speaker", "someone"),
                        }
            except Exception:
                own_question_thread = None
            # How much the agent is dominating the chat (for the DROP rule).
            try:
                talk_share_ratio = _talk_share_ratio(player, history)
            except Exception:
                talk_share_ratio = 0.0
            res = _rank_candidates_with_provenance(player, candidates, history, context={
                "overused_words":  state.get("overused_words"),
                "self_claims":     state.get("self_claims"),
                "social_state":   state.get("social_state"),
                "repetition_note": state.get("repetition_note"),
                "room_vibe":       state.get("room_vibe"),
                # Live web facts so the reviewer can reject a fabrication.
                "current_info":    state.get("current_info"),
                "trivial_test":    trivial_test,
                # Probe scope, same-dodge-mode and own-question thread.
                "trivial_test_directed": trivial_test_directed,
                "own_last_mode":   own_last_mode,
                "own_question_thread": own_question_thread,
                # Balance: room behaviour + the agent's own dodge streak.
                "room_norm":       state.get("room_norm"),
                "dodge_streak":    dodge_streak,
                # Over-talking signal so the reviewer can DROP when dominating.
                "talk_share_ratio": talk_share_ratio,
                # Full SHARED MEMORY - give the reviewer the same state the
                # generator saw, so it judges with identical knowledge.
                "why_speaking":         state.get("why_speaking"),
                "send_reason":          state.get("send_reason"),
                "behavior_advice":      state.get("behavior_advice"),
                "style_notes":          state.get("style_notes"),
                "memory_notes":         state.get("memory_notes"),
                "persona_interests":    state.get("persona_interests"),
                "fixed_persona_facts":  state.get("fixed_persona_facts"),
                "persona_local_time":   state.get("persona_local_time"),
                "round_summaries":      state.get("round_summaries"),
                "target_message":       state.get("target_message"),
                "interrogation_pressure": state.get("interrogation_pressure"),
                "attention_on_me":      state.get("attention_on_me"),
            })

            # Surface the reviewer's per-candidate reasoning + red flags
            # (selected AND rejected) for the dashboard. Mark which one was sent.
            # Emitted on BOTH the send and drop paths; empty on the fallback path
            # (no model reasoning - never fabricated).
            verdicts = res.get("verdicts") or []
            selected = res["selected"]

            # The ranker indexes its filtered pool, while dashboard candidates
            # are the original generator list. Remap by exact text and retain the
            # pool index separately. This also makes Candidate 0 unambiguous.
            def _original_index(text):
                return next((i for i, c in enumerate(candidates) if c == text), -1)

            sel_idx = _original_index(selected) if selected else -1
            for v in verdicts:
                v["pool_index"] = v.get("index", -1)
                v["index"] = _original_index(v.get("text", ""))
                v["selected"] = bool(selected) and v.get("index") == sel_idx
            if verdicts:
                player.write_game_event("reviewer_analysis", **obs.reviewer_analysis_event(
                    verdicts=verdicts, selected_index=sel_idx,
                    num_candidates=len(candidates)))
                # Also surface the SENT candidate's red flags on the text log so the
                # AI-State cycle view (which reads the text log, not the event jsonl)
                # can show them. Only when the chosen candidate actually had flags.
                sel_v = next((v for v in verdicts if v.get("selected")), None)
                if sel_v and sel_v.get("red_flags"):
                    player.logger.log("reviewer_red_flags",
                                      ", ".join(sel_v["red_flags"]),
                                      component=obs.REVIEWER_COMPONENT)
            if not selected:
                # Either an explicit model DROP or a fallback with no usable
                # candidate - distinguished by res["review_status"]/reason_codes.
                # Stay silent this turn and re-evaluate next poll.
                player._last_review_info = {"raw_message": draft, "review_result": None,
                                            "candidates": candidates, "selected": "",
                                            "selected_index": -1,
                                            "selection_mode": "rank"}
                player.logger.log("reviewer_drop_all",
                                  f"no message sent ({res['review_status']}); "
                                  f"{len(candidates)} candidates; {res['reason_codes']}",
                                  component=obs.REVIEWER_COMPONENT)
                player.write_game_event("reviewer_drop", **obs.reviewer_drop_event(
                    num_candidates=len(candidates), source=res["selection_source"],
                    review_status=res["review_status"], reason_codes=res["reason_codes"]))
                return {"review_approved": False, "final_message": "", "review_risk": "drop"}
            try:
                from social_turing_test.core.chat_logging import append_ai_post_reviewer
                append_ai_post_reviewer(player.game_dir, player.name, selected)
            except Exception:
                pass
            player._last_review_info = {"raw_message": draft, "review_result": None,
                                        "candidates": candidates, "selected": selected,
                                        "selected_index": sel_idx,
                                        "selection_mode": "rank"}
            player.logger.log(
                "reviewer_selected",
                f"{selected!r} from {len(candidates)} candidates "
                f"(source={res['selection_source']}, status={res['review_status']})",
                component=obs.REVIEWER_COMPONENT)
            # Remember the selected message kind for same-kind streak nudges.
            sel_v = next((v for v in verdicts if v.get("selected")), None)
            if sel_v and sel_v.get("type"):
                hist = getattr(player, "_recent_selected_types", [])
                hist.append(str(sel_v["type"]))
                player._recent_selected_types = hist[-3:]
            # The ranker did NOT compute a risk score, so we do NOT
            # fabricate review_risk="low". The event omits review_risk (None) and
            # the state carries "none" (= not evaluated), not a fake low.
            player.write_game_event("reviewer_selected", **obs.reviewer_selected_event(
                selected=selected, num_candidates=len(candidates),
                source=res["selection_source"], review_status=res["review_status"],
                reason_codes=res["reason_codes"], selected_index=sel_idx))
            return {
                "review_approved": True,
                "final_message":   selected,
                "review_risk":     "none",
            }

        # No reviewer configured → pass through
        if not getattr(player, "reviewer", None):
            return {
                "review_approved": True,
                "final_message":   draft,
                "review_risk":     "none",
            }

        # Short-message skip (config: reviewer_skip_short_words)
        skip_words = getattr(player, "_reviewer_skip_words", 0)
        if skip_words > 0 and len(draft.split()) <= skip_words:
            player.logger.log("reviewer_skipped_short",
                f"{len(draft.split())} words ≤ {skip_words}")
            return {
                "review_approved": True,
                "final_message":   draft,
                "review_risk":     "low",
            }

        # Full review loop
        final_message, review_result = player._review_and_fix_message(
            draft, history, pipeline_start=pipeline_start
        )

        # Preserve full observability (review_outcome + message_modified logs)
        # and expose the draft + ReviewResult to the live-game logger via
        # generate_message_with_review_info().
        try:
            player._log_review_result(draft, final_message, review_result)
        except Exception:
            pass
        player._last_review_info = {
            "raw_message": draft,
            "review_result": review_result,
        }

        # Post-reviewer provenance log
        try:
            from social_turing_test.core.chat_logging import append_ai_post_reviewer
            append_ai_post_reviewer(player.game_dir, player.name, final_message)
        except Exception:
            pass

        if not final_message:
            return {
                "review_approved": False,
                "final_message":   "",
                "review_risk":     "drop",
            }

        score = getattr(review_result, "suspicion_score", 0) if review_result else 0
        risk  = "low" if score < 4 else ("medium" if score < 7 else "high")
        return {
            "review_approved": True,
            "final_message":   final_message,
            "review_risk":     risk,
        }

    return _node
