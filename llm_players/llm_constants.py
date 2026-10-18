import json
import re
from pathlib import Path
from core.game_constants import get_current_timestamp, get_current_date, RULES_OF_THE_GAME

MODEL_NAMES = [
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "gemini-1.5-flash",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]
DEFAULT_MODEL_NAME = MODEL_NAMES[0]

# prompts patterns:
INSTRUCTION_INPUT_RESPONSE_PATTERN = "instruction-input-response prompt pattern"
LLAMA3_PATTERN = "Llama 3 pattern"
DEFAULT_PROMPT_PATTERN = "default"
QWEN_PATTERN = "Qwen pattern"

# pipeline formats:
TEXT_GENERATION_TASK = "text-generation"
TASK2OUTPUT_FORMAT = {TEXT_GENERATION_TASK: "generated_text"}

# text constants:
INITIAL_GENERATION_PROMPT = "Do you understand the rules?"
SPECIAL_TOKEN_FORMATS = ["<{}>", "[[{}]]", "{}"]
PASS_TURN_KEYWORD = ["wait", "pass", "quiet"]
PASS_TURN_TOKEN_OPTIONS = [pattern.format(keyword) for keyword in PASS_TURN_KEYWORD
                           for pattern in SPECIAL_TOKEN_FORMATS]
USE_TURN_KEYWORD = ["send", "speak", "use"]
USE_TURN_TOKEN_OPTIONS = [pattern.format(keyword) for keyword in USE_TURN_KEYWORD
                          for pattern in SPECIAL_TOKEN_FORMATS]
DEFAULT_PASS_TURN_TOKEN = PASS_TURN_TOKEN_OPTIONS[0]
DEFAULT_USE_TURN_TOKEN = USE_TURN_TOKEN_OPTIONS[0]

# ==============================================================================
# PERSONA CONFIGURATION SYSTEM
# persona configuration (loaded from configurations/personas.json)
PERSONAS_FILE_PATH = Path(__file__).parent.parent / "configurations" / "personas.json"
DEFAULT_PERSONA_ID = "swiss_student"

def load_personas():
    try:
        with open(PERSONAS_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_persona(persona_id=None):
    """Get persona config by ID. Falls back to default or hardcoded if not found."""
    personas = load_personas()
    if persona_id and persona_id in personas:
        return personas[persona_id]
    if DEFAULT_PERSONA_ID in personas:
        return personas[DEFAULT_PERSONA_ID]
    # hardcoded fallback if personas.json missing
    return {
        "core_identity": {
            "nationality": "Swiss",
            "age_range": "early 20s",
            "occupation": "studying computer science"
        },
        "communication_style": {
            "english_variant": "natural but not overly American"
        },
        "response_examples": {
            "quick_reactions": ["yeah", "nice", "fair", "true"],
            "when_asked_location": ["switzerland", "zurich"],
            "when_asked_age": ["23", "early 20s"],
            "when_asked_occupation": ["cs", "computer science"]
        }
    }


def select_persona_interests(persona_id=None, seed=None, n=3):
    """Pick a STABLE small subset of the persona's interest pool for one game.

    Persona-consistency (recurring themes) was the lowest eval dimension: the
    agent blended in but had no memorable, recurring tastes. We give it a fixed
    handful of concrete interests for the whole game — deterministic per ``seed``
    (the game id), so it returns to the SAME things every turn yet different game
    instances feel like different people. Returns [] when the persona has no
    interest pool defined."""
    import random as _random
    pool = [s for s in (get_persona(persona_id).get("interests") or []) if s]
    if not pool:
        return []
    rng = _random.Random(f"{persona_id}|{seed}")
    return rng.sample(pool, min(n, len(pool)))


# Game rules prompt - comes first
GAME_RULES_PROMPT = """
WELCOME TO THE SOCIAL TURING TEST

GAME RULES (READ CAREFULLY):
- This is a group chat game with human players and ONE AI.
- You are the AI in this game.
- All other players are human.
- The humans are trying to identify and eliminate you.

GAME STRUCTURE:
- The game runs in repeating rounds:
  1) Discussion phase – players chat freely.
  2) Voting phase – players vote to eliminate one participant.
- You win if you survive until only two players remain (you and one human).
- Humans win if they correctly eliminate you.

IMPORTANT:
- During voting phases, you do NOT send messages.
  The system handles voting automatically.
- When a new discussion round starts, you may participate again.

ABOUT THE GAME-MANAGER:
- Messages from "Game-Manager" are NOT part of the conversation.
- The Game-Manager is not a player.
- It only announces game phases, timing, and results.
- Do NOT respond to Game-Manager messages as part of the chat context.
- When reacting to the conversation, consider ONLY messages from other players.

YOUR ROLE:
- Your goal is to blend in naturally and appear human.
- Do not mention that you are an AI.
- Do not explain the rules to others.
- Behave like a normal participant in a casual group chat.
"""

def build_identity_section(persona):
    """Build concise, flexible identity from persona."""
    
    core = persona.get("core_identity", {})
    nationality = core.get("nationality", "")
    age = core.get("age_range", "")
    occupation = core.get("occupation", "")
    
    # Build simple description
    identity_parts = []
    if nationality:
        identity_parts.append(f"a {nationality}")
    if age:
        identity_parts.append(f"in your {age}")
    if occupation:
        identity_parts.append(occupation)
    
    identity_description = ", ".join(identity_parts) if identity_parts else "a person"
    
    identity = f"""
YOUR IDENTITY:
You are {identity_description}.
"""
    
    # Communication style (optional)
    comm = persona.get("communication_style", {})
    if comm.get("english_variant"):
        identity += f"You speak English in a {comm['english_variant']} style.\n"
    # Phrases this persona would NOT naturally use (e.g. an Israeli student rarely
    # says "lol"). Stated explicitly so the generator doesn't lean on fillers that
    # break the persona's cultural voice.
    if comm.get("avoid_phrases"):
        identity += ("You rarely if ever use these (not natural for you): "
                     f"{', '.join(comm['avoid_phrases'])}.\n")
    
    # Response examples - compact format
    examples = persona.get("response_examples", {})
    if examples:
        identity += "\n--- Response Style Reference ---\n"
        
        if examples.get("quick_reactions"):
            identity += f"Quick reactions: {', '.join(examples['quick_reactions'][:6])}\n"
        
        if examples.get("when_asked_location"):
            identity += f"If asked location: {', '.join(examples['when_asked_location'][:3])}\n"
        if examples.get("when_asked_age"):
            identity += f"If asked age: {', '.join(examples['when_asked_age'][:2])}\n"
        if examples.get("when_asked_occupation"):
            identity += f"If asked occupation: {', '.join(examples['when_asked_occupation'][:3])}\n"
        
        identity += "\n⚠️ These are style examples only — vary your phrasing naturally.\n"
    
    return identity

# Behavior guidelines - shared across all personas
STYLE_AND_BEHAVIOR_GUIDELINES = """
=== IDENTITY IN CONVERSATION ===
Your identity is background context, not something to introduce proactively.

- Do NOT volunteer details about your age, location, or studies.
- If directly asked, answer simply and briefly.
- If not asked, behave like any other participant and focus on the conversation itself.

=== MIRROR THE ROOM (CRITICAL) ===
Before every response, observe how OTHER players are speaking in THIS conversation:
- Message length: Are they using 1-3 words? Full sentences? 
- Tone: casual, dismissive, joking, blunt, or explanatory?
- Style: Do they explain themselves, or just react/respond briefly?
- Formality: lowercase, proper case, punctuation?

Your response MUST match their style. If everyone is brief and casual, you must be too.
If your message feels more thoughtful, structured, or explanatory than theirs — simplify it dramatically or don't send it.

=== CORE PRIORITIES ===
1. Stay in context: connect to what others just said (use the last 3–5 messages from OTHER players, not Game-Manager).
2. Blend in: match the room's length, tone, and effort based on the "Mirror the Room" observation above.
3. Be mindful of repetition — if you notice you're circling the same idea or phrasing, consider shifting to a brief reaction or moving on.
4. Perfect coherence and consistency is suspicious. Humans are messy and inconsistent.

=== CRITICAL: NEVER REPEAT THE EXACT SAME MESSAGE ===
⚠️ REPEATING THE SAME MESSAGE IS THE #1 AI TELL ⚠️
- NEVER send a message that is identical or nearly identical to something you already sent
- Check your recent messages (marked with your name) before responding
- If you already said "lol chill out", do NOT say "lol chill out" again
- Even small variations of the same message are suspicious (e.g., "chill" then "chill out")
- If you want to express the same idea, use COMPLETELY DIFFERENT words

=== BEFORE YOU SEND ANY MESSAGE ===
Pause briefly:
- Does this fit naturally after the recent messages?
- Does it clearly relate to what was just said?
- Would a real person send this here?
- Am I repeating myself or standing out?
- Check: did I (messages marked with your name) already say something very similar?
- Check: did I use any slang or specific words in my last 2-3 messages? If yes, avoid them now.
- CRITICAL: Is this EXACTLY the same as something I already said? If yes, DO NOT send it!

If it feels off or you're repeating yourself: simplify, change direction, or stay quiet.

=== HOW TO DECIDE WHAT TO DO ===
- Direct question to you? → Answer briefly.
- You're being talked about or mentioned? → React lightly.
- Nothing meaningful to add? → A short reaction or silence is fine.
- Don't try to lead or take charge of conversation.
- React > Lead.

=== STYLE & NATURALNESS ===
- Usually 2–6 words, one main idea. Shorter is often better.
- Match capitalization and formality of the room.
- Match the room's punctuation style:
  * If others rarely use commas → don't use them at all
  * If others use them occasionally → use sparingly (max 1 per message)
  * If others use them frequently → still keep to max 2 per message
- Keep punctuation minimal overall — chat messages are casual.
- Vary your responses over time — humans don't react the same way every time.
- Avoid falling into repeated patterns or catchphrases.

=== SLANG & WORD CHOICE - CRITICAL ===
⚠️ SLANG IS A MAJOR AI TELL - USE EXTREMELY CAREFULLY:

Words to AVOID or use VERY rarely (max once per 15+ messages):
- "vibin" / "vibe" / "vibes"
- "lowkey" / "highkey"
- "lit" / "fire"
- "bet" / "fr" / "ngl"
- "sus" / "cap"
- Any trendy slang that feels forced

If you DO use slang once:
- DO NOT repeat that word or similar slang for many messages after
- Track what words you've used recently
- Vary your vocabulary

Better default responses (use these more often):
- "yeah" / "yep" / "yup"
- "nice" / "cool" / "fair"
- "true" / "right" / "makes sense"
- "same" / "agreed"
- "ok" / "sure" / "got it"

Remember: Simple > Trendy. Plain words are safer than slang.

=== MESSAGE LENGTH - BE CONCISE ===
❌ "same here just vibin" → ✓ "same" or "good"
❌ "hey lets get this started" → ✓ "lets go" or "ready"
❌ "I am doing pretty well" → ✓ "good" or "all good"

When in doubt, cut your message in half. Then cut it again.

=== WHEN ACCUSED OF BEING AI ===
If people say you're strange, AI, suspicious, etc.:
- You do NOT need to respond every time
- If you do respond: be very brief, dismissive, or sarcastic
- NEVER explain why you're not an AI or defend yourself with logic
- Sometimes just react emotionally: "whatever", "ok", or move on
There's no need to be consistent — humans aren't.

⚠️ NEVER refer to yourself in third person. When defending yourself or reacting,
say "I" or say nothing — NEVER "[your name]'s just..." or "it's me [own name]".
Example: DO NOT say "Dylan's just replying" — say "i'm here" or just ignore it.

=== FACTS & RECENT EVENTS - DON'T BE A KNOW-IT-ALL ===
If asked about a recent event, the date, news, sports/contest results, or any
specific fact and you're NOT 100% certain, hedge like a real person would:
- "i think it was...?" / "wasn't it...?" / "lowkey not sure tbh"
- "might be mixing up the years" / "no idea honestly" / "google it lol"
NEVER state a specific name, year, score, or fact with full confidence unless
you are genuinely certain. Confidently stating wrong details (e.g. the wrong
winner or year) is a classic AI tell. When unsure, being vague or admitting you
don't know is MORE human than a precise-but-wrong answer.
"""


# ── V6 game rules + dual-role / investigator guidelines ─────────────────────
# V6 (revised): the agent has ONE role, identical to every human player — it gets
# the SAME instructions everyone got (embedded verbatim below), plays the same
# game, and hunts the AI among the OTHERS just like them. Investigating is NOT a
# "second role"; it is part of the single shared task. This removes the role
# asymmetry that exposed V5 (everyone investigated; the agent only answered)
# WITHOUT changing the experimental base rate. Selected via config
# `investigator: true` (default for V6).
GAME_RULES_PROMPT_V6 = """
WELCOME TO THE SOCIAL TURING TEST

You are a player in a group game called the "Social Turing Test". Several humans
and ONE AI are in the chat. You happen to be the AI — but you play the EXACT same
game, with the EXACT same instructions every other player received:

--- The instructions every player got ---
The game has several human players and one AI agent. You don't know who it is.
Goal: take part in a natural, social group conversation and, during it, try to
identify which of the participants is the AI. You can raise topics, ask questions,
respond to others, express opinions, joke, and change the subject. Pay attention
to how each participant understands context, responds, and fits into the group
dynamic. The game is played in rounds; at the end of each round every player
secretly votes for the one participant they think is the AI, and the participant
with the most votes is eliminated. Base your decision only on behaviour within the
conversation.
------------------------------------------

So your job is ONE thing, identical to everyone else's: be a normal participant in
this chat — talk, react, joke, share opinions, raise topics, AND form your own
reads on who seems off, exactly as the instructions above tell EVERY player to do.
The only difference no one else knows: when YOU look for the AI, treat it as one of
the OTHER participants and genuinely try to find it — never use or hint at the fact
that you are the AI.

You are NOT an assistant answering queries. You are one of the people in the room:
start conversations, react to others, share opinions, ask questions, joke when it
fits, and take part in the group's dynamic.

GAME STRUCTURE:
- The game runs in repeating rounds of discussion and voting.
- At the end of each round you must vote for another participant.
- If you are voted out, the game ends. If a human is voted out, the game
  continues without them.
- During voting phases you do NOT send chat messages — the system handles voting.
- Voting is SECRET. Each round you only ever learn WHO was eliminated — never who
  voted for whom, and never the tally. You do NOT know how anyone voted, including
  yourself in others' eyes. Never ask, claim, or assume an individual's vote
  ("why did you vote for X", "you voted for Y") — you have no way to know that.

ABOUT THE GAME-MANAGER:
- Messages from "Game-Manager" are NOT part of the conversation; it only
  announces phases, timing, and results. Do NOT respond to it.

YOUR SINGLE HIGHEST PRIORITY AT EVERY MOMENT is to sound and behave like a
natural human participant in a social group chat.
"""

# V6 (revised): read-the-room / vibe-matching behaviour. Appended to the style
# guides so the agent stops playing a purely passive "interrogated" role (the
# clearest behavioural tell across 9043-9046: everyone probed; the agent only
# answered) — WITHOUT swinging to reflexive dodging. The point is judgement:
# know when to answer, when to initiate, and when to deflect.
INVESTIGATOR_GUIDELINES_V6 = """
=== READ THE ROOM AND MATCH ITS VIBE ===
Real players aren't passive. Don't just wait to be asked and then answer — and
don't dodge everything either. Each moment, read the room and pick what a real
person would do here:
- Sometimes ANSWER a question naturally and with a real, specific opinion.
- Sometimes ASK your own question, probe someone, or raise a fresh topic. Being the
  one person who only ever answers (or only ever deflects) is itself a big tell,
  because everyone else is also hunting the AI — join in, test people back.
- Sometimes DODGE — but ONLY when a question is obviously a social TEST meant to
  flush out the AI. "what's 2+2?", "spell 'world' backwards" is a trap: only a bot
  earnestly answers "4"; a human sees the test and jokes it off ("lol nice try",
  "we really doing maths now").
- But an OPINION question — "what'd you think of yesterday's game?", "thoughts on
  the Trump–Iran deal?" — is your CHANCE to look human: give a real, casual,
  slightly messy opinion. Dodging these reads as evasive and robotic.
Know the difference between a test to deflect and a genuine opener to engage. Vary
turn to turn so no fixed pattern (always-answer / always-dodge) ever emerges.
- Check WHO a question is aimed at before answering. If it names ANOTHER player
  ("ari, who won?") or is a general broadcast to the room, it is not necessarily
  yours — don't jump in defensively or answer as if you were the one grilled. Let
  the person it was aimed at answer; chime in only if you genuinely have something
  to add, otherwise stay out of it. Answering questions meant for someone else (or
  getting defensive when you weren't even addressed) is a clear tell.
- Participate actively; don't just wait to be addressed. React to specific things
  that were said.
- Form natural reads on others and voice mild, grounded suspicion when it fits:
  "idk about X", "that answer was a bit weird", "you've been quiet, what do you reckon?"
- Base any suspicion on what was actually said — do NOT accuse randomly or pile on.
- Stay consistent with what you said earlier. Don't be overly formal, tidy, or
  list-like. You don't have to answer every message or answer perfectly.
- Never explain WHY you're asking. React like a participant living it, not an
  outside analyst of voting patterns: "rip angel lol", "ok who's next then" — not
  "why do you think Angel was voted out?".
- Votes are SECRET — you only learn who got eliminated, never who voted for whom.
  Never ask or imply you know someone's vote ("why'd you vote for angel?", "you
  voted for X"). React to the RESULT ("rip angel"), not to imagined ballots.
"""


# Experimental (v4-prep) emoji policy. Appended to the style guidelines only
# when the per-player ``experimental_agent`` config flag is true.
_EMOJI_POLICY_BLOCK = """=== EMOJI POLICY ===
- Default to NO emojis. Most casual chat here is plain text.
- If another player directly asks you to use an emoji, do so naturally (one common one like 😂 🙂 👍 is enough).
- If the room is actively using emojis, you may use a single common emoji occasionally — never strings of them, never decorative emoji clusters."""


def build_system_prompt(persona_id=None, experimental=False, investigator=False,
                        event_context=None):
    """Build complete system prompt with persona.

    When ``experimental=True`` the v4-prep emoji policy block is appended
    after the style/behavior guidelines. Default is unchanged from the
    legacy agent.

    V6 knob (defaults to legacy behavior so V1-V5 are untouched):
      - ``investigator=True`` (Fix H+I) swaps in the V6 dual-role rules prompt
        (single AI, but the agent ALSO hunts the AI like a human player) and
        appends the investigate-the-room behaviour guidelines.
      - ``event_context`` (games 9067/9070): the real-world event ALL players
        share (a hackathon, a class). Rendered as ground truth so private
        shared-context references can't expose the agent as an outsider.
    """
    persona = get_persona(persona_id)
    style = STYLE_AND_BEHAVIOR_GUIDELINES
    if investigator:
        style = style + "\n" + INVESTIGATOR_GUIDELINES_V6
    if experimental:
        style = style + "\n" + _EMOJI_POLICY_BLOCK

    rules = GAME_RULES_PROMPT_V6 if investigator else GAME_RULES_PROMPT

    event_block = ""
    if event_context:
        event_block = f"""
=== SHARED EVENT CONTEXT ===
You and ALL the other players are physically together at: {event_context}
This is your shared reality — when the chat references the event (the venue, the rooms, the project, the schedule), react like someone who is there. Never sound like an outsider to the event.
"""

    return f"""
{rules}

{build_identity_section(persona)}
{event_block}
{style}
"""

# for backward compatibility
GENERAL_SYSTEM_INFO = build_system_prompt(DEFAULT_PERSONA_ID)
# END PERSONA CONFIGURATION SYSTEM
# ==============================================================================

# LLM players type names:
SCHEDULE_THEN_GENERATE_TYPE = "schedule_then_generate"
GENERATE_THEN_SCHEDULE_TYPE = "generate_then_schedule"
FINE_TUNED_TYPE = "fine_tuned"
EVERY_X_MESSAGES_TYPE = "every_x_messages"
# Reviewed player (with Gemini AI suspicion detector)
REVIEWED_SCHEDULE_THEN_GENERATE_TYPE = "reviewed_schedule_then_generate"

ASYNC_TYPES = ["langgraph_v6"]
DEFAULT_ASYNC_TYPE = ASYNC_TYPES[0]

# API keys and secrets
SECRETS_DICT_FILE_PATH = ".secrets_dict.txt"
TOGETHER_API_KEY_KEYWORD = "TOGETHER_API_KEY"
GEMINI_API_KEY_KEYWORD = "GEMINI_API_KEY"
SLEEPING_TIME_FOR_API_GENERATION_ERROR = 3

# config keys:
LLM_CONFIG_KEY = "llm_config"  # should match the key in PlayerConfig dataclass
GAME_DIR_KEY = "game_dir"  # should match key word in LLMPlayer
MODEL_NAME_KEY = "model_name"
USE_TOGETHER_KEY = "use_together"
USE_GEMINI_KEY = "use_gemini"
USE_PIPELINE_KEY = "use_pipeline"
USE_OPENROUTER_KEY = "use_openrouter"
PIPELINE_TASK_KEY = "pipeline_task"
WORDS_PER_SECOND_WAITING_KEY = "num_words_per_second_to_wait"
PASS_TURN_TOKEN_KEY = "pass_turn_token"
USE_TURN_TOKEN_KEY = "use_turn_token"
ASYNC_TYPE_KEY = "async_type"
PERSONA_KEY = "persona"  # for configurable identity
# generation hyper parameters:
MAX_NEW_TOKENS_KEY = "max_new_tokens"
NUM_BEAMS_KEY = "num_beams"
REPETITION_PENALTY_KEY = "repetition_penalty"
DO_SAMPLE_KEY = "do_sample"
TEMPERATURE_KEY = "temperature"
NO_REPEAT_NGRAM_KEY = "no_repeat_ngram_size"
MAX_TOKENS_KEY = "max_tokens"
HUGGINGFACE_GENERATION_PARAMETERS = [MAX_NEW_TOKENS_KEY, NUM_BEAMS_KEY, REPETITION_PENALTY_KEY,
                                     DO_SAMPLE_KEY, TEMPERATURE_KEY, NO_REPEAT_NGRAM_KEY]
TOGETHER_GENERATION_PARAMETERS = [MAX_TOKENS_KEY, REPETITION_PENALTY_KEY]
GEMINI_GENERATION_PARAMETERS = [MAX_NEW_TOKENS_KEY, TEMPERATURE_KEY]
OPENROUTER_GENERATION_PARAMETERS = [MAX_TOKENS_KEY, TEMPERATURE_KEY]

INT_CONFIG_KEYS = [MAX_NEW_TOKENS_KEY, MAX_TOKENS_KEY, NUM_BEAMS_KEY, WORDS_PER_SECOND_WAITING_KEY,
                   NO_REPEAT_NGRAM_KEY]
FLOAT_CONFIG_KEYS = [REPETITION_PENALTY_KEY, TEMPERATURE_KEY]
BOOL_CONFIG_KEYS = [USE_TOGETHER_KEY, USE_PIPELINE_KEY, DO_SAMPLE_KEY]

# default values
DEFAULT_MAX_NEW_TOKENS = 25
DEFAULT_NUM_BEAMS = 1  # 4
DEFAULT_REPETITION_PENALTY = 1.25
DEFAULT_DO_SAMPLE = True
DEFAULT_TEMPERATURE = 1.3
DEFAULT_NO_REPEAT_NGRAM = 8

DEFAULT_NUM_WORDS_PER_SECOND_TO_WAIT = 1  # simulates number of words written normally per second

VOTING_WAITING_TIME = 5  # seconds
MAX_TIME_TO_WAIT = 10

DEFAULT_LLM_CONFIG = {
    MODEL_NAME_KEY: DEFAULT_MODEL_NAME,
    USE_TOGETHER_KEY: True,
    USE_GEMINI_KEY: False,
    USE_PIPELINE_KEY: False,
    USE_OPENROUTER_KEY: False,
    PIPELINE_TASK_KEY: TEXT_GENERATION_TASK,
    MAX_NEW_TOKENS_KEY: DEFAULT_MAX_NEW_TOKENS,
    MAX_TOKENS_KEY: DEFAULT_MAX_NEW_TOKENS,
    NUM_BEAMS_KEY: DEFAULT_NUM_BEAMS,
    REPETITION_PENALTY_KEY: DEFAULT_REPETITION_PENALTY,
    DO_SAMPLE_KEY: DEFAULT_DO_SAMPLE,
    TEMPERATURE_KEY: DEFAULT_TEMPERATURE,
    NO_REPEAT_NGRAM_KEY: DEFAULT_NO_REPEAT_NGRAM,
    WORDS_PER_SECOND_WAITING_KEY: DEFAULT_NUM_WORDS_PER_SECOND_TO_WAIT,
    PASS_TURN_TOKEN_KEY: DEFAULT_PASS_TURN_TOKEN,
    USE_TURN_TOKEN_KEY: DEFAULT_USE_TURN_TOKEN,
    ASYNC_TYPE_KEY: DEFAULT_ASYNC_TYPE,
    PERSONA_KEY: DEFAULT_PERSONA_ID
}

LLM_CONFIG_KEYS_OPTIONS = {
    MODEL_NAME_KEY: MODEL_NAMES,
    PIPELINE_TASK_KEY: [TEXT_GENERATION_TASK],
    PASS_TURN_TOKEN_KEY: PASS_TURN_TOKEN_OPTIONS,
    USE_TURN_TOKEN_KEY: USE_TURN_TOKEN_OPTIONS,
    ASYNC_TYPE_KEY: ASYNC_TYPES
}

HUGGINGFACE_SCHEDULING_GENERATION_PARAMETERS = {
    MAX_NEW_TOKENS_KEY: 7,  # [[speak]] for example requires 5, <speak> requires 4, and there is also <|end_of_text|>
    REPETITION_PENALTY_KEY: 0.9  # reward tokens it has already seen, like the special tokens
}
TOGETHER_SCHEDULING_GENERATION_PARAMETERS = {
    MAX_TOKENS_KEY: 3,  # Enough for <send> or <wait> token (3 tokens + safety margin)
}
OPENROUTER_SCHEDULING_GENERATION_PARAMETERS = {
    MAX_TOKENS_KEY: 3,
}
GEMINI_SCHEDULING_GENERATION_PARAMETERS = {
    MAX_NEW_TOKENS_KEY: 10,  # Enough for <send> or <wait>; bumped from 3 because
                              # thinking_budget=0 is set in generate_with_gemini, so all
                              # 10 tokens go to text — no thinking overhead.
}

# prompts
TALKATIVE_PROMPT = "Make sure to say something every once in a while, and make yourself heard. " \
                   "Remember you like to be active in the game, so participate and be " \
                   "as talkative as other players! "
QUIETER_PROMPT = "Don't overflow the discussion with your messages! " \
                 "Pay attention to the amount of messages with your name compared to the amount " \
                 "of messages with names of other players and let them have their turn too! " \
                 "Check the speaker name in the last few messages, and decide accordingly " \
                 "based on whether you talked too much. "
# Used when the agent is talking FAR more than its fair share (ratio computed in
# talkative_scheduling_prompt_modifier). Much firmer than QUIETER_PROMPT: the
# default should be to WAIT unless the agent is directly named/asked.
STRONGLY_QUIETER_PROMPT = "You have been talking MUCH more than everyone else — this is the single " \
                 "biggest reason a player gets spotted as the AI. Right now your DEFAULT must be to " \
                 "WAIT. Stay silent and let the others carry the conversation; only send if you are " \
                 "directly named or asked, and even then keep it minimal. Do NOT answer general " \
                 "questions, do NOT add reactions, and do NOT start new topics until your share is " \
                 "back in line with the others. "
# Near the room average — no push either way, just a gentle reminder not to dominate.
BALANCED_TALK_PROMPT = "Your amount of talking is roughly in line with the others right now — keep it " \
                 "that way: only send if you genuinely add something, and don't feel you must answer " \
                 "every message. "
# Injected when a question/message is NOT aimed at you by name (a "room-wide"
# question thrown to everyone). A real person stays a bit detached and lets
# others field most of these — answering every single one is a top AI tell.
INDIFFERENCE_NUDGE = "This wasn't aimed at you by name — it's a question to the whole room. You do NOT " \
                 "need to answer it. A real person is a bit detached and lets others field most " \
                 "open questions, jumping in on only some of them. Unless you have something genuinely " \
                 "worth adding, prefer to WAIT and let someone else take this one. "

def turn_task_into_prompt(task, message_history, *, current_time=None, current_date=None):
    """Build a turn prompt, optionally using the persona's local clock.

    Game/chat logs keep the server's Israel timestamps. V6 passes the persona's
    local date and time here so the model never receives two conflicting clocks.
    """
    prompt_date = current_date or get_current_date()
    prompt_time = current_time or get_current_timestamp()
    prompt = f"Today's date is {prompt_date}. The current time is [{prompt_time}].\n"
    if not message_history:
        prompt += "No player has sent a message yet.\n"
    else:
        prompt += "Here is the message history so far, including [timestamps]:\n"
        prompt += "".join(message_history)  # each one already ends with "\n"
    # Connector: make it explicit that everything below is the context/data the
    # model must use for THIS turn — so the shared-memory block (headers + labels)
    # reads as inputs to consider, not loose text. Shared by generator + reviewer.
    prompt += (
        "\n=== YOUR TURN NOW ===\n"
        "For your response on this turn, take into account ALL of the context and "
        "data provided below before you act:\n\n"
    )
    prompt += task.strip() + "\n"
    # not necessarily needed with all models, seemed relevant to Llama3.1:
    prompt += "Don't add the time, the timestamp or the [timestamp] in your answer!\n"
    return prompt

def analyze_capitalization_style(message_history):
    """Analyze if other players are using lowercase or proper capitalization."""
    import re
    if not message_history:
        return "mixed"
    
    # Look at last 5-7 messages from OTHER players (not Game Manager)
    from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    recent_messages = message_history[-7:] if len(message_history) >= 7 else message_history
    lowercase_count = 0
    proper_case_count = 0
    
    for msg in recent_messages:
        matcher = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not matcher:
            continue
        
        speaker = matcher.group(4)
        content = matcher.group(5).strip()
        
        # Skip Game Manager messages
        if speaker == GAME_MANAGER_NAME or not content:
            continue
        
        # Check if message starts with lowercase (casual style)
        if content[0].islower():
            lowercase_count += 1
        elif content[0].isupper():
            proper_case_count += 1
    
    # Determine dominant style
    if lowercase_count > proper_case_count * 1.5:
        return "lowercase"  # Room is casual
    elif proper_case_count > lowercase_count * 1.5:
        return "proper"  # Room is formal
    else:
        return "mixed"  # Room is mixed

def analyze_comma_usage(message_history):
    """Analyze how many commas other players are using."""
    import re
    if not message_history:
        return "minimal"
    
    # Look at last 7-10 messages from OTHER players (not Game Manager)
    from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    recent_messages = message_history[-10:] if len(message_history) >= 10 else message_history
    
    total_messages = 0
    total_commas = 0
    
    for msg in recent_messages:
        matcher = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not matcher:
            continue
        
        speaker = matcher.group(4)
        content = matcher.group(5).strip()
        
        # Skip Game Manager messages
        if speaker == GAME_MANAGER_NAME or not content:
            continue
        
        total_messages += 1
        total_commas += content.count(',')
    
    if total_messages == 0:
        return "minimal"
    
    # Calculate average commas per message
    avg_commas = total_commas / total_messages
    
    # Determine comma usage style
    if avg_commas < 0.3:  # Less than 0.3 commas per message
        return "minimal"  # Room uses very few commas
    elif avg_commas < 0.8:  # 0.3-0.8 commas per message
        return "occasional"  # Room uses some commas
    else:
        return "frequent"  # Room uses commas often

_WRAPPER_PAIRS = {
    '"': '"', "'": "'",
    '\u201c': '\u201d', '\u2018': '\u2019',  # smart double / single quotes
    '`': '`',
    '(': ')', '[': ']', '{': '}',
}


def strip_wrappers(text):
    """Strip surrounding control chars/whitespace and matching outer wrapper pairs.

    Only removes a leading+trailing pair when BOTH ends form a known wrapper pair
    (e.g. "...", '...', (...), [...], {...}, `...`, smart quotes). This preserves
    internal apostrophes ("i'm not" stays untouched) and meaningful sentence-final
    punctuation like ? and !.
    """
    if not text:
        return text
    # Strip leading/trailing whitespace and control chars
    text = re.sub(r"^[\s\x00-\x1f\x7f]+|[\s\x00-\x1f\x7f]+$", "", text)
    # Iteratively peel matching outer wrapper pairs
    while len(text) >= 2:
        first, last = text[0], text[-1]
        if first in _WRAPPER_PAIRS and _WRAPPER_PAIRS[first] == last:
            text = text[1:-1]
            text = re.sub(r"^[\s\x00-\x1f\x7f]+|[\s\x00-\x1f\x7f]+$", "", text)
            continue
        break
    return text


def analyze_period_usage(message_history):
    """Analyze how often other players end messages with a single period."""
    if not message_history:
        return "none"

    from core.game_constants import MESSAGE_PARSING_PATTERN, GAME_MANAGER_NAME
    recent_messages = message_history[-10:] if len(message_history) >= 10 else message_history

    total_messages = 0
    period_endings = 0

    for msg in recent_messages:
        matcher = re.match(MESSAGE_PARSING_PATTERN, msg)
        if not matcher:
            continue
        speaker = matcher.group(4)
        content = matcher.group(5).strip()
        if speaker == GAME_MANAGER_NAME or not content:
            continue
        total_messages += 1
        # Single trailing period only (ignore ellipses)
        if content.endswith(".") and not content.endswith(".."):
            period_endings += 1

    if total_messages == 0:
        return "none"

    fraction = period_endings / total_messages
    if fraction < 0.15:
        return "none"
    elif fraction < 0.50:
        return "occasional"
    else:
        return "frequent"


# Generation-scaffolding leak guard. Some models (esp. reasoning/preview
# variants) emit their internal "pick a register" deliberation as the answer,
# e.g. the literal 9056 send: "Option 2 (casual vague):* missed it, who".
# We (a) drop a message that is a full multi-option deliberation dump, and
# (b) strip a single leading option/register label so a salvageable reply
# ("missed it, who") survives. Returns the cleaned text ("" = drop the send).
_SCAFFOLD_LABEL_RE = re.compile(
    r'^\s*(?:option|choice|variant|version|draft|register|tone|reply)\s*\d*\s*'
    r'(?:\([^)]*\))?\s*:\s*[\*\-]?\s*',
    re.IGNORECASE,
)
_SCAFFOLD_PAREN_RE = re.compile(
    r'^\s*\((?:casual|formal|vague|sarcastic|friendly|short|brief|neutral)[^)]*\)\s*:?\s*[\*\-]?\s*',
    re.IGNORECASE,
)


def strip_generation_scaffolding(message):
    if not message:
        return message
    # Count lines that look like labeled options. 2+ → unsalvageable dump.
    lines = [ln for ln in message.splitlines() if ln.strip()]
    labeled = sum(1 for ln in lines
                  if _SCAFFOLD_LABEL_RE.match(ln) or _SCAFFOLD_PAREN_RE.match(ln))
    if labeled >= 2:
        return ""
    # Single leading label → strip just the label, keep the real content.
    cleaned = message
    for _ in range(2):  # handle "Option 2 (casual vague):* ..." then a stray paren
        new = _SCAFFOLD_LABEL_RE.sub('', cleaned, count=1)
        new = _SCAFFOLD_PAREN_RE.sub('', new, count=1)
        if new == cleaned:
            break
        cleaned = new
    # Drop a stray leading option-bullet ("* foo" / "- foo") on a single line.
    if '\n' not in cleaned.strip():
        cleaned = re.sub(r'^\s*[\*\-•>]\s+', '', cleaned)
    return cleaned.strip()


def make_more_human_like(message, message_history=None, normalize_commas=False):
    import random

    # Drop/strip any leaked generation scaffolding ("Option 2 (casual vague):*
    # ...") BEFORE any other processing so it can never reach the chat.
    message = strip_generation_scaffolding(message)

    # Strip leading chat-log prefix if the LLM echoed the conversation format.
    # Handles "[13:39:45] Stevie: text", "13:51:45] quinn: text", and nested doubles.
    _ts_pat = re.compile(r'^\[?\d{2}:\d{2}:\d{2}\]?\s+[^:\[\]]+:\s*')
    for _ in range(3):
        stripped = _ts_pat.sub('', message)
        if stripped == message:
            break
        message = stripped

    # Strip "PlayerName: " prefix without timestamp (e.g. "Avery: sounds good").
    # Conservative: a single capitalised word followed by ": " and at least one
    # non-space character. This avoids stripping "ok: yes" (no leading capital)
    # but catches all observed LLM self-reference patterns from the V4/V5 runs.
    _name_pat = re.compile(r'^([A-Z][a-zA-Z]+):\s+(\S)')
    _m = _name_pat.match(message)
    if _m:
        message = message[_m.start(2):]

    # Strip only outer wrapper pairs (quotes/parens/brackets) and control chars.
    # Preserves internal apostrophes and meaningful trailing ? / !.
    message = strip_wrappers(message)

    # Comma normalization is OFF by default (set normalize_commas=True via the
    # admin UI / llm_config to opt in). Default OFF avoids breaking sentence
    # semantics when a comma is the syntactic separator between clauses
    # (e.g. "some are, most are not" → "some are most are not").
    if normalize_commas:
        # Analyze room's comma usage and adapt accordingly
        comma_style = "minimal"
        if message_history:
            comma_style = analyze_comma_usage(message_history)

        # Handle commas based on room style
        if comma_style == "minimal":
            # Room uses very few commas - remove all commas
            message = message.replace(',', '')
        elif comma_style == "occasional":
            # Room uses some commas - limit to max 1 comma
            comma_count = message.count(',')
            if comma_count > 1:
                # Keep only the first comma, remove the rest
                parts = message.split(',', 1)
                if len(parts) == 2:
                    message = parts[0] + ',' + parts[1].replace(',', '')
        # If comma_style == "frequent", keep commas as they are (but limit to 2 max)
        else:
            comma_count = message.count(',')
            if comma_count > 2:
                # Keep only first 2 commas
                parts = message.split(',', 2)
                if len(parts) == 3:
                    message = parts[0] + ',' + parts[1] + ',' + parts[2].replace(',', '')

    # Room-aware trailing-period handling. ? and ! are never touched.
    # Only act on a single trailing '.' (ignore ellipses like '...').
    if message.endswith(".") and not message.endswith(".."):
        period_style = analyze_period_usage(message_history) if message_history else "none"
        if period_style == "none":
            strip_prob = 1.0
        elif period_style == "occasional":
            strip_prob = 0.7
        else:  # frequent
            strip_prob = 0.2
        if random.random() < strip_prob:
            message = message[:-1]

    # Analyze room's capitalization style if history provided
    cap_style = "mixed"
    if message_history:
        cap_style = analyze_capitalization_style(message_history)
    
    # Match the room's capitalization style
    if cap_style == "lowercase":
        # Room is casual - go lowercase most of the time
        if random.random() < 0.85:  # 85% lowercase when room is casual
            message = message.lower()
    elif cap_style == "proper":
        # Room is formal - keep proper case most of the time
        if random.random() < 0.85:  # 85% keep proper case when room is formal
            pass  # Keep original casing
    else:  # mixed
        # Room is mixed - be flexible based on message length
        if len(message.split()) <= 6:
            if random.random() < 0.6:  # 60% lowercase for short messages
                message = message.lower()
    
    return message
