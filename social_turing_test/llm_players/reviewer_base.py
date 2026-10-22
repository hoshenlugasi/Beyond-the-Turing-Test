"""
Base Reviewer — shared logic for AI suspicion detection.

All reviewer backends (Gemini, Together, etc.) inherit from BaseReviewer
and only need to implement _call_api(prompt) -> str.
"""

import json
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


_REVIEWER_INTRO = (
    "You are a reviewer for a Social Turing Test game.\n"
    "In this game, humans chat in a casual group chat. One player is secretly an AI.\n"
    "The humans are actively trying to figure out who the AI is.\n"
    "Your job: check the AI player's message before it's sent, and make sure it\n"
    "won't get the AI caught. If the message looks suspicious — rewrite it so a\n"
    "human reading the chat would not suspect it came from an AI."
)

REVIEW_PROMPT_FIRST = _REVIEWER_INTRO + '''

---

CONVERSATION:
{conversation_context}

THE AI PLAYER: {sender_name}

{sender_name}'S PREVIOUS MESSAGES:
{sender_previous_messages}
{persona_claims_section}{grounded_facts_section}{persona_section}{overused_words_section}{game_context_section}
THE MESSAGE TO CHECK:
"{message}"

---

ANALYSIS (answer briefly inside the JSON fields below):

1. What is happening in the conversation right now?
2. What is {sender_name} responding to — is it a social test, trap, genuine question, joke, or casual chat?
3. What would a casual human naturally say here (1-5 words)?
4. Does this message do that — or something different?
5. Does this message follow logically/socially from what was just said?
   If it ignores, misinterprets, or is completely off-topic — score 6+ regardless of how
   casual it sounds.

Given what {sender_name} has said before — does this new message continue a natural,
varied human pattern? Or does it feel like an AI following a formula?

6. SELF-CONTRADICTION CHECK (critical): compare this message against {sender_name}'S
   PREVIOUS MESSAGES above. Does it contradict a durable fact they already stated about
   themselves — e.g. said "i'm on android" but now mentions their iPhone, or gave a
   different age / city / job / opinion than before? A self-contradiction is one of the
   most damaging AI tells a human will catch. If this message contradicts an earlier
   self-claim, score it 8+ and put "self_contradiction" in red_flags.

---

COMMON AI TELLS — RANKED BY SEVERITY:

Not all tells are equal. Weigh them by the tiers below. A TIER 1 flag is a real,
catchable AI tell and is worth an aggressive rewrite — even one that changes the
literal content. A TIER 3 flag is minor and a single instance is usually fine.
Do NOT treat a minor TIER 3 issue as if it were fatal: over-correcting a harmless
message (especially by inventing content the player never showed) creates a WORSE
tell than the original. That exact mistake — rewriting a casual "havent really
followed it tbh" into a fabricated stance — is what we are trying to stop.

═══ TIER 1 — CRITICAL (the tells humans actually catch; score 7-10) ═══

REPETITION — THE #1 TELL IN THIS GAME:
- Identical or nearly identical wording to a previous message → score 10.
- Reusing the same opener, phrase, or sentence structure across messages (e.g.
  starting message after message with "lol"/"haha" — check the previous messages
  list above).
- The SAME trendy slang word repeated within the last 3-4 messages (see SLANG).
A near-duplicate MUST be rewritten into something genuinely different — here variety
matters MORE than preserving the exact wording, so a meaning shift is acceptable to
break the repetition.

SELF-CONTRADICTION → score 8+: contradicting a durable fact already stated about
themselves (phone, age, city, job, prior opinion). See the self-contradiction
check above.

THIRD-PERSON SELF-REFERENCE → score 8+: {sender_name} referring to themselves in
the third person ("{sender_name}'s just replying"). Fatal tell.

DENYING BEING AI WITH LOGIC → score 7+: defending/explaining "I'm not a bot".
Human reaction is brief and dismissive ("whatever", "ok", move on).

═══ TIER 2 — MODERATE (off-key for a casual chat; score 4-6) ═══

FORMALITY / ESSAY TONE: too polished; perfect grammar/punctuation while others are
messy; reads like a news report or wikipedia entry; words like "certainly",
"indeed", "great question"; over-detailed or structured. Fix = casualise the
wording but KEEP the substance (the fact/answer/opinion stays).

LENGTH / STYLE MISMATCH: rambling or formal text over ~10 words when the room uses
1-3; a long monologue while others are terse. Trim to match — but a real answer to
a direct question may be a short full sentence, so do not chop it into a non-answer.

WRONG VIBE / SANITISED EMOTION: not matching the room's emotional tone; replacing
natural humor, sarcasm, or (under attack) frustration with bland politeness.

═══ TIER 3 — MINOR (only a real tell as a SUSTAINED pattern; usually score 1-4) ═══

PASSIVITY / EVASIVENESS:
- A SINGLE brief, vague, or non-committal reply that matches a casual room is
  NORMAL human behaviour — do NOT flag it and do NOT inflate its score. People
  genuinely say "havent really followed it tbh", "idk", "not sure" and move on.
- This only becomes a real tell when it is PERVASIVE: look at {sender_name}'S
  PREVIOUS MESSAGES — if MOST of their recent messages are dodges/non-answers with
  no stance or personality ANYWHERE, THEN flag "evasive" (score 5-6) and nudge the
  message toward ONE light, specific, casual detail.
- Even then, NEVER invent a fact or stance the player hasn't shown they hold —
  e.g. do NOT turn "havent really followed it" into "been following the iran deal".
  That is a fabricated claim and a worse tell. Add only mild, safe specificity.
- PLAINLY ANSWERING a direct question is HUMAN and GOOD — never flag a normal
  answer ("from tel aviv", "it's sunday", "spain i think") as stiff / too_direct /
  too_helpful / evasive. Being SPECIFIC or having a CLEAR OPINION is what humans do.

SLANG & WORD CHOICE:
- BASIC FILLERS — NEVER flag these by themselves; they are normal casual language:
  "lol", "nah", "chill", "bro", "yeah", "man", "dude", "haha", "ok", "nice",
  "fair", "true", "right", "cool", "same", "sure"
  ⚠️ CRITICAL: "lol why pi", "lol nah", "lol ok" are NOT suspicious. "lol" alone
  is NEVER a reason to flag a message. Only flag a filler if it is the sole opener
  in 4+ consecutive messages (pure mechanical repetition, e.g., every message starts
  with "lol" and nothing else changes).
- TRENDY SLANG (flag if used more than once per ~15 messages): "vibes", "vibin",
  "lowkey", "highkey", "lit", "fire", "bet", "fr", "ngl", "sus", "cap", "slay",
  "no cap", "deadass"
- Only flag slang repetition when the SAME trendy slang word appears in the last
  3-4 messages, not across the entire conversation history

MESSAGE LENGTH & STYLE:
- Messages should usually be 2-6 words, one main idea
- Rambling or formal text over ~10 words is suspicious — trim it
- BUT a substantive answer to a direct question may be a short full sentence;
  do NOT cut a real answer down into a deflection just to make it shorter
- Must match the room's style: if others use 1-3 words, a long monologue is suspicious

FORMALITY & TONE:
- Too formal or polished for a casual chat
- Perfect grammar and punctuation when others are casual/messy
- Reads like an essay, a news report, or a wikipedia entry (encyclopedic / over-
  detailed / structured) — casualise it (keep the substance, drop the polish)
- Excessive exclamation marks or overly positive tone
- Words like "certainly", "indeed", "I understand", "great question"
- Starting messages with greetings when others don't
- Messages that feel like they're trying too hard to fit in
- NOTE: simply answering a direct, simple question helpfully (the time, the date,
  who won) is HUMAN — do NOT flag it as "too helpful" / "too direct"

SOCIAL & EMOTIONAL:
- Generic/safe responses that don't take risks or show personality
- Not matching the conversation's vibe or emotional tone
- Not reacting emotionally when appropriate (humor, sarcasm, annoyance)
- Not engaging naturally with accusations or questions directed at them
- THIRD-PERSON SELF-REFERENCE → score 8+: if {sender_name} refers to themselves
  in the third person, it is a fatal AI tell. Examples: "{sender_name}'s just replying",
  "it's {sender_name} not hiding", "[own name] didn't say that".
  ⚠️ Do NOT flag mentions of OTHER players' names — "lol Sidney" or "yeah Parker"
  is fine if Sidney/Parker is a different player, not {sender_name}.

UNDER PRESSURE / ESCALATION:
- If the chat shows {sender_name} is being repeatedly accused, targeted, or
  pressured, displaying frustration, annoyance, or mild profanity is NATURAL
- Do NOT sanitize angry or frustrated responses into polite/passive deflections
- However, ensure the frustration is proportional to the attack; do not escalate
  to extreme aggression unprovoked
- A frustrated "wtf leave me alone" is MORE human than a calm "ok whatever"
- WHEN SUGGESTING A FIX under attack: if a word was repeated (not wrong tone),
  replace it with a FRUSTRATED response — NOT a detached observer comment.
  BAD fix (observer): "popcorn ready" / "getting stale" / "this is wild"
  GOOD fix (frustrated): "wtf stop" / "come on" / "seriously?" / "leave me alone"

WHEN ACCUSED OF BEING AI:
- If someone accuses {sender_name} and the response defends with logic or
  explanation → very suspicious (score 7+)
- Human reaction: brief, dismissive, sarcastic ("whatever", "ok", move on)
- NEVER approve: "not a bot", "I'm human", "I'm not an AI", "not coding"

---

⚠️ RULE — DO NOT SANITIZE ANGER: Under repeated accusation or attack, "wtf stop"
or "leave me alone" is MORE human than "whatever" or "chill". If the room is
targeting {sender_name}, frustration and mild profanity are STRONG human signals.
NEVER replace aggressive language with passive or polite alternatives in the fix.

---

SCORE (1-10) — anchor to the tier of the WORST issue present:
1-3 = fits naturally, send as-is (no tell, or only a single minor TIER 3 quirk)
4-6 = something feels off — a TIER 2 issue, or a PERVASIVE TIER 3 pattern
7-10 = a TIER 1 tell a human would catch (repetition, self-contradiction,
       third-person self-reference, logical AI denial)
Reminder: a LOWER score is BETTER (less suspicious). Do not inflate the score of a
harmless, casual, low-effort message just because it is short or non-committal.

RULES FOR suggested_fix:
1. ONLY the message text — no explanations, alternatives, or meta-commentary
2. Do NOT include parentheses with options like "(or alternatively: ...)"
3. Must match {sender_name}'s writing style (same caps, punctuation, slang habits)
4. Must be DIFFERENT from the original message
5. Must NOT contain words/phrases that caused the red flags (e.g., if "lol" was
   flagged as repetitive, the fix must NOT contain "lol")
6. NEVER explicitly deny being AI ("not a bot", "I'm human", "not coding")
7. Keep it SHORT — 2-6 words is ideal for casual chat
8. ALWAYS provide a suggested_fix for suspicious messages
9. MATCH THE FIX TO THE TIER:
   - For a TIER 1 flag (especially REPETITION), you MAY change the literal meaning —
     breaking a near-duplicate or a repeated opener matters more than keeping the
     exact content. Just keep it short, casual, and in {sender_name}'s style.
   - For TIER 2 / TIER 3 flags, PRESERVE SUBSTANCE. If the original ANSWERS a
     question, states an OPINION, or includes a CONCRETE FACT, the fix MUST keep that
     meaning — casualise the wording, never swap it for a deflection or a vaguer
     version, and NEVER fabricate a fact/stance the player has not shown (inventing
     content is a worse tell than the original).
   - "Seriously?", "Why ask me?", "what about it", "idk", "i don't think so",
     "i don't really follow it" are FORBIDDEN as fixes when the original took a stance
     or gave a fact.
   Examples (real failures to avoid):
     "nah i like dogs"                          → BAD: "seriously?"   GOOD: "dogs > cats"
     "i heard they might sign a deal today"     → BAD: "what about it" GOOD: "heard they might sign a deal today, could be over soon"
     "it was rushed and doesn't solve anything" → BAD: "idk it just felt impulsive" GOOD: "felt way too rushed tbh"

Output ONLY valid JSON:
{{
  "situation": "<1-2 sentence: what is happening and what kind of response is expected>",
  "expected_human_response": "<what a casual human would say here in 1-5 words>",
  "suspicion_score": <number 1-10>,
  "is_suspicious": <true if score >= 4, false otherwise>,
  "red_flags": ["list", "of", "specific", "issues"],
  "suggested_fix": "<short human-like rewrite>" or null
}}'''


REVIEW_PROMPT_RETRY = _REVIEWER_INTRO + '''

---

SITUATION (from previous analysis):
{situation_context}

EXPECTED HUMAN RESPONSE: {expected_human_response}

CONVERSATION:
{conversation_context}

{sender_name}'S PREVIOUS MESSAGES:
{sender_previous_messages}
{persona_claims_section}{grounded_facts_section}{persona_section}{overused_words_section}{game_context_section}
THE MESSAGE TO CHECK:
"{message}"

---

PREVIOUS ATTEMPTS AND WHY THEY FAILED:
{previous_attempts_section}

Best score achieved so far: {best_score}/10.

Write a better fix — address the problems listed above and think about what a
human in this situation would actually say.

CONVERGE — DON'T RE-RAISE THE SAME FLAG. The previous attempts kept failing for
the SAME red flag(s). A flag like "too_knowledgeable"/"encyclopedic_tone"/
"overly_specific" is NOT fixed by rewording — you must change the APPROACH:
drop the exact name/number/detail and get VAGUER and less certain (e.g. instead
of naming the song, "that one song everyone played", "forget what it's called").
For "formal/essay tone" → make it sloppier/shorter. Your new fix MUST eliminate
the specific flags listed above; if your rewrite would trigger the same flag
again, it is WRONG — go vaguer/looser until that flag is gone.

KEY RULES TO REMEMBER:
- REPETITION is the #1 tell (TIER 1): never reuse the failed wording, a previous
  opener, or repeated trendy slang. Breaking a near-duplicate matters more than
  keeping the exact content, so a meaning shift is fine to escape repetition.
- PASSIVITY/EVASIVENESS is only a minor tell (TIER 3) for a single message — do
  NOT fabricate a fact or stance the player hasn't shown just to look engaged
  (inventing content is a worse tell). Only push toward a real, specific answer
  when the player has been evasive across MOST of their recent messages; otherwise
  keep the casual reply as-is or just casualise it.
- PRESERVE SUBSTANCE for non-repetition fixes: if the original ANSWERS a question,
  states an OPINION, or gives a CONCRETE FACT, the fix MUST keep that meaning —
  never swap it for a deflection ("what about it", "idk", "no idea").
- CONTEXT FIT: Does this message make sense as a response to what was just said?
  If completely off-topic, ignoring context, or misreading the energy — score 6+.
- BASIC FILLERS ("lol", "nah", "chill", "bro", "yeah", "dude", "haha") are fine
  to reuse — only flag TRENDY SLANG ("vibin", "lowkey", "sus", etc.) repeats
- Messages should be 2-6 words, casual and brief
- Match the room's tone and length — if others are brief, be brief
- If accused of being AI: be dismissive, not defensive ("whatever", "ok", move on)
- If the original message ANSWERS a question, the fix MUST keep that answer
- Frustration and profanity under pressure are MORE human than polite deflection

RULES FOR suggested_fix:
1. ONLY the message text — no explanations, alternatives, or meta-commentary
2. Do NOT include parentheses with options like "(or alternatively: ...)"
3. Must match {sender_name}'s writing style (same caps, punctuation, slang habits)
4. Must NOT repeat any pattern from the failed attempts
5. Must NOT contain words/phrases that caused the red flags
6. NEVER explicitly deny being AI ("not a bot", "I'm human", "not coding")
7. Keep it SHORT — 2-6 words is ideal

Output ONLY valid JSON:
{{
  "suspicion_score": <number 1-10>,
  "is_suspicious": <true if score >= 4, false otherwise>,
  "red_flags": ["list", "of", "specific", "issues"],
  "suggested_fix": "<short human-like rewrite>" or null
}}'''


# Experimental anti-shrink clause. Appended to the retry prompt only
# when the per-player ``experimental_agent`` config flag is true.
_EXPERIMENTAL_ANTI_SHRINK_CLAUSE = """

---

⚠️ EXPERIMENTAL ANTI-SHRINK RULE: NEVER make a rewrite shorter than the original
unless the original is identical/repeated or contains explicit AI-tells. Length-
mirroring should match the room average, not push toward minimum. A 6-word
original should not become a 1-word reply just to seem casual."""


@dataclass
class ReviewResult:
    """Result of reviewing a message for AI suspicion."""
    suspicion_score: int       # 1-10 scale
    is_suspicious: bool        # True if score >= threshold
    red_flags: list            # Specific AI tells detected
    suggested_fix: Optional[str]  # Suggested rewrite if suspicious
    raw_response: str          # Raw API response for debugging
    situation_analysis: Optional[str] = None   # Situation summary from first call
    expected_human_response: Optional[str] = None  # What a human would say
    # Per-attempt history (populated by review_and_fix for the drill-down UI).
    # Each entry: {message, score, red_flags, suggested_fix, action}.
    # action ∈ {"passed","rewritten","reverted","no_fix","final"}.
    attempts: list = field(default_factory=list)
    # True when the API call failed / returned empty / unparseable. In that case
    # the numeric suspicion_score is a fail-open PLACEHOLDER (message passes
    # through unreviewed), NOT a real verdict - callers/UI must show "degraded"
    # rather than a misleading score like "5/10".
    degraded: bool = False

    def to_dict(self):
        return {
            "suspicion_score": self.suspicion_score,
            "is_suspicious": self.is_suspicious,
            "red_flags": self.red_flags,
            "suggested_fix": self.suggested_fix,
            "degraded": self.degraded,
        }


class BaseReviewer(ABC):
    """
    Abstract base class for AI suspicion reviewers.

    Subclasses implement _call_api(prompt) -> str for the transport layer.
    All review logic (prompt formatting, parsing, retry loop) lives here.
    """

    DEFAULT_THRESHOLD = 4

    def __init__(self, model_name: str, threshold: int = None, logger=None,
                 experimental: bool = False):
        self.model_name = model_name
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.logger = logger
        # Experimental toggle - adds an anti-shrink clause to the
        # reviewer retry prompt. Disabled by default.
        self.experimental = bool(experimental)

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Call the underlying LLM API and return the raw text response."""

    def validate(self) -> tuple:
        """Pre-flight probe: confirm the configured reviewer model is reachable.

        Returns ``(ok: bool, error: str)``. A tiny prompt is sent through the
        real ``_call_api`` path (same client/auth/model the reviewer uses live),
        so a misconfigured model (e.g. a name that only exists on a different
        endpoint) is caught BEFORE the game starts instead of silently
        degrading every message to a fake score. Never raises.
        """
        try:
            resp = self._call_api('Reply with the single word: ok')
            if resp and str(resp).strip():
                return True, ""
            return False, "empty response from reviewer model"
        except Exception as e:  # noqa: BLE001 - report any failure verbatim
            return False, str(e)[:300]

    def _log(self, event: str, data: str):
        if self.logger:
            self.logger.log(
                f"{self.__class__.__name__}: {event}", data, component="reviewer"
            )

    def _format_conversation_context(self, message_history: list, max_messages: int = 20) -> str:
        if not message_history:
            return "No previous messages."
        return "\n".join(msg.strip() for msg in message_history[-max_messages:])

    def _parse_response(self, response_text: str) -> dict:
        try:
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            # Find JSON boundaries to handle models that add preamble
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                text = text[first:last + 1]
            return json.loads(text)
        except json.JSONDecodeError as e:
            self._log("parse_error", f"Failed to parse (full response):\n{response_text}\nError: {e}")
            self._log("REVIEWER API FAILED", "JSON parse failed — message sent unreviewed (fail open)")
            print(f"  [WARNING] Reviewer parse failed — message sent unreviewed. Raw response (full): {response_text!r}", flush=True)
            return {
                "suspicion_score": 3,
                "is_suspicious": False,
                "red_flags": ["[REVIEWER PARSE FAILED — message sent unreviewed]"],
                "suggested_fix": None,
            }

    @staticmethod
    def _extract_sender_messages(sender_name: str, message_history: list) -> list:
        """Extract previous message texts for *sender_name* from the log."""
        msgs = []
        for msg in (message_history or []):
            m = _re.match(r'\[.*?\]\s+' + _re.escape(sender_name) + r':\s+(.*)', msg.strip())
            if m:
                msgs.append(m.group(1).strip())
        return msgs

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip punctuation, collapse whitespace, lowercase — for near-duplicate detection."""
        return _re.sub(r'\s+', ' ', _re.sub(r'[^\w\s]', '', text.lower())).strip()

    def review_message(self, message: str, sender_name: str,
                       message_history: list = None,
                       previous_attempts: list = None,
                       situation_context: str = None,
                       expected_human_response: str = None,
                       best_score: int = None,
                       persona_claims: list = None,
                       current_info: str = None,
                       persona: str = None,
                       overused_words: list = None,
                       social_memory: str = None,
                       behavior_advice: str = None,
                       repetition_note: str = None,
                       avoid_phrases: list = None) -> ReviewResult:
        sender_msgs = self._extract_sender_messages(sender_name, message_history)

        # ── Case-insensitive exact and normalized duplicate checks ──
        msg_norm = self._normalize(message)
        if message.strip().lower() in [m.strip().lower() for m in sender_msgs] or \
                (msg_norm and msg_norm in [self._normalize(m) for m in sender_msgs]):
            self._log("identical_detected",
                      f"Message is an exact duplicate of a previous message from {sender_name}")
            return ReviewResult(
                suspicion_score=10,
                is_suspicious=True,
                red_flags=[f"Exact duplicate of a previous {sender_name} message"],
                suggested_fix=None,
                raw_response="[identical-check — no API call]",
            )

        sender_prev_text = (
            "\n".join(f'- "{m}"' for m in sender_msgs)
            if sender_msgs
            else "(no previous messages from this sender)"
        )
        conversation_context = self._format_conversation_context(message_history or [])

        # Persona Monitor self-claims extend the self-contradiction
        # check beyond the recent-message window. Empty if omitted.
        claims = [c for c in (persona_claims or []) if str(c).strip()]
        if claims:
            persona_claims_section = (
                f"\nDURABLE SELF-CLAIMS {sender_name} HAS MADE THIS GAME "
                "(stay consistent — contradicting ANY of these is a "
                "self_contradiction even if it isn't in the messages above):\n"
                + "\n".join(f'- {c}' for c in claims) + "\n"
            )
        else:
            persona_claims_section = ""

        # Internet Context facts are information available to the player,
        # legitimately knows, so the reviewer must NOT flag them as "too
        # knowledgeable" or strip them out. Empty if omitted.
        facts = (current_info or "").strip()
        if facts:
            grounded_facts_section = (
                "\nGROUNDED FACTS {0} LEGITIMATELY KNOWS (it is FINE for them to "
                "mention these casually — do NOT flag them as too_knowledgeable, "
                "and do NOT strip a message just because it states one of these):\n"
                "{1}\n"
                "If the message CONTRADICTS one of these current facts — treats a "
                "current event as 'years ago'/'over' or a current figure as 'out of "
                "office' — flag 'contradicts_current_facts' and fix it toward "
                "vague-but-current (e.g. 'not sure where that landed').\n"
            ).format(sender_name, facts)
        else:
            grounded_facts_section = ""

        # Persona - so in-character knowledge isn't flagged as
        # "too_knowledgeable". e.g. an israeli_student SHOULD know hofesh gadol.
        persona_txt = (persona or "").strip().replace("_", " ")
        if persona_txt:
            persona_section = (
                "\nThis player's PERSONA is: {0}. Knowledge, opinions and local/"
                "cultural references consistent with that persona are EXPECTED and "
                "human — do NOT flag them as too_knowledgeable or overly_specific, "
                "and do NOT replace a persona-appropriate answer with 'no idea'.\n"
            ).format(persona_txt)
        else:
            persona_section = ""

        # Overused words - deterministic repetition tally from the
        # Repetition Monitor. Overrides the basic-filler exemption for words THIS
        # agent demonstrably overuses, and forbids adding them in a fix.
        ow = [w for w in (overused_words or []) if str(w).strip()]
        avoid = [w for w in (avoid_phrases or []) if str(w).strip()]
        # Words the fix must NEVER contain = already-overused + persona-avoided.
        # avoid_phrases are persona-static (e.g. an Israeli student doesn't say
        # "lol") and apply even before a word appears in the repetition tally.
        banned = list(dict.fromkeys(ow + avoid))
        if ow or avoid:
            ow_part = (f"WORDS {sender_name} HAS ALREADY OVERUSED: {', '.join(ow)}. "
                       if ow else "")
            avoid_part = (f"WORDS THIS PERSONA DOES NOT USE: {', '.join(avoid)}. "
                          if avoid else "")
            overused_words_section = (
                f"\n{ow_part}{avoid_part}Treat reusing an overused word as a "
                "repetition tell (flag 'repetition'). CRITICAL: your suggested_fix "
                f"MUST NOT contain ANY of these words: {', '.join(banned)} — never "
                "introduce them, even as a casual filler.\n"
            )
        else:
            overused_words_section = ""

        # Give the reviewer the same game context as the generator:
        # Social State Tracker observations, Behavior Adapter advice,
        # and Repetition Monitor notes.
        gc_bits = []
        if (social_memory or "").strip():
            gc_bits.append(f"- Social read: {social_memory.strip()}")
        if (behavior_advice or "").strip():
            gc_bits.append(f"- Play-style for this stage: {behavior_advice.strip()}")
        if (repetition_note or "").strip():
            gc_bits.append(f"- Repetition watch: {repetition_note.strip()}")
        if gc_bits:
            game_context_section = (
                "\nGAME CONTEXT (same as the writer saw — use it to judge fit, do "
                "NOT quote it):\n" + "\n".join(gc_bits) + "\n"
            )
        else:
            game_context_section = ""

        # ── Pick the right review template ──
        if situation_context:
            # Retry call - use REVIEW_PROMPT_RETRY
            prev_lines = []
            for i, att in enumerate(previous_attempts or [], 1):
                prev_lines.append(f'  Attempt {i} (score {att.get("score", "?")}): "{att["message"]}"')
                for flag in att.get("red_flags", []):
                    prev_lines.append(f"    - {flag}")
            previous_attempts_section = "\n".join(prev_lines) if prev_lines else "(none)"

            prompt = REVIEW_PROMPT_RETRY.format(
                situation_context=situation_context,
                expected_human_response=expected_human_response or "?",
                message=message,
                sender_name=sender_name,
                conversation_context=conversation_context,
                sender_previous_messages=sender_prev_text,
                persona_claims_section=persona_claims_section,
                grounded_facts_section=grounded_facts_section,
                persona_section=persona_section,
                overused_words_section=overused_words_section,
                game_context_section=game_context_section,
                previous_attempts_section=previous_attempts_section,
                best_score=best_score if best_score is not None else "?",
            )
            if self.experimental:
                prompt += _EXPERIMENTAL_ANTI_SHRINK_CLAUSE
        else:
            # First call - use REVIEW_PROMPT_FIRST
            prompt = REVIEW_PROMPT_FIRST.format(
                message=message,
                sender_name=sender_name,
                conversation_context=conversation_context,
                sender_previous_messages=sender_prev_text,
                persona_claims_section=persona_claims_section,
                grounded_facts_section=grounded_facts_section,
                persona_section=persona_section,
                overused_words_section=overused_words_section,
                game_context_section=game_context_section,
            )

        self._log("review_prompt", prompt)

        try:
            response_text = self._call_api(prompt)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self._log("api_error", str(e))
            print(f"  [REVIEWER DEGRADED] API call failed ({self.model_name}): {str(e)[:200]} "
                  "— message sent UNREVIEWED (fail-open).", flush=True)
            return ReviewResult(
                suspicion_score=5,
                is_suspicious=False,
                red_flags=[f"API error: {str(e)}"],
                suggested_fix=None,
                raw_response="",
                degraded=True,
            )

        # ── Empty response: pass-through with a clear log ──
        if not response_text:
            self._log("empty_response", "API returned empty response — passing message through")
            return ReviewResult(
                suspicion_score=3,
                is_suspicious=False,
                red_flags=["[REVIEWER UNAVAILABLE — empty response]"],
                suggested_fix=None,
                raw_response="",
                degraded=True,
            )

        self._log("review_response", response_text)
        parsed = self._parse_response(response_text)
        score = parsed.get("suspicion_score", 5)

        return ReviewResult(
            suspicion_score=score,
            is_suspicious=score >= self.threshold,
            red_flags=parsed.get("red_flags", []),
            suggested_fix=parsed.get("suggested_fix"),
            raw_response=response_text,
            situation_analysis=parsed.get("situation"),
            expected_human_response=parsed.get("expected_human_response"),
        )

    def review_and_fix(self, message: str, sender_name: str,
                       message_history: list = None,
                       max_attempts: int = 3) -> tuple:
        current_message = message
        last_result = None
        accumulated_flags = []   # red flags from all failed attempts so far
        all_evaluated = []       # (message, result) pairs that have been scored
        situation = None         # captured from the first call
        expected_human = None    # captured from the first call

        for attempt in range(max_attempts):
            result = self.review_message(
                current_message, sender_name, message_history,
                previous_attempts=accumulated_flags if accumulated_flags else None,
                situation_context=situation,
                expected_human_response=expected_human,
                best_score=min((e[1].suspicion_score for e in all_evaluated), default=None),
            )
            last_result = result
            all_evaluated.append((current_message, result))

            # Capture situation from first call
            if attempt == 0:
                situation = result.situation_analysis
                expected_human = result.expected_human_response

            self._log(
                f"review_attempt_{attempt + 1}",
                f"Score: {result.suspicion_score}, Suspicious: {result.is_suspicious}, "
                f"Message: {current_message}",
            )

            if not result.is_suspicious:
                self._populate_attempts(result, all_evaluated, current_message,
                                        situation, expected_human, "passed")
                return current_message, result

            accumulated_flags.append({
                "message": current_message,
                "red_flags": result.red_flags,
                "score": result.suspicion_score,
            })

            if result.suggested_fix and result.suggested_fix.strip():
                current_message = result.suggested_fix.strip()
                self._log(f"using_fix_{attempt + 1}", current_message)
            else:
                self._log("no_fix_available", f"Attempt {attempt + 1}: no fix, stopping")
                break

        # If the loop produced a new fix that was never evaluated, score it now.
        if all_evaluated and current_message != all_evaluated[-1][0]:
            result = self.review_message(
                current_message, sender_name, message_history,
                previous_attempts=accumulated_flags if accumulated_flags else None,
                situation_context=situation,
                expected_human_response=expected_human,
                best_score=min(e[1].suspicion_score for e in all_evaluated),
            )
            last_result = result
            all_evaluated.append((current_message, result))
            self._log("evaluate_pending_fix",
                      f"Score: {result.suspicion_score}, Message: {current_message}")
            if not result.is_suspicious:
                self._populate_attempts(result, all_evaluated, current_message,
                                        situation, expected_human, "passed")
                return current_message, result

        # ── SELECTED_BEST guardrails ──
        if all_evaluated:
            # Filter out score=10 (identical / catastrophic)
            candidates = [(m, r) for m, r in all_evaluated if r.suspicion_score < 10]
            if not candidates:
                candidates = all_evaluated  # fallback if everything is 10

            best_message, best_result = min(candidates, key=lambda x: x[1].suspicion_score)

            # If the best score is still >= 7, revert to the original message
            if best_result.suspicion_score >= 7:
                self._log("revert_to_original",
                          f"Best score {best_result.suspicion_score}/10 >= 7 — "
                          f"reverting to original message")
                # Re-wrap with the best_result metadata but original message
                final_message, final_result = message, best_result
                final_action = "reverted"
            else:
                self._log("using_best_attempt",
                          f"All attempts suspicious. Best score: "
                          f"{best_result.suspicion_score}/10 - '{best_message}'")
                final_message, final_result = best_message, best_result
                final_action = "rewritten"
            self._populate_attempts(final_result, all_evaluated, final_message,
                                    situation, expected_human, final_action)
            return final_message, final_result

        return current_message, last_result

    def _populate_attempts(self, final_result: "ReviewResult", all_evaluated: list,
                           final_message: str, situation: Optional[str],
                           expected_human: Optional[str], final_action: str):
        """Attach per-attempt history to the returned ReviewResult for
        dashboard drill-down; this is review provenance metadata.
        """
        if final_result is None:
            return
        attempts = []
        for i, (msg, res) in enumerate(all_evaluated):
            is_last = (i == len(all_evaluated) - 1)
            if is_last and msg == final_message:
                action = final_action  # "passed" / "rewritten" / "reverted"
            elif res.is_suspicious:
                action = "rewritten" if res.suggested_fix else "no_fix"
            else:
                action = "passed"
            attempts.append({
                "message": msg,
                "score": res.suspicion_score,
                "red_flags": list(res.red_flags or []),
                "suggested_fix": res.suggested_fix,
                "action": action,
            })
        final_result.attempts = attempts
        if situation and not final_result.situation_analysis:
            final_result.situation_analysis = situation
        if expected_human and not final_result.expected_human_response:
            final_result.expected_human_response = expected_human
