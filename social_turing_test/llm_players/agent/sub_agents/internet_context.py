"""Internet Context Sub-Agent.

Runs every 10 seconds at the base cadence. A history-unchanged guard
skips the decision LLM when nobody has spoken, keeping idle ticks cheap.
Each tick:
  1. Reads the recent chat history, conversation topic, and prior phase summaries.
    2. Asks the LLM (with date awareness) to pick web-search queries that
     would help the player join the conversation naturally.
  3. For each query: applies year-correction, hits the search backend, asks the
     LLM to summarise into 1–2 casual sentences, and writes the result to the
     SharedSnapshot via `upsert_info_entry`.
    4. Emits one `internet_context_tick` event per query (and one for the
     "no-search" decision) so the audit can prove what happened.

Most ticks are no-ops (searches==[]), which keeps cost low.
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from social_turing_test.llm_players.agent.sub_agents.base import SubAgent
from social_turing_test.llm_players.agent.sub_agents.web_search import grounded_search
from social_turing_test.llm_players.agent.research.query_validator import non_web_query_reason


# Neutral system message for the background query-planning call. Deliberately
# NOT the player's persona prompt - see the call site in tick() for why.
_IC_DECISION_SYSTEM_MSG = (
    "You are a JSON-only research query planner. You do NOT role-play any chat "
    "persona. Read the conversation and output only the requested JSON object."
)

# Neutral system message for the summary call. Same reason as the decision call:
# feeding the player persona here makes the model role-play and emit chat prose
# ("Hey everyone, where would you travel?") instead of summarising the search
# results. A neutral summariser system message keeps it on-task.
_IC_SUMMARY_SYSTEM_MSG = (
    "You summarise web-search results into one short, factual, casual sentence. "
    "You do NOT role-play any chat persona and you do NOT write chat messages — "
    "you only report the key fact from the results given."
)

# A query whose year is part of a HISTORICAL reference must NOT be year-corrected
# (e.g. "October 7 2023", "...war history", "when did the war start").
_HISTORICAL_QUERY_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"history|historical|started?|begin|began|when did|anniversary|founded)\b",
    re.IGNORECASE,
)

# Generic/date/filler tokens that must NOT count toward topic-overlap dedup -
# otherwise shared "june"/"news"/"latest" makes every query look like a repeat.
_GENERIC_QUERY_WORDS = frozenset({
    "news", "latest", "today", "yesterday", "this", "week", "month", "year",
    "update", "updates", "recent", "current", "happening", "results", "result",
    "decision", "about", "what", "when", "where", "match", "game", "report",
    "biggest", "almost", "everyone", "heard", "world", "headlines",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
})

# A grounded-search answer that is actually a training-cutoff REFUSAL / non-answer
# ("the year 2026 has not yet occurred", "as it is a future date", "I cannot
# provide information", "nothing has been reported yet") must NOT be stored as a
# fact. Detect the refusal shape and drop
# it before summarising/storing.
_REFUSAL_RE = re.compile(
    r"(can(?:not|'t| ?not)\s+provide|not able to provide|"
    r"has not yet occurred|hasn'?t (?:happened|occurred)|have not yet (?:happened|occurred)|"
    r"future date|in the future|yet to (?:occur|happen)|"
    r"not yet (?:available|occurred|happened|reported)|"
    r"nothing (?:has been|to) report|no (?:information|data|results?) (?:is |are |yet)?"
    r"(?:available|yet|found)?|"
    r"i (?:cannot|can'?t|do not|don'?t) (?:provide|have|access)|i am unable to|i'?m unable to)",
    re.IGNORECASE,
)


class InternetContextAgent(SubAgent):
    # Base cadence: every 10 s (the web profile sets
    # subagent_interval_scale=1.0). The history-unchanged guard below keeps the
    # frequent ticks cheap - no decision-LLM call when nobody has spoken - and the
    # _CACHE_TTL guard prevents re-searching the same query within 120 s.
    interval_seconds: float = 10.0
    COMPONENT = "internet_context"

    # Don't re-search the same query within this many seconds.
    _CACHE_TTL: float = 120.0

    # Even on a frozen chat, allow one decision pass this often so a fact that
    # expired can be refreshed while the topic is still live.
    _DECISION_REFRESH_S: float = 60.0

    # Configurable upper bound on queries per tick. Per-query dedup and the
    # history-unchanged guard limit repeated work on multi-topic messages.
    _MAX_QUERIES_PER_TICK: int = 5

    def __init__(self, player, snapshot) -> None:
        super().__init__(player, snapshot)
        cfg = getattr(player, "llm_config", {}) or {}
        self._max_queries_per_tick = int(cfg.get("internet_max_queries_per_tick",
                                                  self._MAX_QUERIES_PER_TICK))
        # Optional WEAK/cheap model for the per-tick "is there anything to search?"
        # decision. None → reuse the player's main llm.
        self._detector_model = cfg.get("internet_detector_model")
        self._detector_llm_obj = None
        self._last_queries: dict = {}   # query → last search ts
        # Skip the decision-LLM call when the chat hasn't changed since the last
        # decision (lets us poll faster without burning calls on a static chat).
        self._last_history_sig: str = ""
        self._last_decision_ts: float = 0.0
        self._seeded: bool = False      # One-time world-headlines seed

    def _player_name_words(self) -> set:
        """Lower-cased codenames + real first names of everyone in this game — used
        to gate game-internal search queries. Cached after the first read."""
        cached = getattr(self, "_name_words_cache", None)
        if cached is not None:
            return cached
        words = set()
        try:
            from social_turing_test.core.game_constants import (PLAYER_NAMES_FILE, REAL_NAMES_FILE,
                                             REAL_NAME_CODENAME_DELIMITER)
            gd = self.player.game_dir
            for n in (gd / PLAYER_NAMES_FILE).read_text(encoding="utf-8").splitlines():
                if n.strip():
                    words.add(n.strip().lower())
            try:
                for line in (gd / REAL_NAMES_FILE).read_text(encoding="utf-8").splitlines():
                    real = line.split(REAL_NAME_CODENAME_DELIMITER)[0].strip()
                    if real:
                        words.add(real.split()[0].lower())
            except Exception:
                pass
        except Exception:
            pass
        self._name_words_cache = words
        return words

    _GAME_META_QUERY_RE = re.compile(
        r"\b(suspect|who is the ai|is the ai|the ai player|a bot in|"
        r"turing test game|voted? (?:out|for))\b", re.IGNORECASE)

    def _game_internal_query_reason(self, query: str) -> str:
        """Return a rejection reason when the query is about the game's own
        players / meta rather than the real world; '' when it's fine."""
        if self._GAME_META_QUERY_RE.search(query):
            return "game_meta_query"
        q_words = {w.strip(".,?!'\"").lower() for w in query.split()}
        hits = q_words & self._player_name_words()
        if hits:
            return f"player_name_query:{sorted(hits)[0]}"
        return ""

    def _decision_llm(self):
        """Weak/cheap model for the 'should I search?' decision. Defaults to
        the player's main llm when ``internet_detector_model`` is not configured."""
        if not self._detector_model:
            return self.agent_llm
        if self._detector_llm_obj is None:
            try:
                from social_turing_test.llm_players.llm_wrapper import LLMWrapper
                dcfg = dict(getattr(self.player, "llm_config", {}) or {})
                dcfg["model_name"] = self._detector_model
                self._detector_llm_obj = LLMWrapper(self.player.logger, **dcfg)
            except Exception as e:                       # fall back to the main llm
                self.player.logger.log("internet_detector_init_error", str(e))
                self._detector_llm_obj = self.agent_llm
        return self._detector_llm_obj

    # ── Main tick ─────────────────────────────────────────────────────────

    def lobby_tick(self) -> None:
        """Pre-START seed: while the game is still in the lobby, run the single
        world-headlines search so the agent already knows the week's big stories
        the moment the game begins. This is the ONLY query allowed during the
        lobby; all other (chat-driven) searches wait for START. Marks `_seeded`
        so the first real tick doesn't repeat the seed."""
        if self._seeded:
            return
        self._seeded = True
        try:
            self._seed_world_headlines()
        except Exception as e:
            self.player.logger.log("internet_seed_error", str(e))

    def tick(self) -> None:
        # On the very first tick, seed shared memory with the week's
        # UNIVERSALLY-known world headlines so the agent isn't blank on "what's
        # going on" from message 1 (a real person would have caught the big ones).
        if not self._seeded:
            self._seeded = True
            try:
                self._seed_world_headlines()
            except Exception as e:
                self.player.logger.log("internet_seed_error", str(e))

        history = self._recent_messages(n=8)
        topic = self._read_conversation_topic()
        if not history:
            return

        # History-unchanged guard: if the recent chat is identical to the last
        # time we ran the decision LLM and the refresh window hasn't elapsed,
        # skip this tick entirely (cheap). New messages OR an expired refresh
        # window let it through.
        import hashlib
        sig = hashlib.md5("\n".join(history).encode("utf-8", "replace")).hexdigest()
        now0 = time.time()
        if (sig == self._last_history_sig
                and (now0 - self._last_decision_ts) < self._DECISION_REFRESH_S):
            self.player.logger.log("internet_skip", "history_unchanged")
            return
        self._last_history_sig = sig
        self._last_decision_ts = now0

        # Observability - pair these begin/end so the analyzer can compute
        # tick latency, decision-LLM latency, and per-query LLM latency.
        tick_start = time.time()
        round_summaries_now = self.snapshot.read().get("round_summaries", [])
        self.player.write_game_event(
            "internet_tick_begin",
            history_len=len(history),
            topic=topic or "",
            round_summary_count=len(round_summaries_now),
            wall_t=tick_start,
        )
        searches_run = 0
        entries_written = 0

        # Step 1: ask LLM to decide which queries (if any) to run.
        # The decision parser returns a List[dict] of {query, volatility}.
        decision_prompt = self._build_decision_prompt(history, topic, round_summaries_now)
        self.player.logger.log("internet_decision_prompt", decision_prompt)
        dec_start = time.time()
        # NOTE: do NOT pass the player's persona system message here. This is a
        # background JSON-planning task, and feeding the "you are <player>, blend
        # in, avoid being voted out" persona makes the model role-play the player
        # and emit chat/persona prose instead of JSON. The prompt is self-contained,
        # so a neutral tool system message keeps the model on-task.
        _det = self._decision_llm()
        raw = _det.generate(
            decision_prompt,
            False,
            _IC_DECISION_SYSTEM_MSG,
            max_output_tokens=300,
            component=self.COMPONENT,
        )
        dec_latency = time.time() - dec_start
        # Cost logging reads the MAIN llm's last usage; only accurate when the
        # detector reuses the main model (the default). With a dedicated detector
        # model the decision cost is under-counted - acceptable for this opt-in.
        if _det is self.agent_llm:
            self.player._log_cost("internet_context_decision")
        parsed_list = self._parse_decision(raw)

        # Observability - what the decision LLM said and how we read it.
        self.player.logger.log(
            "internet_decision_response",
            f"raw={raw[:600]!r} parsed={parsed_list} latency={dec_latency:.2f}s",
        )
        self.player.write_game_event(
            "internet_decision_response",
            raw=raw[:2000],
            parsed=parsed_list,
            latency_s=round(dec_latency, 3),
        )

        if not parsed_list:
            self.player.write_game_event(
                "internet_context_tick",
                needs_search=False,
                query="",
                n_results=0,
                summary=None,
            )
            self.player.write_game_event(
                "internet_tick_end",
                tick_latency_s=round(time.time() - tick_start, 3),
                searches_run=0,
                entries_written=0,
            )
            return

        # Step 2: run up to N queries this tick.
        year_now = self._current_year()
        topic_year_m = re.search(r'\b(20\d{2})\b', topic) if topic else None
        topic_year = topic_year_m.group(1) if topic_year_m else None
        now = time.time()

        for entry in parsed_list[: self._max_queries_per_tick]:
            query = (entry.get("query") or "").strip()
            volatility = entry.get("volatility", "volatile")
            if not query:
                # Log the dropped malformed query so audits catch it.
                self.player.write_game_event(
                    "internet_context_tick",
                    needs_search=True,
                    query="",
                    n_results=0,
                    summary=None,
                    dropped=True,
                    reason="malformed_query",
                )
                continue

            # Game-internal gate: queries about the game's own players or
            # meta ("suspect", "who is the ai") have no grounded web answer and only
            # feed the agent misleading facts. Drop them deterministically.
            gate_reason = non_web_query_reason(query) or self._game_internal_query_reason(query)
            if gate_reason:
                self.log("internet_query_gated", f"q={query!r} reason={gate_reason}")
                self.player.write_game_event(
                    "internet_search_rejected",
                    query=query,
                    query_in=query,
                    reason=gate_reason,
                    raw_answer="",
                )
                continue

            # Year-correction safety net:
            # Correct queries with a stale year (e.g. a model-default "2024") →
            # today's year so "Eurovision 2024" becomes "Eurovision <today>".
            # Do NOT rewrite a year that's part of a clearly
            # HISTORICAL reference - "October 7 2023", "...war history",
            # "when did X start".
            query_in = query   # Keep the pre-correction query for the trace
            m = re.search(r'\b(20\d{2})\b', query)
            wanted = topic_year or year_now
            if (m and wanted and int(m.group(1)) < int(year_now)
                    and not _HISTORICAL_QUERY_RE.search(query)):
                query = re.sub(r'\b20\d{2}\b', wanted, query)
                # Year-correction can turn a RANGE ("2024-2026") into a self-equal
                # range ("2026-2026"), which makes the grounded model treat it as a
                # "that's the future" refusal. Collapse any duplicate
                # "YYYY-YYYY" / "YYYY to YYYY" / "YYYY YYYY" into a single year.
                query = re.sub(
                    r'\b(20\d{2})\s*(?:-|–|—|to)\s*\1\b', r'\1', query)
                query = re.sub(r'\b(20\d{2})\b(?:\s+\1\b)+', r'\1', query)
                if wanted not in query:
                    query = f"{query} {wanted}".strip()

            # Cache: skip if we searched this query in the last _CACHE_TTL seconds.
            last_ts = self._last_queries.get(query)
            if last_ts is not None and (now - last_ts) < self._CACHE_TTL:
                continue

            # Topic-level dedup: if the snapshot already has a fresh
            # entry whose topic/query shares ≥2 DISTINCTIVE words with this query,
            # don't re-search the same subject under a slightly different phrasing
            # (e.g. "Eurovision 2026 winner" vs "Israel Eurovision 2026").
            # Exclude generic date/filler tokens from the
            # overlap and skip the broad world-headlines SEED entry so generic
            # shared terms cannot suppress unrelated topical searches.
            def _distinctive(text):
                toks = re.findall(r"[a-z0-9']+", text.lower())
                return {w for w in toks
                        if len(w) >= 4
                        and not re.fullmatch(r"20\d\d", w)
                        and w not in _GENERIC_QUERY_WORDS}
            q_words = _distinctive(query)
            already_covered = False
            for e in self.snapshot.active_entries():
                # The seed is general common-knowledge - it must NOT suppress a
                # specific, chat-driven topical search.
                if (e.get("topic") or "") == "world headlines this week":
                    continue
                e_words = _distinctive(f"{e.get('topic','')} {e.get('query','')}")
                if len(q_words & e_words) >= 2:
                    already_covered = True
                    break
            if already_covered:
                self.player.logger.log("internet_topic_cache_hit", f"q={query!r}")
                continue

            # Search - bracket with begin/end so we can see search backend latency.
            self.player.write_game_event(
                "internet_search_request",
                query=query,
                volatility=volatility,
                wall_t=time.time(),
            )
            # Tell the scheduler a relevant search is in flight so the
            # agent doesn't answer this question from training data while we
            # wait. The flag clears on the matching mark_search_done below.
            self.snapshot.mark_search_started(query)
            search_start = time.time()
            search_error = None
            try:
                results = grounded_search(
                    query, max_results=2, today_str=self._current_date())
            except Exception as e:
                # Don't let the real reason vanish. On the server a grounding
                # failure (quota / region / tool not enabled) otherwise looks
                # identical to a genuine "no results", which is undiagnosable.
                results = []
                search_error = f"{type(e).__name__}: {e}"
                self.player.logger.log(
                    "internet_context_search_error", f"q={query!r} err={search_error}")
                self.player.write_game_event(
                    "internet_search_error", query=query, error=search_error[:500])
            search_latency = time.time() - search_start
            searches_run += 1
            if not results:
                self.snapshot.mark_search_done(query)
                self.player.logger.log("internet_context_no_results", f"q={query!r}")
                self.player.write_game_event(
                    "internet_search_response",
                    query=query,
                    n_results=0,
                    titles=[],
                    summary_text=None,
                    latency_s=round(search_latency, 3),
                )
                self.player.write_game_event(
                    "internet_context_tick",
                    needs_search=True,
                    query=query,
                    n_results=0,
                    summary=None,
                )
                continue

            # Reject training-cutoff REFUSALS and ungrounded volatile answers
            # BEFORE summarising/storing. The grounded backend is inconsistent:
            # some queries truly ground (correct current facts) while others fall
            # back to the model's training prior and answer "that date is in the
            # future / no info available". Do not store these as facts.
            raw_answer = (results[0].get("snippet") or "").strip()
            answer_grounded = bool(results[0].get("grounded"))
            is_refusal = bool(_REFUSAL_RE.search(raw_answer))
            is_ungrounded_volatile = (volatility == "volatile" and not answer_grounded)
            if is_refusal or is_ungrounded_volatile:
                reason = "refusal" if is_refusal else "ungrounded_volatile"
                self.snapshot.mark_search_done(query)
                self.player.logger.log(
                    "internet_context_search_rejected",
                    f"q={query!r} reason={reason} answer={raw_answer!r}")
                self.player.write_game_event(
                    "internet_search_rejected",
                    query=query,
                    query_in=query_in,
                    reason=reason,
                    grounded=answer_grounded,
                    raw_answer=raw_answer[:300],
                    latency_s=round(search_latency, 3),
                )
                self.player.write_game_event(
                    "internet_context_tick",
                    needs_search=True,
                    query=query,
                    n_results=0,
                    summary=None,
                    dropped=True,
                    reason=reason,
                )
                continue

            # Summarise.
            summary_prompt = self._build_summary_prompt(query, results)
            self.player.logger.log("internet_summary_prompt", summary_prompt)
            sum_start = time.time()
            summary = self.agent_llm.generate(
                summary_prompt,
                False,
                _IC_SUMMARY_SYSTEM_MSG,
                max_output_tokens=256,
                component=self.COMPONENT,
            )
            sum_latency = time.time() - sum_start
            self.player._log_cost("internet_context_summary")
            summary = summary.strip()
            # Clean raw results (title+snippet) so the search is fully
            # reconstructable from the log, not buried/truncated in summary_prompt.
            raw_results = [
                {"title": r.get("title", ""), "snippet": (r.get("snippet") or "")[:300]}
                for r in results
            ]
            self.player.write_game_event(
                "internet_search_response",
                query=query,
                query_in=query_in,
                year_corrected=(query != query_in),
                n_results=len(results),
                titles=[r.get("title", "") for r in results],
                raw_results=raw_results,
                summary_prompt=summary_prompt[:1500],
                summary_text=summary,
                latency_s=round(search_latency + sum_latency, 3),
            )

            # Defense-in-depth: even with a usable raw answer the summariser can
            # still emit a non-answer ("nothing has been reported yet"). Don't
            # store that as a fact.
            if not summary or _REFUSAL_RE.search(summary):
                self.snapshot.mark_search_done(query)
                self.player.logger.log(
                    "internet_context_search_rejected",
                    f"q={query!r} reason=summary_refusal summary={summary!r}")
                self.player.write_game_event(
                    "internet_search_rejected",
                    query=query,
                    query_in=query_in,
                    reason="summary_refusal",
                    grounded=answer_grounded,
                    raw_answer=summary[:300],
                    latency_s=round(search_latency + sum_latency, 3),
                )
                continue

            # Write to snapshot via the canonical API (multi-topic cache, TTL).
            stored_topic = topic or query
            stored_conf = "medium" if results else "low"
            self.snapshot.mark_search_done(query)
            self.snapshot.upsert_info_entry(
                topic=stored_topic,
                query=query,
                summary=summary,
                volatility=volatility,
                confidence=stored_conf,
                source_count=len(results),
            )
            entries_written += 1
            self._last_queries[query] = now

            # ONE consolidated trace - query in → corrected → raw results →
            # summary I/O → stored fact. Everything to debug a search in one line.
            self.player.write_game_event(
                "internet_search_trace",
                query_in=query_in,
                corrected_query=query,
                raw_results=raw_results,
                summary_prompt=summary_prompt[:1500],
                summary_out=summary,
                stored={"topic": stored_topic, "volatility": volatility,
                        "confidence": stored_conf, "source_count": len(results)},
            )

            self.player.logger.log(
                "internet_context_update",
                f"q={query!r} info={summary!r}",
            )
            self.player.write_game_event(
                "internet_context_tick",
                needs_search=True,
                query=query,
                n_results=len(results),
                summary=summary,
            )

        self.player.write_game_event(
            "internet_tick_end",
            tick_latency_s=round(time.time() - tick_start, 3),
            searches_run=searches_run,
            entries_written=entries_written,
        )

    # ── World-headlines seed ──────────────────────────────────────────────

    def _seed_world_headlines(self) -> None:
        """One-time: seed shared memory with the common-knowledge baseline a normal
        person would have. Runs a SMALL SET of category searches (general headlines +
        any ongoing major sporting event), each stored as its own entry, so the agent
        isn't blank on big stories (incl. an ongoing World Cup the generic-headlines
        query may miss) from message 1. Sub-agent runs in parallel — no hot-path cost."""
        date = self._current_date()
        year = self._current_year()
        seeds = [
            ("world headlines this week",
             f"biggest world news headlines that almost everyone has heard about, week of {date}"),
            ("major ongoing sports event",
             f"major sports tournament happening now {year} (FIFA World Cup, Olympics) current status"),
        ]
        for topic_label, query in seeds:
            try:
                self._seed_one(topic_label, query, date)
            except Exception as e:
                self.player.logger.log("internet_seed_error", f"{topic_label}: {type(e).__name__}: {e}")

    def _seed_one(self, topic_label: str, query: str, date: str) -> None:
        """Search + summarise + store one seed category as its own current_info entry."""
        fallback_query = (
            f"major world events reported this week {date}"
            if topic_label == "world headlines this week"
            else f"major ongoing sports tournament latest results {date}"
        )
        for attempt, candidate_query in enumerate((query, fallback_query), start=1):
            self.snapshot.mark_search_started(candidate_query)
            try:
                results = grounded_search(
                    candidate_query, max_results=3, today_str=date
                )
            except Exception as e:
                self.player.logger.log("internet_seed_error", f"{type(e).__name__}: {e}")
                self.player.write_game_event(
                    "internet_search_error", query=candidate_query, error=str(e)[:300]
                )
                results = []
            finally:
                self.snapshot.mark_search_done(candidate_query)
            if not results:
                continue

            raw_answer = (results[0].get("snippet") or "").strip()
            if _REFUSAL_RE.search(raw_answer):
                self.player.write_game_event(
                    "internet_search_rejected",
                    query=candidate_query,
                    query_in=query,
                    reason="seed_refusal",
                    raw_answer=raw_answer[:300],
                    retrying=(attempt == 1),
                )
                continue

            seed_prompt = (
                f"Today is {date} — this is the PRESENT (news from the past week has "
                "already happened, it is NOT the future). From the search results below, list ONLY the few "
                "events from the past week or two that almost EVERYONE has heard "
                "about — the kind of thing a regular person would casually know is "
                "going on (a war between countries, a major disaster, a heat wave, a "
                "big outbreak, or a universal sporting event like the FIFA World Cup or "
                "the Olympics that is on RIGHT NOW). EXCLUDE niche/business/celebrity "
                "items only interested people follow. 1-3 short casual clauses, no "
                "preamble. If the results show nothing universally-known for this "
                "category, reply with an empty line.\n\n"
                + "\n".join(f"- {r.get('snippet') or r.get('title','')}" for r in results)
            )
            try:
                summary = self.agent_llm.generate(
                    seed_prompt, False, _IC_SUMMARY_SYSTEM_MSG, max_output_tokens=200,
                    component=self.COMPONENT,
                ).strip()
                self.player._log_cost("internet_context_seed")
            except Exception as e:
                self.player.logger.log("internet_seed_error", f"summarise: {e}")
                return
            if not summary or _REFUSAL_RE.search(summary):
                self.player.write_game_event(
                    "internet_search_rejected",
                    query=candidate_query,
                    query_in=query,
                    reason="seed_summary_refusal",
                    raw_answer=summary[:300],
                    retrying=(attempt == 1),
                )
                continue

            self.snapshot.upsert_info_entry(
                topic=topic_label, query=candidate_query, summary=summary,
                volatility="stable", confidence="medium", source_count=len(results),
                origin="seed",
            )
            self.player.logger.log("internet_seed", f"seeded {topic_label}: {summary!r}")
            self.player.write_game_event(
                "internet_seed", topic=topic_label, summary=summary[:300],
                n_results=len(results), query=candidate_query,
            )
            return

    # ── Date/time helpers ─────────────────────────────────────────────────

    def _current_date(self) -> str:
        """Prefer the game's recorded date (game_date.txt, ISO YYYY-MM-DD) when
        present, so simulating a recorded April 2026 game from June 2026 still
        reports "April 15, 2026" to the LLM. Falls back to wall-clock.
        """
        try:
            game_dir = self.player.game_dir
            game_date_file = game_dir / "game_date.txt"
            if game_date_file.exists():
                iso = game_date_file.read_text(encoding="utf-8", errors="replace").strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
                    return iso
        except (AttributeError, OSError):
            pass
        from social_turing_test.core.game_constants import get_current_date
        return get_current_date()

    @staticmethod
    def _current_timestamp() -> str:
        from social_turing_test.core.game_constants import get_current_timestamp
        return get_current_timestamp()

    def _current_year(self) -> str:
        d = self._current_date()
        # Match either ISO (2026-04-15) or human ("Sunday, May 31, 2026").
        m = re.search(r'(\d{4})', d)
        return m.group(1) if m else "2025"

    def _date_line(self) -> str:
        # Pin the date as the
        # PRESENT explicitly so summaries read as current, not speculative.
        return (
            f"Today's date is {self._current_date()} — this is the PRESENT, the "
            f"current real date, NOT the future. Treat all of {self._current_date()} "
            f"and earlier as having already happened. The current time is "
            f"[{self._current_timestamp()}].\n"
        )

    # ── Prompt builders ───────────────────────────────────────────────────

    def _build_decision_prompt(self, history: list, topic: Optional[str],
                               round_summaries: list) -> str:
        history_text = "\n".join(history[-8:])
        topic_line = f"Conversation topic: {topic}\n" if topic else ""
        phase_line = (
            f"Earlier in this game: {' | '.join(round_summaries[-3:])}\n"
            if round_summaries else ""
        )
        return (
            f"{self._date_line()}"
            f"{topic_line}{phase_line}"
            f"Recent chat:\n{history_text}\n\n"
            "You are a background research assistant for an AI player in a social "
            "chat. Stay one step ahead: look up facts the player may soon need so "
            "it can join naturally.\n\n"
            "List a separate web-search query for EVERY topic below — do NOT cap "
            "yourself at one or two, and when in doubt SEARCH rather than skip (a "
            "missed current fact is far worse than an extra search):\n"
            "  - ANYTHING that could have happened or changed in roughly the LAST "
            "TWO YEARS — a result, score, election, conflict, appointment, release, "
            "award, record, or anyone's 'current' status. Your training memory of "
            "recent events is unreliable, so look them up rather than answer from "
            "memory.\n"
            "  - things the chat is asking about right now (winners, scores, news)\n"
            "  - EACH distinct current/ongoing event someone brings up — a sports "
            "tournament (e.g. the World Cup), an election, a conflict, an awards "
            "show: search EACH ONE separately with the current year (e.g. 'FIFA "
            "World Cup <this year> status'), because your memory of when it last "
            "happened is probably out of date. If two unrelated current topics come "
            "up (say a tournament AND a political event), search BOTH.\n"
            "  - facts tied to the conversation topic the player should know\n"
            "  - likely next questions given where the chat is heading\n\n"
            "Do NOT search casual small talk (food, mood, daily routines, opinions).\n"
            "Do NOT search for the current time or today's date — those are provided "
            "separately and a web search returns the wrong timezone.\n"
            "Do NOT search for the game's PLAYERS or the game itself: player names/"
            "nicknames from this chat, who is 'the AI' / 'a suspect' / 'a bot', or "
            "anything about votes and accusations. Those are game-internal — the web "
            "has nothing real about them and results would only mislead the player. "
            "Search only real-world topics (events, people in the news, shows, "
            "products, places).\n"
            "If nothing is worth looking up, return an empty list.\n\n"
            "Each query MUST include the conversation topic / year verbatim — do NOT "
            "substitute a different year you might remember from training data.\n"
            "Resolve relative time references ('this year', 'last week') against the "
            "date above. Never use a year earlier than today's year unless the topic "
            "explicitly says so.\n"
            "NORMALISE typos and slang: if the chat says 'aurussiion' but means "
            "'Eurovision', search for the correct word. Use proper-noun "
            "capitalisation. Do NOT echo the user's spelling literally.\n\n"
            "Mark each query's volatility:\n"
            "  - \"volatile\": live/changing (today's score, latest news, current standings)\n"
            "  - \"stable\": fixed facts (past winners, dates, historical results)\n\n"
            "Reply with ONLY valid JSON, no extra text:\n"
            '{"searches": [{"query": "<includes topic verbatim>", "volatility": "stable"}]}\n'
            "OR\n"
            '{"searches": []}\n'
        )

    def _build_summary_prompt(self, query: str, results: list) -> str:
        snippets = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in results if r.get("snippet")
        )
        return (
            f"{self._date_line()}"
            f"Search query: {query}\n"
            f"Search results:\n{snippets}\n\n"
            "Write 1–2 very short, casual sentences summarising the key fact. "
            "Do NOT start with 'According to' or 'Based on'. "
            "Write like you'd mention it casually in a chat message. "
            "No bullet points, no headers. Just the fact.\n"
            "Never say the date is 'in the future' or 'hasn't happened yet' — "
            "today's date above is the present. If results for today specifically "
            "aren't in yet, just summarise the most recent available fact instead."
        )

    # ── JSON parser ───────────────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> List[dict]:
        """Parse the decision JSON into a list of {query, volatility} dicts.

        Supports the single-query shape {needs_search, query} too.

        When ``volatility`` is omitted, infer it from the query — queries
        that mention a past year are almost always historical facts that
        deserve the 30-min ``stable`` TTL, not the 5-min ``volatile`` default.
        """
        data = self._loads(raw)
        if not isinstance(data, dict):
            return []

        searches = data.get("searches")
        if isinstance(searches, list):
            out = []
            for it in searches:
                if not isinstance(it, dict):
                    continue
                q = str(it.get("query", "") or "").strip()
                # Keep entries without a query so the tick loop can emit
                # a `dropped=True` event for them (instead of silently
                # losing the LLM's intent).
                vol = it.get("volatility") or (self._infer_volatility(q) if q else "volatile")
                out.append({"query": q, "volatility": vol})
            return out

        # Supported single-query input shape.
        if data.get("needs_search") and data.get("query"):
            q = str(data["query"]).strip()
            return [{"query": q, "volatility": self._infer_volatility(q)}]
        return []

    def _infer_volatility(self, query: str) -> str:
        """If the query mentions a year strictly earlier than the current year,
        treat the fact as historical (stable). Otherwise volatile."""
        m = re.search(r'\b(20\d{2})\b', query)
        if m:
            try:
                if int(m.group(1)) < int(self._current_year()):
                    return "stable"
            except ValueError:
                pass
        return "volatile"

    @staticmethod
    def _loads(raw: str) -> Optional[dict]:
        raw = (raw or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None
