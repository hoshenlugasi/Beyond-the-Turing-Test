from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Dict, List, Optional
import copy
import re
import time


# Differential TTL (seconds): volatile facts (live scores, "latest news") go
# stale fast; stable facts (historical results, dates) stay fresh much longer.
_TTL_BY_VOLATILITY: Dict[str, float] = {
    "volatile": 5 * 60.0,
    "stable":   30 * 60.0,
}
_DEFAULT_TTL = _TTL_BY_VOLATILITY["volatile"]

# Maximum number of cached fact entries. Long games can accumulate many
# search results - without a cap the snapshot would inflate every generator
# prompt with stale context. Cap to the most-recent _MAX_ENTRIES.
_MAX_ENTRIES: int = 8

# A pending search is considered "in flight" only if it started within the
# last _AWAITING_SEARCH_WINDOW_S seconds; older ones are presumed stuck.
_AWAITING_SEARCH_WINDOW_S: float = 15.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


@dataclass
class SharedSnapshot:
    """Compact in-memory snapshot written by background sub-agents and read
    by the main LangGraph agent every tick.

    Thread-safety: all public mutators acquire _lock.  `read()` returns a
    plain dict safe to pass into AgentState without holding the lock.
    """

    # Internet Context Agent slice - multi-topic cache of fact entries.
    # Each entry is a dict:
    #   {topic, query, summary, fetched_at, expires_at, volatility,
    #    confidence, source_count}
    current_info_entries: List[dict] = field(default_factory=list)

    # Round Summarizer slice - ordered list of completed-round summaries
    round_summary_state: List[str] = field(default_factory=list)
    last_round_summarized: Optional[str] = None  # e.g. "Discussion 2"

    # Persona Monitor slice - first-person claims the agent has made about
    # ITSELF this game (e.g. "uses Android", "from Tel Aviv"). NOT TTL'd: you
    # can't un-say something, so these persist for the whole game and the
    # generator/reviewer use them to stay consistent.
    self_claims_state: List[str] = field(default_factory=list)

    # Fixed persona facts: immutable ground truth set at init from
    # the persona config (location, timezone). Distinct from self_claims (which
    # are things the agent SAID): these are facts the agent IS, so the agent
    # must never contradict them even before it has said anything.
    # Fixed facts are kept separate from extracted self-claims.
    fixed_persona_facts: Dict[str, str] = field(default_factory=dict)

    # In-flight searches (query → started_ts). Used by the scheduler's
    # `awaiting_search` guard so the agent doesn't answer from training data
    # while the sub-agent is actively fetching the truth.
    pending_searches: Dict[str, float] = field(default_factory=dict)

    # Social State Tracker slice - the agent's OWN social standing, extracted from the
    # public chat: who has suspected/accused the agent, who the agent has
    # pointed at (so it stays consistent), who's been voted out, the current
    # prime suspect. Lets the generator react to accusers instead of acting
    # oblivious. Keys: suspects_me, i_pointed_at, voted_out (lists), prime_suspect (str).
    social_state: Dict[str, Any] = field(default_factory=dict)

    # Behavior Adapter slice - a single evolving play-style directive for the
    # current stage of the game (e.g. "few players left and you're drawing
    # suspicion → keep replies minimal, mirror others"). Synthesised from game
    # stage, suspicion-on-self, and talk-share.
    behavior_advice_state: str = ""

    # Repetition Monitor slice - words/openers THIS agent has already leaned on
    # too much (e.g. "lol" in 4/10 messages). The reviewer reads these to flag
    # reuse as repetition and to avoid adding them in fixes. Deterministic, no LLM.
    overused_words_state: List[str] = field(default_factory=list)
    # Repetition Monitor's LLM pass: a short note about repeated message
    # SHAPE / approach / stance (not exact words) - e.g. "you keep giving the
    # same 2-word dodge; vary structure / show an opinion". Empty until enough
    # own-messages exist. Fed to the generator (and reviewer).
    repetition_note_state: str = ""
    # Cadence signal for the SCHEDULER: is the agent over/under/balanced
    # in how much it talks vs others (Behavior Adapter), and the room's norm
    # (does everyone answer every question?). The scheduler folds this into its
    # prompt so it holds back when over-participating and not directly addressed.
    cadence_state: Dict[str, Any] = field(default_factory=dict)

    # bookkeeping: slice-name → unix timestamp of last write
    last_updated: Dict[str, float] = field(default_factory=dict)

    # Monotonic revision counter, incremented on every slice write.
    # Surfaced by read() as `snapshot_revision` so consumers (and replays) can tell
    # which revision they consumed and detect stale/mixed-revision reads.
    _revision: int = 0

    # Optional observability hook. The owning player wires this to its
    # `write_game_event` so every upsert/prune emits a structured event into
    # game_events.jsonl. Set via `set_event_callback(...)`; never read directly.
    _event_cb: Optional[Callable[[str, Dict[str, Any]], None]] = field(
        default=None, repr=False, compare=False,
    )

    # internal lock - excluded from repr / equality
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    # ── Revision bookkeeping ─────────────────────────────────────────────

    def _bump(self, slice_name: str) -> None:
        """Record a slice write: stamp last_updated AND increment the monotonic
        revision. MUST be called while holding ``self._lock`` (every setter
        already does)."""
        self.last_updated[slice_name] = time.time()
        self._revision += 1

    # ── Observability hook ────────────────────────────────────────────────

    def set_event_callback(
        self, callback: Optional[Callable[[str, Dict[str, Any]], None]]
    ) -> None:
        """Register a function called as ``callback(event_type, fields)`` from
        every observable mutation. Designed for ``SocialTuringTestAgent`` to wire
        ``write_game_event`` in without creating an import cycle."""
        self._event_cb = callback

    def _emit(self, event_type: str, **fields: Any) -> None:
        cb = self._event_cb
        if cb is None:
            return
        try:
            cb(event_type, fields)
        except Exception:
            pass   # observability must never break the game loop

    # ── Writers (called from sub-agent threads) ────────────────────────────

    def upsert_info_entry(self, topic: str, query: str, summary: str,
                          volatility: str = "volatile",
                          confidence: str = "medium",
                          source_count: int = 0,
                          origin: str = "search") -> None:
        """Insert or replace a fact entry, keyed by normalized topic.

        Newer facts for the same topic overwrite older ones. Expired entries
        are pruned on every write.

        origin — "seed" for the one-time common-knowledge baseline vs
        "search" for chat-driven verified lookups. The admin panel displays
        them differently (seed facts are NOT web-verified evidence).
        """
        now = time.time()
        ttl = _TTL_BY_VOLATILITY.get(volatility, _DEFAULT_TTL)
        entry = {
            "topic": topic,
            "query": query,
            "summary": summary,
            "fetched_at": now,
            "expires_at": now + ttl,
            "volatility": volatility,
            "confidence": confidence,
            "source_count": source_count,
            "origin": origin,
        }
        key = _norm(topic) or _norm(query)
        with self._lock:
            kept_keys = {(_norm(e["topic"]) or _norm(e["query"])) for e in self.current_info_entries}
            replaced = key in kept_keys
            self.current_info_entries = [
                e for e in self.current_info_entries
                if e["expires_at"] > now and (_norm(e["topic"]) or _norm(e["query"])) != key
            ]
            self.current_info_entries.append(entry)
            # Cap snapshot growth - keep newest _MAX_ENTRIES only.
            if len(self.current_info_entries) > _MAX_ENTRIES:
                self.current_info_entries = self.current_info_entries[-_MAX_ENTRIES:]
            self._bump("current_info")
            total_after = len(self.current_info_entries)
        # Emit OUTSIDE the lock so a slow callback can't stall sub-agents.
        self._emit(
            "internet_fact_stored",
            topic=topic, query=query, summary=summary,
            volatility=volatility, confidence=confidence,
            source_count=source_count, origin=origin,
            expires_at_iso=datetime.fromtimestamp(entry["expires_at"], tz=timezone.utc).isoformat(),
            replaced=replaced,
            total_entries_after=total_after,
        )

    def append_round_summary(self, phase: str, summary: str) -> None:
        with self._lock:
            self.round_summary_state.append(f"{phase}: {summary}")
            self.last_round_summarized = phase
            self._bump("round_summary")

    def set_self_claims(self, claims: List[str]) -> None:
        """Replace the persona self-claims slice (Persona Monitor writes this).

        We replace rather than append because the monitor re-derives the full
        claim set from the agent's messages each tick — that naturally drops
        nothing and de-dupes upstream.
        """
        cleaned = [c.strip() for c in claims if c and c.strip()]
        with self._lock:
            self.self_claims_state = cleaned
            self._bump("self_claims")
        self._emit("persona_claims_updated", claims=cleaned, count=len(cleaned))

    def set_social_state(self, memory: Dict[str, Any]) -> None:
        """Replace the Social State Tracker slice.

        Like `set_self_claims`, the agent re-derives the full picture from the
        chat each tick, so we replace rather than merge.
        """
        # DEEP-COPY before storing so the snapshot owns its
        # own object. Otherwise the caller (a sub-agent) could later mutate the
        # same dict and silently change what read() returns - without the lock,
        # _bump, slice_updated_at, or an event. deepcopy runs OUTSIDE the lock
        # (it only touches the caller's object) to keep the lock scope short.
        mem = copy.deepcopy(memory) if isinstance(memory, dict) else {}
        with self._lock:
            changed = mem != self.social_state
            if changed:
                self.social_state = mem
                self._bump("social_state")
        self._emit(
            "social_state_updated",
            suspects_me=list(mem.get("suspects_me", []) or []),
            i_pointed_at=list(mem.get("i_pointed_at", []) or []),
            voted_out=list(mem.get("voted_out", []) or []),
            prime_suspect=mem.get("prime_suspect", "") or "",
            open_threads=list(mem.get("open_threads", []) or []),
            group_mood=mem.get("group_mood", "") or "",
            players=mem.get("players", {}) or {},
            room_norm=mem.get("room_norm", "") or "",
        )

    def get_social_state(self) -> Dict[str, Any]:
        """Thread-safe deep copy of social state for the Behavior Adapter,
        which reads suspicion-on-self."""
        import copy
        with self._lock:
            return copy.deepcopy(self.social_state)

    def set_fixed_persona_facts(self, facts: Dict[str, str]) -> None:
        """Set the immutable persona facts. Called once at init."""
        clean = {str(k): str(v) for k, v in (facts or {}).items() if v}
        with self._lock:
            changed = clean != self.fixed_persona_facts
            self.fixed_persona_facts = clean
            # This mutation changes what read() returns
            # (`fixed_persona_facts`), so it MUST advance the revision. Skip the
            # bump only on a genuine no-op (identical facts re-set).
            if changed:
                self._bump("fixed_persona_facts")
        self._emit("fixed_persona_facts_set", facts=clean)

    def get_fixed_persona_facts(self) -> Dict[str, str]:
        import copy
        with self._lock:
            return copy.deepcopy(self.fixed_persona_facts)

    def set_behavior_advice(self, advice: str) -> None:
        """Replace the Behavior Adapter slice (one play-style directive)."""
        text = (advice or "").strip()
        with self._lock:
            self.behavior_advice_state = text
            self._bump("behavior_advice")
        self._emit("behavior_advice_updated", advice=text)

    def set_overused_words(self, words: List[str]) -> None:
        """Replace the Repetition Monitor slice (words this agent overuses)."""
        cleaned = [w.strip() for w in words if w and w.strip()]
        with self._lock:
            self.overused_words_state = cleaned
            self._bump("overused_words")
        self._emit("overused_words_updated", words=cleaned)

    def set_repetition_note(self, note: str) -> None:
        """Replace the Repetition Monitor LLM-pass note (shape/approach repetition)."""
        text = (note or "").strip()
        with self._lock:
            self.repetition_note_state = text
            self._bump("repetition_note")
        self._emit("repetition_note_updated", note=text)

    def set_cadence(self, cadence: Dict[str, Any]) -> None:
        """Replace the cadence slice written by Behavior Adapter for the scheduler."""
        # Copy-on-store (see set_social_state).
        c = copy.deepcopy(cadence) if isinstance(cadence, dict) else {}
        with self._lock:
            changed = c != self.cadence_state
            if changed:
                self.cadence_state = c
                self._bump("cadence")
        self._emit("cadence_updated", **{k: c.get(k) for k in ("level", "reason", "talk_share")})

    def get_cadence(self) -> Dict[str, Any]:
        import copy
        with self._lock:
            return copy.deepcopy(self.cadence_state)

    # ── Pending-search lifecycle ──────────────────────────────────────────

    def mark_search_started(self, query: str) -> None:
        # Starting a search changes what is_awaiting_search()
        # / read() report, so it advances the revision. (Re-marking the same
        # query refreshes its start time - still a state change.)
        with self._lock:
            self.pending_searches[query] = time.time()
            self._bump("pending_searches")

    def mark_search_done(self, query: str) -> None:
        # Bump ONLY if a real pending entry was removed.
        # Popping a key that was never there is a genuine no-op → no bump.
        with self._lock:
            existed = self.pending_searches.pop(query, None) is not None
            if existed:
                self._bump("pending_searches")

    def is_awaiting_search(self) -> bool:
        """True iff any pending search started within the recency window.

        Pure read: stale "stuck" searches are
        filtered out of the answer but NOT deleted here; removal is the job of
        the explicit ``prune_expired()`` mutator (which advances the revision).
        """
        now = time.time()
        with self._lock:
            return any(now - ts <= _AWAITING_SEARCH_WINDOW_S
                       for ts in self.pending_searches.values())

    # ── Reader (called from main agent thread) ─────────────────────────────

    def active_entries(self) -> List[dict]:
        """Return the non-expired fact entries as OWNED deep copies.

        Pure read: filters by expiry without mutating the backing
        list. Expired entries are removed only by ``prune_expired()``.

        The entries are deep-copied so a caller mutating a
        returned dict cannot reach back into the snapshot's internal entry and
        change it without the lock / a revision bump.
        """
        now = time.time()
        with self._lock:
            return [copy.deepcopy(e) for e in self.current_info_entries
                    if e["expires_at"] > now]

    def prune_expired(self) -> int:
        """Explicit cleanup mutator: drop expired fact entries and
        stale pending searches. Runs under a short lock, advances the revision
        for each slice that actually changed, and emits the prune event OUTSIDE
        the lock. Returns the number of fact entries removed.

        Observability: a structured ``internet_facts_pruned`` event is emitted
        whenever ANYTHING was removed — including the case where only stale
        searches were cleared and no fact entries expired — carrying
        ``expired_fact_count`` / ``stale_search_count`` / ``snapshot_revision``.
        A genuine no-op (nothing removed) bumps nothing and emits nothing.
        """
        now = time.time()
        with self._lock:
            kept_entries = [e for e in self.current_info_entries if e["expires_at"] > now]
            entries_removed = len(self.current_info_entries) - len(kept_entries)
            if entries_removed:
                self.current_info_entries = kept_entries
                self._bump("current_info")

            kept_searches = {q: ts for q, ts in self.pending_searches.items()
                             if now - ts <= _AWAITING_SEARCH_WINDOW_S}
            searches_removed = len(self.pending_searches) - len(kept_searches)
            if searches_removed:
                self.pending_searches = kept_searches
                self._bump("pending_searches")
            remaining = len(self.current_info_entries)
            revision = self._revision
        if entries_removed or searches_removed:
            self._emit("internet_facts_pruned",
                       expired_fact_count=entries_removed,
                       stale_search_count=searches_removed,
                       remaining_count=remaining,
                       snapshot_revision=revision,
                       # Fact-entry count exposed to consumers.
                       pruned_count=entries_removed)
        return entries_removed

    def cached_queries(self) -> List[str]:
        """Normalized queries currently in the (non-expired) cache."""
        return [_norm(e["query"]) for e in self.active_entries()]

    @staticmethod
    def _format_entries(entries: List[dict], now: Optional[float] = None) -> str:
        """Render active entries into a clean bullet list of facts only.

        This string is fed straight into the generator's prompt, so it
        must NOT contain bracketed metadata (`[confidence: medium]`) that the
        LLM might parrot back into a chat message. Confidence/hedging is
        surfaced as a separate top-level instruction by the caller via
        :py:meth:`hedge_advice`.

        ``now`` is passed in by the caller (``read()``) so age computations
        share a single timestamp with the prune step.
        """
        if not entries:
            return ""
        lines = [f"- {e['summary']}" for e in entries]
        return "\n".join(lines)

    @staticmethod
    def hedge_advice(entries: List[dict], now: Optional[float] = None) -> str:
        """Return a single short instruction line if any fact requires hedging.

        Kept separate from the facts block so the generator gets a clean
        list of facts and a separate "hedge if you mention these" directive.
        """
        if not entries:
            return ""
        now = time.time() if now is None else now
        for e in entries:
            conf = e.get("confidence", "medium")
            ttl = _TTL_BY_VOLATILITY.get(e.get("volatility", "volatile"), _DEFAULT_TTL)
            age = now - e.get("fetched_at", now)
            if conf != "high" or (e.get("volatility") == "volatile" and age > ttl / 2):
                return ("If you mention any of the facts above, hedge it "
                        "naturally (e.g. 'i think', 'iirc', 'pretty sure').")
        return ""

    @staticmethod
    def _format_social_state(mem: Dict[str, Any]) -> str:
        """Render the social-state dict into one short line for the generator.

        Empty string when there's nothing actionable yet.
        """
        if not mem:
            return ""
        parts: List[str] = []
        suspects = [s for s in (mem.get("suspects_me") or []) if s]
        pointed = [s for s in (mem.get("i_pointed_at") or []) if s]
        out = [s for s in (mem.get("voted_out") or []) if s]
        threads = [s for s in (mem.get("open_threads") or []) if s]
        mood = (mem.get("group_mood") or "").strip()
        pressure = (mem.get("pressure_type") or "").strip()
        prime = (mem.get("prime_suspect") or "").strip()
        if suspects:
            parts.append(f"{', '.join(suspects)} {'has' if len(suspects) == 1 else 'have'} suspected you")
        if pointed:
            parts.append(f"you pointed at {', '.join(pointed)}")
        if out:
            parts.append(f"{', '.join(out)} {'was' if len(out) == 1 else 'were'} voted out")
        # The player the room suspects most right now. (If it's us, the
        # suspects_me line already covers it; skip the generic "you".)
        if prime and prime.lower() not in ("you", "me", "myself"):
            parts.append(f"the room's prime suspect right now is {prime}")
        if threads:
            # Only the freshest 2 - actionable without nagging.
            parts.append("unanswered to you: " + "; ".join(threads[-2:]))
        if pressure:
            parts.append(f"how you're being tested: {pressure}")
        if mood:
            parts.append(f"mood: {mood}")
        # Compact per-player read (what we understand about each person).
        players = mem.get("players") or {}
        if isinstance(players, dict) and players:
            sketch = "; ".join(f"{n}: {r}" for n, r in list(players.items())[:6])
            parts.append("people: " + sketch)
        return "; ".join(parts)

    def read(self) -> dict:
        """Return a plain dict snapshot safe to pass into AgentState.

        ``current_info`` is a pre-formatted multi-topic facts block (string),
        ready to inject into the generator prompt. Metadata like confidence
        is surfaced separately via ``hedge_advice`` so the generator can't
        leak it into chat.
        """
        import copy
        now = time.time()
        # ATOMIC, PURE read. We hold the lock ONCE so
        # every field, the revision, and the per-slice timestamps describe a
        # single consistent point in time. Crucially this read does NOT mutate:
        # expired fact entries and stale searches are FILTERED out of the
        # returned snapshot but left in the backing collections - removal (and
        # the matching revision bump) is the job of prune_expired(). That keeps
        # the invariant "state never changes without the revision advancing".
        # Lock scope stays short: only in-memory comprehensions/copies run under
        # it - no model calls, search, prompts, or sleeping.
        with self._lock:
            entries = [e for e in self.current_info_entries if e["expires_at"] > now]
            awaiting = any(now - ts <= _AWAITING_SEARCH_WINDOW_S
                           for ts in self.pending_searches.values())
            # The scheduler needs the actual in-flight QUERIES (not
            # just a boolean) to decide whether the pending search matches the
            # question being asked, independently of cached seed facts.
            pending_qs = [q for q, ts in self.pending_searches.items()
                          if now - ts <= _AWAITING_SEARCH_WINDOW_S]

            mem = self.social_state
            snapshot = {
                "current_info":    self._format_entries(entries, now=now),
                "hedge_advice":    self.hedge_advice(entries, now=now),
                "round_summaries": list(self.round_summary_state),
                "awaiting_search": awaiting,
                "pending_search_queries": pending_qs,
                "self_claims":     list(self.self_claims_state),
                "social_state":   self._format_social_state(mem),
                # Raw social-state copy carried in the SAME
                # atomic read so consumers see matching open threads and revision.
                "social_state_raw": copy.deepcopy(mem),
                "behavior_advice": self.behavior_advice_state,
                "overused_words":  list(self.overused_words_state),
                "repetition_note": self.repetition_note_state,
                "pressure_type":   (mem.get("pressure_type", "") or ""),
                # Immutable persona facts for the generator.
                "fixed_persona_facts": dict(self.fixed_persona_facts),
                # Structured numeric social signals; 0.0 when absent.
                "interrogation_pressure": float(mem.get("interrogation_pressure", 0.0) or 0.0),
                "attention_on_me":        float(mem.get("attention_on_me", 0.0) or 0.0),
                "room_norm":              (mem.get("room_norm", "") or ""),
                # Current room mode for the generator to mirror.
                "room_vibe":              (mem.get("room_vibe", "") or ""),
                # Entry count + per-entry metadata carried in
                # the SAME atomic read, so all per-turn observability (memory
                # trace, snapshot_read_full) uses these instead of a second
                # active_entries() call at a possibly-newer revision.
                "active_entry_count":     len(entries),
                "active_entries_metadata": [
                    {k: e.get(k) for k in
                     ("topic", "query", "summary", "volatility", "confidence",
                      "source_count", "fetched_at", "expires_at")}
                    for e in entries
                ],
                # Freshness/consistency metadata for consumers+replays.
                "snapshot_revision": self._revision,
                "read_at":           now,
                "slice_updated_at":  dict(self.last_updated),
            }
        return snapshot

    def staleness_seconds(self, slice_name: str) -> float:
        """Seconds since `slice_name` was last written (inf if never written)."""
        ts = self.last_updated.get(slice_name)
        return (time.time() - ts) if ts is not None else float("inf")
