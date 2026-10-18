import sys
from pathlib import Path

# Ensure project root is on sys.path (mirrors the pattern in factory.py)
_PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from llm_players.reviewed_player import ReviewedPlayer
from llm_players.v4.agent_state import AgentState, DEFAULT_STATE


class LangGraphPlayer(ReviewedPlayer):
    """
    V4: LangGraph-orchestrated pipeline.

    Inherits all existing helpers from ReviewedPlayer / ScheduleThenGeneratePlayer
    (create_scheduling_prompt, _review_and_fix_message, _apply_typing_delay, etc.).

    Overrides only the two public abstract methods:
        generate_message()        → runs the StateGraph
        should_generate_message() → returns True (scheduling is inside the graph)

    async_type = "langgraph"
    """

    TYPE_NAME = "langgraph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # ReviewedPlayer.__init__ wires llm, reviewer, typing
        self._graph = None          # lazy-compiled on first generate_message() call
        # Stash for review provenance, populated by reviewer_node each turn so
        # generate_message_with_review_info() can expose it to the live-game
        # logger without re-running the reviewer outside the graph.
        self._last_review_info = None

    # ── Abstract implementation ────────────────────────────────────────────
    def should_generate_message(self, message_history) -> bool:
        # The scheduling decision is handled by the scheduler_node inside the
        # graph.  This method satisfies the abstract contract but is not called
        # from generate_message() in this class.
        return True

    # ── Main entry point ───────────────────────────────────────────────────
    def generate_message(self, message_history) -> str:
        """
        Run the full V4 LangGraph pipeline and return the final message.

        Returns "" if the agent decides to WAIT or if the reviewer drops
        the message.  Caller (llm_interface.py) treats "" as "no send".
        """
        if self.message_in_progress:
            self.logger.log("generate_message", "BLOCKED: message already in progress")
            return ""

        self.message_in_progress = True
        typing_flag = self.game_dir / f"{self.name}_typing.txt"
        try:
            typing_flag.touch()

            # Lazy graph compilation — happens once per player instance
            if self._graph is None:
                from llm_players.v4.graph import build_graph
                self._graph = build_graph(self)
                self.logger.log("v4_graph_compiled", "LangGraph StateGraph compiled")

            # Build initial state from defaults, then set the inputs
            initial_state: AgentState = {
                **DEFAULT_STATE,
                "message_history": message_history,
                "current_info": "",     # V4.5 will populate before invoking
            }

            result = self._graph.invoke(initial_state)
            return result.get("final_message", "")

        finally:
            self.message_in_progress = False
            typing_flag.unlink(missing_ok=True)
            self.logger.log("generate_message", "UNLOCKED: message generation complete")

    # ── Live-game entry point ──────────────────────────────────────────────
    def generate_message_with_review_info(self, message_history) -> dict:
        """
        Live-game entry point used by llm_interface.py.

        ReviewedPlayer's flat implementation generates a raw message and reviews
        it OUTSIDE the graph — bypassing the scheduler, current-info injection,
        graph-level dedup and (for V5) the snapshot pipeline entirely.  We
        override it here so the LangGraph pipeline is what actually runs.  The
        graph's output_node already applied the typing delay, so we flag
        ``typing_already_applied`` to stop llm_interface from waiting twice.

        ``self.generate_message`` is polymorphic: for a V5 instance it runs the
        V5 graph (snapshot injection + memory trace + dedup); for a plain V4
        instance it runs the V4 graph.  Either way the reviewer_node populates
        ``self._last_review_info`` with the draft + ReviewResult.
        """
        self._last_review_info = None
        final = (self.generate_message(message_history) or "").strip()

        info = self._last_review_info or {}
        raw = info.get("raw_message", final)
        rr = info.get("review_result")

        review_result = None
        if rr is not None:
            review_result = {
                "suspicion_score": getattr(rr, "suspicion_score", None),
                "is_suspicious": getattr(rr, "is_suspicious", None),
                "red_flags": list(getattr(rr, "red_flags", []) or []),
                "suggested_fix": getattr(rr, "suggested_fix", None),
                "budget_exceeded": bool(getattr(rr, "budget_exceeded", False)),
            }

        return {
            "message": final,
            "raw_message": raw,
            "was_reviewed": bool(getattr(self, "reviewer", None)) and bool(final),
            "review_result": review_result,
            "was_modified": bool(final) and raw != final,
            "revision_steps": info.get("revision_steps", []),
            # output_node already called _apply_typing_delay inside the graph.
            "typing_already_applied": True,
        }
