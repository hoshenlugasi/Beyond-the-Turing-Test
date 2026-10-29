from langgraph.graph import StateGraph, END

from social_turing_test.llm_players.agent.agent_state import AgentState
from social_turing_test.llm_players.agent.nodes.state_builder import make_state_builder_node
from social_turing_test.llm_players.agent.nodes.scheduler_node import make_scheduler_node
from social_turing_test.llm_players.agent.nodes.context_builder import (
    make_context_builder_gen_node,
    make_context_builder_rev_node,
)
from social_turing_test.llm_players.agent.nodes.generator_node import make_generator_node
from social_turing_test.llm_players.agent.nodes.reviewer_node import make_reviewer_node
from social_turing_test.llm_players.agent.nodes.output_node import make_output_node


def build_graph(player):
    """
    Compile and return the agent LangGraph StateGraph for a given player.

    Graph topology:

        state_builder → scheduler ──WAIT──→ END (final_message = "")
                            │ SEND
                            ▼
                    context_builder_gen → generator
                                              │ (no reviewer)
                                              ├──────────────→ output → END
                                              │ (reviewer enabled)
                                              ▼
                                      context_builder_rev → reviewer
                                                                │ approved
                                                                ├──────→ output → END
                                                                │ drop
                                                                └──────→ END

    The reviewer loop (up to reviewer_max_attempts) is internal to
    reviewer_node (Python loop).
    """
    has_reviewer = getattr(player, "reviewer", None) is not None

    g = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    g.add_node("state_builder",       make_state_builder_node(player))
    g.add_node("scheduler",           make_scheduler_node(player))
    g.add_node("context_builder_gen", make_context_builder_gen_node(player))
    g.add_node("generator",           make_generator_node(player))
    g.add_node("output",              make_output_node(player))

    if has_reviewer:
        g.add_node("context_builder_rev", make_context_builder_rev_node(player))
        g.add_node("reviewer",            make_reviewer_node(player))

    # ── Edges ──────────────────────────────────────────────────────────────
    g.set_entry_point("state_builder")
    g.add_edge("state_builder", "scheduler")

    # Scheduler → SEND or WAIT
    g.add_conditional_edges(
        "scheduler",
        lambda s: "send" if s["scheduler_decision"] == "SEND" else "wait",
        {"send": "context_builder_gen", "wait": END},
    )

    g.add_edge("context_builder_gen", "generator")

    if has_reviewer:
        # Generator → pre-reviewer context builder, UNLESS the generator chose
        # to skip for lack of substantive content - end with no message.
        g.add_conditional_edges(
            "generator",
            lambda s: "skip" if not s.get("draft_message") else "review",
            {"skip": END, "review": "context_builder_rev"},
        )
        g.add_edge("context_builder_rev", "reviewer")
        # Reviewer → output (approved) or END (dropped)
        g.add_conditional_edges(
            "reviewer",
            lambda s: "approved" if (s.get("review_approved") and s.get("final_message")) else "drop",
            {"approved": "output", "drop": END},
        )
    else:
        # No reviewer: generator feeds directly to output
        # final_message = draft_message (set inside output_node via state)
        # We need to copy draft → final before output runs.
        # Use a tiny pass-through node so the output_node can read final_message.
        def _passthrough(state: AgentState) -> dict:
            return {"final_message": state.get("draft_message", "")}

        g.add_node("set_final", _passthrough)
        # Skip straight to END on an empty draft.
        g.add_conditional_edges(
            "generator",
            lambda s: "skip" if not s.get("draft_message") else "send",
            {"skip": END, "send": "set_final"},
        )
        g.add_edge("set_final", "output")

    g.add_edge("output", END)

    return g.compile()
