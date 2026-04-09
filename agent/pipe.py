from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.functions import (
    AeoPageState,
    keyword_research,
    duplicate_check,
    draft_facts,
    verify_facts,
    generate_faq,
    generate_claims,
    verify_claims,
    assemble_page,
    build_internal_links,
    build_json_ld,
    quality_gate,
    publish_page,
    flag_for_review,
)


def quality_gate_router(state: AeoPageState) -> str:
    """Route to publish or review based on quality gate outcome."""
    if state.get("status") == "PUBLISHED":
        return "publish_page"
    return "flag_for_review"


def duplicate_router(state: AeoPageState) -> str:
    """Early-exit if hard duplicate detected."""
    if state.get("duplicate_status") == "DUPLICATE":
        return "end"
    return "continue"


# ──────────────────────────── Build Graph ────────────────────────────
#
#  [1] keyword_research
#   ↓
#  [2] duplicate_check  ──→ END (if DUPLICATE)
#   ↓
#  [3] draft_facts
#   ↓
#  [4] verify_facts
#   ↓
#  [5] generate_faq
#   ↓
#  [6] generate_claims
#   ↓
#  [7] verify_claims
#   ↓
#  [8] assemble_page
#   ↓
#  [9] build_internal_links
#   ↓
# [10] build_json_ld
#   ↓
# [11] quality_gate  ──→  publish_page  ──→  END
#                    └──→  flag_for_review ──→  END

memory = MemorySaver()
workflow = StateGraph(AeoPageState)

workflow.add_node("keyword_research", keyword_research)
workflow.add_node("duplicate_check", duplicate_check)
workflow.add_node("draft_facts", draft_facts)
workflow.add_node("verify_facts", verify_facts)
workflow.add_node("generate_faq", generate_faq)
workflow.add_node("generate_claims", generate_claims)
workflow.add_node("verify_claims", verify_claims)
workflow.add_node("assemble_page", assemble_page)
workflow.add_node("build_internal_links", build_internal_links)
workflow.add_node("build_json_ld", build_json_ld)
workflow.add_node("quality_gate", quality_gate)
workflow.add_node("publish_page", publish_page)
workflow.add_node("flag_for_review", flag_for_review)

workflow.set_entry_point("keyword_research")
workflow.add_edge("keyword_research", "duplicate_check")

workflow.add_conditional_edges(
    "duplicate_check",
    duplicate_router,
    {
        "end": END,
        "continue": "draft_facts",
    },
)

workflow.add_edge("draft_facts", "verify_facts")
workflow.add_edge("verify_facts", "generate_faq")
workflow.add_edge("generate_faq", "generate_claims")
workflow.add_edge("generate_claims", "verify_claims")
workflow.add_edge("verify_claims", "assemble_page")
workflow.add_edge("assemble_page", "build_internal_links")
workflow.add_edge("build_internal_links", "build_json_ld")
workflow.add_edge("build_json_ld", "quality_gate")

workflow.add_conditional_edges(
    "quality_gate",
    quality_gate_router,
    {
        "publish_page": "publish_page",
        "flag_for_review": "flag_for_review",
    },
)

workflow.add_edge("publish_page", END)
workflow.add_edge("flag_for_review", END)

aeo_agent_app = workflow.compile(checkpointer=memory)
