from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.functions import (
    AeoPageState,
    draft_facts,
    verify_facts,
    generate_faq,
    generate_claims,
    verify_claims,
    assemble_page,
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


# ──────────────────────────── Build Graph ────────────────────────────
#
#  [1] draft_facts
#   ↓
#  [2] verify_facts
#   ↓
#  [3] generate_faq
#   ↓
#  [4] generate_claims
#   ↓
#  [5] verify_claims
#   ↓
#  [6] assemble_page
#   ↓
#  [7] build_json_ld
#   ↓
#  [8] quality_gate  ──→  publish_page  ──→  END
#                    └──→  flag_for_review ──→  END

memory = MemorySaver()
workflow = StateGraph(AeoPageState)

workflow.add_node("draft_facts", draft_facts)
workflow.add_node("verify_facts", verify_facts)
workflow.add_node("generate_faq", generate_faq)
workflow.add_node("generate_claims", generate_claims)
workflow.add_node("verify_claims", verify_claims)
workflow.add_node("assemble_page", assemble_page)
workflow.add_node("build_json_ld", build_json_ld)
workflow.add_node("quality_gate", quality_gate)
workflow.add_node("publish_page", publish_page)
workflow.add_node("flag_for_review", flag_for_review)

workflow.set_entry_point("draft_facts")
workflow.add_edge("draft_facts", "verify_facts")
workflow.add_edge("verify_facts", "generate_faq")
workflow.add_edge("generate_faq", "generate_claims")
workflow.add_edge("generate_claims", "verify_claims")
workflow.add_edge("verify_claims", "assemble_page")
workflow.add_edge("assemble_page", "build_json_ld")
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
