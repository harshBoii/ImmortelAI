from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companyBounty.functions import (
    BountyState,
    discover_niches,
    generate_niche_prompts,
    run_prompts,
    parse_responses,
    build_response,
)


# ──────────────────────────────────────────────────────────────
# Build Graph
#
#  [1] discover_niches        ← LLM: identify niche topics
#       ↓
#  [2] generate_niche_prompts ← LLM: N prompts per niche
#       ↓
#  [3] run_prompts            ← LLM calls for each prompt/model
#       ↓
#  [4] parse_responses         ← LLM parses cited companies/ranks
#       ↓
#  [5] build_response          ← assemble final output
#       ↓
#      END
# ──────────────────────────────────────────────────────────────

memory   = MemorySaver()
workflow = StateGraph(BountyState)

workflow.add_node("discover_niches",        discover_niches)
workflow.add_node("generate_niche_prompts", generate_niche_prompts)
workflow.add_node("run_prompts",            run_prompts)
workflow.add_node("parse_responses",        parse_responses)
workflow.add_node("build_response",         build_response)

workflow.set_entry_point("discover_niches")
workflow.add_edge("discover_niches",        "generate_niche_prompts")
workflow.add_edge("generate_niche_prompts", "run_prompts")
workflow.add_edge("run_prompts",            "parse_responses")
workflow.add_edge("parse_responses",        "build_response")
workflow.add_edge("build_response",         END)

company_bounty_app = workflow.compile(checkpointer=memory)
