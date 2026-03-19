from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companyRadar.functions import (
    GeoRadarState,
    expand_topics,
    generate_prompts,
    run_prompts,
    parse_responses,
    aggregate_citations,
    compute_metrics,
    build_response,
)


# ──────────────────────────────────────────────────────────────
# Build Graph
#
#  [1] expand_topics
#       ↓
#  [2] generate_prompts
#       ↓
#  [3] run_prompts          ← N prompts × M models LLM calls
#       ↓
#  [4] parse_responses      ← regex-first, LLM fallback
#       ↓
#  [5] aggregate_citations  ← pure computation
#       ↓
#  [6] compute_metrics      ← pure computation
#       ↓
#  [7] build_response
#       ↓
#      END
# ──────────────────────────────────────────────────────────────

memory   = MemorySaver()
workflow = StateGraph(GeoRadarState)

workflow.add_node("expand_topics",        expand_topics)
workflow.add_node("generate_prompts",     generate_prompts)
workflow.add_node("run_prompts",          run_prompts)
workflow.add_node("parse_responses",      parse_responses)
workflow.add_node("aggregate_citations",  aggregate_citations)
workflow.add_node("compute_metrics",      compute_metrics)
workflow.add_node("build_response",       build_response)

workflow.set_entry_point("expand_topics")
workflow.add_edge("expand_topics",       "generate_prompts")
workflow.add_edge("generate_prompts",    "run_prompts")
workflow.add_edge("run_prompts",         "parse_responses")
workflow.add_edge("parse_responses",     "aggregate_citations")
workflow.add_edge("aggregate_citations", "compute_metrics")
workflow.add_edge("compute_metrics",     "build_response")
workflow.add_edge("build_response",      END)

geo_radar_app = workflow.compile(checkpointer=memory)
