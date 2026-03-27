from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companyRadar.functions import (
    GeoRadarState,
    logger,
    use_api_topics,
    expand_topics,
    analyze_company_context,
    generate_prompts,
    run_prompts,
    parse_responses,
    aggregate_citations,
    compute_metrics,
    build_response,
)


def topics_source_router(state: GeoRadarState) -> str:
    """
    Route topic source:
    - use_api_topics when API sends llmTopics
    - expand_topics otherwise
    """
    llm_topics = state.get("llm_topics", []) or []
    if llm_topics:
        logger.info("[TOPICS_SOURCE_ROUTER] API llmTopics detected -> use_api_topics")
        return "use_api_topics"
    logger.info("[TOPICS_SOURCE_ROUTER] No API llmTopics -> expand_topics")
    return "expand_topics"


def route_topics(state: GeoRadarState) -> GeoRadarState:
    """No-op node used as graph entry for conditional topic routing."""
    return state


# ──────────────────────────────────────────────────────────────
# Build Graph
#
#  [0] topics_source_router
#       ├─→ [1a] use_api_topics   (if llmTopics provided)
#       └─→ [1b] expand_topics    (fallback)
#                     ↓
#  [2] analyze_company_context
#       ↓
#  [3] generate_prompts
#       ↓
#  [4] run_prompts          ← N prompts × M models LLM calls
#       ↓
#  [5] parse_responses      ← regex-first, LLM fallback
#       ↓
#  [6] aggregate_citations  ← pure computation
#       ↓
#  [7] compute_metrics      ← pure computation
#       ↓
#  [8] build_response
#       ↓
#      END
# ──────────────────────────────────────────────────────────────

memory   = MemorySaver()
workflow = StateGraph(GeoRadarState)

workflow.add_node("route_topics",         route_topics)
workflow.add_node("use_api_topics",       use_api_topics)
workflow.add_node("expand_topics",        expand_topics)
workflow.add_node("analyze_company_context", analyze_company_context)
workflow.add_node("generate_prompts",     generate_prompts)
workflow.add_node("run_prompts",          run_prompts)
workflow.add_node("parse_responses",      parse_responses)
workflow.add_node("aggregate_citations",  aggregate_citations)
workflow.add_node("compute_metrics",      compute_metrics)
workflow.add_node("build_response",       build_response)

workflow.set_entry_point("route_topics")
workflow.add_conditional_edges(
    "route_topics",
    topics_source_router,
    {
        "use_api_topics": "use_api_topics",
        "expand_topics": "expand_topics",
    },
)
workflow.add_edge("use_api_topics",      "analyze_company_context")
workflow.add_edge("expand_topics",       "analyze_company_context")
workflow.add_edge("analyze_company_context", "generate_prompts")
workflow.add_edge("generate_prompts",    "run_prompts")
workflow.add_edge("run_prompts",         "parse_responses")
workflow.add_edge("parse_responses",     "aggregate_citations")
workflow.add_edge("aggregate_citations", "compute_metrics")
workflow.add_edge("compute_metrics",     "build_response")
workflow.add_edge("build_response",      END)

geo_radar_app = workflow.compile(checkpointer=memory)
