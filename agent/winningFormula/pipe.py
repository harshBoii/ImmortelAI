from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.winningFormula.functions import (
    WinningFormulaState,
    normalize_request,
    build_winning_formula,
    post_result_to_webhook,
)


memory = MemorySaver()
workflow = StateGraph(WinningFormulaState)

workflow.add_node("normalize_request", normalize_request)
workflow.add_node("build_winning_formula", build_winning_formula)
workflow.add_node("post_result_to_webhook", post_result_to_webhook)

workflow.set_entry_point("normalize_request")
workflow.add_edge("normalize_request", "build_winning_formula")
workflow.add_edge("build_winning_formula", "post_result_to_webhook")
workflow.add_edge("post_result_to_webhook", END)

winning_formula_app = workflow.compile(checkpointer=memory)

