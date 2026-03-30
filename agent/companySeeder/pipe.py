from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companySeeder.functions import (
    CompanySeedState,
    fetch_company_research_raw,
    structure_company_profile,
)


memory = MemorySaver()
workflow = StateGraph(CompanySeedState)

workflow.add_node("fetch_company_research_raw", fetch_company_research_raw)
workflow.add_node("structure_company_profile", structure_company_profile)

workflow.set_entry_point("fetch_company_research_raw")
workflow.add_edge("fetch_company_research_raw", "structure_company_profile")
workflow.add_edge("structure_company_profile", END)

company_seeder_app = workflow.compile(checkpointer=memory)

