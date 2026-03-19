from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companySeeder.functions import CompanySeedState, build_company_profile


memory = MemorySaver()
workflow = StateGraph(CompanySeedState)

workflow.add_node("build_company_profile", build_company_profile)
workflow.set_entry_point("build_company_profile")
workflow.add_edge("build_company_profile", END)

company_seeder_app = workflow.compile(checkpointer=memory)

