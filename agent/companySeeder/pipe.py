import json
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.companySeeder.functions import (
    CompanySeedState,
    fetch_company_research_raw,
    structure_company_profile,
)

logger = logging.getLogger("company_seeder_pipeline")


memory = MemorySaver()
workflow = StateGraph(CompanySeedState)

workflow.add_node("fetch_company_research_raw", fetch_company_research_raw)
workflow.add_node("structure_company_profile", structure_company_profile)

workflow.set_entry_point("fetch_company_research_raw")
workflow.add_edge("fetch_company_research_raw", "structure_company_profile")
workflow.add_edge("structure_company_profile", END)

company_seeder_app = workflow.compile(checkpointer=memory)


# --- Example input (POST /company/seed) ---
# {
#   "website_url": "https://acme.com",
#   "linkedin_url": "https://www.linkedin.com/company/acme",
#   "session_id": "company-seeder-session"
# }
#
# Graph invoke state (internal):
# {
#   "website_url": "https://acme.com",
#   "linkedin_url": "https://www.linkedin.com/company/acme",
#   "timestamp_iso": null,
#   "company_research_raw": "",
#   "company": {},
#   "brandEntity": {},
#   "offerings": [],
#   "branding": null
# }
#
# --- Example output (HTTP response) ---
# {
#   "company": {
#     "id": "acme_company",
#     "name": "Acme Inc",
#     "slug": "acme-inc",
#     "description": "B2B SaaS for operations teams.",
#     "logoUrl": "https://acme.com/logo.png",
#     "website": "https://acme.com",
#     "email": "hello@acme.com",
#     "createdAt": "2026-03-05T00:00:00.000Z",
#     "updatedAt": "2026-03-05T00:00:00.000Z"
#   },
#   "brandEntity": {
#     "id": "acme-inc_brand_entity",
#     "companyId": "acme_company",
#     "canonicalName": "Acme Inc",
#     "aliases": ["Acme"],
#     "entityType": "Organization",
#     "oneLiner": "Operations software for growing teams.",
#     "about": "Acme builds workflow automation for mid-market companies.",
#     "industry": "Software",
#     "category": "B2B SaaS",
#     "headquartersCity": "San Francisco",
#     "headquartersCountry": "USA",
#     "foundedYear": 2018,
#     "employeeRange": "51-200",
#     "businessModel": "B2B",
#     "topics": ["workflow automation", "operations"],
#     "keywords": ["saas", "automation"],
#     "targetAudiences": ["operations managers"],
#     "authorityScore": null,
#     "citationCount": 0,
#     "lastCrawledAt": null,
#     "completenessScore": 0,
#     "lastEnrichedAt": null,
#     "enrichmentSource": "website + linkedin",
#     "createdAt": "2026-03-05T00:00:00.000Z",
#     "updatedAt": "2026-03-05T00:00:00.000Z"
#   },
#   "offerings": [
#     {
#       "id": "acme-inc_offering_1",
#       "entityId": "acme-inc_brand_entity",
#       "name": "Acme Platform",
#       "slug": "acme-platform",
#       "description": "Core workflow automation product.",
#       "offeringType": "PRODUCT",
#       "url": "https://acme.com/platform",
#       "keywords": ["automation"],
#       "useCases": ["process automation"],
#       "targetAudiences": ["operations teams"],
#       "differentiators": ["no-code setup"],
#       "competitors": ["CompetitorX"],
#       "isPrimary": true,
#       "isActive": true,
#       "createdAt": "2026-03-05T00:00:00.000Z",
#       "updatedAt": "2026-03-05T00:00:00.000Z"
#     }
#   ],
#   "branding": null
# }

