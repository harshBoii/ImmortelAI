from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from agent import aeo_agent_app
from agent.companySeeder import company_seeder_app
from agent.companyRadar.pipe import geo_radar_app
from agent.companyBounty.pipe import company_bounty_app


app = FastAPI(
    title="Immortel AI API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EntityInput(BaseModel):
    name: str
    oneLiner: Optional[str] = None
    website: Optional[str] = None
    offerings: List[Any] = Field(default_factory=list)
    differentiators: List[Any] = Field(default_factory=list)
    competitors: List[Any] = Field(default_factory=list)


class AeoRequest(BaseModel):
    entity: EntityInput
    intelligence: Dict[str, Any]
    query: str
    page_type: str = "COMPARISON"
    locale: str = "en"
    base_url: str = "https://example.com/aeo"
    same_as_links: List[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    published_at: Optional[str] = None
    existing_slugs: List[str] = Field(default_factory=list)
    session_id: str = "api-session"


class CompanySeedRequest(BaseModel):
    website_url: str
    linkedin_url: Optional[str] = None
    session_id: str = "company-seeder-session"


class RadarCompanyInput(BaseModel):
    name: str
    website: Optional[str] = None
    linkedin: Optional[str] = None


class RadarBrandEntityInput(BaseModel):
    category: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class CompanyRadarRequest(BaseModel):
    company: RadarCompanyInput
    brandEntity: RadarBrandEntityInput
    competitors: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    session_id: str = "company-radar-session"


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/aeo/page")
async def generate_aeo_page(payload: AeoRequest):
    """
    Run the AEO LangGraph pipeline and return the final state.
    """
    state = aeo_agent_app.invoke(
        payload.model_dump(),
        config={"configurable": {"thread_id": payload.session_id}},
    )
    # Return just the assembled page & json-ld plus some diagnostics
    return {
        "status": state.get("status"),
        "slug": state.get("slug"),
        "seo_title": state.get("seo_title"),
        "rejection_reason": state.get("rejection_reason"),
        "page": state.get("page"),
        "drafted_facts_count": len(state.get("drafted_facts", [])),
        "verified_facts_count": len(state.get("verified_facts", [])),
        "faq_count": len(state.get("faq", [])),
        "claims_count": len(state.get("claims", [])),
    }


@app.post("/company/seed")
async def seed_company(payload: CompanySeedRequest):
    """
    Run the Company Seeder LangGraph pipeline to build a structured
    company profile (company, brandEntity, offerings, branding).
    """
    state = company_seeder_app.invoke(
        {
            "website_url": payload.website_url,
            "linkedin_url": payload.linkedin_url,
            "timestamp_iso": None,
            "company": {},
            "brandEntity": {},
            "offerings": [],
            "branding": None,
        },
        config={"configurable": {"thread_id": payload.session_id}},
    )

    return {
        "company": state.get("company"),
        "brandEntity": state.get("brandEntity"),
        "offerings": state.get("offerings"),
        "branding": state.get("branding"),
    }


@app.post("/company/radar")
async def company_radar(payload: CompanyRadarRequest):
    """
    Run the Company Radar pipeline to compute topic coverage,
    prompts, citations and metrics for a brand and its competitors.
    """
    state = geo_radar_app.invoke(
        {
            "company": payload.company.model_dump(),
            "brand_entity": payload.brandEntity.model_dump(),
            "competitors": payload.competitors,
            "models": payload.models,
            "topics": [],
            "prompts": [],
            "raw_responses": [],
            "citations": [],
            "aggregated": {},
            "metrics": {},
            "result": {},
            "session_id": payload.session_id,
        },
        config={"configurable": {"thread_id": payload.session_id}},
    )

    # GeoRadarState.build_response stores the final payload in state["result"]
    result = state.get("result") or {}
    return {
        "topics": result.get("topics", []),
        "prompts": result.get("prompts", []),
        "citations": result.get("citations", []),
        "metrics": result.get("metrics", {}),
    }


class CompanyBountyRequest(BaseModel):
    company: RadarCompanyInput
    brandEntity: RadarBrandEntityInput
    competitors: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    session_id: str = "company-bounty-session"


@app.post("/company/bounty")
async def company_bounty(payload: CompanyBountyRequest):
    """
    Run the Company Bounty pipeline to discover niche topics
    and generate AEO-focused prompts for each.
    """
    state = company_bounty_app.invoke(
        {
            "company": payload.company.model_dump(),
            "brand_entity": payload.brandEntity.model_dump(),
            "competitors": payload.competitors,
            "models": payload.models,
            "niches": [],
            "niche_prompts": [],
            "result": {},
            "session_id": payload.session_id,
        },
        config={"configurable": {"thread_id": payload.session_id}},
    )

    return state.get("result") or {}


@app.get("/")
async def root():
    return {"message": "Immortel AI FastAPI server is running"}
