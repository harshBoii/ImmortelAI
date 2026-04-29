from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from agent import aeo_agent_app
from agent.functions import preflight
from agent.companySeeder import company_seeder_app
from agent.companyRadar.functions import build_company_radar_api_response
from agent.companyRadar.pipe import geo_radar_app
from agent.companyBounty.pipe import company_bounty_app
from agent.winningFormula import winning_formula_app


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
    products: List[Any] = Field(default_factory=list)
    differentiators: List[Any] = Field(default_factory=list)
    competitors: List[Any] = Field(default_factory=list)


class AeoRequest(BaseModel):
    entity: EntityInput
    intelligence: Dict[str, Any]
    query: str

    topic: Optional[str] = None
    topic_pages: List[str] = Field(default_factory=list)
    topic_page_names: List[str] 
    
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
    about: Optional[str] = None


class RadarOfferingInput(BaseModel):
    product: str
    productType: Optional[str] = None
    url: Optional[str] = None
    differentiators: List[str] = Field(default_factory=list)
    useCases: List[str] = Field(default_factory=list)
    targetAudiences: List[str] = Field(default_factory=list)
    competitorGroups: List[str] = Field(default_factory=list)


class RadarBrandEntityInput(BaseModel):
    category: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    product: Optional[str] = None
    productType: Optional[str] = None
    url: Optional[str] = None
    differentiators: List[str] = Field(default_factory=list)
    useCases: List[str] = Field(default_factory=list)
    targetAudiences: List[str] = Field(default_factory=list)
    competitorGroups: List[str] = Field(default_factory=list)
    offerings: List[RadarOfferingInput] = Field(default_factory=list)


class CompanyRadarRequest(BaseModel):
    company: RadarCompanyInput
    brandEntity: RadarBrandEntityInput
    llmTopics: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    session_id: str = "company-radar-session"
    webhookUrl: Optional[str] = None


class MetaAsset(BaseModel):
    id: str
    asset_type: str  # VIDEO|IMAGE
    title: str
    filename: str
    intelligence_status: Optional[str] = None  # READY|PROCESSING|FAILED|null


class MetaAssetIntelligence(BaseModel):
    id: Optional[str] = None
    asset_id: str
    company_id: str
    processed_at: Optional[datetime] = None
    language: Optional[str] = None
    content_type: Optional[str] = None
    duration_seconds: Optional[int] = 0
    theme: Optional[str] = None
    sentiment: Optional[str] = None
    intensity_score: Optional[float] = 0
    spiritual_elements: bool = False
    title_primary: Optional[str] = None
    short_summary: Optional[str] = None
    long_description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    tone: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    target_audience: List[str] = Field(default_factory=list)
    best_platforms: List[str] = Field(default_factory=list)
    visual_context: List[str] = Field(default_factory=list)
    video_genres: List[str] = Field(default_factory=list)
    title_variants: Dict[str, Any] = Field(default_factory=dict)
    chapters: List[Any] = Field(default_factory=list)
    shorts_hooks: List[Any] = Field(default_factory=list)
    clipfox_insights: List[Any] = Field(default_factory=list)
    model_version: Optional[str] = None
    confidence: Optional[float] = 0


class MetaMedia(BaseModel):
    id: str


class MetaAdMetricsLatest(BaseModel):
    recorded_at: Optional[datetime] = None
    date_preset: Optional[str] = None
    impressions: int = 0
    clicks: int = 0
    ctr: float = 0
    spend: float = 0
    cpc: Optional[float] = None
    roas: Optional[float] = None


class WinningFormulaItem(BaseModel):
    asset: MetaAsset
    asset_intelligence: Optional[MetaAssetIntelligence] = None
    meta_media: MetaMedia
    meta_ad_metrics_latest: MetaAdMetricsLatest


class WinningFormulaRequest(BaseModel):
    company_id: str
    meta_integration_id: str
    generated_at: datetime
    items: List[WinningFormulaItem]
    webhook_url: Optional[str] = None


class WinningFormulaJobAck(BaseModel):
    ok: bool
    job_id: str
    company_id: str
    meta_integration_id: str
    generated_at: datetime
    input_summary: Dict[str, Any]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/aeo/page")
async def generate_aeo_page(payload: AeoRequest):
    """
    Run the AEO LangGraph pipeline and return the final state.
    """
    state_in = preflight(payload.model_dump())
    state = aeo_agent_app.invoke(
        state_in,
        config={"configurable": {"thread_id": payload.session_id}},
    )
    # Return just the assembled page & json-ld plus some diagnostics

    response = {        
        "status": state.get("status"),
        "slug": state.get("slug"),
        "seo_title": state.get("seo_title"),
        "rejection_reason": state.get("rejection_reason"),
        "duplicate_status": state.get("duplicate_status"),
        "duplicate_reason": state.get("duplicate_reason"),
        "page": state.get("page"),
        "drafted_facts_count": len(state.get("drafted_facts", [])),
        "verified_facts_count": len(state.get("verified_facts", [])),
        "faq_count": len(state.get("faq", [])),
        "claims_count": len(state.get("claims", [])),
        "prompt": payload.query,
    }

    print("Response from AEO Page :   ", response)
    return response


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
            "company_research_raw": "",
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
            "llm_topics": payload.llmTopics,
            "competitors": payload.competitors,
            "models": payload.models,
            "webhook_url": (payload.webhookUrl or "").strip(),
            "topics": [],
            "prompts": [],
            "raw_responses": [],
            "citations": [],
            "aggregated": {},
            "metrics": {},
            "result": {},
            "webhook_delivery": {},
            "session_id": payload.session_id,
        },
        config={"configurable": {"thread_id": payload.session_id}},
    )

    # GeoRadarState.build_response stores the final payload in state["result"]
    res=build_company_radar_api_response(state)
    print("Response from Company Radar :   ", res)
    return res


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

    print("state", state)
    return state.get("result") or {}


def _run_winning_formula_job(job_id: str, payload: WinningFormulaRequest) -> None:
    """
    Background runner for the winning formula LangGraph pipeline.
    """
    state_in = {
        "job_id": job_id,
        "company_id": payload.company_id,
        "meta_integration_id": payload.meta_integration_id,
        "generated_at": payload.generated_at.isoformat(),
        "items": [item.model_dump() for item in payload.items],
        "webhook_url": (payload.webhook_url or "").strip(),
    }
    # meta_id defaults to meta_integration_id as per plan assumptions
    state_in["meta_id"] = payload.meta_integration_id

    winning_formula_app.invoke(
        state_in,
        config={"configurable": {"thread_id": job_id}},
    )


@app.post(
    "/winning-formula/from-meta-analyzed-assets",
    response_model=WinningFormulaJobAck,
)
async def enqueue_winning_formula(
    payload: WinningFormulaRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accept Meta analyzed assets and enqueue a background job
    that builds the winningFormula and posts to the webhook.
    """
    job_id = str(uuid.uuid4())

    background_tasks.add_task(_run_winning_formula_job, job_id, payload)

    input_summary = {
        "items_count": len(payload.items),
        "asset_ids": [item.asset.id for item in payload.items],
    }

    return WinningFormulaJobAck(
        ok=True,
        job_id=job_id,
        company_id=payload.company_id,
        meta_integration_id=payload.meta_integration_id,
        generated_at=payload.generated_at,
        input_summary=input_summary,
    )


@app.get("/")
async def root():
    return {"message": "Immortel AI FastAPI server is running"}

@app.post("/webhook")
async def telnyx_webhook(request: Request):
    # Telnyx call control events — handle later if needed
    return {"ok": True}