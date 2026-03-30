from datetime import datetime, timezone
from typing import TypedDict, Optional, List, Dict, Any

from langchain_openai import ChatOpenAI

from agent.functions import (
    logger,
    llm_strict,
    parse_json_from_llm,
    _llm_call,
    extract_text,
    generate_slug,
)

_search_tool = {"type": "web_search_preview"}

llm_search = ChatOpenAI(
    model="gpt-4.1",
    use_responses_api=True,
).bind_tools([_search_tool])


class CompanySeedState(TypedDict):
    # Inputs
    website_url: str
    linkedin_url: Optional[str]

    # Derived / internal
    timestamp_iso: str
    company_research_raw: str  # LLM narrative from fetch_company_research_raw

    # Outputs (strict schema)
    company: Dict[str, Any]
    brandEntity: Dict[str, Any]
    offerings: List[Dict[str, Any]]
    branding: Optional[Dict[str, Any]]


def _now_midnight_iso() -> str:
    """Return current UTC date at midnight as ISO 8601 with Z suffix."""
    now = datetime.now(timezone.utc)
    midnight = datetime(
        year=now.year,
        month=now.month,
        day=now.day,
        tzinfo=timezone.utc,
    )
    # Match style: 2026-03-05T00:00:00.000Z
    return midnight.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _normalise_company_profile(
    raw: Dict[str, Any],
    timestamp_iso: str,
) -> Dict[str, Any]:
    """Ensure the company profile strictly matches the expected schema."""
    profile: Dict[str, Any] = raw if isinstance(raw, dict) else {}

    company: Dict[str, Any] = profile.get("company") or {}
    brand: Dict[str, Any] = profile.get("brandEntity") or {}
    offerings: List[Dict[str, Any]] = profile.get("offerings") or []
    branding = profile.get("branding")

    # ---- Company ----
    name = company.get("name") or brand.get("canonicalName") or "Unknown Company"
    slug = company.get("slug") or generate_slug(name)
    company_id = company.get("id") or f"{slug}_company"

    company_normalised: Dict[str, Any] = {
        "id": company_id,
        "name": name,
        "slug": slug,
        "description": company.get("description"),
        "logoUrl": company.get("logoUrl"),
        "website": company.get("website"),
        "email": company.get("email"),
        "createdAt": company.get("createdAt") or timestamp_iso,
        "updatedAt": company.get("updatedAt") or timestamp_iso,
    }

    # ---- Brand Entity ----
    brand_id = brand.get("id") or f"{slug}_brand_entity"

    brand_normalised: Dict[str, Any] = {
        "id": brand_id,
        "companyId": brand.get("companyId") or company_id,
        "canonicalName": brand.get("canonicalName") or name,
        "aliases": brand.get("aliases") or [],
        "entityType": brand.get("entityType") or "Organization",
        "oneLiner": brand.get("oneLiner"),
        "about": brand.get("about"),
        "industry": brand.get("industry"),
        "category": brand.get("category"),
        "headquartersCity": brand.get("headquartersCity"),
        "headquartersCountry": brand.get("headquartersCountry"),
        "foundedYear": brand.get("foundedYear"),
        "employeeRange": brand.get("employeeRange"),
        "businessModel": brand.get("businessModel"),
        "topics": brand.get("topics") or [],
        "keywords": brand.get("keywords") or [],
        "targetAudiences": brand.get("targetAudiences") or [],
        "authorityScore": brand.get("authorityScore"),
        "citationCount": brand.get("citationCount") or 0,
        "lastCrawledAt": brand.get("lastCrawledAt"),
        "completenessScore": brand.get("completenessScore") or 0,
        "lastEnrichedAt": brand.get("lastEnrichedAt"),
        "enrichmentSource": brand.get("enrichmentSource") or "website + linkedin",
        "createdAt": brand.get("createdAt") or timestamp_iso,
        "updatedAt": brand.get("updatedAt") or timestamp_iso,
    }

    # ---- Offerings ----
    normalised_offerings: List[Dict[str, Any]] = []
    if not isinstance(offerings, list):
        offerings = []

    if not offerings:
        # Ensure at least one offering shell exists so schema is complete
        offerings = [
            {
                "name": f"{name} Primary Offering",
                "description": None,
                "url": company_normalised.get("website"),
                "isPrimary": True,
            }
        ]

    for idx, off in enumerate(offerings):
        off = off or {}
        off_name = off.get("name") or f"{name} Offering {idx + 1}"
        off_slug = off.get("slug") or generate_slug(off_name)
        off_id = off.get("id") or f"{slug}_offering_{idx + 1}"

        normalised_offerings.append(
            {
                "id": off_id,
                "entityId": off.get("entityId") or brand_id,
                "name": off_name,
                "slug": off_slug,
                "description": off.get("description"),
                "offeringType": off.get("offeringType") or "PRODUCT",
                "url": off.get("url") or company_normalised.get("website"),
                "keywords": off.get("keywords") or [],
                "useCases": off.get("useCases") or [],
                "targetAudiences": off.get("targetAudiences") or [],
                "differentiators": off.get("differentiators") or [],
                "competitors": off.get("competitors") or [],
                "isPrimary": bool(off.get("isPrimary")) if "isPrimary" in off else (idx == 0),
                "isActive": off.get("isActive", True),
                "createdAt": off.get("createdAt") or timestamp_iso,
                "updatedAt": off.get("updatedAt") or timestamp_iso,
            }
        )

    return {
        "company": company_normalised,
        "brandEntity": brand_normalised,
        "offerings": normalised_offerings,
        "branding": branding,
    }


_JSON_SCHEMA_INSTRUCTIONS = """
You MUST return a SINGLE JSON object with the following exact top-level shape:
{{
  "company": {{
    "id": "string",
    "name": "string",
    "slug": "string-kebab-case",
    "description": "string or null",
    "logoUrl": "string or null",
    "website": "string or null",
    "email": "string or null",
    "createdAt": "ISO 8601 timestamp string",
    "updatedAt": "ISO 8601 timestamp string"
  }},
  "brandEntity": {{
    "id": "string",
    "companyId": "string (MUST equal company.id)",
    "canonicalName": "string",
    "aliases": ["string", ...],
    "entityType": "Organization",
    "oneLiner": "short one-line description",
    "about": "multi-sentence description",
    "industry": "high-level industry",
    "category": "more specific category",
    "headquartersCity": "string or null",
    "headquartersCountry": "string or null",
    "foundedYear": 2020,
    "employeeRange": "string, e.g. '51-200'",
    "businessModel": "B2B | B2C | B2B2C | other",
    "topics": ["string", ...],
    "keywords": ["string", ...],
    "targetAudiences": ["string", ...],
    "authorityScore": null,
    "citationCount": 0,
    "lastCrawledAt": null,
    "completenessScore": 0,
    "lastEnrichedAt": null,
    "enrichmentSource": "website + linkedin",
    "createdAt": "ISO 8601 timestamp string",
    "updatedAt": "ISO 8601 timestamp string"
  }},
  "offerings": [
    {{
      "id": "string",
      "entityId": "string (MUST equal brandEntity.id)",
      "name": "string",
      "slug": "string-kebab-case",
      "description": "string or null",
      "offeringType": "PRODUCT" | "SERVICE" | "OTHER",
      "url": "string or null",
      "keywords": ["string", ...],
      "useCases": ["string", ...],
      "targetAudiences": ["string", ...],
      "differentiators": ["string", ...],
      "competitors": ["string", ...],
      "isPrimary": true,
      "isActive": true,
      "createdAt": "ISO 8601 timestamp string",
      "updatedAt": "ISO 8601 timestamp string"
    }}
  ],
  "branding": null
}}

STRICT RULES:
- Follow the field names and nesting EXACTLY.
- Include ALL fields even if some values are null or empty arrays.
- ids and slugs should be consistent and derived from the company/offerings names.
- Do NOT add any extra top-level keys.
- If you are unsure about a field, set it to null (or 0 / [] as appropriate) rather than hallucinating beyond the research memo.
- Return ONLY the JSON object, no backticks or extra text.
"""


def fetch_company_research_raw(state: CompanySeedState) -> CompanySeedState:
    """
    Node 1: Use the LLM to produce a free-form research memo about the company
    (website + optional LinkedIn as anchors). Output is plain text, not JSON.
    """
    node = "COMPANY_SEEDER_FETCH_RESEARCH"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    website_url = state["website_url"]
    linkedin_url = state.get("linkedin_url")

    timestamp_iso = state.get("timestamp_iso") or _now_midnight_iso()
    state["timestamp_iso"] = timestamp_iso

    logger.info("[%s] website_url=%s | linkedin_url=%s", node, website_url, linkedin_url)

    prompt = f"""You are a company research analyst.

Use the URLs below as anchors (browse conceptually / use up-to-date knowledge tied to these domains where possible).
Write a detailed **plain-text research memo** about the organization. Do NOT output JSON.

Cover wherever applicable (use clear section headings in plain text):
- Legal / brand name, aliases, one-line positioning
- What they sell (products, services, modules) with short descriptions
- Target customers, industries, geographies
- Differentiators, competitors mentioned or implied
- Company size signals, HQ location, founding year if known
- Topics and keywords for how buyers would search for them
- Website + contact clues (public email if discoverable; otherwise say unknown)
- Branding notes only if factual (e.g. notable visual identity claims on the site)

WEBSITE URL:
{website_url}

LINKEDIN URL:
{linkedin_url or "N/A"}

Rules:
- Be factual; flag uncertainty in prose rather than inventing specifics.
- No markdown code fences. No JSON. Narrative + headings only.
"""

    import time as _time
    logger.debug("[%s] LLM search call started (%d chars prompt)", node, len(prompt))
    t0 = _time.time()
    try:
        response = llm_search.invoke(prompt)
        elapsed = _time.time() - t0
        raw = extract_text(response)
        if not raw:
            content = getattr(response, "content", None) or []
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text") or block.get("output") or "")
                    else:
                        parts.append(str(block))
                raw = " ".join(p for p in parts if p).strip()
        logger.info("[%s] LLM search call succeeded in %.2fs (%d chars)", node, elapsed, len(raw or ""))
    except Exception as exc:
        elapsed = _time.time() - t0
        logger.error("[%s] LLM search call FAILED after %.2fs: %s", node, elapsed, exc)
        raise

    state["company_research_raw"] = (raw or "").strip()
    logger.info("[%s] Node complete — raw memo length=%d chars", node, len(state["company_research_raw"]))
    return state


def structure_company_profile(state: CompanySeedState) -> CompanySeedState:
    """
    Node 2: Map the research memo into the strict company / brandEntity / offerings JSON.
    """
    node = "COMPANY_SEEDER_STRUCTURE_PROFILE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    website_url = state["website_url"]
    linkedin_url = state.get("linkedin_url")
    research = state.get("company_research_raw") or ""
    timestamp_iso = state.get("timestamp_iso") or _now_midnight_iso()
    state["timestamp_iso"] = timestamp_iso

    prompt = f"""You are a company research and structuring engine.

You will map the research memo below into ONE strict JSON object.
Ground every field in the memo when possible. Use the URLs for canonical website fields.

ANCHOR URLS:
- Website: {website_url}
- LinkedIn: {linkedin_url or "N/A"}

RESEARCH MEMO:
---
{research}
---

{_JSON_SCHEMA_INSTRUCTIONS}
"""

    raw = _llm_call(prompt, node, llm_client=llm_strict)
    parsed = parse_json_from_llm(raw)

    if not isinstance(parsed, dict):
        logger.warning("[%s] LLM returned non-object; using empty profile fallback", node)
        parsed = {}

    profile = _normalise_company_profile(parsed, timestamp_iso=timestamp_iso)

    state["company"] = profile["company"]
    state["brandEntity"] = profile["brandEntity"]
    state["offerings"] = profile["offerings"]
    state["branding"] = profile.get("branding")

    logger.info(
        "[%s] Node complete — company=%s | offerings=%d",
        node,
        state["company"].get("name"),
        len(state["offerings"]),
    )
    return state

