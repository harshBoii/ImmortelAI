import json
from datetime import datetime, timezone
from typing import TypedDict, Optional, List, Dict, Any

from agent.functions import (
    logger,
    llm,
    parse_json_from_llm,
    _llm_call,
    generate_slug,
)


class CompanySeedState(TypedDict):
    # Inputs
    website_url: str
    linkedin_url: Optional[str]

    # Derived / internal
    timestamp_iso: str

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


def build_company_profile(state: CompanySeedState) -> CompanySeedState:
    """
    Single-node pipeline:
    Use the LLM to research a company from its website and LinkedIn URL
    and return a strictly structured profile.
    """
    node = "COMPANY_SEEDER_BUILD_PROFILE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    website_url = state["website_url"]
    linkedin_url = state.get("linkedin_url")

    timestamp_iso = state.get("timestamp_iso") or _now_midnight_iso()
    state["timestamp_iso"] = timestamp_iso

    logger.info("[%s] website_url=%s | linkedin_url=%s", node, website_url, linkedin_url)

    prompt = f"""You are a company research and structuring engine.

You are given the company's primary website URL and (optionally) their LinkedIn company page.
Using your own knowledge and these URLs as anchors, infer a concise but accurate profile.

WEBSITE URL:
- {website_url}

LINKEDIN URL:
- {linkedin_url or "N/A"}

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
- If you are unsure about a field, set it to null (or 0 / [] as appropriate) rather than hallucinating.
- Return ONLY the JSON object, no backticks or extra text.
"""

    raw = _llm_call(prompt, node, llm_client=llm)
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

