import uuid
import json
import re
import time
import logging
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# ──────────────────────────── Logger ────────────────────────────

logger = logging.getLogger("aeo_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# ──────────────────────────── LLM ────────────────────────────

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Generation model (cheaper, used for drafting / content)
llm = ChatOpenAI(model="gpt-4o-mini")

# Guard-rail model (more accurate, used for verification steps)
llm_strict = ChatOpenAI(model="gpt-4o")


# ──────────────────────────── State ────────────────────────────

class AeoPageState(TypedDict):
    entity: dict
    intelligence: dict
    query: str
    page_type: str
    locale: str
    base_url: str
    same_as_links: list
    cluster_id: str | None
    published_at: str | None

    drafted_facts: list
    verified_facts: list
    faq: list
    claims: list
    verified_claims: list

    page: dict
    json_ld: dict
    slug: str
    seo_title: str
    status: str
    rejection_reason: str

    session_id: str
    existing_slugs: list


# ──────────────────────────── Helpers ────────────────────────────

def extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()


def parse_json_from_llm(raw: str):
    """Best-effort extraction of a JSON array or object from LLM output."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{]", cleaned)
        if match:
            try:
                return json.loads(cleaned[match.start():])
            except json.JSONDecodeError:
                pass
    return []


def generate_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or f"page-{uuid.uuid4().hex[:8]}"


def _llm_call(prompt: str, node_name: str, llm_client: ChatOpenAI | None = None) -> str:
    """Wraps an LLM invocation with timing and error logging."""
    client = llm_client or llm
    logger.debug("[%s] LLM call started (%d chars prompt)", node_name, len(prompt))
    t0 = time.time()
    try:
        response = client.invoke(prompt)
        raw = extract_text(response)
        elapsed = time.time() - t0
        logger.info("[%s] LLM call succeeded in %.2fs (%d chars response)", node_name, elapsed, len(raw))
        logger.debug("[%s] Raw LLM response:\n%s", node_name, raw[:500])
        return raw
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[%s] LLM call FAILED after %.2fs: %s", node_name, elapsed, e)
        raise


# ──────────────────────────── Node 1: DRAFT FACTS ────────────────────────────

def draft_facts(state: AeoPageState) -> AeoPageState:
    node = "DRAFT_FACTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    entity = state["entity"]
    intelligence = state["intelligence"]
    query = state["query"]
    source_keys = list(intelligence.keys())

    logger.info("[%s] Entity: %s | Query: %s", node, entity.get("name"), query)
    logger.info("[%s] Intelligence source keys: %s", node, source_keys)

    prompt = f"""You are a rigorous fact-extraction engine.

INPUT ENTITY:
{json.dumps(entity, indent=2)}

INTELLIGENCE CONTEXT (source-keyed):
{json.dumps(intelligence, indent=2)}

QUERY: {query}

TASK:
Extract every factual statement relevant to the query from the intelligence context.
Each fact MUST cite which source key from the intelligence context it came from.

Return a JSON array where each element is:
{{
  "fact": "<factual statement>",
  "source": "<key from intelligence context>"
}}

RULES:
- Only use information present in the intelligence context.
- Every fact MUST have a "source" that matches one of these keys: {source_keys}
- Do NOT invent or hallucinate facts beyond the provided context.
- Return ONLY the JSON array, no extra text.
"""
    raw = _llm_call(prompt, node)
    facts = parse_json_from_llm(raw)
    logger.debug("[%s] Parsed %d total facts from LLM", node, len(facts) if isinstance(facts, list) else 0)

    valid_facts = [f for f in facts if isinstance(f, dict) and f.get("source") in source_keys]
    dropped = len(facts) - len(valid_facts) if isinstance(facts, list) else 0
    if dropped:
        logger.warning("[%s] Dropped %d facts with invalid/missing source key", node, dropped)

    state["drafted_facts"] = valid_facts
    logger.info("[%s] Node complete — %d source-attributed facts drafted", node, len(valid_facts))
    return state


# ──────────────────────────── Node 2: VERIFY FACTS ────────────────────────────

def verify_facts(state: AeoPageState) -> AeoPageState:
    node = "VERIFY_FACTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    drafted = state["drafted_facts"]
    intelligence = state["intelligence"]

    logger.info("[%s] %d drafted facts to verify", node, len(drafted))

    prompt = f"""You are a fact-verification engine. You must be STRICT.

ORIGINAL CONTEXT:
{json.dumps(intelligence, indent=2)}

DRAFTED FACTS:
{json.dumps(drafted, indent=2)}

TASK:
For each drafted fact:
1. Verify it is actually supported by the original context under the cited source key.
2. Assign a confidence score from 0.0 to 1.0.
3. Flag any hallucination (fact not grounded in context) with confidence 0.0.

Return a JSON array where each element is:
{{
  "fact": "<the fact>",
  "source": "<source key>",
  "confidence": <float 0.0-1.0>,
  "flagged": <true if hallucinated, false otherwise>
}}

Return ONLY the JSON array.
"""
    raw = _llm_call(prompt, node, llm_client=llm_strict)
    scored = parse_json_from_llm(raw)

    verified = [
        f for f in scored
        if isinstance(f, dict) and f.get("confidence", 0) >= 0.7 and not f.get("flagged", True)
    ]

    dropped_low_conf = [f for f in scored if isinstance(f, dict) and f.get("confidence", 0) < 0.7]
    dropped_flagged = [f for f in scored if isinstance(f, dict) and f.get("flagged")]
    if dropped_low_conf:
        logger.warning("[%s] %d facts dropped (confidence < 0.7):", node, len(dropped_low_conf))
        for f in dropped_low_conf:
            logger.warning("  - [conf=%.2f] %s", f.get("confidence", 0), f.get("fact", "")[:100])
    if dropped_flagged:
        logger.warning("[%s] %d facts dropped (flagged as hallucination):", node, len(dropped_flagged))
        for f in dropped_flagged:
            logger.warning("  - %s", f.get("fact", "")[:100])

    state["verified_facts"] = verified
    logger.info("[%s] Node complete — %d/%d facts verified (confidence >= 0.7)", node, len(verified), len(drafted))
    return state


# ──────────────────────────── Node 3: GENERATE FAQ ────────────────────────────

def generate_faq(state: AeoPageState) -> AeoPageState:
    node = "GENERATE_FAQ"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    verified = state["verified_facts"]
    query = state["query"]
    entity = state["entity"]
    offerings = entity.get("offerings", [])

    logger.info("[%s] %d verified facts as input | %d offerings", node, len(verified), len(offerings))

    prompt = f"""You are an FAQ generation engine for Answer-Engine-Optimised content.

QUERY: {query}
ENTITY: {json.dumps(entity, indent=2)}
OFFERINGS: {json.dumps(offerings, indent=2)}
VERIFIED FACTS:
{json.dumps(verified, indent=2)}

TASK:
Generate FAQ items in schema.org FAQPage format.
Each answer MUST be derivable from the verified facts above — do NOT add new information.

Return a JSON array:
[
  {{
    "@type": "Question",
    "name": "<question text>",
    "acceptedAnswer": {{
      "@type": "Answer",
      "text": "<answer derived from verified facts>"
    }}
  }}
]

Aim for 3-6 high-quality FAQ items. Return ONLY the JSON array.
"""
    raw = _llm_call(prompt, node)
    faq = parse_json_from_llm(raw)

    state["faq"] = faq if isinstance(faq, list) else []
    logger.info("[%s] Node complete — %d FAQ items generated", node, len(state["faq"]))
    for i, item in enumerate(state["faq"]):
        logger.debug("[%s]   FAQ #%d: %s", node, i + 1, item.get("name", "?")[:80])
    return state


# ──────────────────────────── Node 4: GENERATE CLAIMS ────────────────────────────

def generate_claims(state: AeoPageState) -> AeoPageState:
    node = "GENERATE_CLAIMS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    page_type = state.get("page_type", "").upper()

    if page_type not in ("COMPARISON", "USE_CASE"):
        logger.info("[%s] Skipped — page_type=%s (only COMPARISON/USE_CASE trigger claims)", node, page_type)
        state["claims"] = []
        return state

    verified = state["verified_facts"]
    entity = state["entity"]
    differentiators = entity.get("differentiators", [])
    competitors = entity.get("competitors", [])

    logger.info("[%s] page_type=%s | %d differentiators | %d competitors",
                node, page_type, len(differentiators), len(competitors))

    prompt = f"""You are a claims generation engine for competitive / use-case content.

ENTITY: {json.dumps(entity, indent=2)}
DIFFERENTIATORS: {json.dumps(differentiators, indent=2)}
COMPETITORS: {json.dumps(competitors, indent=2)}
VERIFIED FACTS:
{json.dumps(verified, indent=2)}
PAGE TYPE: {page_type}

TASK:
Generate factual, defensible comparison or use-case claims grounded in the verified facts.
Each claim must reference at least one differentiator or competitor.

STRICT RULES:
- NEVER claim that a competitor LACKS a feature unless the intelligence context explicitly states that absence.
- Prefer positive claims about what the entity HAS, not what competitors do NOT have.
- Every claim must be supportable by at least two distinct verified facts.
- All claims must match the page type:
  - If PAGE TYPE is COMPARISON, only emit type=\"COMPARISON\" claims.
  - If PAGE TYPE is USE_CASE, only emit type=\"USE_CASE\" claims.

Return a JSON array:
[
  {{
    "claim": "<claim statement>",
    "type": "COMPARISON" | "USE_CASE",
    "references": ["<differentiator or competitor name>"],
    "supporting_facts": ["<fact text used>"]
  }}
]

Return ONLY the JSON array.
"""
    raw = _llm_call(prompt, node)
    claims = parse_json_from_llm(raw)

    state["claims"] = claims if isinstance(claims, list) else []
    logger.info("[%s] Node complete — %d claims generated", node, len(state["claims"]))
    for i, c in enumerate(state["claims"]):
        logger.debug("[%s]   Claim #%d: %s", node, i + 1, c.get("claim", "?")[:80])
    return state


# ──────────────────────────── Node 5: VERIFY CLAIMS ────────────────────────────

def verify_claims(state: AeoPageState) -> AeoPageState:
    node = "VERIFY_CLAIMS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    claims = state.get("claims", [])
    verified_facts = state["verified_facts"]

    if not claims:
        logger.info("[%s] No claims to verify — skipping", node)
        state["verified_claims"] = []
        return state

    logger.info("[%s] %d claims to verify against %d verified facts", node, len(claims), len(verified_facts))

    prompt = f"""You are a claim-verification engine. You must be STRICT.

VERIFIED FACTS (ground truth):
{json.dumps(verified_facts, indent=2)}

CLAIMS TO VERIFY:
{json.dumps(claims, indent=2)}

TASK:
For each claim, check whether its "supporting_facts" actually entail the "claim" statement.
A claim is valid ONLY if the supporting facts logically support it without requiring
information beyond what is stated.

Return a JSON array where each element is:
{{
  "claim": "<the claim>",
  "type": "<COMPARISON or USE_CASE>",
  "references": ["<original references>"],
  "supporting_facts": ["<original supporting facts>"],
  "entailed": <true if supporting facts entail the claim, false otherwise>,
  "confidence": <float 0.0-1.0>,
  "reason": "<brief explanation>"
}}

Return ONLY the JSON array.
"""
    raw = _llm_call(prompt, node, llm_client=llm_strict)
    scored = parse_json_from_llm(raw)

    verified = [
        c for c in scored
        if isinstance(c, dict) and c.get("entailed") and c.get("confidence", 0) >= 0.7
    ]

    dropped = [c for c in scored if isinstance(c, dict) and (not c.get("entailed") or c.get("confidence", 0) < 0.7)]
    if dropped:
        logger.warning("[%s] %d claims dropped (not entailed or low confidence):", node, len(dropped))
        for c in dropped:
            logger.warning("  ✗ [conf=%.2f, entailed=%s] %s — %s",
                           c.get("confidence", 0), c.get("entailed"), c.get("claim", "")[:80], c.get("reason", ""))

    state["verified_claims"] = verified
    logger.info("[%s] Node complete — %d/%d claims verified", node, len(verified), len(claims))
    return state


# ──────────────────────────── Node 6: ASSEMBLE PAGE ────────────────────────────

def assemble_page(state: AeoPageState) -> AeoPageState:
    node = "ASSEMBLE_PAGE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    entity = state["entity"]
    query = state["query"]
    verified_facts = state["verified_facts"]
    faq = state["faq"]
    claims = state.get("verified_claims", [])
    products = entity.get("products", [])
    page_type = state.get("page_type", "DEFINITION")
    existing_slugs = state.get("existing_slugs", [])

    # Slug based on entity name + query (session_id is NOT used for slugs)
    entity_name = entity.get("name", "")
    entity_prefix = generate_slug(entity_name)
    query_slug = generate_slug(query)
    base_slug = query_slug if query_slug.startswith(entity_prefix) else f"{entity_prefix}-{query_slug}"
    slug = base_slug
    counter = 2
    while slug in existing_slugs:
        slug = f"{base_slug}-{counter}"
        counter += 1

    logger.info("[%s] Slug: %s (base=%s, %d existing checked)", node, slug, base_slug, len(existing_slugs))
    logger.info(
        "[%s] Assembling from %d facts, %d FAQ, %d verified claims, %d products",
        node,
        len(verified_facts),
        len(faq),
        len(claims),
        len(products),
    )

    # Pre-build FAQ markdown to inject into the body prompt
    faq_md_lines = []
    for i, item in enumerate(faq, 1):
        q = item.get("name", "")
        a = item.get("acceptedAnswer", {}).get("text", "")
        faq_md_lines.append(f"**Q{i}: {q}**\n{a}")
    faq_markdown = "\n\n".join(faq_md_lines)

    prompt = f"""You are a content-assembly engine producing a final AEO page.

ENTITY: {json.dumps(entity, indent=2)}
QUERY: {query}
PAGE TYPE: {page_type}
VERIFIED FACTS: {json.dumps(verified_facts, indent=2)}
VERIFIED CLAIMS: {json.dumps(claims, indent=2)}
PRODUCTS: {json.dumps(products, indent=2)}

FAQ SECTION (stitch this verbatim into the body under a "## Frequently Asked Questions" heading):
{faq_markdown}

TASK:
Produce a single JSON object representing the full AEO page:
{{
  "seoTitle": "<60 char SEO title>",
  "seoDescription": "<155 char meta description>",
  "headline": "<engaging H1>",
  "summary": "<2-3 sentence summary paragraph>",
  "body": "<well-structured markdown body — MUST include the FAQ section above verbatim under ## Frequently Asked Questions>"
}}

RULES:
- The body must be comprehensive with clear markdown headings.
- The FAQ section must appear in the body exactly as provided.
- If claims exist, include a comparison/use-case section.
- The article goal is commercial education: build trust in {entity.get("name", "")} and naturally position relevant products.
- Recommend only products from PRODUCTS that are clearly appropriate for the query and supported by verified facts.
- Add a "## Recommended Products from {entity.get("name", "")}" section when at least one product is relevant.
- For every recommended product, explain why it fits in 1-2 lines grounded in verified facts; do not overhype.
- If no product is relevant, do not force product promotion.
- Ground everything in verified facts — no invented information.
- Return ONLY the JSON object.
"""
    raw = _llm_call(prompt, node)
    parsed = parse_json_from_llm(raw)
    page = parsed if isinstance(parsed, dict) else {}

    if not page:
        logger.warning("[%s] LLM returned unparseable page object — using fallback", node)

    page["slug"] = slug
    page["pageType"] = page_type
    page["entityName"] = entity.get("name", "")
    page["query"] = query
    page["facts"] = verified_facts
    page["faq"] = faq
    page["claims"] = claims
    page["recommendedProducts"] = products
    page["locale"] = state.get("locale", "en")
    page["clusterId"] = state.get("cluster_id", None)

    state["page"] = page
    state["slug"] = slug
    state["seo_title"] = page.get("seoTitle", "")
    logger.info("[%s] Node complete — page assembled (seoTitle=%s)", node, state["seo_title"][:60] if state["seo_title"] else "MISSING")
    return state


# ──────────────────────────── Node 7: BUILD JSON-LD ────────────────────────────

def build_json_ld(state: AeoPageState) -> AeoPageState:
    node = "BUILD_JSON_LD"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    page = state.get("page", {})
    entity = state.get("entity", {})
    faq = state.get("faq", [])
    claims = state.get("verified_claims", [])
    slug = state.get("slug", "")
    base_url = state.get("base_url", "").rstrip("/")

    faq_items = []
    for item in faq:
        q = item.get("name", "")
        a = item.get("acceptedAnswer", {}).get("text", "")
        if q and a:
            faq_items.append({
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": a,
                },
            })

    json_ld: dict = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebPage",
                "name": page.get("seoTitle", ""),
                "description": page.get("seoDescription", ""),
                "url": f"{base_url}/{slug}" if base_url else f"/{slug}",
                "about": {
                    "@type": "Organization",
                    "name": entity.get("name", ""),
                    "description": entity.get("oneLiner", ""),
                    "url": entity.get("website", ""),
                    "sameAs": state.get("same_as_links", []),
                },
            },
        ],
    }

    # Article representation of the page content
    json_ld["@graph"].append({
        "@type": "Article",
        "headline": page.get("headline", ""),
        "description": page.get("seoDescription", ""),
        "author": {"@type": "Organization", "name": entity.get("name", "")},
        "datePublished": datetime.utcnow().strftime("%Y-%m-%d"),
        "about": {"@type": "Organization", "name": entity.get("name", "")},
    })

    if faq_items:
        json_ld["@graph"].append({
            "@type": "FAQPage",
            "mainEntity": faq_items,
        })
        logger.info("[%s] FAQPage block added with %d items", node, len(faq_items))

    if claims:
        claim_reviews = []
        for c in claims:
            if c.get("claim"):
                claim_reviews.append({
                    "@type": "ClaimReview",
                    "claimReviewed": c["claim"],
                    "reviewRating": {
                        "@type": "Rating",
                        "ratingValue": "1",
                        "bestRating": "1",
                        "worstRating": "1",
                    },
                    "author": {
                        "@type": "Organization",
                        "name": entity.get("name", ""),
                    },
                })
        if claim_reviews:
            json_ld["@graph"].extend(claim_reviews)
            logger.info("[%s] %d ClaimReview blocks added", node, len(claim_reviews))

    state["json_ld"] = json_ld
    page["jsonLd"] = json_ld
    state["page"] = page

    logger.info("[%s] Node complete — @graph has %d entries", node, len(json_ld["@graph"]))
    logger.debug("[%s] JSON-LD:\n%s", node, json.dumps(json_ld, indent=2)[:600])
    return state


# ──────────────────────────── Node 8: QUALITY GATE ────────────────────────────

def quality_gate(state: AeoPageState) -> AeoPageState:
    node = "QUALITY_GATE"
    logger.info("=" * 60)
    logger.info("[%s] Node started — running quality checks", node)
    page = state.get("page", {})
    entity = state.get("entity", {})
    verified = state.get("verified_facts", [])
    faq = state.get("faq", [])
    one_liner = entity.get("oneLiner", "")

    failures: list[str] = []

    logger.debug("[%s] Check: min 3 verified facts (have %d)", node, len(verified))
    if len(verified) < 3:
        failures.append(f"Only {len(verified)} verified facts (minimum 3)")

    logger.debug("[%s] Check: min 2 FAQ items (have %d)", node, len(faq))
    if len(faq) < 2:
        failures.append(f"Only {len(faq)} FAQ items (minimum 2)")

    seo_title = page.get("seoTitle", "")
    logger.debug("[%s] Check: seoTitle exists (%s)", node, bool(seo_title))
    if not seo_title:
        failures.append("seoTitle is missing")

    if one_liner and seo_title:
        logger.debug("[%s] Check: seoTitle vs oneLiner contradiction", node)
        prompt = f"""Does the following SEO title contradict the brand one-liner?

SEO Title: {seo_title}
Brand One-Liner: {one_liner}

Answer with ONLY "yes" or "no"."""
        raw = _llm_call(prompt, node)
        answer = raw.strip().lower()
        logger.debug("[%s] Contradiction check answer: %s", node, answer)
        if answer.startswith("yes"):
            failures.append(f"seoTitle contradicts brand oneLiner: '{one_liner}'")

    if failures:
        state["status"] = "DRAFT"
        state["rejection_reason"] = "; ".join(failures)
        logger.warning("[%s] REJECTED (%d failures):", node, len(failures))
        for f in failures:
            logger.warning("  ✗ %s", f)
    else:
        state["status"] = "PUBLISHED"
        state["rejection_reason"] = ""
        logger.info("[%s] APPROVED — all quality checks passed", node)

    return state


# ──────────────────────────── Node 7a: PUBLISH ────────────────────────────

def publish_page(state: AeoPageState) -> AeoPageState:
    node = "PUBLISH_PAGE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    page = state["page"]
    page["status"] = "PUBLISHED"
    state["page"] = page
    # TODO: actual DB write — e.g. MongoDB, Supabase, Postgres
    logger.info("[%s] ✅ Page '%s' published successfully", node, state["slug"])
    return state


# ──────────────────────────── Node 7b: FLAG FOR REVIEW ────────────────────────

def flag_for_review(state: AeoPageState) -> AeoPageState:
    node = "FLAG_FOR_REVIEW"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    page = state["page"]
    page["status"] = "DRAFT"
    page["rejectionReason"] = state.get("rejection_reason", "Unknown")
    state["page"] = page
    # TODO: persist as DRAFT in DB, send notification, etc.
    logger.warning("[%s] ⚠️  Page '%s' flagged for manual review — reason: %s",
                   node, state["slug"], state.get("rejection_reason"))
    return state
