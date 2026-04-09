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
llm = ChatOpenAI(model="gpt-5.4-mini")

# Guard-rail model (more accurate, used for verification steps)
llm_strict = ChatOpenAI(model="gpt-5.4")

# Advanced Reasoning model (for complex tasks)
llm_complex = ChatOpenAI(model="gpt-5.4")

# ──────────────────────────── State ────────────────────────────

class AeoPageState(TypedDict):
    entity: dict
    intelligence: dict
    query: str
    topic: str
    topic_pages: list
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
    internal_links: list
    duplicate_status: str
    duplicate_reason: str

    primary_kw: str
    secondary_kws: list
    search_intent: str
    target_slug: str

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


def strip_markdown_fence(text: str) -> str:
    """Remove ```markdown ... ``` or ``` ... ``` wrappers the LLM sometimes adds."""
    cleaned = re.sub(
        r"^```(?:markdown)?\s*\n(.*?)\n```\s*$",
        r"\1",
        text.strip(),
        flags=re.DOTALL,
    )
    return cleaned.strip()


def flatten_intelligence(intelligence: dict) -> dict:
    """
    Flatten a potentially nested intelligence dict into {source_key: dict}.
    Keeps only dict-valued leaves (best-effort) so downstream can do rule-based extraction.
    """
    flat: dict = {}
    if not isinstance(intelligence, dict):
        return flat

    for key, value in intelligence.items():
        if isinstance(value, dict):
            # If this dict looks like a product/intel record, keep it.
            if any(not isinstance(v, dict) for v in value.values()):
                flat[str(key)] = value
                continue
            # Otherwise, flatten one level deeper.
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    flat[f"{key}.{sub_key}"] = sub_val
        elif isinstance(value, list):
            # Lists of dicts: index them
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat[f"{key}[{i}]"] = item
    return flat


def source_is_valid(source: str, valid_sources: list[str]) -> bool:
    return bool(source) and source in set(valid_sources)


def extract_product_facts(flat_intelligence: dict) -> list:
    """
    Rule-based pre-extraction for product-type intelligence.
    Guarantees at least basic facts exist before LLM runs.
    """
    facts: list[dict] = []
    if not isinstance(flat_intelligence, dict):
        return facts

    for key, value in flat_intelligence.items():
        if not isinstance(value, dict):
            continue
        # Price fact
        price = value.get("price") or value.get("Price")
        if price:
            facts.append({"fact": f"{key} is priced at {price}.", "source": key})
        # Material/fabric fact
        desc = str(value.get("description", "") or "")
        desc_l = desc.lower()
        if "bamboo" in desc_l:
            facts.append({"fact": f"{key} is made from bamboo fabric.", "source": key})
        if "wire-free" in desc_l or "non-wired" in desc_l:
            facts.append({"fact": f"{key} is wire-free.", "source": key})
        if "nursing" in desc_l or "breastfeeding" in desc_l:
            facts.append({"fact": f"{key} supports nursing or breastfeeding access.", "source": key})
    return facts


def generate_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or f"page-{uuid.uuid4().hex[:8]}"


def build_topic_internal_links(topic: str, topic_page_names: list, base_url: str = "") -> list[dict]:
    """
    Build canonical internal URLs as {topic_slug}/{page_slug} for each cluster page name/slug.
    Returned objects are passed to the LLM so it injects these exact hrefs into markdown.
    """
    topic_part = generate_slug(topic or "topic")
    base = (base_url or "").rstrip("/")
    links: list[dict] = []
    for raw in topic_page_names or []:
        name = str(raw).strip()
        if not name:
            continue
        page_part = generate_slug(name)
        path = f"{topic_part}/{page_part}"
        href = f"{base}/{path}" if base else f"/{path}"
        links.append({"label": name, "path": path, "href": href})
    return links


def _llm_call(prompt: str, node_name: str, llm_client: ChatOpenAI | None = None) -> str:
    """Wraps an LLM invocation with timing and error logging."""
    client = llm_client or llm
    logger.debug("[%s] LLM call started (%d chars prompt)", node_name, len(prompt))
    logger.debug("[%s] Prompt:\n%s", node_name, prompt)
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



# ──────────────────────────── Node 0: PREFLIGHT ────────────────────────────

def preflight(state: AeoPageState) -> AeoPageState:
    node = "PREFLIGHT"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    # Core rule: no existing cluster pages → this IS the pillar
    topic_pages = state.get("topic_pages") or []
    if len(topic_pages) == 0:
        state["page_type"] = "PILLAR_PAGE"

    return state

def duplicate_router(state: AeoPageState) -> str:
    if state.get("duplicate_status") == "DUPLICATE":
        return "early_exit"   # → END with DUPLICATE status in response
    return "draft_facts"      # SAFE or REVIEW both proceed

# ──────────────────────────── Node 1: KEYWORD RESEARCH ────────────────────────────


def keyword_research(state: AeoPageState) -> AeoPageState:
    node = "KEYWORD_RESEARCH"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    query = (state.get("query") or "").strip()
    topic = (state.get("topic") or "").strip()
    entity = state.get("entity") or {}
    locale = (state.get("locale") or "en").strip()

    prompt = f"""
You are a keyword research engine.

QUERY: {query}
TOPIC: {topic}
ENTITY: {entity.get("name", "")} — {entity.get("oneLiner", "")}
LOCALE: {locale}

Derive the best SEO keyword targeting for this query.

Return JSON:
{{
  "primary_kw": "<the single best target keyword phrase>",
  "secondary_kws": ["<related keyword 1>", "<related keyword 2>", ...],
  "search_intent": "informational" | "commercial" | "navigational",
  "target_slug": "<url-safe slug derived from primary_kw>"
}}
""".strip()

    raw = _llm_call(prompt, node, llm_client=llm)
    parsed = parse_json_from_llm(raw)
    obj = parsed if isinstance(parsed, dict) else {}

    primary_kw = (obj.get("primary_kw") or "").strip()
    secondary_kws_raw = obj.get("secondary_kws")
    secondary_kws = (
        [str(x).strip() for x in secondary_kws_raw if str(x).strip()]
        if isinstance(secondary_kws_raw, list)
        else []
    )
    search_intent = (obj.get("search_intent") or "").strip().lower()
    if search_intent not in {"informational", "commercial", "navigational"}:
        search_intent = "informational"

    target_slug = (obj.get("target_slug") or "").strip()
    if not target_slug and primary_kw:
        target_slug = generate_slug(primary_kw)
    elif target_slug:
        target_slug = generate_slug(target_slug)

    state["primary_kw"] = primary_kw
    state["secondary_kws"] = secondary_kws
    state["search_intent"] = search_intent
    state["target_slug"] = target_slug

    logger.info(
        "[%s] Node complete — primary_kw=%s | intent=%s | slug=%s",
        node,
        primary_kw[:80] if primary_kw else "MISSING",
        search_intent,
        target_slug or "MISSING",
    )
    return state


def duplicate_check(state: AeoPageState) -> AeoPageState:
    """
    Stateless duplicate / cannibalization check.
    Uses only existing_slugs + topic_pages (no LLM) and sets:
      - duplicate_status: SAFE | DUPLICATE | REVIEW
      - duplicate_reason: human-readable reason
    """
    existing_slugs = [str(s).strip() for s in (state.get("existing_slugs") or []) if str(s).strip()]
    topic_pages = [str(s).strip() for s in (state.get("topic_pages") or []) if str(s).strip()]

    base_slug = (state.get("target_slug") or "").strip()
    kw_slug = generate_slug(state.get("primary_kw", "") or "")
    if not base_slug:
        base_slug = kw_slug

    # Exact slug match against existing slugs is a hard DUPLICATE.
    if base_slug and base_slug in existing_slugs:
        state["duplicate_status"] = "DUPLICATE"
        state["duplicate_reason"] = f"Exact slug already exists: {base_slug}"
        return state

    # Also treat exact match vs provided topic_pages as DUPLICATE.
    if base_slug and base_slug in topic_pages:
        state["duplicate_status"] = "DUPLICATE"
        state["duplicate_reason"] = f"Exact match in topic_pages: {base_slug}"
        return state

    # Soft overlap: if the keyword slug appears inside any existing slug, flag REVIEW.
    if kw_slug:
        for existing in existing_slugs:
            if kw_slug in existing or existing in kw_slug:
                state["duplicate_status"] = "REVIEW"
                state["duplicate_reason"] = f"Potential cannibalization vs existing slug: {existing}"
                return state

        for existing in topic_pages:
            if kw_slug in existing or existing in kw_slug:
                state["duplicate_status"] = "REVIEW"
                state["duplicate_reason"] = f"Potential overlap vs topic_pages: {existing}"
                return state

    state["duplicate_status"] = "SAFE"
    state["duplicate_reason"] = ""
    return state

# ──────────────────────────── Node 1: DRAFT FACTS ────────────────────────────

def draft_facts(state: AeoPageState) -> AeoPageState:
    node = "DRAFT_FACTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)
    entity = state["entity"]
    intelligence = state["intelligence"]
    query = state["query"]
    flat_intelligence = flatten_intelligence(intelligence)
    source_keys = list(flat_intelligence.keys())
    if not source_keys:
        source_keys = list(intelligence.keys())

    logger.info("[%s] Entity: %s | Query: %s", node, entity.get("name"), query)
    logger.info("[%s] Intelligence source keys: %s", node, source_keys)

    baseline_facts = extract_product_facts(flat_intelligence)

    prompt = f"""You are a  fact-extraction engine.

INPUT ENTITY:
{json.dumps(entity, indent=2)}

INTELLIGENCE CONTEXT (source-keyed):
{json.dumps(flat_intelligence or intelligence, indent=2)}

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
    llm_facts = parse_json_from_llm(raw)
    logger.debug("[%s] Parsed %d total facts from LLM", node, len(llm_facts) if isinstance(llm_facts, list) else 0)

    all_facts = baseline_facts + [
        f for f in (llm_facts if isinstance(llm_facts, list) else [])
        if isinstance(f, dict) and f.get("fact") and source_is_valid(str(f.get("source", "")), source_keys)
    ]

    seen: set[str] = set()
    valid_facts: list[dict] = []
    for f in all_facts:
        fact_text = str(f.get("fact", "")).strip()
        source = str(f.get("source", "")).strip()
        if not fact_text or not source:
            continue
        if fact_text in seen:
            continue
        seen.add(fact_text)
        valid_facts.append({"fact": fact_text, "source": source})

    state["drafted_facts"] = valid_facts
    logger.info("[%s] %d baseline + LLM facts drafted", node, len(valid_facts))
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
    primary_kw = (state.get("primary_kw") or "").strip()
    secondary_kws = state.get("secondary_kws") or []
    search_intent = (state.get("search_intent") or "").strip()
    verified_facts = state["verified_facts"]
    faq = state["faq"]
    claims = state.get("verified_claims", [])
    products = entity.get("products", [])
    page_type = state.get("page_type", "DEFINITION")
    existing_slugs = state.get("existing_slugs", [])

    # Slug uses keyword_research output (fallback to primary_kw/query)
    base_slug = (state.get("target_slug") or "").strip()
    if not base_slug:
        base_slug = generate_slug(primary_kw or query)
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
PRIMARY KEYWORD: {primary_kw}
SECONDARY KEYWORDS: {json.dumps(secondary_kws, indent=2)}
SEARCH INTENT: {search_intent}
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
- SEO RULES (mandatory):
  - seoTitle MUST contain the primary keyword, max 60 chars
  - seoDescription MUST contain the primary keyword, max 155 chars
  - The H1 headline MUST contain the primary keyword
  - First paragraph of body MUST naturally include the primary keyword
  - Use secondary keywords in subheadings (H2/H3) where naturally appropriate
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
- Do NOT wrap the body in markdown code fences (no ```markdown or ```).
- Do NOT add any text outside the JSON object.

"""
    raw = _llm_call(prompt, node)
    parsed = parse_json_from_llm(raw)
    page = parsed if isinstance(parsed, dict) else {}

    if page.get("body"):
        page["body"] = strip_markdown_fence(str(page["body"]))

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

    indian_names = [
        "Aarav", "Vivaan", "Aditya", "Arjun", "Ishaan", "Reyansh", "Krishna", "Rohan",
        "Ananya", "Aadhya", "Diya", "Isha", "Kavya", "Meera", "Saanvi", "Priya",
        "Rahul", "Karan", "Vikram", "Sanjay", "Neha", "Pooja", "Sneha", "Aditi",
    ]

    author_name = (entity.get("author_name") or "").strip()
    if not author_name:
        # Deterministic fallback so the same page doesn't "change author" between runs
        seed = (state.get("slug") or state.get("target_slug") or state.get("query") or "")
        author_name = indian_names[abs(hash(seed)) % len(indian_names)] if seed else "Editorial Team"

    author_type = "Person" 
    author: dict = {
        "@type": author_type,
        "name": author_name,
    }
    if entity.get("author_url"):
        author["url"] = entity["author_url"]
    if entity.get("author_same_as"):
        author["sameAs"] = entity.get("author_same_as", [])

    # Article representation of the page content
    json_ld["@graph"].append({
        "@type": "Article",
        "headline": page.get("headline", ""),
        "description": page.get("seoDescription", ""),
        "author": author,
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
        json_ld["@graph"].append({
            "@type": "ItemList",
            "name": f"{entity.get('name')} Key Differentiators",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": i + 1,
                    "name": c["claim"],
                }
                for i, c in enumerate(claims)
                if isinstance(c, dict) and c.get("claim")
            ],
        })
        logger.info("[%s] ItemList block added with %d items", node, len([c for c in claims if isinstance(c, dict) and c.get("claim")]))

    state["json_ld"] = json_ld
    page["jsonLd"] = json_ld
    state["page"] = page

    logger.info("[%s] Node complete — @graph has %d entries", node, len(json_ld["@graph"]))
    logger.debug("[%s] JSON-LD:\n%s", node, json.dumps(json_ld, indent=2)[:600])
    return state


# ──────────────────────────── Node X: BUILD INTERNAL LINKS ────────────────────────────

def build_internal_links(state: AeoPageState) -> AeoPageState:
    topic_pages = state.get("topic_pages") or []
    page_type = state.get("page_type", "")
    page = state.get("page") or {}
    body = page.get("body", "")
    topic = (state.get("topic") or "").strip()
    base_url = (state.get("base_url") or "").strip()
    links: list = build_topic_internal_links(topic, topic_pages, base_url)

    if page_type == "PILLAR_PAGE":
        # Pillar links OUT to all cluster pages.
        # topic_pages here would be empty (that's why we're pillar),
        # but pillar body gets a placeholder ul for future retroactive updates.
        body += "\n\n## Related Articles\n<ul id='cluster-links'></ul>"
    else:
        # Cluster pages: use LLM to contextually inject links; must use PREBUILT_LINKS only.
        if topic_pages and links:
            prompt = f"""
You are an internal linking engine.

ARTICLE BODY (markdown):
{body}

PREBUILT INTERNAL LINKS (use these exact href values in markdown [text](href); do not invent URLs):
{json.dumps(links, indent=2)}

BASE URL: {base_url}

TASK:
1. Identify up to 3 locations in the body where linking to a relevant cluster page
   would feel somewhat natural for the reader.
2. Inject markdown links using ONLY the href values from PREBUILT INTERNAL LINKS. Match each link to the appropriate label/path.
4. ALWAYS add one link back to the pillar page (the first one in the prebuilt links list)

Return the FULL updated body markdown with links injected. Return ONLY the markdown.
""".strip()
            body = _llm_call(prompt, "BUILD_INTERNAL_LINKS", llm_client=llm_complex)

    state.setdefault("page", {})
    state["page"]["body"] = body
    state["internal_links"] = links
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

    primary_kw = state.get("primary_kw", "").lower()
    slug = state.get("slug", "")
    body = page.get("body", "")
    seo_desc = page.get("seoDescription", "")

    if primary_kw and primary_kw not in seo_title.lower():
        failures.append(f"primary_kw '{primary_kw}' missing from seoTitle")

    if len(seo_title) > 60:
        failures.append(f"seoTitle too long: {len(seo_title)} chars (max 60)")

    if len(seo_desc) > 155:
        failures.append(f"seoDescription too long: {len(seo_desc)} chars (max 155)")

    if primary_kw and primary_kw.replace(" ", "-") not in slug:
        failures.append(f"primary_kw not reflected in slug: {slug}")

    word_count = len(body.split())
    if word_count < 600:
        failures.append(f"Body too short: {word_count} words (min 600)")

    h1_count = body.count("\n# ")
    if h1_count == 0 and not body.startswith("# "):
        failures.append("No H1 found in body")
    elif h1_count > 1:
        failures.append(f"Multiple H1s found ({h1_count}) — only one allowed")

    if "## Frequently Asked Questions" not in body:
        failures.append("FAQ section missing from body")

    # Duplicate review warning (not a hard failure, but flagged)
    if state.get("duplicate_status") == "REVIEW":
        failures.append(f"Possible cannibalization: {state.get('duplicate_reason')}")

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
