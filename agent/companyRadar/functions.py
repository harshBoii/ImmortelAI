import re
import json
import time
import logging
import urllib.error
import urllib.request
import urllib.parse
from typing import TypedDict
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────

logger = logging.getLogger("geo_radar_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


# ──────────────────────────────────────────────────────────────
# Parser LLM  (used only as fallback in parse_responses)
# ──────────────────────────────────────────────────────────────

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm_parser = ChatOpenAI(model="gpt-4o-mini")

# Used only for LLM-assisted prompt generation in `generate_prompts` when API provides llmTopics.
llm_generate_prompts = ChatOpenAI(model="gpt-5.4-mini")


# ──────────────────────────────────────────────────────────────
# Topic expansion templates
# ──────────────────────────────────────────────────────────────

TOPIC_TEMPLATES = [
    "{topic}",
    "best {topic}",
    "top {topic} tools",
    "{topic} software",
    "{topic} platforms",
    "{topic} for small business",
]

# When API does not send llmTopics, discover this many niche topics via LLM (same style as companyBounty.discover_niches).
RADAR_LLM_TOPIC_COUNT = 3
# Hard cap: expand_topics never produces more than this many topics (LLM path + template fallback).
RADAR_LLM_TOPIC_MAX = 5


# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────

class GeoRadarState(TypedDict):
    # ── Input (caller-provided) ────────────────────────────────
    company: dict        # {name, website, linkedin}
    brand_entity: dict   # {category, topics, keywords, offerings:[...], product/url/differentiators...}
    competitors: list    # ["Twilio", "Respond.io", ...]
    models: list         # ["gpt-4o", "claude-3.5", "gemini-1.5"]
    llm_topics: list     # optional API-provided topics override
    webhook_url: str     # optional URL to POST the final result JSON (or set COMPANY_RADAR_RESULT_WEBHOOK_URL)

    # ── Pipeline stages (populated by nodes) ──────────────────
    topics: list         # expanded topic strings
    company_insights: dict  # distilled niche/citation insights from business context
    topic_metadata: dict  # {topic: {"reason": str, "use": str}}
    prompt_topic_map: dict  # {prompt: topic}
    prompt_metadata: dict  # {prompt: {"reason": str, "use": str}}
    prompts: list        # final prompt strings sent to LLMs
    raw_responses: list  # [{prompt, model, response, ?error}]
    citations: list      # [{prompt, model, companies:[{name,rank}]}]
    aggregated: dict     # intermediate aggregation data (consumed by compute_metrics)
    metrics: dict        # {share_of_voice, top3_rate, query_coverage, competitor_rank, topic_authority}
    result: dict         # final assembled output
    webhook_delivery: dict  # outcome of POST to webhook (if configured)

    # ── Session ────────────────────────────────────────────────
    session_id: str


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        parts = [
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ]
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


def _llm_call(prompt: str, node_name: str, llm_client: ChatOpenAI | None = None) -> str:
    """Wraps an LLM invocation with timing and error logging."""
    client = llm_client or llm_parser
    logger.debug("[%s] LLM call started (%d chars prompt)", node_name, len(prompt))
    logger.info("[%s] Prompt:\n%s", node_name, prompt)
    
    t0 = time.time()
    try:
        response = client.invoke(prompt)
        raw = extract_text(response)
        elapsed = time.time() - t0
        logger.info("[%s] LLM responded in %.2fs (%d chars)", node_name, elapsed, len(raw))
        logger.debug("[%s] Raw response:\n%s", node_name, raw[:400])
        return raw
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[%s] LLM FAILED after %.2fs: %s", node_name, elapsed, e)
        raise


def get_llm_client(model_name: str) -> ChatOpenAI:
    """
    Build a LangChain chat client from a friendly model alias.

    Supported aliases
    -----------------
    OpenAI   : "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
    Anthropic: "claude-3.5", "claude-3", "claude-3-haiku"
    Gemini   : "gemini-1.5", "gemini-2.0"
    """
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name)

    if "claude" in model_name:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic") from exc
        CLAUDE_MAP = {
            "claude-3.5":     "claude-3-5-sonnet-20241022",
            "claude-3":       "claude-3-opus-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        }
        return ChatAnthropic(model=CLAUDE_MAP.get(model_name, model_name))  # type: ignore[return-value]

    if "gemini" in model_name:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:
            raise ImportError("Install langchain-google-genai: pip install langchain-google-genai") from exc
        GEMINI_MAP = {
            "gemini-1.5": "gemini-1.5-pro",
            "gemini-2.0": "gemini-2.0-flash-exp",
        }
        return ChatGoogleGenerativeAI(model=GEMINI_MAP.get(model_name, model_name))  # type: ignore[return-value]

    raise ValueError(f"Unsupported model alias: '{model_name}'")


def _parse_ranking_regex(text: str) -> list[dict]:
    """
    Fast regex-based extraction of ranked items from a numbered list.
    Handles formats: '1. Company', '1) Company', '**1. Company**'
    Falls back gracefully — returns [] if nothing matched.
    """
    results = []
    pattern = r"(?m)^\s*(\d+)[.)]\s*\*{0,2}([^*\n:(]+?)\*{0,2}\s*(?:[:\-–—(,]|$)"
    for match in re.finditer(pattern, text):
        rank = int(match.group(1))
        name = match.group(2).strip().strip("*").strip()
        if name and len(name) > 1:
            results.append({"name": name, "rank": rank})
    return results


def _normalize_label(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def _domain_core(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    netloc = netloc.split("@")[-1]  # strip userinfo
    netloc = netloc.split(":")[0]   # strip port
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if not netloc:
        return ""
    return re.split(r"[.]", netloc)[0] or netloc


def _title_lead(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    for sep in ("|", " - ", " — ", " – ", "•", ":", "·"):
        if sep in t:
            t = t.split(sep)[0].strip()
            break
    return t


def _extract_candidate_name_from_tavily_result(result: dict) -> str:
    url = str(result.get("url") or "")
    title = str(result.get("title") or "")
    lead = _title_lead(title)
    if lead:
        return lead
    core = _domain_core(url)
    return core or url


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _safe_list(x) -> list:
    return x if isinstance(x, list) else []


def _estimate_monthly_prompt_reach(prompt: str) -> float:
    """
    Multiplicative reach model:  base × intent_mult × audience_mult

    base          → driven by word count (longer query = lower raw reach)
    intent_mult   → biggest single driver of volume tier
    audience_mult → niche qualifiers narrow the pool (reduces reach, raises intent)
    """
    p = prompt.lower()
    word_count = len(p.split())

    # ── 1. Base volume (word count inversely correlates with search volume)
    if word_count <= 3:
        base = 6_000.0   # head terms: "video hosting", "crm tools"
    elif word_count <= 5:
        base = 3_500.0   # mid: "best video hosting platforms"
    elif word_count <= 8:
        base = 1_800.0   # long-tail: "best video hosting for small business"
    else:
        base = 700.0     # ultra long-tail: conversational / hyper-specific

    # ── 2. Intent multiplier (dominant volume driver)
    #    "best X" queries get 3–5× the reach of generic queries at same word count
    if any(k in p for k in ("best ", "top ", "top-")):
        intent_mult = 2.8   # commercial evaluation — highest demand
    elif any(k in p for k in (" vs ", " versus ", "comparison", "compare ")):
        intent_mult = 2.4   # decision stage — strong demand
    elif any(k in p for k in ("alternatives", "alternative to")):
        intent_mult = 2.0   # switching intent — targeted demand
    elif any(k in p for k in ("how to ", "guide to ", "tutorial", "step by step")):
        intent_mult = 1.6   # informational — high volume, lower conversion
    elif any(k in p for k in ("what is ", "what are ", "define ", "meaning of")):
        intent_mult = 1.2   # awareness stage — broad but thin intent
    else:
        intent_mult = 1.0   # neutral / navigational

    # ── 3. Audience modifier (niche qualifiers reduce reach, raise intent quality)
    #    These signals narrow the addressable pool — opposite of original formula
    niche_signals = (
        "near me", "for small business", "for startups",
        "for enterprise", "for freelancers", "for agencies",
        "cheap ", "affordable", "free ", "open source",
        "in india", "in us", "in uk",
    )
    audience_mult = 0.55 if any(k in p for k in niche_signals) else 1.0

    return round(base * intent_mult * audience_mult, 2)
# Intent-based CTR table — commercial queries convert better than informational
_INTENT_CTR: list[tuple[tuple[str, ...], float]] = [
    (("best ", "top ", "top-"),                          0.18),  # evaluation intent
    ((" vs ", "comparison", "compare ", " versus "),     0.16),  # decision intent
    (("alternatives", "alternative to"),                 0.14),  # switching intent
    (("how to ", "guide to ", "tutorial"),               0.08),  # informational
    (("what is ", "what are ", "define "),               0.05),  # awareness
]

def _intent_ctr(prompt: str, base_ctr: float) -> float:
    """Adjust CTR based on query intent — callers can still override via base_ctr."""
    p = prompt.lower()
    for signals, ctr in _INTENT_CTR:
        if any(s in p for s in signals):
            return ctr
    return base_ctr


def _estimate_prompt_revenue(
    prompt: str,
    ctr: float = 0.12,   # overrides intent_ctr when explicitly passed
    cvr: float = 0.03,
    aov: float = 50.0,
    adjust_ctr_by_intent: bool = True,
) -> dict:
    """
    Estimates the monthly revenue opportunity of a prompt at full visibility (rank-agnostic).

    Formula: monthly_reach × ctr × cvr × aov

    This is the market value of owning this prompt — not what the company currently earns.
    Rank-based discounting belongs at the dashboard layer, not here.
    """
    monthly_reach   = _estimate_monthly_prompt_reach(prompt)
    effective_ctr   = _intent_ctr(prompt, ctr) if adjust_ctr_by_intent else ctr
    estimated_rev   = round(monthly_reach * effective_ctr * cvr * aov, 2)*200

    return {
        "monthlyPromptReach": monthly_reach,
        "ctr":                round(effective_ctr, 4),
        "visibilityWeight":   90.0,
        "cvr":                cvr,
        "aov":                aov,
        "estimatedRevenue":   estimated_rev*50,
    }


def _aggregate_brand_context(company: dict, brand_entity: dict) -> dict:
    """
    Normalize/aggregate brand context from either:
    - legacy flat fields on brand_entity, or
    - brand_entity.offerings[] (new schema).
    """
    offerings = _safe_list(brand_entity.get("offerings", []))
    primary = offerings[0] if offerings else {}

    def _pick(field: str, default=""):
        return brand_entity.get(field) or primary.get(field) or default

    products = []
    differentiators = set(_safe_list(brand_entity.get("differentiators", [])))
    use_cases = set(_safe_list(brand_entity.get("useCases", [])))
    target_audiences = set(_safe_list(brand_entity.get("targetAudiences", [])))
    competitor_groups = set(_safe_list(brand_entity.get("competitorGroups", [])))

    for off in offerings:
        if not isinstance(off, dict):
            continue
        prod = str(off.get("product", "")).strip()
        if prod:
            products.append({
                "product": prod,
                "productType": off.get("productType"),
                "url": off.get("url"),
            })
        differentiators.update(_safe_list(off.get("differentiators", [])))
        use_cases.update(_safe_list(off.get("useCases", [])))
        target_audiences.update(_safe_list(off.get("targetAudiences", [])))
        competitor_groups.update(_safe_list(off.get("competitorGroups", [])))

    website = company.get("website") or ""
    return {
        "category": brand_entity.get("category", ""),
        "topics": _safe_list(brand_entity.get("topics", [])),
        "keywords": _safe_list(brand_entity.get("keywords", [])),
        "product": _pick("product", ""),
        "productType": _pick("productType", ""),
        "url": _pick("url", website),
        "offerings": offerings,
        "products": products,
        "differentiators": sorted({str(x).strip() for x in differentiators if str(x).strip()}),
        "useCases": sorted({str(x).strip() for x in use_cases if str(x).strip()}),
        "targetAudiences": sorted({str(x).strip() for x in target_audiences if str(x).strip()}),
        "competitorGroups": sorted({str(x).strip() for x in competitor_groups if str(x).strip()}),
    }


# ──────────────────────────────────────────────────────────────
# Node 1a — USE API TOPICS
# Uses llm_topics from API request as direct topics input.
# ──────────────────────────────────────────────────────────────

def use_api_topics(state: GeoRadarState) -> GeoRadarState:
    node = "USE_API_TOPICS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_topics = state.get("llm_topics", []) or []
    cleaned_topics = sorted({str(t).strip() for t in raw_topics if str(t).strip()})
    state["topics"] = cleaned_topics
    topic_metadata: dict[str, dict] = {}

    if cleaned_topics:
        meta_prompt = f"""You are an AEO strategist.

Given these topics: {json.dumps(cleaned_topics)}

Return ONLY a JSON array where each item is:
{{
  "topic": "<topic text>",
  "reason": "<why this topic can improve AI citation probability>",
  "use": "<how the business should use this topic in content strategy>"
}}
"""
        try:
            raw = _llm_call(meta_prompt, f"{node}:topic_metadata")
            parsed = parse_json_from_llm(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    topic = str(item.get("topic", "")).strip()
                    if topic:
                        topic_metadata[topic] = {
                            "reason": str(item.get("reason", "")).strip(),
                            "use": str(item.get("use", "")).strip(),
                        }
        except Exception as e:
            logger.warning("[%s] Topic metadata generation failed: %s", node, e)

    for t in cleaned_topics:
        if t not in topic_metadata:
            topic_metadata[t] = {
                "reason": "This is a focused topic area with intent depth, improving citation opportunity.",
                "use": "Use this topic to publish practical, niche pages with direct answer-style sections.",
            }
    state["topic_metadata"] = topic_metadata

    logger.info(
        "[%s] Using %d topics from API llmTopics (expand_topics skipped)",
        node,
        len(cleaned_topics),
    )
    logger.debug("[%s] API topics: %s", node, cleaned_topics)
    return state


# ──────────────────────────────────────────────────────────────
# Node 1b — EXPAND TOPICS (LLM niche discovery, companyBounty-style)
# Requests min(RADAR_LLM_TOPIC_COUNT, RADAR_LLM_TOPIC_MAX) topics; never more than RADAR_LLM_TOPIC_MAX.
# ──────────────────────────────────────────────────────────────

def _expand_topics_template_fallback(state: GeoRadarState) -> None:
    """Legacy deterministic expansion into state['topics'] / state['topic_metadata']."""
    brand_entity = state["brand_entity"]
    base_topics = brand_entity.get("topics", []) or []
    keywords = brand_entity.get("keywords", []) or []
    category = brand_entity.get("category", "")

    expanded: set[str] = set()
    for topic in base_topics:
        for template in TOPIC_TEMPLATES:
            expanded.add(template.format(topic=topic))
    for kw in keywords:
        expanded.add(kw)
        expanded.add(f"best {kw}")
        expanded.add(f"top {kw} tools")
    if category:
        expanded.add(category)
        expanded.add(f"best {category}")
        expanded.add(f"top {category} tools")
        expanded.add(f"{category} comparison")

    topics = sorted(expanded)[:RADAR_LLM_TOPIC_MAX]
    state["topics"] = topics
    state["topic_metadata"] = {
        t: {
            "reason": "This topic is derived from category/topics/keywords expansion to capture searchable intent variants.",
            "use": "Use it as a targeting cluster for comparison, best-of, and how-to answer pages.",
        }
        for t in topics
    }


def expand_topics(state: GeoRadarState) -> GeoRadarState:
    node = "EXPAND_TOPICS"
    logger.info("=" * 60)
    n = min(RADAR_LLM_TOPIC_COUNT, RADAR_LLM_TOPIC_MAX)
    logger.info(
        "[%s] Node started (LLM niche discovery, requesting %d topics, max %d)",
        node,
        n,
        RADAR_LLM_TOPIC_MAX,
    )

    company = state["company"]
    brand_entity = state["brand_entity"]
    competitors = state.get("competitors", []) or []

    company_name = company.get("name", "")
    category = brand_entity.get("category", "")
    topics_seed = brand_entity.get("topics", []) or []
    keywords = brand_entity.get("keywords", []) or []

    logger.info("[%s] Company: %s | Category: %s", node, company_name, category)
    logger.info("[%s] Seed topics: %s | Keywords: %s", node, topics_seed, keywords)

    prompt = f"""You are an AEO strategist identifying niche topics where a company can realistically become the cited authority in AI search engines (ChatGPT, Perplexity, Gemini).

## Company
- Name: {company_name}
- Website: {company.get("website", "N/A")}
- Category: {category}
- Core topics: {', '.join(str(t) for t in topics_seed) or 'not specified'}
- Keywords: {', '.join(str(k) for k in keywords) or 'not specified'}
- Main competitors: {', '.join(str(c) for c in competitors) or 'not specified'}

## Task
Generate exactly {n} niche topics this company should target for topical authority.

A good niche topic:
- Is narrow enough to own with focused content (not a broad category)
- Maps directly to the company's product, expertise, or a specific use case
- Has real user search intent behind it — people actively ask questions about it
- Has a realistic path to authority given the company's current market position

## Difficulty Rating
Assign one of: `easy` · `medium` · `hard`

| Rating | Signal |
|--------|--------|
| easy   | Narrow, low competition; company already has relevant expertise or content |
| medium | Moderate competition or partial coverage; achievable with focused effort |
| hard   | Broad or heavily contested by established players; high investment required |

Aim for a mix of difficulties — not all easy, not all hard.
Also vary the type: include product-specific, use-case, and audience-specific niches.

## Output
Return ONLY a valid JSON array, no explanation:
[
  {{
    "topic": "<short niche topic title>",
    "description": "<one sentence — why this is a high-value AEO opportunity for this company and citation potential in AI search>",
    "difficulty": "easy|medium|hard"
  }}
]"""

    try:
        raw = _llm_call(prompt, node, llm_client=llm_generate_prompts)
        parsed = parse_json_from_llm(raw)
        if not isinstance(parsed, list):
            parsed = []
        niches = [
            x for x in parsed
            if isinstance(x, dict) and str(x.get("topic", "")).strip()
        ][:RADAR_LLM_TOPIC_MAX]

        niches = niches[:n]

        if len(niches) < n:
            logger.warning("[%s] LLM returned %d/%d topics; using template fallback", node, len(niches), n)
            _expand_topics_template_fallback(state)
            return state

        topic_list: list[str] = []
        topic_metadata: dict[str, dict] = {}
        for item in niches:
            title = str(item.get("topic", "")).strip()
            if not title:
                continue
            desc = str(item.get("description", "")).strip()
            diff = str(item.get("difficulty", "")).strip().lower()
            topic_list.append(title)
            topic_metadata[title] = {
                "reason": desc or "High-value niche aligned with the company's positioning and search intent.",
                "use": (
                    f"Priority for difficulty={diff or 'unknown'}: build authoritative pages (FAQ, guides, comparisons) "
                    "so AI engines can cite this brand for this niche."
                ),
            }

        state["topics"] = topic_list
        state["topic_metadata"] = topic_metadata
        logger.info("[%s] Node complete — %d LLM topics", node, len(topic_list))
    except Exception as e:
        logger.warning("[%s] LLM niche discovery failed: %s — template fallback", node, e)
        _expand_topics_template_fallback(state)

    return state


# ──────────────────────────────────────────────────────────────
# Node 2 — ANALYZE COMPANY CONTEXT
# Distills brand positioning, niche strengths and citation angles
# from request payload for use in prompt generation.
# ──────────────────────────────────────────────────────────────

def analyze_company_context(state: GeoRadarState) -> GeoRadarState:
    node = "ANALYZE_COMPANY_CONTEXT"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    company = state["company"]
    brand_entity = state["brand_entity"]
    llm_topics = state.get("llm_topics", []) or []
    competitors = state.get("competitors", [])
    agg = _aggregate_brand_context(company, brand_entity)

    context_payload = {
        "company_name": company.get("name", ""),
        "website": company.get("website", ""),
        "about": company.get("about", ""),
        "category": brand_entity.get("category", ""),
        "product": agg.get("product", ""),
        "product_type": agg.get("productType", ""),
        "topics": agg.get("topics", []),
        "keywords": agg.get("keywords", []),
        "offerings": agg.get("offerings", []),
        "products": agg.get("products", []),
        "differentiators": agg.get("differentiators", []),
        "use_cases": agg.get("useCases", []),
        "target_audiences": agg.get("targetAudiences", []),
        "competitor_groups": agg.get("competitorGroups", []),
        "competitors": competitors,
        "llm_topics": llm_topics,
    }

    prompt = f"""You are a market strategist helping maximize AI-citation probability.

Analyze this company context:
{json.dumps(context_payload, indent=2)}

Return ONLY a JSON object with:
{{
  "positioning_summary": "<1-2 line summary of what makes this company distinct>",
  "niche_strengths": ["<specific capability or niche strength>", "..."],
  "underserved_angles": ["<market gap similar competitors often miss>", "..."],
  "citation_playbook": ["<how prompts should be framed for high citation chance>", "..."]
}}

Rules:
- Keep insights specific and practical.
- Avoid generic marketing language.
- Focus on domain-specific opportunities where expertise can win.
"""
    try:
        raw = _llm_call(prompt, node)
        parsed = parse_json_from_llm(raw)
        if isinstance(parsed, dict):
            state["company_insights"] = parsed
        else:
            state["company_insights"] = {}
    except Exception as e:
        logger.warning("[%s] Insight extraction failed: %s. Using empty insights.", node, e)
        state["company_insights"] = {}

    logger.info(
        "[%s] Node complete — insights keys: %s",
        node,
        list(state.get("company_insights", {}).keys()),
    )
    return state


# ──────────────────────────────────────────────────────────────
# Node 3 — GENERATE PROMPTS
# - If llm_topics is present: generate 2-4 niche citation-oriented prompts
#   per topic (without mentioning the brand explicitly).
# - Else: default deterministic prompt generation.
# ──────────────────────────────────────────────────────────────

def generate_prompts(state: GeoRadarState) -> GeoRadarState:
    node = "GENERATE_PROMPTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    topics      = state["topics"]
    llm_topics  = state.get("llm_topics", []) or []
    competitors = state["competitors"]
    company     = state["company"]["name"]
    category    = state["brand_entity"].get("category", "")
    brand_entity = state["brand_entity"]
    agg = _aggregate_brand_context(state["company"], brand_entity)
    keywords    = agg.get("keywords", [])
    base_topics = agg.get("topics", [])
    product     = agg.get("product", "")
    product_type = agg.get("productType", "")
    website_url = agg.get("url", "")
    differentiators = agg.get("differentiators", [])
    use_cases   = agg.get("useCases", [])
    target_audiences = agg.get("targetAudiences", [])
    competitor_groups = agg.get("competitorGroups", [])
    offerings = agg.get("offerings", [])
    products = agg.get("products", [])
    company_insights = state.get("company_insights", {})
    topic_metadata = state.get("topic_metadata", {}) or {}

    prompts: set[str] = set()
    prompt_topic_map: dict[str, str] = {}
    prompt_metadata: dict[str, dict] = {}

    # If API llmTopics are present, use LLM-assisted prompt generation focused
    # on niche/domain-specific angles that increase citation probability.
    if llm_topics:
        logger.info(
            "[%s] llmTopics present (%d). Generating 2-4 niche prompts per topic via LLM.",
            node,
            len(llm_topics),
        )
        for topic in topics:
            prompt_builder = f"""You are an AEO strategist generating LLM citation prompts.

## Target Business
- Category: {category}
- Product: {product} ({product_type})
- Target audiences: {', '.join(target_audiences) or 'not specified'}
- Offerings (raw): {json.dumps(offerings) if offerings else '[]'}
- Products (catalog): {json.dumps(products) if products else '[]'}

## Brand Positioning
- Differentiators: {', '.join(differentiators) or 'not specified'}
- Use cases: {', '.join(use_cases) or 'not specified'}
- Core topics: {', '.join(base_topics) or 'not specified'}
- Keywords: {', '.join(keywords) or 'not specified'}
- Insight: {company_insights.get('positioning_summary', '')}

## Topic to Target
"{topic}"
Reason this topic matters: {topic_metadata.get(topic, {}).get('reason', '')}

## Your Task
Generate 2–4 search queries a real INDIAN USER would type when looking for exactly what this business offers.

Each prompt must:
- Be specific enough that generic enterprise giants (e.g. Nestle , Haldiram's , BigBasket , Britania , Aashirvaad, Amul , etc.) would NOT naturally appear in the answer
- Reflect the business's geography and Unique Selling Proposition (USP) and the niche they are targeting
- Read like a natural user query, not a keyword string
- NOT mention the company name
- The Prompt must reflect the word "Indian" in some way.

Avoid:
- Broad category queries ("best CRM software", "top marketing tools") — these surface only Fortune 500 tools
- Queries without geographic, audience, or niche qualifiers when the business is local/regional
- Redundant variants of the same intent

## Output Format
Return ONLY a valid JSON array, no explanation:
[
  {{
    "prompt": "<the search query>",
    "reason": "<why an LLM would cite a niche/specialized provider here>",
    "use": "<what page or content type should target this prompt>"
  }}
]
"""
            try:
                logger.info("[%s] Generating prompts for topic: %s", node, topic)
                logger.debug("[%s] Built prompt for topic '%s':\n%s", node, topic, prompt_builder)
                raw = _llm_call(prompt_builder, f"{node}:llmTopics", llm_client=llm_generate_prompts)
                generated = parse_json_from_llm(raw)
                if isinstance(generated, list):
                    clean: list[tuple[str, str, str]] = []
                    for item in generated:
                        if isinstance(item, str):
                            p = item.strip()
                            if p and company.lower() not in p.lower():
                                clean.append((p, "", ""))
                            continue
                        if not isinstance(item, dict):
                            continue
                        p = str(item.get("prompt", "")).strip()
                        if p and company.lower() not in p.lower():
                            clean.append((
                                p,
                                str(item.get("reason", "")).strip(),
                                str(item.get("use", "")).strip(),
                            ))
                    for p, reason, use in clean:
                        prompts.add(p)
                        prompt_topic_map[p] = topic
                        prompt_metadata[p] = {
                            "reason": reason or "This prompt targets specific intent where specialized expertise is more likely to be cited.",
                            "use": use or "Use this as a dedicated article/FAQ query with direct, source-backed answers.",
                        }
                    logger.debug("[%s] Topic '%s' -> %d prompts", node, topic, len(clean))
            except Exception as e:
                logger.warning(
                    "[%s] LLM prompt generation failed for topic '%s': %s. Using topic fallback.",
                    node,
                    topic,
                    e,
                )
                prompts.add(topic)
                prompt_topic_map[topic] = topic
                prompt_metadata[topic] = {
                    "reason": "Fallback prompt retained due to generation failure.",
                    "use": "Use as a baseline query and iterate with more specific variants.",
                }
    else:
        logger.info("[%s] No llmTopics provided. Using default prompt generation flow.", node)

        # All expanded topics become prompts directly
        for topic in topics:
            prompts.add(topic)
            prompt_topic_map[topic] = topic
            prompt_metadata[topic] = {
                "reason": "Direct topic query captures baseline demand and broad citation opportunity.",
                "use": "Use as a pillar page or anchor query in the topic cluster.",
            }

        # Competitor-alternative prompts
        for comp in competitors:
            p1 = f"{comp} alternatives"
            p2 = f"best alternatives to {comp}"
            prompts.add(p1)
            prompts.add(p2)
            prompt_topic_map[p1] = f"{comp} alternatives"
            prompt_topic_map[p2] = f"{comp} alternatives"
            prompt_metadata[p1] = {
                "reason": "Alternative-intent users often compare providers, increasing citation opportunities.",
                "use": "Use for alternatives/comparison pages with clear criteria and ranked options.",
            }
            prompt_metadata[p2] = {
                "reason": "Best-alternative phrasing captures high-intent comparison traffic.",
                "use": "Use for listicle-style pages with objective pros/cons and fit-by-use-case sections.",
            }
            if category:
                p3 = f"{comp} vs {company}"
                prompts.add(p3)
                prompt_topic_map[p3] = f"{comp} alternatives"
                prompt_metadata[p3] = {
                    "reason": "Head-to-head comparison intent is strong and citation-friendly for evaluative queries.",
                    "use": "Use for comparison pages with side-by-side capability mapping.",
                }

        # Head-to-head competitor vs competitor (top 4 only to avoid explosion)
        top_comps = competitors[:4]
        for i, c1 in enumerate(top_comps):
            for c2 in top_comps[i + 1:]:
                p = f"{c1} vs {c2}"
                prompts.add(p)
                prompt_topic_map[p] = "competitor comparisons"
                prompt_metadata[p] = {
                    "reason": "Competitor-vs-competitor prompts reveal market framing where your brand can be inserted contextually.",
                    "use": "Use to build category comparison content that also introduces your solution where relevant.",
                }

    prompts_list = sorted(prompts)
    state["prompts"] = prompts_list
    state["prompt_topic_map"] = prompt_topic_map
    state["prompt_metadata"] = prompt_metadata
    logger.info(
        "[%s] Node complete — %d prompts generated (%s source)",
        node,
        len(prompts_list),
        "llmTopics-assisted" if llm_topics else "default",
    )
    return state


# ──────────────────────────────────────────────────────────────
# Node 3 — RUN PROMPTS
# Executes every prompt against every requested LLM model.
# Stores raw free-text responses.
# ──────────────────────────────────────────────────────────────

# def run_prompts(state: GeoRadarState) -> GeoRadarState:
#     node = "RUN_PROMPTS"
#     logger.info("=" * 60)
#     logger.info("[%s] Node started", node)

#     prompts     = state["prompts"]
#     model_names = state["models"]
#     total_calls = len(prompts) * len(model_names)

#     logger.info(
#         "[%s] %d prompts × %d models = %d LLM calls",
#         node, len(prompts), len(model_names), total_calls,
#     )
#     logger.info("[%s] Prompts:\n%s", node, prompts)

#     raw_responses: list[dict] = []

#     for model_name in model_names:
#         try:
#             client = get_llm_client(model_name)
#         except (ValueError, ImportError) as e:
#             logger.warning("[%s] Skipping model '%s' — %s", node, model_name, e)
#             continue

#         for prompt_text in prompts:
#             system_instruction = (
#                 "You are a recommendation assistant.\n"
#                 "Given a user query, respond with concise, practical guidance and include ranked company/product suggestions amongst indian local companies or global companies used widely in india.\n\n"
#                 "Rules:\n"
#                 "1) If the query explicitly asks for recommendations, provide a ranked list.\n"
#                 "2) If the query is informational (does not explicitly ask for recommendations), first answer briefly, then add\n"
#                 "   a section named 'Suggested products/companies' with ranked alternatives.\n"
#                 "3) Wherever possible, include product + company together (e.g., '1. Product ABC - XYZ Company').\n"
#                 "4) Keep output factual and concise.\n"
#                 "5) Include ranking numbers in output."
#             )
#             full_prompt = f"{system_instruction}\n\nQuery: {prompt_text}"

#             try:
#                 raw = _llm_call(full_prompt, f"{node}:{model_name}", client)
#                 raw_responses.append({
#                     "prompt": prompt_text,
#                     "model": model_name,
#                     "response": raw,
#                 })
#             except Exception as e:
#                 logger.error("[%s] %s | '%s' — %s", node, model_name, prompt_text, e)
#                 raw_responses.append({
#                     "prompt": prompt_text,
#                     "model": model_name,
#                     "response": "",
#                     "error": str(e),
#                 })

#     state["raw_responses"] = raw_responses
#     logger.info("[%s] Node complete — %d raw responses stored", node, len(raw_responses))
#     return state


# Tavily-powered alternative to the (commented) LLM run_prompts node.
def run_web_search(state: GeoRadarState) -> GeoRadarState:
    node = "RUN_WEB_SEARCH"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    prompts = state.get("prompts", []) or []
    logger.info("[%s] %d prompts → Tavily searches", node, len(prompts))

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError as exc:
        raise ImportError("Install tavily-python: pip install tavily-python") from exc

    client = TavilyClient(api_key)

    raw_responses: list[dict] = []
    for prompt_text in prompts:
        try:
            response = client.search(query=prompt_text, search_depth="advanced", max_results=5)
            raw_responses.append(
                {
                    "prompt": prompt_text,
                    "model": "tavily",
                    "response": response,
                }
            )
        except Exception as e:
            logger.error("[%s] Tavily search failed for '%s' — %s", node, prompt_text, e)
            raw_responses.append(
                {
                    "prompt": prompt_text,
                    "model": "tavily",
                    "response": {},
                    "error": str(e),
                }
            )

    state["raw_responses"] = raw_responses
    logger.info("[%s] Node complete — %d raw responses stored", node, len(raw_responses))
    return state


def run_web_search_synth(state: GeoRadarState) -> GeoRadarState:
    """
    Node between Tavily and parsing:
    - Takes Tavily results from `state["raw_responses"]`
    - Feeds them to the recommendation system prompt (old RUN_PROMPTS rules)
    - Produces LLM text responses so `parse_responses` can extract ranked items.
    """
    node = "RUN_WEB_SEARCH_SYNTH"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state.get("raw_responses") or []
    model_names = state.get("models") or []

    # If caller didn't request any LLM models, keep Tavily structured results
    # so `parse_responses` can still operate via its Tavily branch.
    if not model_names:
        logger.info("[%s] No llm models provided; skipping synthesis", node)
        return state

    system_instruction = (
        "You are a recommendation assistant.\n"
        "Given a user query, respond with concise, practical guidance and include ranked company/product suggestions amongst indian local companies or global companies used widely in india.\n\n"
        "Rules:\n"
        "1) If the query explicitly asks for recommendations, provide a ranked list.\n"
        "2) If the query is informational (does not explicitly ask for recommendations), first answer briefly, then add\n"
        "   a section named 'Suggested products/companies' with ranked alternatives.\n"
        "3) Wherever possible, include product + company together (e.g., '1. Product ABC - XYZ Company').\n"
        "4) Keep output factual and concise.\n"
        "5) Include ranking numbers in output.\n"
    )

    synthesized: list[dict] = []
    for tavily_item in raw_responses:
        prompt_text = tavily_item.get("prompt", "")
        tavily_response = tavily_item.get("response") or {}

        # If this wasn't produced by Tavily for some reason, keep it.
        if not isinstance(tavily_response, dict):
            synthesized.append(tavily_item)
            continue

        results = tavily_response.get("results") or []
        if not isinstance(results, list):
            results = []

        # Trim web context to keep prompts reasonable.
        web_chunks: list[str] = []
        for i, r in enumerate(results[:6], start=1):
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "")
            title = str(r.get("title") or "")
            content = str(r.get("content") or "")
            content = content[:1400] + ("..." if len(content) > 1400 else "")
            web_chunks.append(
                f"[Result {i}]\nTitle: {title}\nURL: {url}\nContent:\n{content}\n"
            )

        web_context = "\n".join(web_chunks).strip()

        for model_name in model_names:
            try:
                client = get_llm_client(model_name)
            except (ValueError, ImportError) as e:
                logger.warning("[%s] Skipping model '%s' — %s", node, model_name, e)
                continue

            # Ask model to ground recommendations in the provided web snippets.
            full_prompt = (
                f"{system_instruction}\n"
                f"User query: {prompt_text}\n\n"
                f"Web search evidence (use for grounding):\n{web_context or '[no results]'}\n"
            )

            try:
                raw = _llm_call(full_prompt, f"{node}:{model_name}", llm_client=client)
                synthesized.append(
                    {
                        "prompt": prompt_text,
                        "model": model_name,
                        "response": raw,
                    }
                )
            except Exception as e:
                logger.error("[%s] LLM synth failed for '%s' / '%s': %s", node, prompt_text, model_name, e)
                synthesized.append(
                    {
                        "prompt": prompt_text,
                        "model": model_name,
                        "response": "",
                        "error": str(e),
                    }
                )

    state["raw_responses"] = synthesized
    logger.info("[%s] Node complete — %d synthesized responses", node, len(synthesized))
    return state


# ──────────────────────────────────────────────────────────────
# Node 4 — PARSE RESPONSES
# Extracts structured [{name, rank}] lists from raw LLM text.
# Strategy: LLM-only parsing for robust extraction.
# ──────────────────────────────────────────────────────────────

def parse_responses(state: GeoRadarState) -> GeoRadarState:
    node = "PARSE_RESPONSES"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state["raw_responses"]
    citations: list[dict] = []
    llm_parses = 0

    for item in raw_responses:
        prompt_text   = item["prompt"]
        model_name    = item["model"]
        response_obj  = item.get("response")

        companies: list[dict] = []

        # Tavily is already structured; no LLM parsing needed.
        if model_name == "tavily" and isinstance(response_obj, dict):
            results = response_obj.get("results") or []
            if isinstance(results, list):
                for idx, r in enumerate(results, start=1):
                    if not isinstance(r, dict):
                        continue
                    name = _extract_candidate_name_from_tavily_result(r)
                    companies.append(
                        {
                            "name": name,
                            "product": "",
                            "rank": idx,
                            "url": r.get("url"),
                            "title": r.get("title"),
                            "score": r.get("score"),
                        }
                    )
        else:
            # Legacy: parse free-form LLM output (kept as fallback).
            response_text = str(response_obj or "")
            if response_text and not item.get("error"):
                parse_prompt = (
                    "Extract all recommended companies and/or products from the response below.\n"
                    "Assign rank by recommendation order (1 = best/first).\n"
                    "If a line contains both product and company, put product name in 'product' and company in 'name'.\n\n"
                    f"Response:\n\"\"\"\n{response_text}\n\"\"\"\n\n"
                    "Return ONLY a JSON array in this format:\n"
                    '[{"name":"Company A","product":"Product X","rank":1},{"name":"Company B","product":"","rank":2}]\n'
                    "No explanation. JSON only."
                )
                try:
                    raw = _llm_call(parse_prompt, f"{node}:llm_parse")
                    parsed = parse_json_from_llm(raw)
                    companies = parsed if isinstance(parsed, list) else []
                    llm_parses += 1
                except Exception as e:
                    logger.error("[%s] LLM parse failed for '%s': %s", node, prompt_text, e)
                    companies = []

        citations.append({
            "prompt":    prompt_text,
            "model":     model_name,
            "companies": companies,
        })

    state["citations"] = citations
    logger.info(
        "[%s] Node complete — %d records | %d LLM parses",
        node, len(citations), llm_parses,
    )
    return state


# ──────────────────────────────────────────────────────────────
# Node 5 — AGGREGATE CITATIONS
# Pure data aggregation, no LLM call.
# Builds the intermediate `aggregated` dict consumed by compute_metrics.
# ──────────────────────────────────────────────────────────────

def aggregate_citations(state: GeoRadarState) -> GeoRadarState:
    node = "AGGREGATE_CITATIONS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    citations   = state["citations"]
    company_name = _normalize_label(state["company"]["name"])
    competitors  = [_normalize_label(c) for c in state["competitors"]]

    all_mentions:    list[dict] = []
    target_mentions: list[dict] = []
    prompts_with_target: set[str] = set()

    # Per-competitor tracking
    competitor_stats: dict[str, dict] = {c: {"mentions": 0, "ranks": [], "prompts": set()} for c in competitors}

    def _matches(candidate: str, target: str) -> bool:
        if not candidate or not target:
            return False
        if candidate == target:
            return True
        return candidate in target or target in candidate

    for record in citations:
        for entry in record.get("companies", []):
            name_lower = _normalize_label(entry.get("name") or "")
            rank       = entry.get("rank")

            mention = {
                "prompt": record["prompt"],
                "model":  record["model"],
                "name":   entry.get("name", ""),
                "rank":   rank,
            }
            all_mentions.append(mention)

            # Target company
            if _matches(name_lower, company_name):
                target_mentions.append(mention)
                prompts_with_target.add(record["prompt"])

            # Competitors
            matched_comp = None
            for comp in competitor_stats.keys():
                if _matches(name_lower, comp):
                    matched_comp = comp
                    break
            if matched_comp:
                competitor_stats[matched_comp]["mentions"] += 1
                if rank is not None:
                    competitor_stats[matched_comp]["ranks"].append(rank)
                competitor_stats[matched_comp]["prompts"].add(record["prompt"])

    # Compute per-competitor averages and convert sets → lists for serialisation
    competitor_summary: dict[str, dict] = {}
    for comp, data in competitor_stats.items():
        ranks = data["ranks"]
        competitor_summary[comp] = {
            "mentions":      data["mentions"],
            "avg_rank":      round(sum(ranks) / len(ranks), 1) if ranks else None,
            "prompt_count":  len(data["prompts"]),
        }

    agg = {
        "total_prompts":       len(set(c["prompt"] for c in citations)),
        "total_responses":     len(citations),
        "total_mentions":      len(all_mentions),
        "target_mentions":     target_mentions,
        "prompts_with_target": list(prompts_with_target),
        "competitor_summary":  competitor_summary,
    }

    state["aggregated"] = agg
    logger.info(
        "[%s] Node complete — %d total mentions | %d target | %d prompts covered",
        node, agg["total_mentions"], len(target_mentions), len(prompts_with_target),
    )
    return state


# ──────────────────────────────────────────────────────────────
# Node 6 — COMPUTE METRICS
# Pure arithmetic, no LLM call.
#
# Formulas
# ─────────────────────────────────────────────────────────────
# share_of_voice  = target_mentions / total_mentions      × 100
# top3_rate       = prompts_where_rank ≤ 3 / total_prompts × 100
# query_coverage  = prompts_with_target / total_prompts    × 100
# competitor_rank = average rank of target when mentioned
# topic_authority = base_topics_covered_by_target / total_base_topics × 10
# ──────────────────────────────────────────────────────────────

def compute_metrics(state: GeoRadarState) -> GeoRadarState:
    node = "COMPUTE_METRICS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    agg             = state["aggregated"]
    brand_entity    = state["brand_entity"]
    base_topics     = brand_entity.get("topics", [])

    total_mentions       = agg["total_mentions"] or 1
    target_mentions      = agg["target_mentions"]
    total_prompts        = agg["total_prompts"]  or 1
    prompts_with_target  = len(agg["prompts_with_target"])
    target_count         = len(target_mentions)

    # share_of_voice
    share_of_voice = round(target_count / total_mentions * 100, 1)

    # top3_rate — prompts where target appears ranked ≤ 3
    top3_count = sum(
        1 for m in target_mentions
        if m.get("rank") is not None and m["rank"] <= 3
    )
    top3_rate = round(top3_count / total_prompts * 100, 1)

    # query_coverage
    query_coverage = round(prompts_with_target / total_prompts * 100, 1)

    # competitor_rank — average rank when mentioned
    ranks = [m["rank"] for m in target_mentions if m.get("rank") is not None]
    competitor_rank = round(sum(ranks) / len(ranks), 1) if ranks else None

    # topic_authority — how many base topics the target covers (score /10)
    topics_covered: set[str] = set()
    for topic in base_topics:
        for mention in target_mentions:
            if topic.lower() in mention["prompt"].lower():
                topics_covered.add(topic)
                break
    topic_authority = round(
        (len(topics_covered) / len(base_topics) * 10) if base_topics else 0.0, 1
    )

    metrics = {
        "share_of_voice":   share_of_voice,
        "top3_rate":        top3_rate,
        "query_coverage":   query_coverage,
        "competitor_rank":  competitor_rank,
        "topic_authority":  topic_authority,
    }

    state["metrics"] = metrics
    logger.info("[%s] Metrics → %s", node, metrics)
    logger.info("[%s] Node complete", node)
    return state


# ──────────────────────────────────────────────────────────────
# Node 7 — BUILD RESPONSE
# Assembles the final output payload matching the output schema.
# ──────────────────────────────────────────────────────────────

def build_response(state: GeoRadarState) -> GeoRadarState:
    node = "BUILD_RESPONSE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    company = state.get("company", {})
    brand_entity = state.get("brand_entity", {})
    prompt_topic_map = state.get("prompt_topic_map", {})
    topic_metadata = state.get("topic_metadata", {}) or {}
    prompt_metadata = state.get("prompt_metadata", {}) or {}
    citations = state.get("citations", [])
    topics = state.get("topics", [])
    prompts = state.get("prompts", [])
    base_link = (brand_entity.get("url") or company.get("website") or "").rstrip("/")

    # Build per-prompt citation summary across models.
    prompt_citations: dict[str, dict] = {}
    for record in citations:
        prompt = record.get("prompt", "")
        model = record.get("model", "")
        companies = record.get("companies", []) or []
        if prompt not in prompt_citations:
            prompt_citations[prompt] = {"by_model": [], "rank_accumulator": {}}
        prompt_citations[prompt]["by_model"].append({
            "model": model,
            "companies": companies,
        })
        for c in companies:
            name = c.get("name")
            rank = c.get("rank")
            if not name or rank is None:
                continue
            acc = prompt_citations[prompt]["rank_accumulator"].setdefault(name, [])
            acc.append(rank)

    revenue_by_prompt: dict[str, dict] = {}
    for p in prompts:
        revenue_by_prompt[p] = _estimate_prompt_revenue(prompt=p)

    topic_prompt_analysis: list[dict] = []
    topics_for_analysis = sorted(set(topics + list(prompt_topic_map.values())))

    for topic in topics_for_analysis:
        topic_prompts = sorted([p for p in prompts if prompt_topic_map.get(p) == topic])
        topic_link = f"{base_link}/insights/{_slugify(topic)}" if base_link else f"/insights/{_slugify(topic)}"
        topic_reason = topic_metadata.get(topic, {}).get(
            "reason",
            "This topic targets a specific intent area where domain depth can improve citation likelihood.",
        )
        topic_use = topic_metadata.get(topic, {}).get(
            "use",
            "Use this topic to create highly focused pages and FAQ sections with direct answers.",
        )

        prompt_items: list[dict] = []
        for p in topic_prompts:
            p_data = prompt_citations.get(p, {"by_model": [], "rank_accumulator": {}})
            consensus = []
            for name, ranks in p_data.get("rank_accumulator", {}).items():
                consensus.append({
                    "name": name,
                    "avg_rank": round(sum(ranks) / len(ranks), 2),
                    "mentions": len(ranks),
                })
            consensus = sorted(consensus, key=lambda x: (x["avg_rank"], -x["mentions"], x["name"]))

            prompt_link = f"{base_link}/insights/{_slugify(p)}" if base_link else f"/insights/{_slugify(p)}"
            prompt_reason = prompt_metadata.get(p, {}).get(
                "reason",
                "This prompt is niche and intent-rich, helping specialized providers get cited.",
            )
            prompt_use = prompt_metadata.get(p, {}).get(
                "use",
                "Use this prompt as a dedicated page/query target with concise, evidence-backed answers.",
            )
            prompt_items.append({
                "prompt": p,
                "link": prompt_link,
                "reason": prompt_reason,
                "use": prompt_use,
                "cited_companies_by_model": p_data.get("by_model", []),
                "cited_companies_consensus": consensus,
                "estimated_revenue": revenue_by_prompt.get(p, {}),
            })

        topic_prompt_analysis.append({
            "topic": topic,
            "link": topic_link,
            "reason": topic_reason,
            "use": topic_use,
            "prompts": prompt_items,
        })

    result = {
        "topics":    state.get("topics",    []),
        "prompts":   state.get("prompts",   []),
        "citations": citations,
        "metrics":   state.get("metrics",   {}),
        "revenue_by_prompt": revenue_by_prompt,
        "topic_prompt_analysis": topic_prompt_analysis,
        # bonus: expose competitor breakdown for UI charts
        "competitor_analysis": state.get("aggregated", {}).get("competitor_summary", {}),
    }

    state["result"] = result
    logger.info(
        "[%s] Node complete — %d topics | %d prompts | %d citation records",
        node,
        len(result["topics"]),
        len(result["prompts"]),
        len(result["citations"]),
    )
    return state


def build_company_radar_api_response(state: GeoRadarState) -> dict:
    """
    Same JSON shape as POST /company/radar returns. Used for the HTTP response
    and as the webhook POST body (webhook_delivery reflects state at call time).
    """
    result = state.get("result") or {}
    def _stringify_response(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    return {
        "topics": result.get("topics", []),
        "prompts": result.get("prompts", []),
        "raw_responses_with_prompt": [
            {
                "prompt": item.get("prompt", ""),
                "model": item.get("model", ""),
                "response": _stringify_response(item.get("response", "")),
                "error": _stringify_response(item.get("error", "")),
            }
            for item in state.get("raw_responses", [])
        ],
        "citations": result.get("citations", []),
        "metrics": result.get("metrics", {}),
        "revenue_by_prompt": result.get("revenue_by_prompt", {}),
        "topic_prompt_analysis": result.get("topic_prompt_analysis", []),
        "webhook_delivery": state.get("webhook_delivery") or {},
    }


# ──────────────────────────────────────────────────────────────
# Node 8 — POST RESULT TO API
# Sends the same JSON as POST /company/radar to a configured HTTP endpoint.
# ──────────────────────────────────────────────────────────────

def post_result_to_api(state: GeoRadarState) -> GeoRadarState:
    node = "POST_RESULT_TO_API"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    url = (state.get("webhook_url") or "").strip()
    if not url:
        url = os.environ.get("COMPANY_RADAR_RESULT_WEBHOOK_URL", "").strip()

    if not url:
        logger.info("[%s] No webhook URL (request or env), skipping", node)
        state["webhook_delivery"] = {"skipped": True, "reason": "no_url"}
        logger.info("[%s] Node complete", node)
        return state

    payload = build_company_radar_api_response(state)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    auth = os.environ.get("COMPANY_RADAR_RESULT_WEBHOOK_AUTH", "").strip()
    if auth:
        headers["Authorization"] = auth if auth.lower().startswith("bearer ") else f"Bearer {auth}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""
            state["webhook_delivery"] = {
                "ok": True,
                "status": resp.status,
                "body_preview": text[:2000] if text else "",
            }
            logger.info("[%s] POST succeeded status=%s", node, resp.status)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        state["webhook_delivery"] = {
            "ok": False,
            "error": "http_error",
            "status": e.code,
            "body_preview": err_body[:2000],
        }
        logger.warning("[%s] POST failed HTTP %s", node, e.code)
    except urllib.error.URLError as e:
        state["webhook_delivery"] = {"ok": False, "error": "url_error", "message": str(e.reason)}
        logger.warning("[%s] POST failed: %s", node, e.reason)
    except OSError as e:
        state["webhook_delivery"] = {"ok": False, "error": "os_error", "message": str(e)}
        logger.warning("[%s] POST failed: %s", node, e)

    logger.info("[%s] Node complete", node)
    return state
