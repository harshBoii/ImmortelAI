import json
import time
import logging
from typing import TypedDict
from dotenv import load_dotenv
import os
import re

from langchain_openai import ChatOpenAI

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────

logger = logging.getLogger("company_bounty_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


# ──────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm = ChatOpenAI(model="gpt-4o-mini")


# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────

class BountyState(TypedDict):
    # ── Input (caller-provided) ────────────────────────────────
    company: dict        # {name, website, linkedin}
    brand_entity: dict   # {category, topics, keywords}
    competitors: list    # ["Twilio", "Respond.io", ...]
    models: list         # ["gpt-4o", "claude-3.5", ...]

    # ── Pipeline stages (populated by nodes) ──────────────────
    niches: list         # [{topic, description, difficulty, reason?, use?}]
    niche_prompts: list  # [{topic, description, difficulty, prompts:[{prompt,reason,use}]}]
    prompts: list        # flattened prompt strings for LLM calls
    raw_responses: list  # [{prompt, model, response, ?error}]
    citations: list      # [{prompt, model, companies:[{name,product?,rank}]}]
    result: dict         # final assembled output

    # ── Session ────────────────────────────────────────────────
    session_id: str


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        parts = [
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ]
        return " ".join(parts).strip()
    return str(content).strip()


def _parse_json(raw: str):
    import re
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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _estimate_monthly_prompt_reach(prompt: str) -> float:
    p = prompt.lower()
    score = 800.0
    if any(k in p for k in ["best", "top", "alternatives", "vs", "comparison"]):
        score += 1200.0
    if any(k in p for k in ["how to", "guide", "tutorial"]):
        score += 500.0
    if any(k in p for k in ["near me", "for moms", "for kids", "for small business"]):
        score += 350.0
    score += min(len(prompt.split()) * 45.0, 450.0)
    return round(score, 2)


def _estimate_prompt_revenue(
    prompt: str,
    company_name: str,
    by_model: list[dict],
    rank_accumulator: dict[str, list[int]],
    ctr: float = 0.12,
    cvr: float = 0.03,
    aov: float = 50.0,
) -> dict:
    monthly_prompt_reach = _estimate_monthly_prompt_reach(prompt)
    company_key = (company_name or "").strip().lower()
    company_ranks = rank_accumulator.get(company_name, []) or rank_accumulator.get(company_key, [])
    if not company_ranks and company_key:
        for k, ranks in rank_accumulator.items():
            if str(k).strip().lower() == company_key:
                company_ranks = ranks
                break

    total_model_responses = max(len(by_model), 1)
    mention_rate = (len(company_ranks) / total_model_responses) if company_ranks else 0.0
    avg_rank = (sum(company_ranks) / len(company_ranks)) if company_ranks else None
    rank_factor = max(0.0, min(1.0, (11.0 - (avg_rank or 11.0)) / 10.0))
    visibility_weight = round((mention_rate * rank_factor) if company_ranks else 0.05, 4)

    estimated_revenue = round(
        monthly_prompt_reach * visibility_weight * ctr * cvr * aov,
        2,
    )
    return {
        "monthlyPromptReach": monthly_prompt_reach,
        "visibilityWeight": visibility_weight,
        "ctr": ctr,
        "cvr": cvr,
        "aov": aov,
        "estimatedRevenue": estimated_revenue,
    }


def _llm_call(prompt: str, node_name: str) -> str:
    logger.debug("[%s] LLM call started (%d chars prompt)", node_name, len(prompt))
    t0 = time.time()
    try:
        response = llm.invoke(prompt)
        raw = _extract_text(response)
        elapsed = time.time() - t0
        logger.info("[%s] LLM responded in %.2fs (%d chars)", node_name, elapsed, len(raw))
        logger.debug("[%s] Raw response:\n%s", node_name, raw[:500])
        return raw
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[%s] LLM FAILED after %.2fs: %s", node_name, elapsed, e)
        raise


# ──────────────────────────────────────────────────────────────
# Node 1 — DISCOVER NICHES
# Uses an LLM to identify high-potential niche topics
# where the company can establish topical authority for AEO.
# ──────────────────────────────────────────────────────────────

NICHE_COUNT = 5

def discover_niches(state: BountyState) -> BountyState:
    node = "DISCOVER_NICHES"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    company      = state["company"]
    brand_entity = state["brand_entity"]
    competitors  = state["competitors"]

    company_name = company.get("name", "")
    category     = brand_entity.get("category", "")
    topics       = brand_entity.get("topics", [])
    keywords     = brand_entity.get("keywords", [])

    logger.info("[%s] Company: %s | Category: %s", node, company_name, category)
    logger.info("[%s] Topics: %s | Keywords: %s", node, topics, keywords)

    prompt = f"""You are an AEO strategist identifying niche topics where a company can realistically become the cited authority in AI search engines (ChatGPT, Perplexity, Gemini).

## Company
- Name: {company_name}
- Website: {company.get("website", "N/A")}
- Category: {category}
- Core topics: {', '.join(topics) or 'not specified'}
- Keywords: {', '.join(keywords) or 'not specified'}
- Main competitors: {', '.join(competitors) or 'not specified'}

## Task
Generate exactly {NICHE_COUNT} niche topics this company should target for topical authority.

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
    "description": "<one sentence — why this is a high-value AEO opportunity for this company and why this company's function or products can become the cited authority in AI search engines >",
    "difficulty": "easy|medium|hard"
  }}
]"""
    raw = _llm_call(prompt, node)
    niches = _parse_json(raw)

    if not isinstance(niches, list):
        niches = []

    niches = [n for n in niches if isinstance(n, dict) and "topic" in n]

    state["niches"] = niches
    logger.info("[%s] Node complete — %d niches discovered", node, len(niches))
    return state


# ──────────────────────────────────────────────────────────────
# Node 2 — GENERATE NICHE PROMPTS
# For each niche topic, generates multiple AEO-focused
# prompts/questions that users might search for.
# ──────────────────────────────────────────────────────────────

PROMPTS_PER_NICHE = 3

def generate_niche_prompts(state: BountyState) -> BountyState:
    node = "GENERATE_NICHE_PROMPTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    niches       = state["niches"]
    company      = state["company"]
    brand_entity = state["brand_entity"]

    company_name = company.get("name", "")
    category     = brand_entity.get("category", "")

    logger.info("[%s] Generating prompts for %d niches", node, len(niches))

    niche_prompts: list[dict] = []

    for i, niche in enumerate(niches):
        topic       = niche.get("topic", "")
        description = niche.get("description", "")

        logger.info("[%s] Niche %d/%d: %s", node, i + 1, len(niches), topic)

        prompt = f"""You are an AEO (Answer Engine Optimization) content strategist.

Company: {company_name} (Category: {category})
Niche topic: {topic}
Context: {description}

Generate {PROMPTS_PER_NICHE} search prompts/questions that real users would type
into AI search engines (ChatGPT, Perplexity, Google AI Overview, etc.) related to this niche.

Requirements:
1. Each prompt must be a natural-language question or search query.
2. Prompts should cover different angles: comparison, how-to, best-of, use-case, pros/cons.
3. Prompts should be specific enough that a focused content page could rank for them.
4. Prompts should give the company a realistic chance of being cited if it publishes
   authoritative content on that niche.
5. Do NOT include the company name in the prompts — they should be generic searches.

Return ONLY a JSON array (no extra text). Each item must include:
{{
  "prompt": "<query text>",
  "reason": "<why this prompt can increase citation probability>",
  "use": "<what content/page should target this prompt>"
}}
"""

        raw = _llm_call(prompt, f"{node}:{topic}")
        prompts_obj = _parse_json(raw)

        prompts_list: list[dict] = []
        if isinstance(prompts_obj, list):
            for item in prompts_obj:
                if isinstance(item, str) and item.strip():
                    prompts_list.append({"prompt": item.strip(), "reason": "", "use": ""})
                elif isinstance(item, dict) and str(item.get("prompt", "")).strip():
                    prompts_list.append({
                        "prompt": str(item.get("prompt", "")).strip(),
                        "reason": str(item.get("reason", "")).strip(),
                        "use": str(item.get("use", "")).strip(),
                    })

        prompts_list = prompts_list[:PROMPTS_PER_NICHE]

        niche_prompts.append({
            "topic": topic,
            "description": description,
            "difficulty": niche.get("difficulty", "medium"),
            "prompts": prompts_list,
        })

        logger.info("[%s] → %d prompts generated for '%s'", node, len(prompts_list), topic)

    state["niche_prompts"] = niche_prompts
    flat_prompts = [p["prompt"] for np in niche_prompts for p in np.get("prompts", []) if p.get("prompt")]
    state["prompts"] = sorted({p for p in flat_prompts if p})
    total = sum(len(np["prompts"]) for np in niche_prompts)
    logger.info("[%s] Node complete — %d niches, %d total prompts", node, len(niche_prompts), total)
    return state


# ──────────────────────────────────────────────────────────────
# Node 3 — RUN PROMPTS
# Executes prompts against requested LLM models.
# ──────────────────────────────────────────────────────────────

def run_prompts(state: BountyState) -> BountyState:
    node = "RUN_PROMPTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    prompts = state.get("prompts", [])
    model_names = state.get("models", [])
    total_calls = len(prompts) * len(model_names)
    logger.info("[%s] %d prompts × %d models = %d LLM calls", node, len(prompts), len(model_names), total_calls)

    raw_responses: list[dict] = []
    for model_name in model_names:
        client = ChatOpenAI(model=model_name) if str(model_name).startswith("gpt") else llm
        for prompt_text in prompts:
            system_instruction = (
                "You are a recommendation assistant.\n"
                "Given a user query, respond with concise, practical guidance and include ranked company/product suggestions amongst indian local companies or global companies used widely in india.\n\n"
                "Rules:\n"
                "1) If the query explicitly asks for recommendations, provide a ranked list.\n"
                "2) If the query is informational (does not explicitly ask for recommendations), first answer briefly, then add\n"
                "   a section named 'Suggested products/companies' with ranked alternatives.\n"
                "3) Wherever possible, include product + company together (e.g., '1. Product ABC - XYZ Company').\n"
                "4) Keep output factual and concise.\n"
                "5) Include ranking numbers in output."
            )
            full_prompt = f"{system_instruction}\n\nQuery: {prompt_text}"
            try:
                raw = _llm_call(full_prompt, f"{node}:{model_name}")
                raw_responses.append({"prompt": prompt_text, "model": model_name, "response": raw})
            except Exception as e:
                raw_responses.append({"prompt": prompt_text, "model": model_name, "response": "", "error": str(e)})

    state["raw_responses"] = raw_responses
    logger.info("[%s] Node complete — %d raw responses stored", node, len(raw_responses))
    return state


# ──────────────────────────────────────────────────────────────
# Node 4 — PARSE RESPONSES
# LLM-only extraction of cited companies/products and ranks.
# ──────────────────────────────────────────────────────────────

def parse_responses(state: BountyState) -> BountyState:
    node = "PARSE_RESPONSES"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state.get("raw_responses", [])
    citations: list[dict] = []
    parsed_count = 0

    for item in raw_responses:
        if not item.get("response") or item.get("error"):
            continue
        prompt_text = item.get("prompt", "")
        model_name = item.get("model", "")
        response_text = item.get("response", "")

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
            companies = _parse_json(raw)
            if not isinstance(companies, list):
                companies = []
            parsed_count += 1
        except Exception as e:
            logger.error("[%s] LLM parse failed for '%s': %s", node, prompt_text, e)
            companies = []

        citations.append({"prompt": prompt_text, "model": model_name, "companies": companies})

    state["citations"] = citations
    logger.info("[%s] Node complete — %d citation records | %d parses", node, len(citations), parsed_count)
    return state


# ──────────────────────────────────────────────────────────────
# Node 5 — BUILD RESPONSE
# Assembles the final structured output.
# ──────────────────────────────────────────────────────────────

def build_response(state: BountyState) -> BountyState:
    node = "BUILD_RESPONSE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    niche_prompts = state.get("niche_prompts", [])
    citations = state.get("citations", [])
    raw_responses = state.get("raw_responses", [])
    company = state.get("company", {})
    base_link = (state.get("brand_entity", {}).get("url") or company.get("website") or "").rstrip("/")

    # build citation index by prompt
    prompt_citations: dict[str, dict] = {}
    for record in citations:
        p = record.get("prompt", "")
        if p not in prompt_citations:
            prompt_citations[p] = {"by_model": [], "rank_accumulator": {}}
        prompt_citations[p]["by_model"].append({
            "model": record.get("model", ""),
            "companies": record.get("companies", []),
        })
        for c in record.get("companies", []) or []:
            name = c.get("name")
            rank = c.get("rank")
            if not name or rank is None:
                continue
            prompt_citations[p]["rank_accumulator"].setdefault(name, []).append(rank)

    all_prompts = [p.get("prompt", "") for np in niche_prompts for p in np.get("prompts", []) if p.get("prompt")]
    revenue_by_prompt: dict[str, dict] = {}
    company_name = str(company.get("name", "")).strip()
    for p in all_prompts:
        p_data = prompt_citations.get(p, {"by_model": [], "rank_accumulator": {}})
        revenue_by_prompt[p] = _estimate_prompt_revenue(
            prompt=p,
            company_name=company_name,
            by_model=p_data.get("by_model", []),
            rank_accumulator=p_data.get("rank_accumulator", {}),
        )

    result = {
        "niches": [
            {
                "topic": np["topic"],
                "description": np.get("description", ""),
                "difficulty": np.get("difficulty", "medium"),
                "prompts": np.get("prompts", []),
                "prompt_count": len(np.get("prompts", [])),
            }
            for np in niche_prompts
        ],
        "summary": {
            "total_niches": len(niche_prompts),
            "total_prompts": sum(len(np["prompts"]) for np in niche_prompts),
            "by_difficulty": {
                "easy": sum(1 for np in niche_prompts if np.get("difficulty") == "easy"),
                "medium": sum(1 for np in niche_prompts if np.get("difficulty") == "medium"),
                "hard": sum(1 for np in niche_prompts if np.get("difficulty") == "hard"),
            },
        },
    }

    # Enriched analysis similar to company/radar
    topic_prompt_analysis: list[dict] = []
    for np in niche_prompts:
        topic = np.get("topic", "")
        if not topic:
            continue
        topic_link = f"{base_link}/insights/{_slugify(topic)}" if base_link else f"/insights/{_slugify(topic)}"
        topic_reason = np.get("description", "")
        topic_use = "Use this niche topic to create dedicated AEO pages that answer high-intent questions."

        prompt_items: list[dict] = []
        for p_obj in np.get("prompts", []):
            p_text = p_obj.get("prompt", "") if isinstance(p_obj, dict) else str(p_obj)
            if not p_text:
                continue
            p_link = f"{base_link}/insights/{_slugify(p_text)}" if base_link else f"/insights/{_slugify(p_text)}"
            prompt_items.append({
                "prompt": p_text,
                "link": p_link,
                "reason": (p_obj.get("reason") if isinstance(p_obj, dict) else "") or "Niche, intent-rich query more likely to cite specialized providers.",
                "use": (p_obj.get("use") if isinstance(p_obj, dict) else "") or "Target this prompt with a focused FAQ/article page.",
                "cited_companies_by_model": prompt_citations.get(p_text, {}).get("by_model", []),
                "estimated_revenue": revenue_by_prompt.get(p_text, {}),
            })

        topic_prompt_analysis.append({
            "topic": topic,
            "link": topic_link,
            "reason": topic_reason,
            "use": topic_use,
            "prompts": prompt_items,
        })

    result["topic_prompt_analysis"] = topic_prompt_analysis
    result["raw_responses_with_prompt"] = [
        {
            "prompt": item.get("prompt", ""),
            "model": item.get("model", ""),
            "response": item.get("response", ""),
            "error": item.get("error"),
        }
        for item in raw_responses
    ]
    responses_by_prompt: dict[str, list[dict]] = {}
    for item in raw_responses:
        p = item.get("prompt", "")
        if not p:
            continue
        responses_by_prompt.setdefault(p, []).append({
            "model": item.get("model", ""),
            "response": item.get("response", ""),
            "error": item.get("error"),
        })
    result["responses_by_prompt"] = responses_by_prompt
    result["revenue_by_prompt"] = revenue_by_prompt
    result["citations"] = citations

    state["result"] = result
    logger.info(
        "[%s] Node complete — %d niches | %d total prompts",
        node,
        result["summary"]["total_niches"],
        result["summary"]["total_prompts"],
    )
    return state
