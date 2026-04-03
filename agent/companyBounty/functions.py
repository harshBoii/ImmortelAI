import json
import time
import logging
from typing import TypedDict
from dotenv import load_dotenv
import os
import re
import urllib.parse

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

llm = ChatOpenAI(model="gpt-5.4-mini")

# Used only for LLM-assisted prompt generation in `generate_niche_prompts`.
llm_generate_niche_prompts = ChatOpenAI(model="gpt-5.4-mini")


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


def parse_json_from_llm(raw: str):
    """Best-effort extraction of JSON from LLM output (alias for _parse_json)."""
    return _parse_json(raw)


def _title_lead(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    for sep in ("|", " - ", " — ", " – ", "•", ":", "·"):
        if sep in t:
            t = t.split(sep)[0].strip()
            break
    return t


def _domain_core(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if not netloc:
        return ""
    return re.split(r"[.]", netloc)[0] or netloc


def _extract_candidate_name_from_tavily_result(result: dict) -> str:
    url = str(result.get("url") or "")
    title = str(result.get("title") or "")
    lead = _title_lead(title)
    if lead:
        return lead
    core = _domain_core(url)
    return core or url


def _tavily_results_from_response(response_obj) -> list:
    """Tavily payload may be a dict or a JSON string (see `run_web_search_synth` no-model path)."""
    if isinstance(response_obj, dict):
        results = response_obj.get("results") or []
        return results if isinstance(results, list) else []
    if isinstance(response_obj, str):
        s = response_obj.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            results = data.get("results") or []
            return results if isinstance(results, list) else []
    return []


def _unwrap_llm_company_list(parsed) -> list:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("companies", "items", "results", "data", "recommendations"):
            v = parsed.get(key)
            if isinstance(v, list):
                return v
    return []


def _normalize_parsed_companies(parsed) -> list[dict]:
    items = _unwrap_llm_company_list(parsed)
    out: list[dict] = []
    for i, x in enumerate(items, start=1):
        if isinstance(x, str):
            name = x.strip()
            if name:
                out.append({"name": name, "product": "", "rank": i})
            continue
        if not isinstance(x, dict):
            continue
        name = str(x.get("name") or x.get("company") or x.get("brand") or "").strip()
        if not name:
            continue
        rank = x.get("rank")
        try:
            rank_i = int(rank) if rank is not None else i
        except (TypeError, ValueError):
            rank_i = i
        product = str(x.get("product") or "").strip()
        out.append({"name": name, "product": product, "rank": rank_i})
    return out


def _parse_ranking_regex(text: str) -> list[dict]:
    """Extract numbered recommendations when JSON LLM parse fails or returns empty."""
    results: list[dict] = []
    pattern = r"(?m)^\s*(\d+)[.)]\s*\*{0,2}([^*\n:(]+?)\*{0,2}\s*(?:[:\-–—(,]|$)"
    for match in re.finditer(pattern, text):
        rank = int(match.group(1))
        name = match.group(2).strip().strip("*").strip()
        if name and len(name) > 1:
            results.append({"name": name, "product": "", "rank": rank})
    return results


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


def _llm_call(prompt: str, node_name: str, llm_client: ChatOpenAI | None = None) -> str:
    logger.debug("[%s] LLM call started (%d chars prompt)", node_name, len(prompt))
    t0 = time.time()
    client = llm_client or llm
    try:
        response = client.invoke(prompt)
        raw = _extract_text(response)
        elapsed = time.time() - t0
        logger.info("[%s] LLM responded in %.2fs (%d chars)", node_name, elapsed, len(raw))
        logger.debug("[%s] Raw response:\n%s", node_name, raw[:500])
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
    OpenAI   
    Anthropic
    Gemini   
    """
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name, api_key=os.environ.get("OPENAI_API_KEY"))

    if "claude" in model_name:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic") from exc
        return ChatAnthropic(model=model_name, api_key=os.environ.get("ANTHROPIC_API_KEY"))  # type: ignore[return-value]

    if "gemini" in model_name:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:
            raise ImportError("Install langchain-google-genai: pip install langchain-google-genai") from exc

        return ChatGoogleGenerativeAI(model=model_name, api_key=os.environ.get("GEMINI_API_KEY"))  # type: ignore[return-value]

    raise ValueError(f"Unsupported model alias: '{model_name}'")


# ──────────────────────────────────────────────────────────────
# Node 1 — DISCOVER NICHES
# Uses an LLM to identify high-potential niche topics
# where the company can establish topical authority for AEO.
# ──────────────────────────────────────────────────────────────

NICHE_COUNT = 3

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

Generate {PROMPTS_PER_NICHE} search prompts/questions that real indian users would type
into AI search engines (ChatGPT, Perplexity, Google AI Overview, etc.) related to this niche.

Requirements:
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


Return ONLY a JSON array (no extra text). Each item must include:
{{
  "prompt": "<query text>",
  "reason": "<why this prompt can increase citation probability>",
  "use": "<what content/page should target this prompt>"
}}
"""

        raw = _llm_call(prompt, f"{node}:{topic}", llm_client=llm_generate_niche_prompts)
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

def run_web_search(state: BountyState) -> BountyState:
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
            raw_responses.append({"prompt": prompt_text, "model": "tavily", "response": response})
        except Exception as e:
            logger.error("[%s] Tavily search failed for '%s' — %s", node, prompt_text, e)
            raw_responses.append({"prompt": prompt_text, "model": "tavily", "response": "", "error": str(e)})

    state["raw_responses"] = raw_responses
    logger.info("[%s] Node complete — %d raw responses stored", node, len(raw_responses))
    return state


def run_web_search_synth(state: BountyState) -> BountyState:
    """
    Convert Tavily structured results into LLM recommendation text responses
    using the same system rules as the legacy `run_prompts` node.
    """
    node = "RUN_WEB_SEARCH_SYNTH"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state.get("raw_responses", []) or []
    model_names = state.get("models", []) or []

    # If no models requested, keep Tavily responses as strings so downstream stays stable.
    if not model_names:
        for item in raw_responses:
            if isinstance(item.get("response"), dict):
                item["response"] = json.dumps(item["response"], ensure_ascii=False)
        state["raw_responses"] = raw_responses
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
        response_obj = tavily_item.get("response")
        results = []
        if isinstance(response_obj, dict):
            results = response_obj.get("results") or []
        if not isinstance(results, list):
            results = []

        web_chunks: list[str] = []
        for i, r in enumerate(results[:6], start=1):
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "")
            title = str(r.get("title") or "")
            content = str(r.get("content") or "")
            content = content[:1400] + ("..." if len(content) > 1400 else "")
            web_chunks.append(f"[Result {i}]\nTitle: {title}\nURL: {url}\nContent:\n{content}\n")
        web_context = "\n".join(web_chunks).strip()

        for model_name in model_names:
            try:
                client = get_llm_client(model_name)
            except (ValueError, ImportError) as e:
                logger.warning("[%s] Skipping model '%s' — %s", node, model_name, e)
                continue
            
            full_prompt = (
                f"{system_instruction}\n"
                f"User query: {prompt_text}\n\n"
                f"Web search evidence (use for grounding):\n{web_context or '[no results]'}\n"
            )
            try:
                raw = _llm_call(full_prompt, f"{node}:{model_name}")
                synthesized.append({"prompt": prompt_text, "model": model_name, "response": raw})
            except Exception as e:
                synthesized.append({"prompt": prompt_text, "model": model_name, "response": "", "error": str(e)})

    state["raw_responses"] = synthesized
    logger.info("[%s] Node complete — %d synthesized responses", node, len(synthesized))
    return state


# Back-compat: keep name used by older pipe wiring.
def run_prompts(state: BountyState) -> BountyState:
    state = run_web_search(state)
    state = run_web_search_synth(state)
    return state


# ──────────────────────────────────────────────────────────────
# Node 4 — PARSE RESPONSES
# LLM-only extraction of cited companies/products and ranks.
# ──────────────────────────────────────────────────────────────

def parse_responses(state: BountyState) -> BountyState:
    node = "PARSE_RESPONSES"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state.get("raw_responses") or []
    citations: list[dict] = []
    llm_parses = 0

    for item in raw_responses:
        prompt_text = item.get("prompt", "")
        model_name = str(item.get("model", ""))
        response_obj = item.get("response")

        companies: list[dict] = []

        if model_name == "tavily":
            results = _tavily_results_from_response(response_obj)
            for idx, r in enumerate(results, start=1):
                if not isinstance(r, dict):
                    continue
                name = (_extract_candidate_name_from_tavily_result(r) or "").strip()
                if not name:
                    continue
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
            response_text = str(response_obj or "")
            if response_text and not item.get("error"):
                parse_prompt = (
                    "Extract all recommended companies and/or products from the response below.\n"
                    "Assign rank by recommendation order (1 = best/first).\n"
                    "If a line contains both product and company, put the company name in 'name' and the product name in 'product'.\n\n"
                    f"Response:\n\"\"\"\n{response_text}\n\"\"\"\n\n"
                    "Return ONLY a JSON array in this format:\n"
                    '[{"name":"Company A","product":"Product X","rank":1},{"name":"Company B","product":"","rank":2}]\n'
                    "No explanation. JSON only."
                )
                try:
                    raw = _llm_call(parse_prompt, f"{node}:llm_parse")
                    parsed = parse_json_from_llm(raw)
                    companies = _normalize_parsed_companies(parsed)
                    llm_parses += 1
                    if not companies:
                        companies = _parse_ranking_regex(response_text)
                except Exception as e:
                    logger.error("[%s] LLM parse failed for '%s': %s", node, prompt_text, e)
                    companies = _parse_ranking_regex(response_text)

        citations.append(
            {
                "prompt": prompt_text,
                "model": model_name,
                "companies": companies,
            }
        )

    state["citations"] = citations
    logger.info(
        "[%s] Node complete — %d records | %d LLM parses",
        node,
        len(citations),
        llm_parses,
    )
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
