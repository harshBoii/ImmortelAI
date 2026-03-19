import json
import time
import logging
from typing import TypedDict
from dotenv import load_dotenv
import os

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
    niches: list         # [{topic, description}]
    niche_prompts: list  # [{topic, description, prompts: [str]}]
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

NICHE_COUNT = 10

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

    prompt = f"""You are an AEO (Answer Engine Optimization) strategist.

A company needs to discover high-potential niche topics where it can establish
topical authority and increase its chances of being cited by AI search engines
(ChatGPT, Perplexity, Gemini, etc.).

Company details:
- Name: {company_name}
- Website: {company.get("website", "N/A")}
- Category: {category}
- Core topics: {json.dumps(topics)}
- Keywords: {json.dumps(keywords)}
- Main competitors: {json.dumps(competitors)}

Generate exactly {NICHE_COUNT} highly specific niche topics for this company.

Requirements for each niche:
1. Must be specific enough to target with focused content (not generic).
2. Must be closely aligned with the company's product, expertise, or category.
3. Should represent a topic where users actively search for answers.
4. Should be areas where the company can realistically become an authority.
5. Include a mix of product-specific, use-case, and audience-specific niches.

For each niche, also assess a difficulty level — "easy", "medium", or "hard" — based on:
- The company's current market position and existing authority in that area.
- How many strong competitors already dominate the topic.
- How broad or narrow the topic is (narrower = easier to own).

Difficulty guidelines:
- "easy"   → Narrow niche with few competitors; the company already has relevant expertise.
- "medium" → Moderate competition or the company has partial coverage; achievable with focused effort.
- "hard"   → Broad topic, heavily contested by established competitors; requires significant investment.

Return ONLY a JSON array with this structure (no extra text):
[
  {{"topic": "short niche topic title", "description": "one sentence explaining why this niche is valuable for AEO", "difficulty": "easy|medium|hard"}},
  ...
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

PROMPTS_PER_NICHE = 8

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

Generate exactly {PROMPTS_PER_NICHE} search prompts/questions that real users would type
into AI search engines (ChatGPT, Perplexity, Google AI Overview, etc.) related to this niche.

Requirements:
1. Each prompt must be a natural-language question or search query.
2. Prompts should cover different angles: comparison, how-to, best-of, use-case, pros/cons.
3. Prompts should be specific enough that a focused content page could rank for them.
4. Prompts should give the company a realistic chance of being cited if it publishes
   authoritative content on that niche.
5. Do NOT include the company name in the prompts — they should be generic searches.

Return ONLY a JSON array of strings (no extra text):
["prompt 1", "prompt 2", ...]"""

        raw = _llm_call(prompt, f"{node}:{topic}")
        prompts = _parse_json(raw)

        if not isinstance(prompts, list):
            prompts = []

        prompts = [p for p in prompts if isinstance(p, str) and p.strip()]

        niche_prompts.append({
            "topic": topic,
            "description": description,
            "difficulty": niche.get("difficulty", "medium"),
            "prompts": prompts,
        })

        logger.info("[%s] → %d prompts generated for '%s'", node, len(prompts), topic)

    state["niche_prompts"] = niche_prompts
    total = sum(len(np["prompts"]) for np in niche_prompts)
    logger.info("[%s] Node complete — %d niches, %d total prompts", node, len(niche_prompts), total)
    return state


# ──────────────────────────────────────────────────────────────
# Node 3 — BUILD RESPONSE
# Assembles the final structured output.
# ──────────────────────────────────────────────────────────────

def build_response(state: BountyState) -> BountyState:
    node = "BUILD_RESPONSE"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    niche_prompts = state.get("niche_prompts", [])

    result = {
        "niches": [
            {
                "topic": np["topic"],
                "description": np.get("description", ""),
                "difficulty": np.get("difficulty", "medium"),
                "prompts": np["prompts"],
                "prompt_count": len(np["prompts"]),
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

    state["result"] = result
    logger.info(
        "[%s] Node complete — %d niches | %d total prompts",
        node,
        result["summary"]["total_niches"],
        result["summary"]["total_prompts"],
    )
    return state
