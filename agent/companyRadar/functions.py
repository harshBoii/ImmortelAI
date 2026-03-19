import re
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


# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────

class GeoRadarState(TypedDict):
    # ── Input (caller-provided) ────────────────────────────────
    company: dict        # {name, website, linkedin}
    brand_entity: dict   # {category, topics, keywords}
    competitors: list    # ["Twilio", "Respond.io", ...]
    models: list         # ["gpt-4o", "claude-3.5", "gemini-1.5"]

    # ── Pipeline stages (populated by nodes) ──────────────────
    topics: list         # expanded topic strings
    prompts: list        # final prompt strings sent to LLMs
    raw_responses: list  # [{prompt, model, response, ?error}]
    citations: list      # [{prompt, model, companies:[{name,rank}]}]
    aggregated: dict     # intermediate aggregation data (consumed by compute_metrics)
    metrics: dict        # {share_of_voice, top3_rate, query_coverage, competitor_rank, topic_authority}
    result: dict         # final assembled output

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


# ──────────────────────────────────────────────────────────────
# Node 1 — EXPAND TOPICS
# Pure template expansion, no LLM call.
# ──────────────────────────────────────────────────────────────

def expand_topics(state: GeoRadarState) -> GeoRadarState:
    node = "EXPAND_TOPICS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    brand_entity = state["brand_entity"]
    base_topics   = brand_entity.get("topics", [])
    keywords      = brand_entity.get("keywords", [])
    category      = brand_entity.get("category", "")

    logger.info("[%s] Base topics: %s", node, base_topics)
    logger.info("[%s] Keywords: %s",    node, keywords)

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

    topics = sorted(expanded)
    state["topics"] = topics
    logger.info("[%s] Node complete — %d expanded topics", node, len(topics))
    return state


# ──────────────────────────────────────────────────────────────
# Node 2 — GENERATE PROMPTS
# Combines topic queries + competitor-alternative queries.
# No LLM call.
# ──────────────────────────────────────────────────────────────

def generate_prompts(state: GeoRadarState) -> GeoRadarState:
    node = "GENERATE_PROMPTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    topics      = state["topics"]
    competitors = state["competitors"]
    company     = state["company"]["name"]
    category    = state["brand_entity"].get("category", "")

    prompts: set[str] = set()

    # All expanded topics become prompts directly
    prompts.update(topics)

    # Competitor-alternative prompts
    for comp in competitors:
        prompts.add(f"{comp} alternatives")
        prompts.add(f"best alternatives to {comp}")
        if category:
            prompts.add(f"{comp} vs {company}")

    # Head-to-head competitor vs competitor (top 4 only to avoid explosion)
    top_comps = competitors[:4]
    for i, c1 in enumerate(top_comps):
        for c2 in top_comps[i + 1:]:
            prompts.add(f"{c1} vs {c2}")

    prompts_list = sorted(prompts)
    state["prompts"] = prompts_list
    logger.info("[%s] Node complete — %d prompts generated", node, len(prompts_list))
    return state


# ──────────────────────────────────────────────────────────────
# Node 3 — RUN PROMPTS
# Executes every prompt against every requested LLM model.
# Stores raw free-text responses.
# ──────────────────────────────────────────────────────────────

def run_prompts(state: GeoRadarState) -> GeoRadarState:
    node = "RUN_PROMPTS"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    prompts     = state["prompts"]
    model_names = state["models"]
    total_calls = len(prompts) * len(model_names)

    logger.info(
        "[%s] %d prompts × %d models = %d LLM calls",
        node, len(prompts), len(model_names), total_calls,
    )

    raw_responses: list[dict] = []

    for model_name in model_names:
        try:
            client = get_llm_client(model_name)
        except (ValueError, ImportError) as e:
            logger.warning("[%s] Skipping model '%s' — %s", node, model_name, e)
            continue

        for prompt_text in prompts:
            system_instruction = (
                "You are a helpful assistant recommending software tools and services.\n"
                "Answer the user's query with a concise numbered list of the best matching "
                "companies or tools, most relevant first.\n"
                "Format: 1. Company Name, 2. Company Name, ...\n"
                "Be factual. Do not add lengthy explanations."
            )
            full_prompt = f"{system_instruction}\n\nQuery: {prompt_text}"

            try:
                raw = _llm_call(full_prompt, f"{node}:{model_name}", client)
                raw_responses.append({
                    "prompt": prompt_text,
                    "model": model_name,
                    "response": raw,
                })
            except Exception as e:
                logger.error("[%s] %s | '%s' — %s", node, model_name, prompt_text, e)
                raw_responses.append({
                    "prompt": prompt_text,
                    "model": model_name,
                    "response": "",
                    "error": str(e),
                })

    state["raw_responses"] = raw_responses
    logger.info("[%s] Node complete — %d raw responses stored", node, len(raw_responses))
    return state


# ──────────────────────────────────────────────────────────────
# Node 4 — PARSE RESPONSES
# Extracts structured [{name, rank}] lists from raw LLM text.
# Strategy: regex first (fast/free), LLM fallback if regex yields nothing.
# ──────────────────────────────────────────────────────────────

def parse_responses(state: GeoRadarState) -> GeoRadarState:
    node = "PARSE_RESPONSES"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    raw_responses = state["raw_responses"]
    citations: list[dict] = []
    regex_hits = 0
    llm_fallbacks = 0

    for item in raw_responses:
        if not item.get("response") or item.get("error"):
            continue

        prompt_text   = item["prompt"]
        model_name    = item["model"]
        response_text = item["response"]

        # ── Regex pass ────────────────────────────────────────
        companies = _parse_ranking_regex(response_text)

        if companies:
            regex_hits += 1
            logger.debug("[%s] Regex parsed %d companies for '%s'", node, len(companies), prompt_text)
        else:
            # ── LLM fallback ──────────────────────────────────
            llm_fallbacks += 1
            logger.debug("[%s] Regex found nothing for '%s', falling back to LLM", node, prompt_text)
            fallback_prompt = (
                "Extract every company or tool name mentioned in the response below.\n"
                "For each, assign a rank based on its position (1 = first mentioned).\n\n"
                f"Response:\n\"\"\"\n{response_text}\n\"\"\"\n\n"
                "Return ONLY a JSON array:\n"
                '[{"name": "Company A", "rank": 1}, {"name": "Company B", "rank": 2}]\n'
                "No explanation. JSON only."
            )
            try:
                raw = _llm_call(fallback_prompt, f"{node}:fallback")
                companies = parse_json_from_llm(raw)
                if not isinstance(companies, list):
                    companies = []
            except Exception as e:
                logger.error("[%s] LLM fallback failed for '%s': %s", node, prompt_text, e)
                companies = []

        citations.append({
            "prompt":    prompt_text,
            "model":     model_name,
            "companies": companies,
        })

    state["citations"] = citations
    logger.info(
        "[%s] Node complete — %d records | %d regex hits | %d LLM fallbacks",
        node, len(citations), regex_hits, llm_fallbacks,
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
    company_name = state["company"]["name"].lower()
    competitors  = [c.lower() for c in state["competitors"]]

    all_mentions:    list[dict] = []
    target_mentions: list[dict] = []
    prompts_with_target: set[str] = set()

    # Per-competitor tracking
    competitor_stats: dict[str, dict] = {
        c: {"mentions": 0, "ranks": [], "prompts": set()} for c in competitors
    }

    for record in citations:
        for entry in record.get("companies", []):
            name_lower = (entry.get("name") or "").lower()
            rank       = entry.get("rank")

            mention = {
                "prompt": record["prompt"],
                "model":  record["model"],
                "name":   entry.get("name", ""),
                "rank":   rank,
            }
            all_mentions.append(mention)

            # Target company
            if name_lower == company_name:
                target_mentions.append(mention)
                prompts_with_target.add(record["prompt"])

            # Competitors
            if name_lower in competitor_stats:
                competitor_stats[name_lower]["mentions"] += 1
                if rank is not None:
                    competitor_stats[name_lower]["ranks"].append(rank)
                competitor_stats[name_lower]["prompts"].add(record["prompt"])

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

    result = {
        "topics":    state.get("topics",    []),
        "prompts":   state.get("prompts",   []),
        "citations": state.get("citations", []),
        "metrics":   state.get("metrics",   {}),
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
