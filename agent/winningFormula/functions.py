import json
import hmac
import time
import hashlib
import logging
import urllib.error
import urllib.request
from typing import TypedDict, List, Dict, Any, Optional

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


logger = logging.getLogger("winning_formula_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")


llm = ChatOpenAI(model="gpt-5.5")


def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        parts = [
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ]
        return " ".join(parts).strip()
    return str(content).strip()


def _parse_json_from_llm(raw: str) -> Any:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


class WinningFormulaState(TypedDict, total=False):
    # input
    job_id: str
    company_id: str
    meta_integration_id: str
    generated_at: str
    items: List[Dict[str, Any]]
    webhook_url: str
    meta_id: str

    # output
    winningFormula: Dict[str, Any]
    error: Optional[str]
    webhook_delivery: Dict[str, Any]


def normalize_request(state: WinningFormulaState) -> WinningFormulaState:
    """
    Ensure basic fields and sensible defaults are present on the state.
    """
    logger.info("=" * 60)
    logger.info("[NORMALIZE_REQUEST] job_id=%s company_id=%s", state.get("job_id"), state.get("company_id"))

    state["meta_id"] = state.get("meta_id") or state.get("meta_integration_id", "")
    state["items"] = state.get("items") or []
    state["webhook_url"] = (state.get("webhook_url") or "").strip()
    return state


def _summarize_items_for_prompt(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    for it in items:
        asset = it.get("asset") or {}
        intelligence = it.get("asset_intelligence") or {}
        metrics = it.get("meta_ad_metrics_latest") or {}

        enriched = {
            "id": asset.get("id"),
            "asset_type": asset.get("asset_type"),
            "title": asset.get("title"),
            "filename": asset.get("filename"),
            "language": intelligence.get("language"),
            "content_type": intelligence.get("content_type"),
            "duration_seconds": intelligence.get("duration_seconds"),
            "theme": intelligence.get("theme"),
            "sentiment": intelligence.get("sentiment"),
            "tone": intelligence.get("tone") or [],
            "tags": intelligence.get("tags") or [],
            "topics": intelligence.get("topics") or [],
            "target_audience": intelligence.get("target_audience") or [],
            "best_platforms": intelligence.get("best_platforms") or [],
            "visual_context": intelligence.get("visual_context") or [],
            "video_genres": intelligence.get("video_genres") or [],
            "shorts_hooks": intelligence.get("shorts_hooks") or [],
            "chapters": intelligence.get("chapters") or [],
            "impressions": metrics.get("impressions"),
            "clicks": metrics.get("clicks"),
            "ctr": metrics.get("ctr"),
            "spend": metrics.get("spend"),
            "cpc": metrics.get("cpc"),
            "roas": metrics.get("roas"),
        }

        if asset.get("asset_type") == "VIDEO":
            videos.append(enriched)
        elif asset.get("asset_type") == "IMAGE":
            images.append(enriched)

    return {"videos": videos, "images": images}


def build_winning_formula(state: WinningFormulaState) -> WinningFormulaState:
    """
    Use LLM to construct the winningFormula JSON, preserving the exact schema.
    """
    node = "BUILD_WINNING_FORMULA"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    summary = _summarize_items_for_prompt(state.get("items", []))
    prompt_payload = {
        "company_id": state.get("company_id"),
        "meta_integration_id": state.get("meta_integration_id"),
        "generated_at": state.get("generated_at"),
        "assets": summary,
    }

    system_instructions = """
You are a senior performance creative strategist.
You will receive analyzed ad assets (videos and images) with rich intelligence and performance metrics.
Your task is to synthesize a single JSON object called `winningFormula` that can guide another AI to write high-performing ad scripts.

VERY IMPORTANT:
- The JSON MUST follow exactly this schema (all keys present, even if you leave them empty):
{
  "purpose": "",
  "brand": "",

  "winning_video_dna": {
    "reference_asset": "",
    "why_it_won": "",
    "duration_seconds": null,
    "content_type": "",
    "language": "",
    "pacing": "",
    "dialogue_style": "",
    "opening_words": "",
    "offer_repetition_count": null,
    "on_screen_text_style": "",
    "color_palette": [],
    "emotional_trigger": "",
    "chapters": [
      { "seconds": "", "role": "" }
    ]
  },

  "ugc_video_dna": {
    "reference_assets": [],
    "duration_range_seconds": "",
    "content_type": "",
    "pacing": "",
    "dialogue_style": "",
    "opening_frame": "",
    "main_character": "",
    "emotional_trigger": "",
    "color_palette": [],
    "scene_topics_pool": [],
    "recurring_props": [],
    "setting": "",
    "shot_types": [],
    "chapters_pattern": [
      { "seconds": "", "role": "" }
    ],
    "brand_reveal_position": "",
    "music_character": ""
  },

  "regional_video_dna": {
    "reference_assets": [],
    "languages": [],
    "what_works": "",
    "what_fails": "",
    "critical_rule": "",
    "opener_style": {
      "Tamil": "",
      "Telugu": ""
    },
    "cultural_visual_cues": {
      "Tamil": [],
      "Telugu": []
    },
    "caption_rule": "",
    "duration_range_seconds": "",
    "pacing": ""
  },

  "static_image_dna": {
    "reference_asset": "",
    "performance": "",
    "layout": "",
    "color_palette": [],
    "hierarchy": [],
    "typography_style": "",
    "design_principle": "",
    "urgency_elements": [],
    "compliance_element": ""
  },

  "universal_creative_rules": {
    "first_3_seconds_law": "",
    "no_logo_at_start": "",
    "no_dialogue_in_ugc": "",
    "repetition_in_promos": "",
    "music_is_mandatory": "",
    "baby_is_always_present": "",
    "positive_only": "",
    "brand_safety": "",
    "promo_code_format": ""
  },

  "emotional_arc_by_content_type": {
    "DIRECT_PROMO": "",
    "UGC_LIFESTYLE": "",
    "REGIONAL": ""
  },

  "audience_signals_from_content": {
    "who_the_videos_are_made_for": [],
    "what_they_respond_to": [],
    "what_they_do_not_respond_to": []
  },

  "hooks_library": {
    "promo_hooks": [],
    "ugc_hooks": [],
    "regional_hooks": {
      "Tamil": [],
      "Telugu": []
    }
  },

  "shorts_hooks_from_top_assets": [],

  "topics_and_tags_master_list": [],

  "platforms_ranked_by_fit": [
    { "platform": "", "fit": "" }
  ],

  "generated_by": "",
  "analysis_date": "",
  "intended_consumer": ""
}

- Do NOT add or remove keys.
- You may leave arrays empty and strings as "" when you lack information, but preserve structure.
- Ground your analysis in the performance metrics (roas, ctr, impressions, spend) and intelligence fields to capture the \"winning ads\" tone.
"""

    user_prompt = (
        "Here is the analyzed asset payload as JSON. "
        "Respond with ONLY the winningFormula JSON object, no explanations, no markdown:\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
    )

    t0 = time.time()
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ]
        )
        raw = _extract_text(response)
        elapsed = time.time() - t0
        logger.info("[%s] LLM responded in %.2fs (%d chars)", node, elapsed, len(raw))
        parsed = _parse_json_from_llm(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object")
        state["winningFormula"] = parsed
        state["error"] = None
    except Exception as e:
        logger.error("[%s] FAILED to build winning formula: %s", node, e)
        state["winningFormula"] = _empty_winning_formula()
        state["error"] = str(e)

    return state


def _empty_winning_formula() -> Dict[str, Any]:
    """
    Deterministic fallback ensuring the exact schema shape even on failures.
    """
    return {
        "purpose": "",
        "brand": "",
        "winning_video_dna": {
            "reference_asset": "",
            "why_it_won": "",
            "duration_seconds": None,
            "content_type": "",
            "language": "",
            "pacing": "",
            "dialogue_style": "",
            "opening_words": "",
            "offer_repetition_count": None,
            "on_screen_text_style": "",
            "color_palette": [],
            "emotional_trigger": "",
            "chapters": [{"seconds": "", "role": ""}],
        },
        "ugc_video_dna": {
            "reference_assets": [],
            "duration_range_seconds": "",
            "content_type": "",
            "pacing": "",
            "dialogue_style": "",
            "opening_frame": "",
            "main_character": "",
            "emotional_trigger": "",
            "color_palette": [],
            "scene_topics_pool": [],
            "recurring_props": [],
            "setting": "",
            "shot_types": [],
            "chapters_pattern": [{"seconds": "", "role": ""}],
            "brand_reveal_position": "",
            "music_character": "",
        },
        "regional_video_dna": {
            "reference_assets": [],
            "languages": [],
            "what_works": "",
            "what_fails": "",
            "critical_rule": "",
            "opener_style": {"Tamil": "", "Telugu": ""},
            "cultural_visual_cues": {"Tamil": [], "Telugu": []},
            "caption_rule": "",
            "duration_range_seconds": "",
            "pacing": "",
        },
        "static_image_dna": {
            "reference_asset": "",
            "performance": "",
            "layout": "",
            "color_palette": [],
            "hierarchy": [],
            "typography_style": "",
            "design_principle": "",
            "urgency_elements": [],
            "compliance_element": "",
        },
        "universal_creative_rules": {
            "first_3_seconds_law": "",
            "no_logo_at_start": "",
            "no_dialogue_in_ugc": "",
            "repetition_in_promos": "",
            "music_is_mandatory": "",
            "baby_is_always_present": "",
            "positive_only": "",
            "brand_safety": "",
            "promo_code_format": "",
        },
        "emotional_arc_by_content_type": {
            "DIRECT_PROMO": "",
            "UGC_LIFESTYLE": "",
            "REGIONAL": "",
        },
        "audience_signals_from_content": {
            "who_the_videos_are_made_for": [],
            "what_they_respond_to": [],
            "what_they_do_not_respond_to": [],
        },
        "hooks_library": {
            "promo_hooks": [],
            "ugc_hooks": [],
            "regional_hooks": {"Tamil": [], "Telugu": []},
        },
        "shorts_hooks_from_top_assets": [],
        "topics_and_tags_master_list": [],
        "platforms_ranked_by_fit": [{"platform": "", "fit": ""}],
        "generated_by": "",
        "analysis_date": "",
        "intended_consumer": "",
    }


def _compute_signature(body: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256)
    return mac.hexdigest()


def post_result_to_webhook(state: WinningFormulaState) -> WinningFormulaState:
    """
    POST the ready/failed event to the configured webhook URL,
    following the async webhook contract.
    """
    node = "POST_RESULT_TO_WEBHOOK"
    logger.info("=" * 60)
    logger.info("[%s] Node started", node)

    url = (state.get("webhook_url") or "").strip()
    if not url:
        url = os.environ.get("WINNING_FORMULA_WEBHOOK_URL", "").strip()

    if not url:
        logger.info("[%s] No webhook URL, skipping delivery", node)
        state["webhook_delivery"] = {"skipped": True, "reason": "no_url"}
        return state

    items = state.get("items", []) or []
    asset_ids = [str((it.get("asset") or {}).get("id")) for it in items if (it.get("asset") or {}).get("id")]

    event = "meta.winning_formula.ready" if not state.get("error") else "meta.winning_formula.failed"
    payload: Dict[str, Any] = {
        "event": event,
        "job_id": state.get("job_id", ""),
        "meta_id": state.get("meta_id", ""),
        "company_id": state.get("company_id", ""),
        "meta_integration_id": state.get("meta_integration_id", ""),
        "generated_at": state.get("generated_at", ""),
        "input_summary": {
            "items_count": len(items),
            "asset_ids": asset_ids,
        },
        "winningFormula": state.get("winningFormula") if not state.get("error") else {},
        "error": state.get("error") if state.get("error") else None,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    secret = os.environ.get("WINNING_FORMULA_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["x-webhook-signature"] = _compute_signature(body, secret)

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""
            state["webhook_delivery"] = {
                "ok": True,
                "status": resp.status,
                "body_preview": text[:2000],
            }
            logger.info("[%s] Webhook POST succeeded status=%s", node, resp.status)
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
        logger.warning("[%s] Webhook POST failed HTTP %s", node, e.code)
    except urllib.error.URLError as e:
        state["webhook_delivery"] = {
            "ok": False,
            "error": "url_error",
            "message": str(e.reason),
        }
        logger.warning("[%s] Webhook POST failed: %s", node, e.reason)
    except OSError as e:
        state["webhook_delivery"] = {
            "ok": False,
            "error": "os_error",
            "message": str(e),
        }
        logger.warning("[%s] Webhook POST failed: %s", node, e)

    return state

