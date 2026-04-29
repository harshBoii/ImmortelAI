# Winning Formula API Contract

This document defines:
1. The **trigger API** your app calls to start a Winning Formula build.
2. The **outbound webhook payload** your pipeline sends when the build is complete (success or failure).

The pipeline takes Meta-analyzed assets, synthesizes a `winningFormula` JSON object (matching the exact schema you provided), and sends it to your webhook endpoint.

---

## 1) Trigger API (sync ack, async processing)

### Endpoint
- **Method**: `POST`
- **URL**: `/winning-formula/from-meta-analyzed-assets`

### Headers
- `Content-Type: application/json`

### Request body
The request mirrors your Meta contract (plus an optional `webhook_url`):

```json
{
  "company_id": "string",
  "meta_integration_id": "string",
  "generated_at": "2026-04-28T12:34:56.000Z",
  "items": [
    {
      "asset": {
        "id": "string",
        "asset_type": "VIDEO|IMAGE",
        "title": "string",
        "filename": "string",
        "intelligence_status": "READY|PROCESSING|FAILED|null"
      },
      "asset_intelligence": {
        "id": "string|null",
        "asset_id": "string",
        "company_id": "string",
        "processed_at": "2026-04-28T12:34:56.000Z|null",
        "language": "string|null",
        "content_type": "string|null",
        "duration_seconds": 0,
        "theme": "string|null",
        "sentiment": "string|null",
        "intensity_score": 0,
        "spiritual_elements": false,
        "title_primary": "string|null",
        "short_summary": "string|null",
        "long_description": "string|null",
        "tags": ["string"],
        "tone": ["string"],
        "topics": ["string"],
        "target_audience": ["string"],
        "best_platforms": ["string"],
        "visual_context": ["string"],
        "video_genres": ["string"],
        "title_variants": {},
        "chapters": [],
        "shorts_hooks": [],
        "clipfox_insights": [],
        "model_version": "string|null",
        "confidence": 0
      },
      "meta_media": {
        "id": "string"
      },
      "meta_ad_metrics_latest": {
        "recorded_at": "2026-04-28T12:34:56.000Z|null",
        "date_preset": "string|null",
        "impressions": 0,
        "clicks": 0,
        "ctr": 0,
        "spend": 0,
        "cpc": null,
        "roas": null
      }
    }
  ],
  "webhook_url": "https://your-domain.com/whatever" 
}
```

Notes:
1. `webhook_url` is optional.
2. `asset_intelligence` can be `null` (pipeline tolerates missing intelligence and uses schema-safe defaults).
3. `generated_at` must be an ISO-8601 timestamp.

### Response (immediate acknowledgement)
The API returns a job id immediately; the `winningFormula` is delivered later via webhook.

```json
{
  "ok": true,
  "job_id": "uuid",
  "company_id": "string",
  "meta_integration_id": "string",
  "generated_at": "2026-04-28T12:34:56.000Z",
  "input_summary": {
    "items_count": 1,
    "asset_ids": ["string"]
  }
}
```

---

## 2) Outbound Webhook (async callback)

### Webhook URL
The pipeline POSTs to:
1. `webhook_url` from the incoming request body, or
2. `WINNING_FORMULA_WEBHOOK_URL` environment variable (if `webhook_url` not provided).

If neither is set, the pipeline skips delivery and records that internally.

### Headers
- `Content-Type: application/json`
- Optional: `x-webhook-signature`
  - If `WINNING_FORMULA_WEBHOOK_SECRET` is set, the header is:
    - `hmac_sha256_hex(body, WINNING_FORMULA_WEBHOOK_SECRET)`

### Webhook request body (success)
On success the payload is:

```json
{
  "event": "meta.winning_formula.ready",
  "job_id": "string",
  "meta_id": "string",
  "company_id": "string",
  "meta_integration_id": "string",
  "generated_at": "2026-04-28T12:34:56.000Z",
  "input_summary": {
    "items_count": 0,
    "asset_ids": ["string"]
  },
  "winningFormula": { 
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
  },
  "error": null
}
```

### Webhook request body (failure)
On failure the payload is:

```json
{
  "event": "meta.winning_formula.failed",
  "job_id": "string",
  "meta_id": "string",
  "company_id": "string",
  "meta_integration_id": "string",
  "generated_at": "2026-04-28T12:34:56.000Z",
  "input_summary": {
    "items_count": 0,
    "asset_ids": ["string"]
  },
  "winningFormula": {},
  "error": "string|null"
}
```

---

## 3) Important schema guarantees
- The `winningFormula` object sent on success is produced to follow your provided schema structure.
- Even when some intelligence/metrics are missing, the pipeline uses schema-safe defaults (`""`, `[]`, `null`) to preserve the exact shape.
- On failure, `winningFormula` is `{}` and `error` is set.

