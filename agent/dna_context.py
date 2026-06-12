"""Format and validate brand DNA for AEO content generation."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _bullet_lines(fields: list[tuple[str, Any]]) -> str:
    lines = [f"- {label}: {value}" for label, value in fields if _non_empty(value)]
    return "\n".join(lines)


def format_communication_context(dna: dict | None) -> str:
    if not dna:
        return ""
    return _bullet_lines([
        ("Tone", dna.get("tone")),
        ("Voice", dna.get("voice")),
        ("Brand personality", dna.get("brandPersonality")),
        ("Emotional intensity", dna.get("emotionalIntensity")),
        ("Headline style", dna.get("headlineStyle")),
        ("CTA style", dna.get("ctaStyle")),
        ("Urgency level", dna.get("urgencyLevel")),
        ("Social proof usage", dna.get("socialProofUsage")),
        ("Primary messaging theme", dna.get("primaryMessagingTheme")),
        ("Secondary messaging theme", dna.get("secondaryMessagingTheme")),
        ("Avoided messaging themes", dna.get("avoidedMessagingTheme")),
        ("Reading level", dna.get("readingLevel")),
        ("Avg sentence length", dna.get("avgSentenceLength")),
        ("Paragraph density", dna.get("paragraphDensity")),
        ("Active voice percentage", dna.get("activeVoicePercentage")),
        ("Positioning statement", dna.get("positioningStatement")),
        ("Value proposition style", dna.get("valuePropositionStyle")),
        ("Differentiation strategy", dna.get("differentiationStrategy")),
        ("Intro pattern", dna.get("introPattern")),
        ("Storytelling pattern", dna.get("storytellingPattern")),
        ("Conclusion pattern", dna.get("conclusionPattern")),
    ])


def format_audience_context(dna: dict | None) -> str:
    if not dna:
        return ""
    pain = dna.get("audiencePainPoints") or []
    motivations = dna.get("audienceMotivations") or []
    objections = dna.get("audienceObjections") or []
    return _bullet_lines([
        ("Primary persona", dna.get("primaryPersona")),
        ("Secondary persona", dna.get("secondaryPersona")),
        ("Industry focus", dna.get("industryFocus")),
        ("Technical level", dna.get("technicalLevel")),
        ("Domain knowledge level", dna.get("domainKnowledgeLevel")),
        ("Pain points", ", ".join(pain) if pain else None),
        ("Motivations", ", ".join(motivations) if motivations else None),
        ("Objections to address", ", ".join(objections) if objections else None),
    ])


def format_compliance_rules(dna: dict | None) -> str:
    if not dna:
        return ""
    banned_abs = dna.get("bannedAbsoluteClaims") or []
    banned_cmp = dna.get("bannedComparativeClaims") or []
    allowed = dna.get("allowedClaims") or []
    banned_words = dna.get("bannedWords") or []
    allowed_words = dna.get("allowedWords") or []

    lines: list[str] = []
    if banned_abs:
        lines.append(f"- NEVER make these absolute claims: {', '.join(banned_abs)}")
    if banned_cmp:
        lines.append(f"- NEVER make these comparative claims: {', '.join(banned_cmp)}")
    if allowed:
        lines.append(f"- Prefer claims aligned with allowed claims: {', '.join(allowed)}")
    if banned_words:
        lines.append(f"- NEVER use these words/phrases: {', '.join(banned_words)}")
    if allowed_words:
        lines.append(f"- When natural, prefer vocabulary from: {', '.join(allowed_words)}")
    if not dna.get("fearBasedMarketingAllowed", False):
        lines.append("- Fear-based marketing is NOT allowed.")
    if not dna.get("sensationalLanguageAllowed", False):
        lines.append("- Sensational or hyperbolic language is NOT allowed.")
    if not dna.get("politicalContentAllowed", False):
        lines.append("- Political content is NOT allowed.")
    if not dna.get("religiousContentAllowed", False):
        lines.append("- Religious content is NOT allowed.")
    if not dna.get("controversialTopicsAllowed", False):
        lines.append("- Controversial topics are NOT allowed.")
    return "\n".join(lines)


def format_brand_audience_block(
    communication_dna: dict | None,
    audience_dna: dict | None,
) -> str:
    """Shared prompt appendix for content-generation nodes."""
    comm = format_communication_context(communication_dna)
    aud = format_audience_context(audience_dna)
    if not comm and not aud:
        return ""
    parts: list[str] = []
    if comm:
        parts.append(f"BRAND COMMUNICATION DNA (follow strictly):\n{comm}")
    if aud:
        parts.append(f"TARGET AUDIENCE DNA (write for this reader):\n{aud}")
    return "\n\n".join(parts)


def communication_tone_override(communication_dna: dict | None, default_tone: str) -> str:
    if not communication_dna:
        return default_tone
    tone = (communication_dna.get("tone") or "").strip()
    voice = (communication_dna.get("voice") or "").strip()
    if tone and voice:
        return f"{tone}; voice: {voice}"
    return tone or voice or default_tone


def normalize_dna_from_request(state: dict) -> dict:
    """Map camelCase request DNA keys to snake_case state keys."""
    mapping = {
        "communicationDna": "communication_dna",
        "audienceDna": "audience_dna",
        "complianceDna": "compliance_dna",
    }
    for src, dest in mapping.items():
        if src in state and state[src] is not None:
            state[dest] = state.pop(src)
        elif dest not in state:
            state[dest] = None
    return state


def _word_boundary_pattern(word: str) -> re.Pattern[str]:
    escaped = re.escape(word.strip())
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def check_compliance_deterministic(text: str, dna: dict) -> list[str]:
    failures: list[str] = []
    if not text or not dna:
        return failures

    for word in dna.get("bannedWords") or []:
        w = str(word).strip()
        if w and _word_boundary_pattern(w).search(text):
            failures.append(f"Banned word/phrase found: '{w}'")

    # Soft check: only flag if allowedWords is non-empty and none appear
    allowed_words = [str(w).strip() for w in (dna.get("allowedWords") or []) if str(w).strip()]
    if allowed_words:
        found_any = any(_word_boundary_pattern(w).search(text) for w in allowed_words)
        if not found_any:
            failures.append(
                "Content does not use any allowed vocabulary from compliance DNA"
            )
    return failures


def build_compliance_llm_prompt(content: str, claims: list, dna: dict) -> str:
    rules = format_compliance_rules(dna)
    return f"""You are a compliance review engine. You must be STRICT.

COMPLIANCE RULES:
{rules}

CONTENT TO REVIEW:
{content}

CLAIMS TO REVIEW:
{json.dumps(claims, indent=2)}

TASK:
Determine whether the content and claims violate ANY compliance rule above.
Check for banned absolute/comparative claim patterns, banned words, disallowed
content types (fear-based, sensational, political, religious, controversial),
and misalignment with allowed claims when provided.

Return a JSON object:
{{
  "compliant": <true if fully compliant, false otherwise>,
  "violations": ["<specific violation 1>", "<specific violation 2>", ...]
}}

Return ONLY the JSON object.
"""


def parse_compliance_llm_response(raw: str, parse_json_fn: Callable[[str], Any]) -> list[str]:
    parsed = parse_json_fn(raw)
    if not isinstance(parsed, dict):
        return ["Compliance LLM check returned unparseable response"]
    if parsed.get("compliant"):
        return []
    violations = parsed.get("violations") or []
    if isinstance(violations, list) and violations:
        return [str(v) for v in violations if str(v).strip()]
    return ["Content failed compliance review"]


def check_compliance_llm(
    content: str,
    claims: list,
    dna: dict,
    node: str,
    llm_invoke: Callable[[str, str], str],
    parse_json_fn: Callable[[str], Any],
) -> list[str]:
    if not dna or not content:
        return []
    prompt = build_compliance_llm_prompt(content, claims, dna)
    raw = llm_invoke(prompt, node)
    return parse_compliance_llm_response(raw, parse_json_fn)
