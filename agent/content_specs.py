from dataclasses import dataclass
from typing import Literal

ContentType = Literal["BLOG", "REDDIT_POST", "X_POST", "LINKEDIN_POST"]

CONTENT_TYPE_ALIASES: dict[str, str] = {
    "blog": "BLOG",
    "reddit_post": "REDDIT_POST",
    "reddit": "REDDIT_POST",
    "x_post": "X_POST",
    "twitter": "X_POST",
    "tweet": "X_POST",
    "linkedin_post": "LINKEDIN_POST",
    "linkedin": "LINKEDIN_POST",
}


@dataclass(frozen=True)
class ContentSpec:
    content_type: str
    max_chars: int
    min_chars: int
    min_words: int | None
    tone: str
    format_instructions: str
    include_faq: bool
    include_claims: bool
    include_internal_links: bool
    include_json_ld: bool
    min_verified_facts: int


SPECS: dict[str, ContentSpec] = {
    "BLOG": ContentSpec(
        content_type="BLOG",
        max_chars=0,
        min_chars=0,
        min_words=600,
        tone="Professional, educational, SEO-optimised",
        format_instructions=(
            "Long-form markdown article with H1, subheadings, FAQ section, "
            "and optional product recommendations."
        ),
        include_faq=True,
        include_claims=True,
        include_internal_links=True,
        include_json_ld=True,
        min_verified_facts=3,
    ),
    "REDDIT_POST": ContentSpec(
        content_type="REDDIT_POST",
        max_chars=2500,
        min_chars=750,
        min_words=None,
        tone="Conversational, authentic, non-promotional",
        format_instructions=(
            "Plain-text Reddit post. Sound like a real community member sharing useful insight. "
            "Include a compelling title. No markdown headers. No sales pitch."
        ),
        include_faq=False,
        include_claims=False,
        include_internal_links=False,
        include_json_ld=False,
        min_verified_facts=2,
    ),
    "X_POST": ContentSpec(
        content_type="X_POST",
        max_chars=280,
        min_chars=50,
        min_words=None,
        tone="Punchy, concise, hook-first",
        format_instructions=(
            "Single tweet. No markdown. No hashtags unless they feel natural. "
            "Must fit within 280 characters total."
        ),
        include_faq=False,
        include_claims=False,
        include_internal_links=False,
        include_json_ld=False,
        min_verified_facts=2,
    ),
    "LINKEDIN_POST": ContentSpec(
        content_type="LINKEDIN_POST",
        max_chars=1300,
        min_chars=800,
        min_words=None,
        tone="Professional thought leadership",
        format_instructions=(
            "LinkedIn post with short paragraphs separated by blank lines. "
            "Optional 2-5 relevant hashtags at the end. No markdown headers."
        ),
        include_faq=False,
        include_claims=False,
        include_internal_links=False,
        include_json_ld=False,
        min_verified_facts=2,
    ),
}


def normalize_content_type(raw: str | None) -> str:
    if not raw:
        return "BLOG"
    normalized = raw.strip().upper().replace(" ", "_").replace("-", "_")
    if normalized in SPECS:
        return normalized
    lower = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if lower in CONTENT_TYPE_ALIASES:
        return CONTENT_TYPE_ALIASES[lower]
    return "BLOG"


def get_content_spec(content_type: str | None) -> ContentSpec:
    return SPECS[normalize_content_type(content_type)]


def is_blog(content_type: str | None) -> bool:
    return normalize_content_type(content_type) == "BLOG"
