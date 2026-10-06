"""catalog — load candidate model capability profiles.

Priority:
  1. ROUTER_CATALOG_PATH JSON file (operator roster) — fully explicit profiles.
  2. Upstream /models ids (discovered at startup) mapped to a built-in family
     capability map — so the service works against whatever the gateway exposes
     with zero config beyond the required upstream vars.

No host or id is hardcoded as a REQUIRED default; the family map is capability
METADATA keyed by id prefix (a reviewed judgement about model families), matched
to whatever concrete ids the live gateway returns.
"""

from __future__ import annotations

import json
from pathlib import Path

from .scorer import CAPABILITIES, ModelProfile

# Capability metadata by stable id prefix. cost_rank: 1 = cheapest tier.
_FAMILY_PROFILES: list[dict] = [
    {"prefix": "claude_4_5_haiku", "name": "Claude Haiku", "capabilities": ["coding", "fast", "cheap"], "context": 200000, "cost_rank": 1},
    {"prefix": "gemini-2.5-flash-lite", "name": "Gemini Flash Lite", "capabilities": ["coding", "fast", "cheap", "vision", "long_context"], "context": 1000000, "cost_rank": 1},
    {"prefix": "gemini-2.5-flash", "name": "Gemini Flash", "capabilities": ["coding", "fast", "cheap", "vision", "long_context"], "context": 1000000, "cost_rank": 2},
    {"prefix": "llama", "name": "Llama", "capabilities": ["coding", "fast", "cheap"], "context": 128000, "cost_rank": 1},
    {"prefix": "cohere", "name": "Cohere", "capabilities": ["coding", "cheap", "fast"], "context": 128000, "cost_rank": 1},
    {"prefix": "claude_4_5_sonnet", "name": "Claude Sonnet", "capabilities": ["coding", "reasoning", "vision"], "context": 200000, "cost_rank": 3},
    {"prefix": "claude_4_6_sonnet", "name": "Claude 4.6 Sonnet", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 3},
    {"prefix": "claude-sonnet-5", "name": "Claude Sonnet 5", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 3},
    {"prefix": "gemini-2.5-pro", "name": "Gemini Pro", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 3},
    {"prefix": "gpt-5.2", "name": "GPT-5.2", "capabilities": ["coding", "reasoning", "long_context", "vision"], "context": 400000, "cost_rank": 4},
    {"prefix": "gpt-5.4", "name": "GPT-5.4", "capabilities": ["coding", "reasoning", "long_context", "vision"], "context": 1050000, "cost_rank": 4},
    {"prefix": "gpt-5.5", "name": "GPT-5.5", "capabilities": ["coding", "reasoning", "long_context", "vision"], "context": 1050000, "cost_rank": 4},
    {"prefix": "gpt-5", "name": "GPT-5", "capabilities": ["coding", "reasoning", "long_context", "vision"], "context": 400000, "cost_rank": 4},
    {"prefix": "claude_4_7_opus", "name": "Claude 4.7 Opus", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 5},
    {"prefix": "claude_4_8_opus", "name": "Claude 4.8 Opus", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 5},
    {"prefix": "claude-opus-5", "name": "Claude Opus 5", "capabilities": ["coding", "reasoning", "vision", "long_context"], "context": 1000000, "cost_rank": 5},
    {"prefix": "claude", "name": "Claude", "capabilities": ["coding", "reasoning", "vision"], "context": 200000, "cost_rank": 3},
]

# Embedding/non-chat ids we must never route a chat completion to.
_NON_CHAT_PREFIXES = ("text-embedding", "embed")


def _match_family(model_id: str) -> dict | None:
    best: dict | None = None
    for fam in _FAMILY_PROFILES:
        if model_id.startswith(fam["prefix"]):
            if best is None or len(fam["prefix"]) > len(best["prefix"]):
                best = fam
    return best


def _profile_from_dict(d: dict) -> ModelProfile:
    caps = frozenset(c for c in d.get("capabilities", []) if c in CAPABILITIES)
    return ModelProfile(
        id=str(d["id"]),
        name=str(d.get("name", d["id"])),
        capabilities=caps,
        context=int(d.get("context", 128000)),
        cost_rank=int(d.get("cost_rank", 3)),
        input_cost=d.get("input_cost"),
    )


def from_catalog_file(path: str) -> list[ModelProfile]:
    data = json.loads(Path(path).read_text())
    raw = data["models"] if isinstance(data, dict) else data
    return [_profile_from_dict(d) for d in raw]


def from_upstream_ids(ids: list[str]) -> list[ModelProfile]:
    """Map live gateway ids to family capability profiles (chat models only)."""
    profiles: list[ModelProfile] = []
    for mid in ids:
        if any(mid.startswith(p) for p in _NON_CHAT_PREFIXES):
            continue
        fam = _match_family(mid)
        if not fam:
            continue
        profiles.append(
            ModelProfile(
                id=mid,
                name=fam["name"],
                capabilities=frozenset(fam["capabilities"]),
                context=int(fam["context"]),
                cost_rank=int(fam["cost_rank"]),
            )
        )
    return profiles
