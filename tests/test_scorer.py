"""Tests for the deterministic scorer and catalog mapping."""

from __future__ import annotations

from model_router_service.catalog import from_catalog_file, from_upstream_ids
from model_router_service.scorer import (
    ModelProfile,
    detect_demands,
    score,
    shortlist,
)


def _profiles():
    return [
        ModelProfile("haiku", "Haiku", frozenset({"coding", "fast", "cheap"}), 200000, 1),
        ModelProfile("sonnet", "Sonnet", frozenset({"coding", "reasoning", "vision"}), 200000, 3),
        ModelProfile("opus", "Opus", frozenset({"coding", "reasoning", "vision", "long_context"}), 1000000, 5),
    ]


def test_detect_demands():
    assert "fast" in detect_demands("fix a quick typo")
    assert "reasoning" in detect_demands("redesign the architecture")
    assert "long_context" in detect_demands("summarize the whole repository")
    assert "vision" in detect_demands("read this screenshot")


def test_score_prefers_cheapest_covering():
    r = score("fix a quick typo", _profiles())
    assert r.suggested == "haiku"


def test_score_reasoning_excludes_cheap_only():
    r = score("refactor and redesign the auth architecture", _profiles())
    assert r.suggested == "sonnet"


def test_shortlist_hard_demand_vision_filters():
    demands, cands = shortlist("read the diagram in this screenshot", _profiles())
    assert "vision" in demands
    ids = {p.id for p in cands}
    assert "haiku" not in ids  # haiku lacks vision
    assert "sonnet" in ids and "opus" in ids


def test_shortlist_long_context_filters():
    demands, cands = shortlist("process the whole repository at once", _profiles())
    assert "long_context" in demands
    assert {p.id for p in cands} == {"opus"}


def test_score_empty_profiles():
    assert score("anything", []).suggested is None


def test_from_upstream_ids_maps_and_drops_embeddings():
    ids = ["claude_4_5_haiku", "gemini-2.5-flash", "text-embedding-005", "unknown-xyz"]
    profiles = from_upstream_ids(ids)
    got = {p.id for p in profiles}
    assert "claude_4_5_haiku" in got
    assert "gemini-2.5-flash" in got
    assert "text-embedding-005" not in got  # non-chat dropped
    assert "unknown-xyz" not in got  # no family match dropped


def test_from_catalog_file(tmp_path):
    import json

    p = tmp_path / "roster.json"
    p.write_text(json.dumps({"models": [
        {"id": "m1", "name": "M1", "capabilities": ["coding", "cheap"], "context": 1000, "cost_rank": 1},
    ]}))
    profiles = from_catalog_file(str(p))
    assert profiles[0].id == "m1"
    assert "coding" in profiles[0].capabilities
