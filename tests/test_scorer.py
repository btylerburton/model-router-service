"""Tests for the graded reasoning scorer and catalog mapping."""

from __future__ import annotations

import json

from model_router_service import scorer
from model_router_service.catalog import from_catalog_file, from_upstream_ids
from model_router_service.scorer import (
    ModelProfile,
    band_for,
    detect_demands,
    reasoning_score,
    score,
    shortlist,
)


def _profiles():
    # A spread of cost ranks with IDENTICAL capability sets at the top (the real
    # USAi situation that made the old boolean-reasoning path unreachable).
    return [
        ModelProfile("haiku", "Haiku", frozenset({"coding", "fast", "cheap"}), 200000, 1),
        ModelProfile("flash", "Flash", frozenset({"coding", "fast", "cheap", "vision", "long_context"}), 1000000, 1),
        ModelProfile("sonnet", "Sonnet", frozenset({"coding", "reasoning", "vision", "long_context"}), 1000000, 3),
        ModelProfile("gpt5", "GPT5", frozenset({"coding", "reasoning", "vision", "long_context"}), 400000, 4),
        ModelProfile("opus", "Opus", frozenset({"coding", "reasoning", "vision", "long_context"}), 1000000, 5),
    ]


def _rank(profiles, model_id):
    return next((p.cost_rank for p in profiles if p.id == model_id), None)


def test_detect_demands_capabilities_only():
    # reasoning is NOT a capability demand anymore (graded separately).
    d = detect_demands("refactor the architecture and prove correctness")
    assert "reasoning" not in d
    assert "vision" in detect_demands("read this screenshot")
    assert "long_context" in detect_demands("review the whole repository")
    assert "fast" in detect_demands("fix a quick typo")


def test_band_for_edges():
    assert band_for(0.0)[0] == "none"
    assert band_for(0.95)[0] == "extreme"
    assert band_for(0.95)[1] == 5


def test_trivial_prompt_abstains():
    r = score("fix a typo in the comment", _profiles())
    assert r.suggested is None  # band none, no hard demand → abstain


def test_mechanical_bulk_edit_abstains():
    # "rename across 3 files" is wide but mechanical — must NOT score as reasoning.
    r = score("rename a variable across 3 files", _profiles())
    assert r.suggested is None


def test_moderate_task_reaches_mid_tier():
    r = score("refactor this module and add unit tests", _profiles())
    assert r.suggested is not None
    assert _rank(_profiles(), r.suggested) >= 3  # moderate floor


def test_hard_proof_reaches_top_tier():
    # A correctness proof must be able to climb past Sonnet (the old ceiling).
    r = score("prove this Paxos variant correct with formal verification", _profiles())
    assert r.suggested is not None
    assert _rank(_profiles(), r.suggested) >= 4


def test_vision_gate_still_applies():
    demands, _rs, cands = shortlist("read the diagram in this screenshot", _profiles())
    assert "vision" in demands
    ids = {p.id for p in cands}
    assert "haiku" not in ids  # haiku lacks vision


def test_shortlist_returns_triple_and_floors():
    demands, rs, cands = shortlist(
        "prove correctness with formal verification", _profiles()
    )
    assert rs.band in {"hard", "extreme"}
    # Every candidate sits at or above the band floor.
    assert all(p.cost_rank >= rs.floor for p in cands)


def test_grade_from_subscores_depth_dominates():
    rs = scorer.grade_from_subscores(depth=1.0, breadth=0.0, novelty=0.0)
    assert rs.band in {"hard", "extreme"}  # saturated depth alone climbs high


def test_score_empty_profiles():
    assert score("anything", []).suggested is None


def test_reasoning_dict_present_for_logging():
    r = score("refactor the module and add tests", _profiles())
    assert r.reasoning is not None
    assert set(r.reasoning) >= {"r", "depth", "breadth", "novelty", "band", "floor"}


def test_from_upstream_ids_maps_and_drops_embeddings():
    ids = ["claude_4_5_haiku", "gemini-2.5-flash", "text-embedding-005", "unknown-xyz"]
    profiles = from_upstream_ids(ids)
    got = {p.id for p in profiles}
    assert "claude_4_5_haiku" in got
    assert "gemini-2.5-flash" in got
    assert "text-embedding-005" not in got
    assert "unknown-xyz" not in got


def test_from_catalog_file(tmp_path):
    p = tmp_path / "roster.json"
    p.write_text(json.dumps({"models": [
        {"id": "m1", "name": "M1", "capabilities": ["coding", "cheap"], "context": 1000, "cost_rank": 1},
    ]}))
    profiles = from_catalog_file(str(p))
    assert profiles[0].id == "m1"
    assert "coding" in profiles[0].capabilities


def test_tuning_overrides_change_bands(tmp_path, monkeypatch):
    # Raise the extreme edge so a saturated-depth prompt lands 'hard' not 'extreme'.
    f = tmp_path / "tuning.json"
    f.write_text(json.dumps({"band_edges": [
        ["extreme", 0.95, 5], ["hard", 0.75, 4],
        ["moderate", 0.50, 3], ["light", 0.28, 1], ["none", 0.0, 0],
    ]}))
    monkeypatch.setenv("MODEL_ROUTER_TUNING", str(f))
    scorer.reload_overrides()
    try:
        rs = reasoning_score("prove this Paxos variant correct with formal verification")
        # r≈0.90 now falls below the raised 0.95 extreme edge → hard.
        assert rs.band == "hard"
    finally:
        monkeypatch.delenv("MODEL_ROUTER_TUNING", raising=False)
        scorer.reload_overrides()
