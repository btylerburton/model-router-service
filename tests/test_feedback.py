"""Tests for feedback capture (back-solving a model → reasoning target) and the
offline recalibrate pass."""

from __future__ import annotations

import json

from model_router_service import scorer
from model_router_service.catalog import from_upstream_ids
from model_router_service.feedback import (
    back_solve,
    load_corrections,
    record_correction,
)
from model_router_service.recalibrate import recalibrate


def _profiles():
    return from_upstream_ids([
        "claude_4_5_haiku", "claude_4_5_sonnet", "gemini-2.5-pro",
        "gpt-5.2", "claude_4_8_opus", "claude-opus-5",
    ])


def test_back_solve_maps_model_to_target_rank_and_r():
    profiles = _profiles()
    rank, target_r = back_solve("claude_4_8_opus", profiles)  # rank 5
    assert rank == 5
    # The min r must land in a band whose floor >= 5 (extreme).
    name, floor = scorer.band_for(target_r)
    assert floor >= 5


def test_back_solve_unknown_model_raises():
    import pytest

    with pytest.raises(ValueError):
        back_solve("totally-made-up", _profiles())


def test_record_correction_appends(tmp_path):
    profiles = _profiles()
    store = tmp_path / "feedback.jsonl"
    c = record_correction(
        "design a distributed consensus protocol", "claude_4_8_opus", profiles,
        path=str(store),
    )
    assert c.target_rank == 5
    rows = load_corrections(str(store))
    assert len(rows) == 1
    assert rows[0]["target_model"] == "claude_4_8_opus"


def test_recalibrate_satisfies_a_correction(tmp_path, monkeypatch):
    """A user says a moderate-scoring prompt should have gone UP a tier; after
    recalibrate (edge-only search), the scorer routes it at/above that tier
    without regressing a known-good trivial prompt."""
    profiles = _profiles()
    store = tmp_path / "feedback.jsonl"
    tuning = tmp_path / "tuning.json"

    # A refactor prompt scores ~moderate (r≈0.56 → rank 3). User says it needs a
    # rank-4 model (gpt-5.2). Edge-only recalibration can satisfy this by
    # lowering the 'hard' edge beneath the prompt's score.
    prompt = "refactor this module and add unit tests"
    c = record_correction(prompt, "gpt-5.2", profiles, path=str(store))
    assert c.target_rank == 4

    monkeypatch.setenv("MODEL_ROUTER_TUNING", str(tuning))
    scorer.reload_overrides()
    try:
        result = recalibrate(
            profiles,
            feedback_path=str(store),
            good_prompts=["fix a typo in the comment"],  # must stay abstain
            tuning_path=str(tuning),
        )
        assert prompt in result.satisfied
        assert result.written_to is not None
        # Known-good trivial prompt did not regress into a paid model.
        assert "fix a typo in the comment" not in result.regressions
        # After writing, the scorer honors the new edges.
        scorer.reload_overrides()
        pick = scorer.score(prompt, profiles).suggested
        rank = next((p.cost_rank for p in profiles if p.id == pick), 0)
        assert rank >= 4  # climbed to the user's target tier
    finally:
        monkeypatch.delenv("MODEL_ROUTER_TUNING", raising=False)
        scorer.reload_overrides()


def test_recalibrate_dry_run_writes_nothing(tmp_path):
    profiles = _profiles()
    store = tmp_path / "feedback.jsonl"
    tuning = tmp_path / "tuning.json"
    record_correction("prove X", "claude_4_8_opus", profiles, path=str(store))
    result = recalibrate(
        profiles, feedback_path=str(store), tuning_path=str(tuning), dry_run=True
    )
    assert result.dry_run is True
    assert not tuning.exists()
