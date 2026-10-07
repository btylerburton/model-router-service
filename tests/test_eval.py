"""Routing evaluation corpus — the regression guard + evidence for the scorer.

Loads tests/fixtures/routing-eval.yaml (labeled prompt -> expected band) and
scores each prompt with the DETERMINISTIC scorer.

Two kinds of case (see the fixture header):
  * status: assert     — a locked behavior; a mismatch FAILS the suite.
  * status: known_gap  — a documented heuristic limitation; the current band is
                         recorded in the fixture's `rationale` and verified to be
                         STABLE (so we notice if it drifts), but it does not fail.

Run just this file for a readable scorecard:
    PYTHONPATH=src pytest tests/test_eval.py -q
    PYTHONPATH=src pytest tests/test_eval.py -s -k report   # prints the full table
"""

from __future__ import annotations

import pathlib

import pytest

from model_router_service.scorer import reasoning_score

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "routing-eval.yaml"
_VALID_BANDS = {"none", "light", "moderate", "hard", "extreme"}


def _load_cases() -> list[dict]:
    # Minimal YAML dependency: pyyaml is already a transitive dep via the stack.
    import yaml

    data = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]
    # Validate the corpus itself (a malformed label is a corpus bug).
    for c in cases:
        assert c["expect"] in _VALID_BANDS, f"{c['id']}: bad expect {c['expect']!r}"
        assert c["status"] in {"assert", "known_gap"}, f"{c['id']}: bad status"
    return cases


_CASES = _load_cases()
_ASSERT = [c for c in _CASES if c["status"] == "assert"]
_GAPS = [c for c in _CASES if c["status"] == "known_gap"]


@pytest.mark.parametrize("case", _ASSERT, ids=[c["id"] for c in _ASSERT])
def test_asserted_band(case):
    got = reasoning_score(case["prompt"]).band
    assert got == case["expect"], (
        f"\n  case:     {case['id']}"
        f"\n  prompt:   {case['prompt'][:80]}"
        f"\n  expected: {case['expect']}"
        f"\n  got:      {got}"
        f"\n  -> routing behavior changed. If intended, update the fixture label;"
        f"\n     if not, this is a regression in the scorer."
    )


@pytest.mark.parametrize("case", _GAPS, ids=[c["id"] for c in _GAPS])
def test_known_gap_is_stable(case):
    """A known_gap must NOT silently become MORE expensive than its asserted
    target without us noticing. We allow it to sit at-or-below the target band
    (graceful cheap degradation is fine); we fail only if it over-promotes past
    the documented target — that would be an unreviewed cost increase."""
    order = ["none", "light", "moderate", "hard", "extreme"]
    got = reasoning_score(case["prompt"]).band
    assert order.index(got) <= order.index(case["expect"]), (
        f"\n  known_gap {case['id']} OVER-PROMOTED: got {got} > target {case['expect']}."
        f"\n  A gap drifting UP (more expensive) must be reviewed — update the"
        f"\n  fixture or the scorer deliberately."
    )


def test_corpus_covers_every_band():
    """Evidence completeness: the asserted set must exercise every band, so a
    reviewer sees the full spectrum is locked, not just the easy ends."""
    covered = {c["expect"] for c in _ASSERT}
    missing = _VALID_BANDS - covered
    assert not missing, f"asserted corpus does not cover bands: {sorted(missing)}"


def test_report(capsys):
    """Not an assertion — prints the full scorecard (assert + known_gap) so
    `pytest -s -k report` is a one-command view of routing behavior."""
    order = ["none", "light", "moderate", "hard", "extreme"]
    lines = ["", "ROUTING EVAL SCORECARD", "=" * 72]
    for c in sorted(_CASES, key=lambda c: (order.index(c["expect"]), c["id"])):
        rs = reasoning_score(c["prompt"])
        mark = "OK " if rs.band == c["expect"] else ("gap" if c["status"] == "known_gap" else "!! ")
        lines.append(
            f"{mark} {c['expect']:9}->{rs.band:9} r={rs.r:.2f} "
            f"[{c['status']:9}] {c['id']}"
        )
    lines.append("=" * 72)
    with capsys.disabled():
        print("\n".join(lines))
