"""recalibrate — search tuning constants that satisfy user corrections.

Given the labeled corrections (feedback.jsonl) and, optionally, a set of
"known-good" prompts to NOT regress, this pass searches a small grid of band
edges (and, optionally, signal weights) for the configuration that:

  * makes each corrected prompt reach AT LEAST its target band floor
    (so "this should have been Opus" now routes to an Opus-tier model), while
  * minimizing collateral change to the known-good prompts' decisions.

It writes the winning config to the tuning overrides file the scorer reads
(MODEL_ROUTER_TUNING). The search is deterministic and cheap — it replays
cached classifier outputs, never calls an LLM — so re-tuning costs nothing and
is fully diffable. Baseline is restored by deleting the overrides file.

This is a HEURISTIC calibrator, intentionally conservative: it only moves band
EDGES by default (the safest lever), reports exactly what it changed and which
labels it could/couldn't satisfy, and never writes a config that regresses more
good decisions than it fixes unless you pass --allow-regressions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from . import scorer
from .feedback import DEFAULT_TUNING_PATH, load_corrections
from .scorer import ModelProfile, reasoning_score


@dataclass
class RecalResult:
    satisfied: list[str]
    unsatisfied: list[str]
    regressions: list[str]
    band_edges: list
    written_to: Optional[str]
    dry_run: bool

    def summary(self) -> str:
        lines = [
            f"corrections satisfied:   {len(self.satisfied)}",
            f"corrections unsatisfied: {len(self.unsatisfied)}",
            f"good-decision regressions: {len(self.regressions)}",
        ]
        if self.unsatisfied:
            lines.append("  unsatisfied: " + ", ".join(s[:40] for s in self.unsatisfied))
        if self.regressions:
            lines.append("  regressed:   " + ", ".join(s[:40] for s in self.regressions))
        lines.append(f"band_edges: {self.band_edges}")
        lines.append(("DRY RUN — not written" if self.dry_run else f"written to {self.written_to}"))
        return "\n".join(lines)


# Candidate edge grids to try (lower-inclusive). We search the moderate/hard/
# extreme edges — the ones that control whether hard work climbs the ladder.
# Each tuple: (light, moderate, hard, extreme). The grid spans from the
# conservative baseline down to aggressive edges, so a user correction that
# "this moderate-scoring task needs a top tier" is reachable by lowering the
# hard/extreme edges beneath that task's score.
_EDGE_GRID = [
    (0.28, 0.50, 0.75, 0.90),  # baseline
    (0.28, 0.48, 0.70, 0.86),
    (0.28, 0.45, 0.68, 0.82),
    (0.25, 0.45, 0.65, 0.80),
    (0.30, 0.52, 0.72, 0.88),
    (0.25, 0.42, 0.62, 0.78),
    (0.22, 0.40, 0.58, 0.74),
    (0.22, 0.38, 0.52, 0.70),
    (0.20, 0.35, 0.50, 0.66),
    (0.18, 0.32, 0.46, 0.62),
]

_FLOORS = {"none": 0, "light": 1, "moderate": 3, "hard": 4, "extreme": 5}


def _edges_from_tuple(t: tuple) -> list:
    light, moderate, hard, extreme = t
    return [
        ["extreme", extreme, 5],
        ["hard", hard, 4],
        ["moderate", moderate, 3],
        ["light", light, 1],
        ["none", 0.0, 0],
    ]


def _band_floor_with_edges(r: float, edges: list) -> int:
    for _name, edge, floor in sorted(edges, key=lambda x: -x[1]):
        if r >= edge:
            return floor
    return 0


def recalibrate(
    profiles: list[ModelProfile],
    *,
    feedback_path: Optional[str] = None,
    good_prompts: Optional[list[str]] = None,
    tuning_path: Optional[str] = None,
    dry_run: bool = False,
    allow_regressions: bool = False,
) -> RecalResult:
    """Search edge configs; write the best to the tuning file (unless dry-run)."""
    corrections = load_corrections(feedback_path)
    good_prompts = good_prompts or []

    # Precompute each correction's reasoning r (classifier is deterministic;
    # band edges don't affect r, only the band it maps to).
    labeled = []
    for c in corrections:
        r = reasoning_score(c["prompt"]).r
        labeled.append((c["prompt"], r, int(c["target_rank"])))

    # Baseline decisions for the good prompts (current overrides in effect).
    scorer.reload_overrides()
    good_baseline = {p: scorer.score(p, profiles).suggested for p in good_prompts}

    best = None  # (satisfied_count, -regressions, -spread, edges)
    for t in _EDGE_GRID:
        edges = _edges_from_tuple(t)
        satisfied, unsatisfied = [], []
        for prompt, r, target_rank in labeled:
            floor = _band_floor_with_edges(r, edges)
            (satisfied if floor >= target_rank else unsatisfied).append(prompt)

        # Evaluate regressions on good prompts under these edges.
        regressions = []
        for p in good_prompts:
            r = reasoning_score(p).r
            floor = _band_floor_with_edges(r, edges)
            # Re-pick under this floor using the scorer's selection on a temp
            # override.
            pick = _pick_under_edges(p, profiles, edges)
            if good_baseline.get(p) is not None and pick != good_baseline[p]:
                regressions.append(p)

        key = (len(satisfied), -len(regressions), -_edge_spread(t))
        if best is None or key > best[0]:
            best = (key, edges, satisfied, unsatisfied, regressions)

    _key, edges, satisfied, unsatisfied, regressions = best

    # Guard: don't write a config that regresses more than it fixes, unless told.
    will_write = not dry_run
    if regressions and not allow_regressions and len(regressions) > len(satisfied):
        will_write = False

    written_to = None
    if will_write:
        path = tuning_path or DEFAULT_TUNING_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"band_edges": edges}, f, indent=2)
            f.write("\n")
        written_to = path
        scorer.reload_overrides()

    return RecalResult(
        satisfied=satisfied,
        unsatisfied=unsatisfied,
        regressions=regressions,
        band_edges=edges,
        written_to=written_to,
        dry_run=not will_write,
    )


def _edge_spread(t: tuple) -> float:
    # Prefer edge sets closest to baseline (least disruptive) as a tiebreak.
    base = (0.28, 0.50, 0.75, 0.90)
    return sum(abs(a - b) for a, b in zip(t, base))


def _pick_under_edges(prompt: str, profiles: list[ModelProfile], edges: list) -> Optional[str]:
    """Replay the scorer's selection with a temporary edge override."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"band_edges": edges}, tf)
        tmp = tf.name
    prev = os.environ.get("MODEL_ROUTER_TUNING")
    os.environ["MODEL_ROUTER_TUNING"] = tmp
    try:
        scorer.reload_overrides()
        return scorer.score(prompt, profiles).suggested
    finally:
        if prev is None:
            os.environ.pop("MODEL_ROUTER_TUNING", None)
        else:
            os.environ["MODEL_ROUTER_TUNING"] = prev
        os.unlink(tmp)
        scorer.reload_overrides()
