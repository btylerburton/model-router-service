"""feedback — capture "the router chose poorly" corrections and recalibrate.

A correction is simple to give: name a prompt (or the last routed turn) and the
model you think it SHOULD have gone to. From that, we:

  1. BACK-SOLVE the reasoning target: the target model sits at some cost_rank;
     the minimum reasoning score `r` that produces a band whose floor >= that
     rank becomes the desired r for that prompt. ("This should have been Opus"
     → the prompt's r should be at least the 'hard'/'extreme' edge.)
  2. STORE it as a labeled example in a JSONL feedback file (append-only, so the
     history is auditable).
  3. RECALIBRATE (an explicit, offline pass) replays all cached prompts + all
     corrections and searches the weights/band-edges that satisfy the labels
     WITHOUT regressing previously-good decisions, then writes a tuning
     overrides file the scorer loads (MODEL_ROUTER_TUNING).

Nothing changes routing silently: corrections are inert until `recalibrate`
runs and writes the overrides file. The result is fully diffable (a small JSON
of numbers) and reversible (delete the overrides file to revert to baseline).

Zero third-party deps.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .scorer import (
    BAND_EDGES,
    ModelProfile,
    band_for,
    grade_from_subscores,
    reasoning_score,
)

# Default store locations (override via env). JSONL append-only.
DEFAULT_FEEDBACK_PATH = os.environ.get(
    "MODEL_ROUTER_FEEDBACK", os.path.expanduser("~/.model-router/feedback.jsonl")
)
# The tuning overrides file the scorer reads (MODEL_ROUTER_TUNING). recalibrate
# writes here; if unset we default next to the feedback store.
DEFAULT_TUNING_PATH = os.environ.get(
    "MODEL_ROUTER_TUNING", os.path.expanduser("~/.model-router/tuning.json")
)


@dataclass
class Correction:
    prompt: str
    target_model: str
    target_rank: int
    target_r: float          # back-solved minimum r to reach target_rank's band
    source: str = "cli"      # cli | endpoint
    note: str = ""
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "prompt": self.prompt,
                "target_model": self.target_model,
                "target_rank": self.target_rank,
                "target_r": round(self.target_r, 4),
                "source": self.source,
                "note": self.note,
                "ts": self.ts,
            }
        )


def _min_r_for_rank(rank: int) -> float:
    """Smallest reasoning score whose band floor is >= rank.

    Scans the band edges low→high and returns the lowest edge whose floor meets
    the target rank. If no band reaches that rank (target higher than any floor),
    returns the highest edge (best effort).
    """
    edges = sorted(BAND_EDGES, key=lambda t: t[1])  # low → high
    for _name, edge, floor in edges:
        if floor >= rank:
            return edge
    return edges[-1][1]


def back_solve(target_model: str, profiles: list[ModelProfile]) -> tuple[int, float]:
    """Return (target_rank, target_r) for the model the user says it should be."""
    prof = next((p for p in profiles if p.id == target_model), None)
    if prof is None:
        raise ValueError(
            f"target model {target_model!r} is not in the current catalog; "
            "use an id from GET /v1/models"
        )
    return prof.cost_rank, _min_r_for_rank(prof.cost_rank)


def record_correction(
    prompt: str,
    target_model: str,
    profiles: list[ModelProfile],
    *,
    source: str = "cli",
    note: str = "",
    path: Optional[str] = None,
) -> Correction:
    """Back-solve and append a correction to the feedback store."""
    rank, target_r = back_solve(target_model, profiles)
    c = Correction(
        prompt=prompt,
        target_model=target_model,
        target_rank=rank,
        target_r=target_r,
        source=source,
        note=note,
    )
    p = path or DEFAULT_FEEDBACK_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(c.to_json() + "\n")
    return c


def load_corrections(path: Optional[str] = None) -> list[dict]:
    p = path or DEFAULT_FEEDBACK_PATH
    if not os.path.isfile(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
