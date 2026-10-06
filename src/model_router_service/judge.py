"""judge — rank candidate models for a request using a cheap LLM (the "judge").

Mirrors the hermes/jev two-call shape: before forwarding the user's request to a
chosen model, we make ONE small call to a cheap judge model that picks the best
candidate from a shortlist. The judge's own inference endpoint is configurable
(defaults to the same upstream gateway + a cheap model id).

Contract with the judge model: it is given the user's request summary and a
compact list of candidates (id + capability tags + relative cost), and must
reply with STRICT JSON: {"model": "<one id from the list>", "why": "<short>"}.
We constrain it to the provided ids and validate the reply — a hallucinated id,
malformed JSON, timeout, or any error raises, and the caller FAILS OPEN to the
deterministic scorer / default model (never crashes the turn).

Judged results are cached (LRU) by a hash of (request-fingerprint, candidate-set)
so repeated/similar prompts don't pay the judge latency twice.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass

import httpx

from .scorer import ModelProfile


@dataclass
class JudgeDecision:
    model: str
    why: str
    via: str = "judge"
    # Optional reasoning sub-scores the judge may return in the same call. When
    # present, the router uses these (via scorer.grade_from_subscores) as the
    # authoritative grade instead of the offline heuristic. None → use heuristic.
    depth: float | None = None
    breadth: float | None = None
    novelty: float | None = None

    def has_subscores(self) -> bool:
        return None not in (self.depth, self.breadth, self.novelty)


class _LRU:
    """Tiny dependency-free LRU for judged decisions."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._d: "OrderedDict[str, JudgeDecision]" = OrderedDict()

    def get(self, key: str) -> JudgeDecision | None:
        if self.maxsize <= 0 or key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value: JudgeDecision) -> None:
        if self.maxsize <= 0:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)


_JUDGE_SYSTEM = (
    "You are a model-routing judge. Given a user's coding request and a list of "
    "candidate models with capability tags and a relative cost rank (1=cheapest), "
    "choose the SINGLE cheapest candidate that can do the task well. Prefer cheaper, "
    "faster models for simple/mechanical work; reserve expensive high-reasoning "
    "models for genuinely hard tasks (architecture, cross-file refactors, hard "
    "debugging, proofs/formal verification) or large context. "
    "Also grade the task's reasoning difficulty on three axes, each 0.0-1.0:\n"
    "  depth   = multi-step derivation / proof / formal reasoning\n"
    "  breadth = cross-file / cross-module / whole-system scope\n"
    "  novelty = design-from-scratch vs. apply-a-known-pattern\n"
    "Reply with STRICT JSON only: "
    '{"model":"<exact id from the list>","why":"<one short sentence>",'
    '"depth":<0..1>,"breadth":<0..1>,"novelty":<0..1>}. '
    "The model MUST be one of the provided ids."
)


def _request_fingerprint(request: str) -> str:
    # Normalize whitespace/case so trivially-different prompts share a cache slot.
    norm = " ".join(request.lower().split())[:2000]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _cache_key(request: str, candidates: list[ModelProfile]) -> str:
    ids = ",".join(sorted(p.id for p in candidates))
    return hashlib.sha256(
        (_request_fingerprint(request) + "|" + ids).encode("utf-8")
    ).hexdigest()


def _candidates_block(candidates: list[ModelProfile]) -> str:
    lines = []
    for p in candidates:
        caps = ",".join(sorted(p.capabilities))
        lines.append(f"- {p.id} | caps: {caps} | cost_rank: {p.cost_rank} | ctx: {p.context}")
    return "\n".join(lines)


class Judge:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
        cache_size: int,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._timeout = timeout_s
        self._cache = _LRU(cache_size)

    async def decide(
        self,
        request: str,
        candidates: list[ModelProfile],
        *,
        client: httpx.AsyncClient,
    ) -> JudgeDecision:
        """Pick one candidate id via the judge model. Raises on any failure so
        the caller can fail open to the scorer."""
        if not candidates:
            raise ValueError("no candidates to judge")

        valid_ids = {p.id for p in candidates}
        key = _cache_key(request, candidates)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        user = (
            "Request:\n"
            f"{request.strip()[:4000]}\n\n"
            "Candidates:\n"
            f"{_candidates_block(candidates)}\n\n"
            'Reply with JSON only: {"model":"<id>","why":"<reason>"}'
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 120,
            "stream": False,
        }
        resp = await client.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        decision = _parse_decision(content, valid_ids)
        self._cache.put(key, decision)
        return decision


def _parse_decision(content: str, valid_ids: set[str]) -> JudgeDecision:
    """Extract and validate the judge's JSON reply. Raises on bad output."""
    text = content.strip()
    # Tolerate a fenced ```json block.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Find the first {...} object if the model added prose.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    obj = json.loads(text)
    model = str(obj["model"]).strip()
    if model not in valid_ids:
        raise ValueError(f"judge returned non-candidate model id: {model!r}")

    def _sub(key: str) -> float | None:
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return None

    return JudgeDecision(
        model=model,
        why=str(obj.get("why", ""))[:200],
        via="judge",
        depth=_sub("depth"),
        breadth=_sub("breadth"),
        novelty=_sub("novelty"),
    )
