# 0001 — Deterministic scorer is the engine; a local distilled model is deferred

- Status: accepted
- Date: 2026-10-07

## Context

The service must decide, per prompt, which upstream model to route to. Three
engine classes were considered:

1. **Deterministic graded scorer** (what ships): offline, zero-dependency,
   sub-millisecond, no network call. Scores reasoning difficulty from
   depth/breadth/novelty cues → a band → a cost-rank floor.
2. **LLM-as-judge**: a cheap model call ranks candidates per prompt. Accurate but
   adds a full inference round-trip *before every turn* — the latency that made
   an all-judge build unusable beyond a POC. Kept as an **opt-in** escalation
   (`ROUTER_JUDGE_ENABLED`, default off).
3. **A small local classifier / distilled encoder** — e.g.
   [laya](https://github.com/NandhaKishorM/laya) (Apache-2.0; a non-autoregressive
   ModernBERT-large decision model, ~421M params, published on HF as
   `convaiinnovations/laya`). This is a *real, verifiable, correctly-licensed*
   candidate — unlike several other names floated during design that could not
   be verified and were not pursued.

## Decision

Ship the **deterministic scorer as the default** engine; keep the LLM judge as
opt-in. **Defer a local distilled model (laya or equivalent)** until there is
measured evidence the scorer misroutes AND the practical blockers below are
cleared.

## Why laya is deferred (not rejected)

- **Could not be benchmarked yet.** laya depends on `torch` (+`transformers`);
  the only wheel for the dev/sandbox interpreter (Python 3.14 / aarch64) is
  ~1.5 GB unpacked, which did not fit the constrained sandbox disk. It must be
  benchmarked where it would actually run.
- **Latency is a GPU figure.** The quoted ~33 ms is on a T4 GPU; cloud.gov and
  the sandbox are CPU-only, so real latency is meaningfully higher (still likely
  faster than an LLM-judge round-trip, but not free versus the sub-ms scorer).
- **Footprint & provenance.** It adds ~1.5 GB of torch + a model download to any
  deployment — a real ATO/footprint consideration against the zero-dependency
  scorer.
- **Zero-shot accuracy is modest.** Published zero-shot numbers are low on
  typed-decision sets and rise mainly after fine-tuning on domain data, which is
  a multi-week ML effort (data collection, training, eval, hosting, a model
  card) — out of scope for the current need (a ~15-model USAi catalog the
  deterministic scorer routes well).

## Revisit when

- The scorer is **measured** misrouting on real traffic (the decision log +
  the feedback/recalibrate loop are the evidence path), AND
- laya (or an equivalent) can be benchmarked on the actual target (CPU, where
  USAi is reachable) for latency and zero-shot/fine-tuned accuracy.

At that point, evaluate wiring it as a third engine behind a `ROUTER_ENGINE`
selector (scorer | judge | laya), gated so it never loads unless explicitly
enabled, mapping its score to the existing band→floor machinery.

## Consequences

- Default stays fast, offline, dependency-free, and ATO-light.
- The upgrade path is a bounded, opt-in engine seam — not a rewrite.
- This record exists so the laya evaluation is not lost and is not re-litigated
  from scratch.
