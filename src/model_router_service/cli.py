"""cli — offline steering commands for the router's reasoning heuristic.

Two subcommands:

  feedback   Record "the router chose poorly here; it should have been <MODEL>".
             Back-solves the reasoning target from the model's cost tier and
             appends a labeled correction to the feedback store. Inert until you
             recalibrate.

  recalibrate  Replay the corrections (and optional known-good prompts) and
               search band edges that satisfy the corrections without regressing
               good decisions; write the tuning overrides file the service reads.

Both are OFFLINE: they operate on the feedback/catalog files and never call an
LLM. The catalog of candidate models is loaded the same way the service does
(a roster file via ROUTER_CATALOG_PATH, else a fixed built-in family list — see
--models to pass ids explicitly when no gateway is reachable).

Usage:
  model-router feedback --prompt "…" --model claude_4_8_opus [--note "…"]
  model-router feedback --last --model claude_4_8_opus            # from decision log
  model-router recalibrate [--good-prompts file.txt] [--dry-run] [--allow-regressions]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .catalog import from_catalog_file, from_upstream_ids
from .feedback import (
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_TUNING_PATH,
    load_corrections,
    record_correction,
)
from .recalibrate import recalibrate

# Built-in id list used only when neither a roster nor explicit --models is
# given — the known USAi families, so back-solving a rank works offline.
_FALLBACK_IDS = [
    "claude_4_5_haiku", "gemini-2.5-flash-lite", "gemini-2.5-flash",
    "llama_4_maverick", "cohere_english_v3", "claude_4_5_sonnet",
    "claude_4_6_sonnet", "claude-sonnet-5", "gemini-2.5-pro",
    "gpt-5.2", "gpt-5.4", "gpt-5.5", "claude_4_7_opus", "claude_4_8_opus",
    "claude-opus-5",
]


def _load_profiles(models_arg: str | None):
    if models_arg:
        ids = [m.strip() for m in models_arg.split(",") if m.strip()]
        return from_upstream_ids(ids)
    roster = os.environ.get("ROUTER_CATALOG_PATH")
    if roster and os.path.isfile(roster):
        return from_catalog_file(roster)
    return from_upstream_ids(_FALLBACK_IDS)


def _last_prompt_from_log() -> str | None:
    """Read the most recent routed prompt from the decision log, if present."""
    log_path = os.environ.get(
        "MODEL_ROUTER_DECISION_LOG", os.path.expanduser("~/.model-router/decisions.jsonl")
    )
    if not os.path.isfile(log_path):
        return None
    last = None
    for line in open(log_path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return (last or {}).get("prompt")


def _cmd_feedback(args: argparse.Namespace) -> int:
    profiles = _load_profiles(args.models)
    prompt = args.prompt
    if args.last and not prompt:
        prompt = _last_prompt_from_log()
        if not prompt:
            print(
                "no --prompt given and no decision log to read --last from "
                "(set MODEL_ROUTER_DECISION_LOG or pass --prompt).",
                file=sys.stderr,
            )
            return 2
    if not prompt:
        print("provide --prompt \"…\" or --last", file=sys.stderr)
        return 2
    try:
        c = record_correction(
            prompt, args.model, profiles, source="cli", note=args.note or "",
            path=args.feedback,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"recorded: prompt→{args.model} (rank {c.target_rank}, target r≥{c.target_r:.2f})\n"
        f"stored in {args.feedback or DEFAULT_FEEDBACK_PATH}\n"
        f"run `model-router recalibrate` to apply."
    )
    return 0


def _cmd_recalibrate(args: argparse.Namespace) -> int:
    profiles = _load_profiles(args.models)
    good = []
    if args.good_prompts and os.path.isfile(args.good_prompts):
        good = [ln.strip() for ln in open(args.good_prompts) if ln.strip()]
    result = recalibrate(
        profiles,
        feedback_path=args.feedback,
        good_prompts=good,
        tuning_path=args.tuning,
        dry_run=args.dry_run,
        allow_regressions=args.allow_regressions,
    )
    print(result.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-router", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("feedback", help="record a routing correction by target model")
    f.add_argument("--prompt", help="the prompt that routed poorly")
    f.add_argument("--last", action="store_true", help="use the last routed prompt from the decision log")
    f.add_argument("--model", required=True, help="the model it SHOULD have used (an id from /v1/models)")
    f.add_argument("--note", help="optional note stored with the correction")
    f.add_argument("--models", help="comma-separated candidate ids (else roster/built-in)")
    f.add_argument("--feedback", help=f"feedback store path (default {DEFAULT_FEEDBACK_PATH})")
    f.set_defaults(func=_cmd_feedback)

    r = sub.add_parser("recalibrate", help="search + write tuning overrides from corrections")
    r.add_argument("--good-prompts", help="file of known-good prompts (one per line) to NOT regress")
    r.add_argument("--models", help="comma-separated candidate ids (else roster/built-in)")
    r.add_argument("--feedback", help=f"feedback store path (default {DEFAULT_FEEDBACK_PATH})")
    r.add_argument("--tuning", help=f"tuning overrides output path (default {DEFAULT_TUNING_PATH})")
    r.add_argument("--dry-run", action="store_true", help="search and report, but do not write")
    r.add_argument("--allow-regressions", action="store_true", help="write even if it regresses more good decisions than it fixes")
    r.set_defaults(func=_cmd_recalibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
