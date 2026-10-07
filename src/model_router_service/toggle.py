"""toggle — flip a coding harness between the direct USAi gateway and the
model-router proxy, by editing that harness's AGENT-SIDE config.

Harness-agnostic orchestration. The per-harness specifics (where the config is,
which field holds the base URL) live in adapters/ — OpenCode today, a generic
OpenAI-compatible env adapter, and more later. Everything here is shared:

  * save the ORIGINAL base URL the first time we flip ON (sidecar), so OFF
    restores it exactly;
  * idempotent (ON when already on / OFF when already off is a no-op);
  * probe the proxy /readyz and warn if it isn't up;
  * print an honest RESTART notice (no harness hot-swaps a running session).

Harness selection: `--harness <name>` explicit, else auto-detect. Auto-detect
FAILS SAFE: if zero or more-than-one harness is detected, we refuse to guess and
ask for `--harness`, rather than edit the wrong tool's config.

Pure standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from . import adapters

DEFAULT_PROXY_URL = os.environ.get("MODEL_ROUTER_URL", "http://127.0.0.1:8080")
SIDECAR = os.environ.get(
    "MODEL_ROUTER_TOGGLE_STATE",
    os.path.expanduser("~/.model-router/toggle-state.json"),
)
# A durable, auditable record of the LAST toggle action per harness: what state
# was set, to what target, when, and by what. `status` reads this (not just the
# live config), the agent-context acknowledgement is generated from it, and it is
# the on/off audit trail. Separate from SIDECAR (which only stores the original
# URL to restore); kept in the same dir.
STATE_LOG = os.environ.get(
    "MODEL_ROUTER_TOGGLE_LOG",
    os.path.expanduser("~/.model-router/toggle-log.jsonl"),
)


# --- shared state (original base URLs, keyed per harness) ------------------
def _read_sidecar() -> dict:
    if os.path.isfile(SIDECAR):
        try:
            return json.loads(open(SIDECAR, encoding="utf-8").read())
        except Exception:
            return {}
    return {}


def _write_sidecar(d: dict) -> None:
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    tmp = SIDECAR + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, SIDECAR)


def _probe(proxy: str) -> bool:
    try:
        with urllib.request.urlopen(proxy.rstrip("/") + "/readyz", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _record_state(harness: str, *, on: bool, target: str | None) -> None:
    """Append an auditable on/off record. Best-effort; never raises."""
    try:
        os.makedirs(os.path.dirname(STATE_LOG), exist_ok=True)
        with open(STATE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "harness": harness,
                "state": "ROUTER ACTIVE" if on else "ROUTER BYPASSED",
                "on": on,
                "target": target,
                "ts": time.time(),
            }) + "\n")
    except Exception:
        pass


def _last_state(harness: str) -> dict | None:
    """The most recent recorded toggle for a harness (or None)."""
    if not os.path.isfile(STATE_LOG):
        return None
    last = None
    try:
        for line in open(STATE_LOG, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("harness") == harness:
                last = rec
    except Exception:
        return None
    return last


def _is_proxy_url(url: str | None, proxy: str) -> bool:
    return bool(url) and url.rstrip("/").startswith(proxy.rstrip("/"))


def _target_url(proxy: str) -> str:
    return proxy.rstrip("/") + "/v1"


# --- harness selection -----------------------------------------------------
def _select(args) -> adapters.HarnessAdapter | None:
    if args.harness:
        if args.harness not in adapters.available():
            print(
                f"unknown harness {args.harness!r}; available: {', '.join(adapters.available())}",
                file=sys.stderr,
            )
            return None
        return adapters.make(args.harness)
    found = adapters.autodetect()
    if len(found) == 1:
        return adapters.make(found[0])
    if not found:
        print(
            "no harness detected; pass --harness "
            f"({', '.join(adapters.available())}).",
            file=sys.stderr,
        )
    else:
        print(
            f"multiple harnesses detected ({', '.join(found)}); pass --harness to choose.",
            file=sys.stderr,
        )
    return None


# --- commands --------------------------------------------------------------
def cmd_on(args) -> int:
    proxy = args.url or DEFAULT_PROXY_URL
    ad = _select(args)
    if ad is None:
        return 2
    try:
        current = ad.get_base_url()
    except FileNotFoundError:
        print(
            f"{ad.name}: config not found at {ad.describe_target()}.\n"
            "  This path lives INSIDE the sandbox, not on your host. Run the "
            "toggle inside the sandbox (e.g. `acq exec <name> -- model-router-toggle on`),\n"
            "  or point it at the right file with OPENCODE_GLOBAL_CONFIG=<path>.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"{ad.name}: cannot read config ({exc}); not modifying.", file=sys.stderr)
        return 1

    if _is_proxy_url(current, proxy):
        print(f"{ad.name}: already ON (baseURL={current}). No change.")
        return 0

    # Save original the first time we flip on.
    if current and not _is_proxy_url(current, proxy):
        sc = _read_sidecar()
        sc.setdefault(ad.name, {})["original_base_url"] = current
        _write_sidecar(sc)

    target = _target_url(proxy)
    try:
        ad.set_base_url(target)
    except Exception as exc:
        print(f"{ad.name}: {exc}", file=sys.stderr)
        return 1

    print(f"{ad.name}: router ON -> {target}")
    _record_state(ad.name, on=True, target=target)
    _after_write(ad, proxy, on=True, target=target)
    return 0


def cmd_off(args) -> int:
    proxy = args.url or DEFAULT_PROXY_URL
    ad = _select(args)
    if ad is None:
        return 2
    try:
        current = ad.get_base_url()
    except FileNotFoundError:
        print(f"{ad.name}: config not found; nothing to turn off.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"{ad.name}: cannot read config ({exc}); not modifying.", file=sys.stderr)
        return 1

    if not _is_proxy_url(current, proxy):
        print(f"{ad.name}: already OFF (baseURL={current}). No change.")
        return 0

    original = _read_sidecar().get(ad.name, {}).get("original_base_url")
    try:
        if original:
            ad.set_base_url(original)
        else:
            ad.clear_base_url()
    except Exception as exc:
        print(f"{ad.name}: {exc}", file=sys.stderr)
        return 1

    restored = original or "(harness default)"
    print(f"{ad.name}: router OFF -> {restored}")
    _record_state(ad.name, on=False, target=original)
    _after_write(ad, proxy, on=False, target=original)
    return 0


def cmd_status(args) -> int:
    proxy = args.url or DEFAULT_PROXY_URL
    # status with no --harness reports ALL detected harnesses.
    names = [args.harness] if args.harness else (adapters.autodetect() or adapters.available())
    for n in names:
        try:
            ad = adapters.make(n)
        except KeyError:
            print(f"unknown harness {n!r}", file=sys.stderr)
            continue
        detected = ad.detect()
        try:
            base = ad.get_base_url() if detected else None
        except Exception:
            base = None
        on = _is_proxy_url(base, proxy)
        print(f"[{n}] {'detected' if detected else 'not configured'}")
        print(f"    target:  {ad.describe_target()}")
        print(f"    baseURL: {base}")
        print(f"    router:  {'ON' if on else 'OFF'}  ({'ROUTER ACTIVE' if on else 'ROUTER BYPASSED'})")
        # The last recorded toggle action (audit trail), if any.
        last = _last_state(n)
        if last:
            when = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(last.get("ts", 0)))
            print(f"    last set: {last.get('state')} @ {when} -> {last.get('target')}")
    if args.probe or any(_is_proxy_url(adapters.make(n).get_base_url() if adapters.make(n).detect() else None, proxy) for n in names if n in adapters.available()):
        print(f"proxy {proxy}/readyz: {'reachable' if _probe(proxy) else 'NOT reachable'}")
    return 0


def cmd_ack(args) -> int:
    """Emit a one-line, user-facing acknowledgement of routing state for the
    agent to surface at session start (A3). Suppressed to NOTHING when
    MODEL_ROUTER_ACK=off|0|false (log-only mode) — the state log still records
    everything; only the user-facing note is silenced.

    Designed to be written into the harness agent-context on session start.
    """
    mode = os.environ.get("MODEL_ROUTER_ACK", "on").strip().lower()
    if mode in {"off", "0", "false", "no"}:
        return 0  # log-only: no user-facing acknowledgement
    proxy = args.url or DEFAULT_PROXY_URL
    names = [args.harness] if args.harness else (adapters.autodetect() or adapters.available())
    for n in names:
        try:
            ad = adapters.make(n)
        except KeyError:
            continue
        if not ad.detect():
            continue
        try:
            base = ad.get_base_url()
        except Exception:
            base = None
        on = _is_proxy_url(base, proxy)
        if on:
            print(
                "Model routing is ON for this session: prompts are auto-routed to "
                "the model that best fits each request (the chosen model is in the "
                "`x-model-router-decision` response header; the decision log is at "
                "~/.local/state/model-router-proxy/decisions.jsonl). "
                "To stop routing: `model-router-toggle off` (takes effect on the "
                "next session). To pin a model for one turn, send the "
                "`x-model-router-bypass: 1` header with an explicit model."
            )
        else:
            print(
                "Model routing is OFF for this session (talking directly to USAi). "
                "Enable with `model-router-toggle on` (takes effect next session)."
            )
    return 0


def _after_write(ad, proxy: str, *, on: bool, target: str | None) -> None:
    # option C for the env adapter: also print an export line for immediate use.
    exporter = getattr(ad, "export_line", None)
    if callable(exporter):
        print("\n# apply to the current shell now:")
        print(exporter(target if on else None))
    print(f"\nNOTE: {ad.restart_hint()}", file=sys.stderr)
    if on and not _probe(proxy):
        print(
            f"warning: proxy {proxy}/readyz not reachable — start model-router-service "
            "before your next turn.",
            file=sys.stderr,
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="model-router-toggle",
        description="Toggle a coding harness between the direct gateway and the model-router proxy.",
    )
    # Global flags. Also attached to each subparser under distinct dests so they
    # work BEFORE or AFTER the subcommand — `--harness opencode on` and
    # `on --harness opencode` both parse. A post-subcommand value wins.
    p.add_argument("--harness", dest="harness", help=f"one of: {', '.join(adapters.available())} (default: auto-detect)")
    p.add_argument("--url", dest="url", help=f"proxy base URL (default {DEFAULT_PROXY_URL})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("--harness", dest="harness_sub", help=argparse.SUPPRESS)
        sp.add_argument("--url", dest="url_sub", help=argparse.SUPPRESS)
        return sp

    s_on = _common(sub.add_parser("on", help="route via the proxy"))
    s_on.set_defaults(func=cmd_on)
    s_off = _common(sub.add_parser("off", help="restore the direct gateway"))
    s_off.set_defaults(func=cmd_off)
    s_st = _common(sub.add_parser("status", help="show current mode (all detected harnesses)"))
    s_st.add_argument("--probe", action="store_true", help="also probe the proxy /readyz")
    s_st.set_defaults(func=cmd_status)
    s_ack = _common(sub.add_parser("ack", help="emit a session-start routing acknowledgement for the agent to surface (A3; silenced by MODEL_ROUTER_ACK=off)"))
    s_ack.set_defaults(func=cmd_ack)

    args = p.parse_args(argv)
    # Merge pre/post-subcommand flags; post-subcommand value wins when given.
    args.harness = getattr(args, "harness_sub", None) or args.harness
    args.url = getattr(args, "url_sub", None) or args.url
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
