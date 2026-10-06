"""adapters — per-harness glue for the model-router toggle.

Each harness (OpenCode, a generic OpenAI-compatible env, and later Cursor /
Claude Code / Continue / etc.) differs only in WHERE its config lives, WHAT
format it is, and WHICH field holds the base URL. Everything else about the
toggle (save/restore the original, idempotence, /readyz probe, restart notice)
is harness-agnostic and lives in toggle.py.

An adapter implements the small HarnessAdapter interface below. To add a new
harness, drop a new module in this package and register it in REGISTRY — nothing
in toggle.py changes.
"""

from __future__ import annotations

from typing import Protocol


class HarnessAdapter(Protocol):
    #: short stable id, e.g. "opencode"
    name: str

    def detect(self) -> bool:
        """True if this harness appears configured on this machine (its config
        file exists). Used for fail-safe auto-selection."""
        ...

    def describe_target(self) -> str:
        """Human-readable 'what/where' this adapter edits (for status/errors)."""
        ...

    def get_base_url(self) -> str | None:
        """Current base URL this harness points its model API at (or None)."""
        ...

    def set_base_url(self, url: str) -> None:
        """Point the harness at `url` (merge-not-clobber; preserve everything
        else and the file's format where feasible)."""
        ...

    def clear_base_url(self) -> None:
        """Remove the override so the harness falls back to its own default."""
        ...

    def restart_hint(self) -> str:
        """How THIS harness picks up the change (all read at init today)."""
        ...


from .opencode import OpenCodeAdapter  # noqa: E402
from .openai_env import OpenAIEnvAdapter  # noqa: E402

# Registered adapters, in auto-detect preference order.
REGISTRY: dict[str, type] = {
    OpenCodeAdapter.name: OpenCodeAdapter,
    OpenAIEnvAdapter.name: OpenAIEnvAdapter,
}


def available() -> list[str]:
    return list(REGISTRY)


def make(name: str) -> HarnessAdapter:
    if name not in REGISTRY:
        raise KeyError(name)
    return REGISTRY[name]()


def autodetect() -> list[str]:
    """Return the names of harnesses that look configured on this machine."""
    found = []
    for n, cls in REGISTRY.items():
        try:
            if cls().detect():
                found.append(n)
        except Exception:
            continue
    return found
