"""utility — detect harness UTILITY calls that should bypass routing.

Agent harnesses (OpenCode / Paseo) make non-task calls through the same
OpenAI-compatible endpoint the proxy fronts: thread-title generation,
summarization, conversation compaction. These are not work to reason about —
routing them on task difficulty is meaningless, and they should go to the
cheapest model unconditionally.

Detection is on the SYSTEM prompt only (the stable persona/instruction the
harness sets), matched against known signatures. This is intentionally
conservative: an unknown call is NOT treated as utility (it routes normally),
so a false negative merely routes a cheap call normally — never the reverse.
"""

from __future__ import annotations

import re

# Signatures matched (case-insensitive) against the SYSTEM message text. Kept
# narrow and specific so a real task prompt can't accidentally match.
_UTILITY_PATTERNS = (
    r"you are a title generator",
    r"output only a thread title",
    r"generate a (brief|short) title",
    r"summariz(e|ing) (the|this) (conversation|thread|session)",
    r"you are a conversation summarizer",
    r"condense the (conversation|following messages)",
    r"compact(ing)? the (conversation|context)",
)
_UTILITY_RE = re.compile("|".join(_UTILITY_PATTERNS), re.IGNORECASE)


def is_utility_call(system_text: str) -> bool:
    """True if the system prompt matches a known harness utility signature."""
    if not system_text:
        return False
    return bool(_UTILITY_RE.search(system_text))
