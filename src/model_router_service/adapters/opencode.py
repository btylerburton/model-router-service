"""OpenCode adapter — edits provider.<PROVIDER>.options.baseURL in the
agent-side global opencode.jsonc (JSONC), merge-not-clobber.

The durable config in a sandbox is the KIT-MERGED global file
(/home/agent/.config/opencode/opencode.jsonc), not a host copy — so we edit
that. We parse JSONC tolerantly but only WRITE when the file parses, round-
tripping as plain JSON (OpenCode accepts JSON; comments are dropped on a write,
same limitation the usai-provider kit's merge accepts).
"""

from __future__ import annotations

import json
import os
import re


def _default_config() -> str:
    return os.environ.get(
        "OPENCODE_GLOBAL_CONFIG",
        os.path.expanduser("~/.config/opencode/opencode.jsonc"),
    )


def _strip_jsonc(text: str) -> str:
    """Strip JSONC comments and trailing commas, as a SINGLE PASS that tracks
    string state across the whole document (not per line).

    A per-line scanner desyncs on real OpenCode configs: a string value that
    contains `//` (a URL, or a comment-like phrase), or `//` inside one line
    while a `"` opened on a previous line, made the scanner treat code as a
    comment (or vice versa) and drop a brace/comma — leaving two top-level
    fragments and a json "Extra data" error. This walks the text once:
      * inside a string, only `\\` escaping and the closing `"` matter;
      * outside a string, `//` runs to end-of-line and `/* */` to its close.
    Then trailing commas before } or ] are removed.
    """
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        # not in a string
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            # line comment: skip to newline (keep the newline)
            j = text.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            # block comment: skip to the closing */
            j = text.find("*/", i + 2)
            i = (j + 2) if j != -1 else n
            continue
        out.append(ch)
        i += 1
    joined = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", joined)


class OpenCodeAdapter:
    name = "opencode"

    def __init__(self, config_path: str | None = None, provider: str | None = None) -> None:
        self.config_path = config_path or _default_config()
        self.provider = provider or os.environ.get("MODEL_ROUTER_PROVIDER", "usai")

    # --- interface ---------------------------------------------------------
    def detect(self) -> bool:
        return os.path.isfile(self.config_path)

    def describe_target(self) -> str:
        return f"{self.config_path} (provider.{self.provider}.options.baseURL)"

    def get_base_url(self) -> str | None:
        data = self._load()
        opts = self._opts(data, create=False)
        return opts.get("baseURL") if opts else None

    def set_base_url(self, url: str) -> None:
        data = self._load()
        opts = self._opts(data, create=True)
        if opts is None:
            raise RuntimeError(
                f"provider '{self.provider}' not found in {self.config_path}; "
                "is the usai-provider kit applied?"
            )
        opts["baseURL"] = url
        self._save(data)

    def clear_base_url(self) -> None:
        data = self._load()
        opts = self._opts(data, create=False)
        if opts and "baseURL" in opts:
            opts.pop("baseURL", None)
            self._save(data)

    def restart_hint(self) -> str:
        return "restart your OpenCode session (baseURL is read at provider init)."

    # --- helpers -----------------------------------------------------------
    def _load(self) -> dict:
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(self.config_path)
        raw = open(self.config_path, encoding="utf-8").read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(_strip_jsonc(raw))

    def _save(self, data: dict) -> None:
        tmp = self.config_path + ".model-router-toggle.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, self.config_path)

    def _opts(self, data: dict, *, create: bool) -> dict | None:
        prov = data.get("provider")
        if not isinstance(prov, dict) or self.provider not in prov:
            return None
        entry = prov[self.provider]
        if not isinstance(entry, dict):
            return None
        opts = entry.get("options")
        if opts is None and create:
            opts = entry["options"] = {}
        return opts if isinstance(opts, dict) else None
