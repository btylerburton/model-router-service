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
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    out = []
    for line in text.splitlines():
        in_str = esc = False
        cut = None
        for i, ch in enumerate(line):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                cut = i
                break
        out.append(line[:cut] if cut is not None else line)
    joined = "\n".join(out)
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
