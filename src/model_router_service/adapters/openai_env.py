"""openai-env adapter — point ANY OpenAI-compatible harness at the proxy via the
standard OPENAI_BASE_URL environment variable.

Most OpenAI-compatible tools (Aider, LiteLLM clients, raw SDKs, many agents)
honor OPENAI_BASE_URL (some also OPENAI_API_BASE). This adapter manages a dotenv
file the user sources, AND the CLI prints an `export` line for the current shell
(option C: persistence + immediate use).

It writes a managed BLOCK delimited by sentinels so it never clobbers other
content in the file and `off` removes exactly what `on` added.
"""

from __future__ import annotations

import os
import re

_BEGIN = "# >>> model-router (managed) >>>"
_END = "# <<< model-router (managed) <<<"

# Both common spellings, so tools that read either are covered.
_VARS = ("OPENAI_BASE_URL", "OPENAI_API_BASE")


def _default_env_file() -> str:
    return os.environ.get(
        "MODEL_ROUTER_ENV_FILE", os.path.expanduser("~/.model-router/env")
    )


class OpenAIEnvAdapter:
    name = "openai-env"

    def __init__(self, env_file: str | None = None) -> None:
        self.env_file = env_file or _default_env_file()

    # --- interface ---------------------------------------------------------
    def detect(self) -> bool:
        # Present if our managed env file exists OR the var is already set in the
        # environment (some other tool established it). We can manage either.
        return os.path.isfile(self.env_file) or any(os.environ.get(v) for v in _VARS)

    def describe_target(self) -> str:
        return f"{self.env_file} (OPENAI_BASE_URL / OPENAI_API_BASE)"

    def get_base_url(self) -> str | None:
        # Prefer the managed-file value; fall back to the live environment.
        block = self._read_block()
        for v in _VARS:
            m = re.search(rf"^{v}=(.*)$", block, flags=re.MULTILINE)
            if m:
                return m.group(1).strip().strip('"').strip("'")
        for v in _VARS:
            if os.environ.get(v):
                return os.environ[v]
        return None

    def set_base_url(self, url: str) -> None:
        lines = [_BEGIN] + [f'{v}="{url}"' for v in _VARS] + [_END]
        self._write_block("\n".join(lines))

    def clear_base_url(self) -> None:
        self._write_block(None)

    def restart_hint(self) -> str:
        return (
            f"source the env file in your shell/session: `set -a; . {self.env_file}; set +a` "
            "(or use the printed `export` line). Harnesses read OPENAI_BASE_URL at startup."
        )

    # --- extra: the export line the CLI prints (option C immediate use) ----
    def export_line(self, url: str | None) -> str:
        if url:
            return "; ".join(f'export {v}="{url}"' for v in _VARS)
        return "; ".join(f"unset {v}" for v in _VARS)

    # --- helpers -----------------------------------------------------------
    def _read_all(self) -> str:
        if os.path.isfile(self.env_file):
            return open(self.env_file, encoding="utf-8").read()
        return ""

    def _read_block(self) -> str:
        text = self._read_all()
        m = re.search(re.escape(_BEGIN) + r"(.*?)" + re.escape(_END), text, flags=re.DOTALL)
        return m.group(1) if m else ""

    def _write_block(self, block: str | None) -> None:
        text = self._read_all()
        # Remove any existing managed block first (idempotent).
        text = re.sub(
            re.escape(_BEGIN) + r".*?" + re.escape(_END) + r"\n?",
            "",
            text,
            flags=re.DOTALL,
        )
        if block is not None:
            if text and not text.endswith("\n"):
                text += "\n"
            text += block + "\n"
        os.makedirs(os.path.dirname(self.env_file), exist_ok=True)
        tmp = self.env_file + ".model-router-toggle.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.env_file)
