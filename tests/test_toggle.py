"""Tests for the multi-harness toggle: OpenCode + openai-env adapters, plus the
shared on/off/status orchestration and fail-safe auto-detect."""

from __future__ import annotations

import json

import pytest

from model_router_service import adapters, toggle
from model_router_service.adapters.opencode import OpenCodeAdapter
from model_router_service.adapters.openai_env import OpenAIEnvAdapter

GATEWAY = "https://api.gsa.usai.gov/api/v1"
PROXY = "http://127.0.0.1:8080"
PROXY_V1 = "http://127.0.0.1:8080/v1"


# --- OpenCode adapter ------------------------------------------------------
def _write_opencode(path, base_url=GATEWAY):
    path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["usai"],
        "provider": {"usai": {"npm": "x", "options": {"baseURL": base_url, "timeout": 600000}}},
        "permission": {"edit": "allow"},
    }, indent=2))


def test_opencode_get_set_clear(tmp_path):
    cfg = tmp_path / "opencode.jsonc"
    _write_opencode(cfg)
    a = OpenCodeAdapter(config_path=str(cfg))
    assert a.detect()
    assert a.get_base_url() == GATEWAY
    a.set_base_url(PROXY_V1)
    assert a.get_base_url() == PROXY_V1
    # other keys preserved
    data = json.loads(cfg.read_text())
    assert data["provider"]["usai"]["options"]["timeout"] == 600000
    assert data["permission"]["edit"] == "allow"
    assert data["enabled_providers"] == ["usai"]
    a.clear_base_url()
    assert a.get_base_url() is None


def test_opencode_jsonc_with_comments(tmp_path):
    cfg = tmp_path / "opencode.jsonc"
    cfg.write_text(
        "{\n"
        "  // a comment the loader must tolerate\n"
        '  "provider": { "usai": { "options": { "baseURL": "%s" } } },\n'
        "  \"model\": \"usai/foo\", // trailing comment\n"
        "}\n" % GATEWAY
    )
    a = OpenCodeAdapter(config_path=str(cfg))
    assert a.get_base_url() == GATEWAY
    a.set_base_url(PROXY_V1)
    # After write it is strict JSON; value applied, model preserved.
    data = json.loads(cfg.read_text())
    assert data["provider"]["usai"]["options"]["baseURL"] == PROXY_V1
    assert data["model"] == "usai/foo"


def test_opencode_real_merged_config_round_trips(tmp_path):
    """Regression: the real kit-merged global config from a sandbox (p1) failed
    to parse with the old per-line JSONC stripper — a `https://` URL in a string
    value and `//` inside comments desynced its line-local string tracking,
    truncating the document ("Extra data: line 73"). The single-pass stripper
    must read it, return the baseURL, and round-trip a flip without corrupting
    the top-level keys / permission globs."""
    import pathlib
    src = pathlib.Path(__file__).parent / "fixtures" / "opencode-global-real.jsonc"
    cfg = tmp_path / "opencode.jsonc"
    cfg.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    a = OpenCodeAdapter(config_path=str(cfg))
    assert a.detect()
    assert a.get_base_url() == GATEWAY  # it reads the real file now
    a.set_base_url(PROXY_V1)
    data = json.loads(cfg.read_text())  # strict JSON after write
    assert data["provider"]["usai"]["options"]["baseURL"] == PROXY_V1
    # nothing else clobbered: the permission block + other top-level keys stay
    assert "permission" in data
    assert data["permission"]["read"]["*.env"] == "deny"
    assert {"$schema", "provider", "permission", "model"}.issubset(data)
    a.clear_base_url()
    assert a.get_base_url() is None


def test_opencode_missing_provider_raises(tmp_path):
    cfg = tmp_path / "opencode.jsonc"
    cfg.write_text(json.dumps({"provider": {}}))
    a = OpenCodeAdapter(config_path=str(cfg))
    with pytest.raises(RuntimeError):
        a.set_base_url(PROXY_V1)


# --- openai-env adapter ----------------------------------------------------
def test_openai_env_roundtrip(tmp_path):
    env = tmp_path / "env"
    env.write_text("# pre-existing user content\nFOO=bar\n")
    a = OpenAIEnvAdapter(env_file=str(env))
    a.set_base_url(PROXY_V1)
    assert a.get_base_url() == PROXY_V1
    text = env.read_text()
    assert "FOO=bar" in text  # preserved
    assert "model-router (managed)" in text
    # export line (option C immediate-use)
    assert "export OPENAI_BASE_URL=" in a.export_line(PROXY_V1)
    assert "unset OPENAI_BASE_URL" in a.export_line(None)
    a.clear_base_url()
    assert a.get_base_url() is None
    assert "FOO=bar" in env.read_text()  # still preserved
    assert "model-router (managed)" not in env.read_text()


def test_openai_env_block_is_idempotent(tmp_path):
    env = tmp_path / "env"
    a = OpenAIEnvAdapter(env_file=str(env))
    a.set_base_url(PROXY_V1)
    a.set_base_url(PROXY_V1)
    # Only one managed block.
    assert env.read_text().count("model-router (managed)") == 2  # begin + end sentinel


# --- orchestration + auto-detect ------------------------------------------
def test_on_off_roundtrip_via_cli(tmp_path, monkeypatch):
    cfg = tmp_path / "opencode.jsonc"
    _write_opencode(cfg)
    state = tmp_path / "state.json"
    monkeypatch.setenv("OPENCODE_GLOBAL_CONFIG", str(cfg))
    monkeypatch.setenv("MODEL_ROUTER_TOGGLE_STATE", str(state))
    # avoid network probe affecting result
    monkeypatch.setattr(toggle, "_probe", lambda proxy: True)

    rc = toggle.main(["--harness", "opencode", "on"])
    assert rc == 0
    assert json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == PROXY_V1

    rc = toggle.main(["--harness", "opencode", "off"])
    assert rc == 0
    # restored to the exact original gateway
    assert json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == GATEWAY


def test_on_is_idempotent(tmp_path, monkeypatch):
    cfg = tmp_path / "opencode.jsonc"
    _write_opencode(cfg)
    monkeypatch.setenv("OPENCODE_GLOBAL_CONFIG", str(cfg))
    monkeypatch.setenv("MODEL_ROUTER_TOGGLE_STATE", str(tmp_path / "s.json"))
    monkeypatch.setattr(toggle, "_probe", lambda proxy: True)
    assert toggle.main(["--harness", "opencode", "on"]) == 0
    assert toggle.main(["--harness", "opencode", "on"]) == 0  # no-op, still 0
    assert json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == PROXY_V1


def test_autodetect_fails_safe_when_none(tmp_path, monkeypatch, capsys):
    # Point both adapters at non-existent files so nothing is detected.
    monkeypatch.setenv("OPENCODE_GLOBAL_CONFIG", str(tmp_path / "nope.jsonc"))
    monkeypatch.setenv("MODEL_ROUTER_ENV_FILE", str(tmp_path / "nope-env"))
    for v in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(v, raising=False)
    rc = toggle.main(["on"])
    assert rc == 2
    assert "no harness detected" in capsys.readouterr().err


def test_autodetect_single(tmp_path, monkeypatch):
    cfg = tmp_path / "opencode.jsonc"
    _write_opencode(cfg)
    monkeypatch.setenv("OPENCODE_GLOBAL_CONFIG", str(cfg))
    monkeypatch.setenv("MODEL_ROUTER_ENV_FILE", str(tmp_path / "nope-env"))
    for v in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MODEL_ROUTER_TOGGLE_STATE", str(tmp_path / "s.json"))
    monkeypatch.setattr(toggle, "_probe", lambda proxy: True)
    assert toggle.main(["on"]) == 0  # auto-detected opencode


def test_registry_lists_both():
    assert "opencode" in adapters.available()
    assert "openai-env" in adapters.available()


# --- state log + acknowledgement (A3) --------------------------------------
def _opencode_env(tmp_path, monkeypatch):
    cfg = tmp_path / "opencode.jsonc"
    _write_opencode(cfg)
    monkeypatch.setenv("OPENCODE_GLOBAL_CONFIG", str(cfg))
    monkeypatch.setenv("MODEL_ROUTER_TOGGLE_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("MODEL_ROUTER_TOGGLE_LOG", str(tmp_path / "toggle-log.jsonl"))
    monkeypatch.setattr(toggle, "_probe", lambda proxy: True)
    # module-level path constants are read at import; patch them to the tmp files
    monkeypatch.setattr(toggle, "SIDECAR", str(tmp_path / "state.json"))
    monkeypatch.setattr(toggle, "STATE_LOG", str(tmp_path / "toggle-log.jsonl"))
    monkeypatch.setattr(toggle, "PREF_FILE", str(tmp_path / "pref.json"))
    return cfg


def test_toggle_writes_state_log(tmp_path, monkeypatch):
    _opencode_env(tmp_path, monkeypatch)
    toggle.main(["--harness", "opencode", "on"])
    toggle.main(["--harness", "opencode", "off"])
    log = (tmp_path / "toggle-log.jsonl").read_text().strip().splitlines()
    recs = [json.loads(x) for x in log]
    assert recs[0]["state"] == "ROUTER ACTIVE" and recs[0]["on"] is True
    assert recs[-1]["state"] == "ROUTER BYPASSED" and recs[-1]["on"] is False
    assert all(r["harness"] == "opencode" for r in recs)


def test_status_reports_last_state(tmp_path, monkeypatch, capsys):
    _opencode_env(tmp_path, monkeypatch)
    toggle.main(["--harness", "opencode", "on"])
    capsys.readouterr()
    toggle.main(["--harness", "opencode", "status"])
    out = capsys.readouterr().out
    assert "ROUTER ACTIVE" in out
    assert "last set:" in out


def test_ack_on_emits_user_message(tmp_path, monkeypatch, capsys):
    _opencode_env(tmp_path, monkeypatch)
    toggle.main(["--harness", "opencode", "on"])
    capsys.readouterr()
    rc = toggle.main(["--harness", "opencode", "ack"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Model routing is ON" in out
    assert "model-router-toggle off" in out
    assert "x-model-router-bypass" in out  # points to the per-turn escape hatch


def test_ack_suppressed_when_disabled(tmp_path, monkeypatch, capsys):
    _opencode_env(tmp_path, monkeypatch)
    toggle.main(["--harness", "opencode", "on"])
    capsys.readouterr()
    monkeypatch.setenv("MODEL_ROUTER_ACK", "off")
    rc = toggle.main(["--harness", "opencode", "ack"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == ""  # log-only mode: no user-facing acknowledgement


def test_ack_off_state_message(tmp_path, monkeypatch, capsys):
    _opencode_env(tmp_path, monkeypatch)
    # never turned on → router OFF
    capsys.readouterr()
    rc = toggle.main(["--harness", "opencode", "ack"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Model routing is OFF" in out


# --- sticky off: the restart-reflip bug ------------------------------------
def _simulate_startup_flip(cfg_path, proxy="http://127.0.0.1:8080"):
    """Reproduce what the kit's startup script does: consult the saved pref and
    only flip ON when it is not 'off'. Mirrors model-router-proxy-install.sh."""
    from model_router_service import toggle as t
    desired = (t._read_pref().get("opencode") or {}).get("desired", "on")
    if desired == "off":
        return "skipped"
    return t.main(["--harness", "opencode", "--url", proxy, "on"])


def test_off_persists_across_simulated_restart(tmp_path, monkeypatch):
    """Regression: `off` must survive a sandbox restart. The kit startup used to
    re-flip ON every boot, so off was unreachable. With a persisted pref, startup
    must SKIP the flip when the user chose off."""
    cfg = _opencode_env(tmp_path, monkeypatch)
    # user turns routing ON then OFF
    toggle.main(["--harness", "opencode", "on"])
    toggle.main(["--harness", "opencode", "off"])
    import json as _json
    assert _json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == GATEWAY
    # simulate a sandbox restart running the kit startup flip
    result = _simulate_startup_flip(str(cfg))
    assert result == "skipped", "startup must NOT re-flip when pref is off"
    # baseURL still the gateway — off survived the restart
    assert _json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == GATEWAY


def test_on_clears_optout_so_next_boot_routes(tmp_path, monkeypatch):
    cfg = _opencode_env(tmp_path, monkeypatch)
    toggle.main(["--harness", "opencode", "off"])
    toggle.main(["--harness", "opencode", "on"])  # user re-enables
    # a subsequent startup should now flip ON (pref back to on)
    result = _simulate_startup_flip(str(cfg))
    assert result == 0
    import json as _json
    assert _json.loads(cfg.read_text())["provider"]["usai"]["options"]["baseURL"] == PROXY_V1


def test_default_pref_is_on(tmp_path, monkeypatch, capsys):
    _opencode_env(tmp_path, monkeypatch)
    # no toggle run yet -> pref defaults to on
    rc = toggle.main(["--harness", "opencode", "pref"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "on"
