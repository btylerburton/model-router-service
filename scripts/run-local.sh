#!/usr/bin/env bash
#
# run-local.sh — run model-router-service on your HOST so a sandbox (acq/Docker)
# can reach it, with an honest exposure warning.
#
# WHY THIS EXISTS: the service defaults to binding 127.0.0.1 (loopback), which a
# container CANNOT reach even via host.docker.internal. To let an in-sandbox
# OpenCode route through a host-run service you must bind an interface the
# Docker bridge can see. This script does that deliberately and tells you what
# it exposed. It is a LOCAL-DEV convenience ONLY — on cloud.gov you do none of
# this: the buildpack sets $PORT and the app binds 0.0.0.0:$PORT automatically.
#
# SECURITY: binding beyond loopback makes the service reachable from other hosts
# on your network, and it forwards your USAi key upstream. The script defaults
# to the SAFEST reachable option (the Docker bridge gateway IP if it can detect
# one) and only falls back to 0.0.0.0 (ALL interfaces) with an explicit warning.
# It never prints the API key.
#
# Usage:
#   scripts/run-local.sh                 # bind for sandbox reachability, start
#   scripts/run-local.sh --print-only    # show the bind + sandbox URL, do NOT start
#   scripts/run-local.sh --host 0.0.0.0  # force a specific bind host
#   scripts/run-local.sh --port 8080     # override port (default 8080)
#   scripts/run-local.sh --help

set -euo pipefail
IFS=$'\n\t'

PORT="${ROUTER_PORT:-8080}"
HOST_OVERRIDE=""
PRINT_ONLY=0

usage() {
  cat <<'USAGE'
run-local.sh — start model-router-service reachable from a sandbox.

  --host <addr>   bind address (default: auto-detect the Docker bridge gateway,
                  else 0.0.0.0 with a warning)
  --port <n>      listen port (default: $ROUTER_PORT or 8080)
  --print-only    print the chosen bind + the URL a sandbox should use, then
                  exit WITHOUT starting the server
  --help          this help

cloud.gov note: do NOT use this there. The buildpack sets $PORT and the app
binds 0.0.0.0:$PORT on its own; the sandbox reaches it at the public app route.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host) HOST_OVERRIDE="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --print-only) PRINT_ONLY=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# --- choose a bind host ------------------------------------------------------
# Prefer the Docker/OrbStack bridge gateway (reachable from containers as
# host.docker.internal) over 0.0.0.0, so we expose the NARROWEST interface that
# still works. Detection is best-effort and non-fatal.
detect_bridge_ip() {
  # Linux docker0 bridge
  if command -v ip >/dev/null 2>&1; then
    ip -4 addr show docker0 2>/dev/null \
      | awk '/inet /{sub(/\/.*/,"",$2); print $2; exit}'
  fi
}

BIND_HOST=""
BIND_SOURCE=""
if [ -n "$HOST_OVERRIDE" ]; then
  BIND_HOST="$HOST_OVERRIDE"
  BIND_SOURCE="--host override"
else
  bridge_ip="$(detect_bridge_ip || true)"
  if [ -n "${bridge_ip:-}" ]; then
    BIND_HOST="$bridge_ip"
    BIND_SOURCE="detected docker0 bridge"
  else
    # macOS Docker Desktop / OrbStack expose host.docker.internal to containers
    # but there is no host-side bridge IP to bind narrowly, so we must bind all
    # interfaces. This is the honest exposure the warning below describes.
    BIND_HOST="0.0.0.0"
    BIND_SOURCE="fallback (no detectable bridge; ALL interfaces)"
  fi
fi

# --- report ------------------------------------------------------------------
echo "model-router-service local bind:"
echo "  host : ${BIND_HOST}   (${BIND_SOURCE})"
echo "  port : ${PORT}"
echo
echo "A sandbox should point MODEL_ROUTER_URL at the HOST BRIDGE, not this bind:"
echo "  MODEL_ROUTER_URL=http://host.docker.internal:${PORT}"
echo "  (0.0.0.0 / a bridge IP is a BIND address, never a connect address.)"
echo

if [ "$BIND_HOST" = "0.0.0.0" ]; then
  echo "WARNING: binding 0.0.0.0 exposes this service on EVERY network interface" >&2
  echo "         on this machine, not just the Docker bridge. Anything on your" >&2
  echo "         LAN can reach it, and it forwards your USAi key upstream. Prefer" >&2
  echo "         a firewalled/managed network, stop it when done, and never run" >&2
  echo "         it this way on an untrusted network." >&2
  echo >&2
fi

if [ "$PRINT_ONLY" -eq 1 ]; then
  echo "--print-only: not starting the server."
  exit 0
fi

# --- preflight: .env present? (do NOT read or print its contents) ------------
if [ ! -f ".env" ]; then
  echo "note: no .env in $(pwd); the service reads config from the environment." >&2
  echo "      It will FAIL CLOSED if ROUTER_UPSTREAM_API_KEY / required vars are unset." >&2
fi

# --- launch ------------------------------------------------------------------
# ROUTER_HOST/ROUTER_PORT drive the bind (see config.py / __main__.py). We do
# NOT set $PORT: that is the PaaS signal that would force 0.0.0.0 and is only
# correct on cloud.gov. Prefer `uv run` if present, else the active python3.
echo "starting: ROUTER_HOST=${BIND_HOST} ROUTER_PORT=${PORT} model_router_service"
if command -v uv >/dev/null 2>&1; then
  exec env ROUTER_HOST="${BIND_HOST}" ROUTER_PORT="${PORT}" \
    uv run python -m model_router_service
else
  exec env ROUTER_HOST="${BIND_HOST}" ROUTER_PORT="${PORT}" \
    python3 -m model_router_service
fi
