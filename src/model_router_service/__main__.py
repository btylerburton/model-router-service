"""__main__ — entrypoint. Maps cloud.gov's $PORT to the server and runs uvicorn.

cloud.gov (and most PaaS) inject the listen port as $PORT and expect the app to
bind 0.0.0.0:$PORT. We read $PORT if present (overriding ROUTER_PORT) and always
bind 0.0.0.0 in that case, so the same artifact runs locally (ROUTER_HOST/PORT)
and on cloud.gov (PORT) with no code change.
"""

from __future__ import annotations

import os

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    host = settings.host
    port = settings.port
    # PaaS-provided port wins and forces a public bind.
    paas_port = os.environ.get("PORT")
    if paas_port:
        host = "0.0.0.0"  # noqa: S104 — required for the platform-published port
        port = int(paas_port)

    uvicorn.run(
        "model_router_service.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
