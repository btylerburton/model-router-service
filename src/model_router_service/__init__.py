"""Package marker for model_router_service.

`create_app` is imported LAZILY (via __getattr__) so the dependency-free core
modules (scorer, catalog) can be imported and tested without the FastAPI/httpx
stack installed. Importing `model_router_service.create_app` (or the app module)
is what pulls in the web dependencies.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

