"""The app factory and uvicorn entry point. Thin, mostly wiring.

Run:
    FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run uvicorn web.server:app --reload

Same env vars as everything else in this repo; `FF_EDGE_PROFILE` sets the
default profile and the header selector overrides it per request.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web import api, pages


def create_app() -> FastAPI:
    app = FastAPI(title="ff-edge", docs_url="/api/docs", redoc_url=None)
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )
    app.include_router(api.router)
    app.include_router(pages.router)
    return app


app = create_app()
