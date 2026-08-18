"""JSON routes. The same board the pages render, as data.

The response contract mirrors `board.build()`: warnings are first-class, never
an empty list standing in for a sentence. An unpriceable board must say why —
an empty `data` array looks identical to a network blip and invites the caller
to retry forever.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.config import SEASON
from web import data as wd

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "season": SEASON}


@router.get("/board")
def api_board(profile: str | None = None) -> dict[str, Any]:
    try:
        data = wd.board(profile)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err.args[0]))
    players = data["players"]
    return {
        "data": players.to_dicts(),
        "warnings": data.get("warnings", []),
        "meta": {
            "profile": getattr(data.get("profile"), "name", profile),
            "season": SEASON,
            "rows": players.height,
            "columns": players.columns,
        },
    }
