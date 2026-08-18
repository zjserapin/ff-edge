"""HTML routes. Thin on purpose: numbers come from web.data, names from src.

Pages are server-rendered Jinja; htmx swaps fragments by re-requesting the
partial routes with the current controls as query params. **State lives in the
URL, never in a session.** That is the design decision that kills the
five-selectbox problem (checklist E1), and it is load-bearing: every view is
bookmarkable and nothing hides in a session dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from src import archetypes as ar
from src import board as bd
from src import ids
from src import profiles as pf
from src.config import ENV_WEIGHT, FANTASY_POSITIONS, SEASON, SLEEPER_USERNAME
from web import charts
from web import data as wd
from web import memo
from web.format import fmt, header, table_ctx

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals.update(
    season=SEASON,
    profile_names=sorted(pf.PROFILES),
    env_weight=ENV_WEIGHT,
    positions_all=list(FANTASY_POSITIONS),
    signals_all=["both up", "split", "both down", "quiet"],
)

# Column order is the Big Board tab's, and the reasoning travels with it: the
# board opens on `block` rather than `board_rank` because the block is measured
# and the order inside it is not; the three layers that build `par_env` read
# left to right in the order they are applied; `ffb_spread`/`stalest_days` sit
# beside `ffb_par` because they qualify it. Usage columns splice in before
# `value_gap` when toggled.
_COLS_PRE = (
    "block", "name", "position", "need", "pos_rank", "demand", "team", "bye",
    "tier", "same", "adp", "adj_adp", "par", "ffb_par", "ffb_spread",
    "stalest_days", "ecr", "ecr_sd", "blend_par", "env_swing", "par_env",
    "drop", "quality_pct",
)
_COLS_POST = ("value_gap", "vegas_gap", "signal", "board_rank")


def _page(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    """Render a full page. Fragments use TemplateResponse directly.

    Every full page gets `league`, so the "which draft am I actually pricing"
    tag is on screen no matter which page you landed on — see
    `data.league_identity`. A dead network costs the tag, never the page.
    """
    if "league" not in ctx:
        try:
            ctx["league"] = wd.league_identity()
        except Exception:  # noqa: BLE001
            ctx["league"] = None
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def _profile_error(request: Request, err: KeyError) -> HTMLResponse:
    # `profiles.resolve` refuses to fall back — a typo that silently returned
    # the Shiva Bowl would price a standard league as a superflex keeper league
    # and look entirely plausible doing it. The refusal becomes a sentence.
    return _page(
        request, "error.html",
        {"title": "Unknown profile", "message": str(err.args[0])},
        status_code=404,
    )


# --- board assembly ---------------------------------------------------------


def _clean(values: list[str]) -> list[str]:
    # An empty <option value=""> submits `signals=`, and filtering on [""]
    # would silently empty the board — a blank filter must mean "all",
    # never "match nothing".
    return [v for v in values if v]


def _board_view(
    profile: str | None, positions: list[str], signals: list[str], usage: bool
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """The board dict, the full player frame, and the filtered/selected view."""
    data = wd.board(profile)
    players = data["players"]
    if players.height and "indist_n" in players.columns:
        # A column of 1s reads as a rating; the only useful signal is the
        # groups bigger than one.
        players = players.with_columns(
            pl.when(pl.col("indist_n") > 1)
            .then(pl.col("indist_n"))
            .otherwise(None)
            .alias("same")
        )
    view = players
    if players.height:
        view = players.filter(
            pl.col("position").is_in(positions or list(FANTASY_POSITIONS))
        )
        if signals:
            view = view.filter(pl.col("signal").is_in(signals))
        usage_cols = (
            [c for c in bd._USAGE_COLUMNS if c in view.columns] if usage else []
        )
        columns = [
            c for c in (*_COLS_PRE, *usage_cols, *_COLS_POST) if c in view.columns
        ]
        # `nulls_last` on every sort: polars defaults it to False, so an
        # unscored player would otherwise open the board.
        view = view.sort("board_rank", nulls_last=True).select(columns)
    return data, players, view


def _board_table_ctx(
    view: pl.DataFrame,
    positions: list[str],
    signals: list[str],
    usage: bool,
    rows: int,
    profile: str | None,
) -> dict[str, Any]:
    shown = view.head(rows) if rows else view
    base = table_ctx(shown)
    body = []
    for i, r in enumerate(shown.iter_rows(named=True)):
        below = "block" in shown.columns and r.get("block") is None
        body.append(
            {
                "cells": base["rows"][i],
                "position": r.get("position"),
                "below": below,
            }
        )
    # The demand line is drawn where `block` first goes blank — the point past
    # which the board stops claiming and hands the order back to ADP.
    line_at = next(
        (i for i, row in enumerate(body) if row["below"] and i > 0 and not body[i - 1]["below"]),
        None,
    )
    query = _query(profile, positions, signals, usage, rows)
    return {
        "headers": base["headers"],
        "body": body,
        "line_at": line_at,
        "total": view.height,
        "shown": shown.height,
        "usage": usage,
        "rows": rows,
        "positions": positions,
        "signals": signals,
        "csv_url": f"/board.csv?{query}",
        "query": query,
    }


def _query(
    profile: str | None, positions: list[str], signals: list[str], usage: bool, rows: int
) -> str:
    parts = []
    if profile:
        parts.append(f"profile={profile}")
    parts += [f"positions={p}" for p in positions]
    parts += [f"signals={s}" for s in signals]
    if usage:
        parts.append("usage=1")
    if rows:
        parts.append(f"rows={rows}")
    return "&".join(parts)


def _metrics(data: dict[str, Any], players: pl.DataFrame) -> list[dict[str, Any]]:
    priced = players.get_column("vegas_gap").drop_nulls().len() if "vegas_gap" in players.columns else 0
    tiles = [
        {"label": "Draftable", "value": players.height, "help": ""},
        {"label": "Off the board (kept)", "value": data["kept"].height, "help": ""},
    ]
    if "in_demand" in players.columns:
        tiles.append(
            {
                "label": "Inside roster demand",
                "value": int(players.get_column("in_demand").sum()),
                "help": (
                    "Players inside the number of their position the draft still "
                    "has to fill. Below this line the board is in ADP order and "
                    "block is blank, because PAR has no lineup value under "
                    "replacement."
                ),
            }
        )
    tiles.append({"label": "With a book line", "value": priced, "help": ""})
    return tiles


# --- cost of waiting --------------------------------------------------------


def _cost_ctx(
    players: pl.DataFrame, data: dict[str, Any], profile: str | None, horizon: int | None
) -> dict[str, Any]:
    """Port of app.py's `_cost_of_waiting_panel`, decisions included.

    Costed on `par`, not the `par_env` the board ranks on — availability is a
    fact about the market's curve, and the board's own blocks say the `par_env`
    separation at the top is a distinction it cannot make. The recommendation
    is restricted to positions with open starting slots, because league
    scarcity is maximally misleading to the manager who caused it.
    """
    picks = wd.my_picks()
    usable = (
        picks.filter(pl.col("usable")).get_column("pick_no").to_list()
        if picks.height and "usable" in picks.columns
        else []
    )
    if len(usable) < 2:
        return {"missing": True}

    ceiling = min(6, len(usable))
    horizon = max(2, min(int(horizon or min(4, len(usable))), ceiling))
    waiting = bd.cost_of_waiting(players, usable[:horizon])
    if not waiting.height:
        return {"missing": True, "why": "No cost-of-waiting estimate under these settings."}

    first = usable[0]
    at_first = waiting.filter(pl.col("pick_no") == first).sort(
        "cost_of_waiting", descending=True, nulls_last=True
    )

    need = wd.my_need(profile)
    filled: list[str] = []
    if need.height and "slots_open" in need.columns:
        open_positions = set(
            need.filter(pl.col("slots_open") > 0).get_column("position")
        )
        filled = sorted(set(at_first.get_column("position").to_list()) - open_positions)
        if open_positions:
            at_first = at_first.filter(pl.col("position").is_in(list(open_positions)))

    ctx: dict[str, Any] = {
        "missing": False,
        "first": first,
        "horizon": horizon,
        "ceiling": ceiling,
        "tiles": [
            {**row, "text": "—" if row["cost_of_waiting"] is None else f"{row['cost_of_waiting']:.1f}"}
            for row in at_first.iter_rows(named=True)
        ],
        "filled": filled,
        "kept_names": "",
    }
    if at_first.height >= 2:
        top, nxt = at_first.row(0, named=True), at_first.row(1, named=True)
        ctx.update(
            top=top, nxt=nxt,
            gap=(top["cost_of_waiting"] or 0) - (nxt["cost_of_waiting"] or 0),
        )

    if filled:
        my_keepers = data.get("kept", pl.DataFrame())
        if my_keepers.height and {"owner", "player_name", "position"} <= set(my_keepers.columns):
            names = (
                my_keepers.filter(
                    (pl.col("owner") == SLEEPER_USERNAME)
                    & pl.col("position").is_in(filled)
                )
                .get_column("player_name")
                .to_list()
            )
            if names:
                ctx["kept_names"] = f" — you keep {' and '.join(names)}"

    remaining = need.filter(pl.col("slots_open") > 0) if need.height else need
    if remaining.height:
        ctx["slots"] = ", ".join(
            f"{r['position']} {r['open_dedicated']}"
            + (f"+{r['open_flex']}flex" if r["open_flex"] else "")
            for r in remaining.iter_rows(named=True)
        )
    ctx["table"] = table_ctx(
        waiting.sort(["pick_no", "cost_of_waiting"], descending=[False, True], nulls_last=True)
    )
    return ctx


# --- assumptions ------------------------------------------------------------


def _assumptions_ctx(data: dict[str, Any]) -> dict[str, Any] | None:
    summary, replacement = data.get("summary"), data.get("replacement")
    if summary is None or not summary.height:
        return None
    joined = summary
    if replacement is not None and replacement.height:
        joined = summary.join(
            replacement.select("position", "replacement_rank", "replacement_points"),
            on="position",
            how="left",
        )
    return table_ctx(joined)


# --- block similarity -------------------------------------------------------


def _block_ctx(
    players: pl.DataFrame, block: int | None, anchor: str | None
) -> dict[str, Any] | None:
    """Port of app.py's `_block_similarity_panel`, traps included.

    Null blocks are dropped before grouping (below the line the board resolves
    nothing to compare), the name join is deduped to one row per player before
    use, and the gsis id is aliased on the way in so the board's FFC Int64
    `player_id` never collides with the String id space.
    """
    if "block" not in players.columns or not players.height:
        return None
    counts = (
        players.filter(pl.col("block").is_not_null())
        .group_by("block")
        .agg(pl.len().alias("n"), pl.col("position").first().alias("pos"))
        .filter(pl.col("n") > 1)
        .sort("block")
    )
    if not counts.height:
        return {"empty": True}

    options = counts.to_dicts()
    valid = {o["block"] for o in options}
    block = block if block in valid else options[0]["block"]
    members = players.filter(pl.col("block") == block)
    names = members.get_column("name").to_list()
    anchor = anchor if anchor in names else names[0]
    ctx: dict[str, Any] = {
        "empty": False, "options": options, "block": block,
        "names": names, "anchor": anchor,
    }

    feats = wd.features()
    if not feats.height:
        ctx["message"] = "No features built yet. Run the bootstrap."
        return ctx
    season = int(feats.get_column("season").max())
    scored = wd.scores(season, 8)
    if not scored.height:
        ctx["message"] = "No quality scores available to compare on."
        return ctx

    key = (
        scored.select(
            ids.normalize("player_name").alias("_norm"),
            pl.col("player_id").alias("gsis_id"),
        )
        .unique(subset=["_norm"], keep="first")
    )
    mapped = (
        members.with_columns(ids.normalize("name").alias("_norm"))
        .join(key, on="_norm", how="left")
        .drop("_norm")
    )
    ids_in_block = mapped.get_column("gsis_id").drop_nulls().to_list()
    target = mapped.filter(pl.col("name") == anchor).get_column("gsis_id")

    unmatched = mapped.height - len(ids_in_block)
    if unmatched:
        ctx["unmatched"] = (
            f"{unmatched} of {mapped.height} in this block have no quality "
            "score — a season under the volume floor is not measured, which is "
            "not the same as being unlike everyone."
        )
    if len(target) == 0 or target[0] is None or len(ids_in_block) < 2:
        ctx["message"] = (
            f"Not enough scored players in block {block} to compare {anchor} against."
        )
        return ctx

    near = ar.neighbors(
        target[0], scored, feats, n=20, season=season, restrict_to=ids_in_block
    )
    if not near.height:
        ctx["message"] = "No comparison available inside this block."
        return ctx
    ctx["table"] = table_ctx(
        near.select("player_name", "team", "distance", "ppg", "pos_rank", "games")
    )
    return ctx


# --- routes -----------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def big_board(
    request: Request,
    profile: str | None = None,
    positions: list[str] = Query(default=[]),
    signals: list[str] = Query(default=[]),
    usage: bool = False,
    rows: int = 75,
    horizon: int | None = None,
    block: int | None = None,
    anchor: str | None = None,
) -> HTMLResponse:
    positions, signals = _clean(positions), _clean(signals)
    try:
        data, players, view = _board_view(profile, positions, signals, usage)
    except KeyError as err:
        return _profile_error(request, err)
    ctx: dict[str, Any] = {
        "active": "board",
        "profile": profile,
        "warnings": data.get("warnings", []),
        "empty": not players.height,
    }
    if players.height:
        ctx.update(
            metrics=_metrics(data, players),
            unpriced=players.height - (players.get_column("vegas_gap").drop_nulls().len() if "vegas_gap" in players.columns else 0),
            cost=_cost_ctx(players, data, profile, horizon),
            assumptions=_assumptions_ctx(data),
            table=_board_table_ctx(view, positions, signals, usage, rows, profile),
            similarity=_block_ctx(players, block, anchor),
        )
    return _page(request, "board.html", ctx)


@router.get("/partials/board", response_class=HTMLResponse)
def board_partial(
    request: Request,
    profile: str | None = None,
    positions: list[str] = Query(default=[]),
    signals: list[str] = Query(default=[]),
    usage: bool = False,
    rows: int = 75,
) -> HTMLResponse:
    positions, signals = _clean(positions), _clean(signals)
    try:
        data, players, view = _board_view(profile, positions, signals, usage)
    except KeyError as err:
        return _profile_error(request, err)
    if not players.height:
        return HTMLResponse('<div id="board-region"></div>')
    ctx = {"table": _board_table_ctx(view, positions, signals, usage, rows, profile)}
    return _page(request, "partials/_board_table.html", ctx)


@router.get("/partials/cost", response_class=HTMLResponse)
def cost_partial(
    request: Request, profile: str | None = None, horizon: int | None = None
) -> HTMLResponse:
    try:
        data, players, _ = _board_view(profile, [], [], False)
    except KeyError as err:
        return _profile_error(request, err)
    ctx = {"cost": _cost_ctx(players, data, profile, horizon), "profile": profile}
    return _page(request, "partials/_cost_panel.html", ctx)


@router.get("/partials/block", response_class=HTMLResponse)
def block_partial(
    request: Request,
    profile: str | None = None,
    block: int | None = None,
    anchor: str | None = None,
) -> HTMLResponse:
    try:
        data, players, _ = _board_view(profile, [], [], False)
    except KeyError as err:
        return _profile_error(request, err)
    if players.height and "indist_n" in players.columns:
        players = players.with_columns(
            pl.when(pl.col("indist_n") > 1).then(pl.col("indist_n")).otherwise(None).alias("same")
        )
    ctx = {"similarity": _block_ctx(players, block, anchor), "profile": profile}
    return _page(request, "partials/_block_similarity.html", ctx)


@router.get("/partials/dropoff", response_class=HTMLResponse)
def dropoff_partial(
    request: Request, profile: str | None = None, dark: int = 0
) -> HTMLResponse:
    try:
        data = wd.board(profile)
    except KeyError as err:
        return _profile_error(request, err)
    spec = charts.dropoff(data["players"], bool(dark))
    if spec is None:
        return HTMLResponse('<p class="note">No curve to draw yet.</p>')
    # "</" is escaped so a player name can never close the script element early.
    payload = json.dumps(spec).replace("</", "<\\/")
    return HTMLResponse(
        '<div id="dropoff-chart" class="chart"></div>'
        f'<script type="application/json" class="vega-spec" data-target="dropoff-chart">{payload}</script>'
    )


@router.get("/board.csv")
def board_csv(
    profile: str | None = None,
    positions: list[str] = Query(default=[]),
    signals: list[str] = Query(default=[]),
    usage: bool = False,
) -> Response:
    """The full filtered board with raw column names — a board you cannot mark
    up is not a board you can prepare with (checklist A4)."""
    positions, signals = _clean(positions), _clean(signals)
    try:
        _, players, view = _board_view(profile, positions, signals, usage)
    except KeyError as err:
        return Response(str(err.args[0]), status_code=404, media_type="text/plain")
    return Response(
        view.write_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="ff-edge-big-board-{SEASON}.csv"'
        },
    )


@router.post("/refresh")
def refresh(request: Request) -> RedirectResponse:
    """Drop the in-process memo so the next render re-reads every feed.

    Draft-day control: ADP moves all day and picks land by the minute. The
    disk cache in src/cache.py has its own TTLs and is deliberately not
    touched here — this clears the web layer only.
    """
    memo.clear_all()
    target = request.headers.get("referer") or "/"
    return RedirectResponse(target, status_code=303)


# --- pages that arrive in later blocks --------------------------------------

_PLACEHOLDERS = {
    "draft": (
        "Draft Day",
        "W2",
        "The pick-by-pick companion moves here after the Big Board. Until it "
        "does — and until it has been trusted through a dry run — draft from "
        "the Streamlit app:",
        "FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run streamlit run app.py",
    ),
    "player": (
        "Player",
        "W3",
        "One page, one player, everything the repo knows — board row, the four "
        "layers as an argument, usage against position, comparables, screens, "
        "claims. This page is the reason the site exists (checklist E1).",
        "",
    ),
    "research": (
        "Research",
        "W4",
        "Screens, the Footballers disagreement panel, quality against price, "
        "stability, the honest nulls, rookies, the strategy simulator and the "
        "claims ledger — ported section by section after Draft Day.",
        "",
    ),
    "reference": (
        "Reference",
        "W5",
        "The glossary, plus the pipeline's shape. Last, because every table "
        "already carries its definitions on the headers.",
        "",
    ),
}


def _placeholder(request: Request, key: str, profile: str | None) -> HTMLResponse:
    title, phase, text, command = _PLACEHOLDERS[key]
    return _page(
        request, "placeholder.html",
        {
            "active": key, "profile": profile, "title": title,
            "phase": phase, "text": text, "command": command,
        },
    )


@router.get("/draft", response_class=HTMLResponse)
def draft_day(request: Request, profile: str | None = None) -> HTMLResponse:
    return _placeholder(request, "draft", profile)


@router.get("/player", response_class=HTMLResponse)
def player(request: Request, profile: str | None = None) -> HTMLResponse:
    return _placeholder(request, "player", profile)


@router.get("/research", response_class=HTMLResponse)
def research(request: Request, profile: str | None = None) -> HTMLResponse:
    return _placeholder(request, "research", profile)


@router.get("/reference", response_class=HTMLResponse)
def reference(request: Request, profile: str | None = None) -> HTMLResponse:
    return _placeholder(request, "reference", profile)
