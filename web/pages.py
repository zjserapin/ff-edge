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
from src import glossary
from src import ids
from src import profiles as pf
from src import stability as stab
from src import valuation as val
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


# --- Draft Day --------------------------------------------------------------

_DRAFT_COLS = (
    "board_rank", "name", "position", "team", "adp", "adj_adp", "exp_pick",
    "par", "se", "same", "tier", "quality_pct", "value_gap", "env_swing",
)


@router.get("/draft", response_class=HTMLResponse)
def draft_day(
    request: Request,
    profile: str | None = None,
    positions: list[str] = Query(default=[]),
    pick: int | None = None,
    floor: float = 0.35,
    qpos: str | None = None,
    dark: int = 1,
) -> HTMLResponse:
    """The tab with a deadline on it. Everything here is read on a clock.

    Ported from `app.py:_tab_draft_day`. The one deliberate omission is the
    tie-break screens, which live on Research: they are read *about* a
    shortlist rather than at the pick, and duplicating them here would mean
    two places to keep in step.
    """
    positions = _clean(positions)
    try:
        data = wd.board(profile)
    except KeyError as err:
        return _profile_error(request, err)

    players = data["players"]
    ctx: dict[str, Any] = {
        "active": "draft",
        "profile": profile,
        "warnings": data.get("warnings", []),
        "empty": not players.height,
        "positions": positions,
    }
    if not players.height:
        return _page(request, "draft.html", ctx)

    if "indist_n" in players.columns:
        players = players.with_columns(
            pl.when(pl.col("indist_n") > 1).then(pl.col("indist_n")).otherwise(None).alias("same")
        )

    prof = data.get("profile")
    gone = data.get("drafted", pl.DataFrame())
    ctx["header"] = {
        "label": getattr(prof, "label", ""),
        "market": getattr(prof, "adp_scoring", ""),
        "available": players.height,
        "drafted": gone.height,
    }
    if gone.height and {"player_name", "pick_no"} <= set(gone.columns):
        last = gone.sort("pick_no")
        ctx["last_pick"] = {
            "name": last.get_column("player_name")[-1],
            "no": int(last.get_column("pick_no")[-1]),
        }

    # --- your picks ---------------------------------------------------------
    picks = wd.my_picks()
    if picks.height:
        usable = picks.filter(pl.col("usable"))
        ctx["picks"] = {
            "owned": picks.height,
            "usable": usable.height,
            "first": int(usable.get_column("pick_no")[0]) if usable.height else None,
            "table": table_ctx(
                picks.select(
                    [c for c in ("round", "pick_no", "from_owner", "keeper", "usable")
                     if c in picks.columns]
                ),
                pretty=False,
            ),
        }

    kept = data.get("kept", pl.DataFrame())
    if kept.height:
        kept_cols = [c for c in ("player_name", "position", "team", "kept_by", "round")
                     if c in kept.columns]
        ctx["kept"] = {
            "n": kept.height,
            "table": table_ctx(kept.select(kept_cols) if kept_cols else kept, pretty=False),
        }
    unmatched = data.get("unmatched", pl.DataFrame())
    if unmatched.height:
        ctx["unmatched"] = {"n": unmatched.height, "table": table_ctx(unmatched, pretty=False)}

    # --- how many of each asset are left ------------------------------------
    tiers = bd.tier_map(players)
    if tiers.height:
        ctx["tiers"] = {
            "spec": charts.tiers_left(tiers, bool(dark)),
            "table": table_ctx(tiers),
        }

    # --- the board ----------------------------------------------------------
    view = players.filter(pl.col("position").is_in(positions or list(FANTASY_POSITIONS)))
    cols = [c for c in _DRAFT_COLS if c in view.columns]
    ctx["board"] = table_ctx(view.select(cols).head(60))

    # --- availability at a chosen pick --------------------------------------
    options = (
        picks.filter(pl.col("usable")).get_column("pick_no").to_list()
        if picks.height else []
    )
    chosen = pick if pick is not None else (options[0] if options else 24)
    floor = min(0.95, max(0.05, float(floor)))
    got = bd.targets(players, int(chosen), min_available=float(floor), top=15)
    ctx["targets"] = {
        "options": options,
        "pick": int(chosen),
        "floor": floor,
        "floors": [round(0.05 + 0.05 * i, 2) for i in range(19)],
        "table": table_ctx(got, pretty=False) if got.height else None,
    }

    # --- what it costs to wait ---------------------------------------------
    if options:
        waiting = bd.cost_of_waiting(players, options[:6])
        if waiting.height:
            ctx["waiting"] = {
                "best": table_ctx(
                    waiting.pivot(on="pick_no", index="position", values="best_par"),
                    pretty=False,
                ),
                "cost": table_ctx(
                    waiting.pivot(on="pick_no", index="position", values="cost_of_waiting"),
                    pretty=False,
                ),
            }

    # --- one position at a time --------------------------------------------
    if {"quality_pct", "market_pct"} <= set(players.columns):
        scored = players.filter(
            pl.col("quality_pct").is_not_null() & pl.col("market_pct").is_not_null()
        )
        if scored.height:
            available = [
                p for p in FANTASY_POSITIONS
                if scored.filter(pl.col("position") == p).height
            ]
            which = qpos if qpos in available else (available[0] if available else None)
            if which:
                detail = scored.filter(pl.col("position") == which).sort(
                    "value_gap", descending=True, nulls_last=True
                )
                dcols = [c for c in ("name", "team", "adp", "par", "tier", "quality_pct",
                                     "market_pct", "value_gap") if c in detail.columns]
                ctx["quality"] = {
                    "positions": available,
                    "position": which,
                    "spec": charts.quality_against_price(scored, which, bool(dark)),
                    "table": table_ctx(detail.select(dcols)),
                    "thin_qb": which == "QB",
                }
    return _page(request, "draft.html", ctx)


# --- Player -----------------------------------------------------------------


@router.get("/player", response_class=HTMLResponse)
def player(
    request: Request,
    profile: str | None = None,
    name: str | None = None,
    dark: int = 1,
) -> HTMLResponse:
    """One page, one player, everything the repo knows about him.

    **This is checklist E1.** Researching a player previously meant five
    selectboxes across four tabs, each with its own session key, and joining by
    eye. Here the identity is the URL (`/player?name=...`), so the whole
    picture is one request and one bookmark.

    Every panel degrades independently: a player with no prior season has no
    usage chart, one under the volume floor has no comparables, and each says
    so rather than rendering an empty box.
    """
    try:
        data = wd.board(profile)
    except KeyError as err:
        return _profile_error(request, err)

    players = data["players"]
    ctx: dict[str, Any] = {
        "active": "player", "profile": profile, "empty": not players.height,
    }
    if not players.height:
        return _page(request, "player.html", ctx)

    names = players.get_column("name").to_list()
    ctx["all_names"] = names
    if not name:
        # No player chosen yet: offer the top of the board as a starting point
        # rather than an empty page.
        ctx["suggestions"] = players.head(12).select(
            [c for c in ("name", "position", "team", "board_rank") if c in players.columns]
        ).to_dicts()
        return _page(request, "player.html", ctx)

    match = players.filter(pl.col("name") == name)
    if not match.height:
        # Fall back to a normalized match so a URL typed by hand still lands.
        target = ids.normalize_name(name) if hasattr(ids, "normalize_name") else name.lower()
        match = players.with_columns(ids.normalize("name").alias("_n")).filter(
            pl.col("_n") == str(target).lower()
        )
        if match.height:
            match = match.drop("_n")
    if not match.height:
        ctx["missing"] = name
        return _page(request, "player.html", ctx)

    row = match.row(0, named=True)
    ctx["player"] = row
    ctx["name"] = row["name"]

    # The argument, left to right, in the order the layers are applied.
    ctx["layers"] = [
        {"key": "par", "label": "ADP curve", "adds": "the draft slot — blind to player and team"},
        {"key": "ffb_par", "label": "Fantasy Footballers", "adds": "the player"},
        {"key": "ecr", "label": "FantasyPros ECR", "adds": "the crowd"},
        {"key": "env_swing", "label": "Team environment", "adds": "the team"},
        {"key": "par_env", "label": "What the board ranks on", "adds": ""},
    ]
    ctx["layer_values"] = {l["key"]: row.get(l["key"]) for l in ctx["layers"]}

    ctx["usage_spec"] = charts.player_usage(row, players, bool(dark))
    ctx["usage_row"] = {
        c: row.get(c) for c in bd._USAGE_COLUMNS if c in players.columns
    }

    # Comparables, restricted to the position rather than the block — this page
    # asks "who is he like", not "who is he like at this price".
    feats = wd.features()
    if feats.height:
        season = int(feats.get_column("season").max())
        scored = wd.scores(season, 8)
        if scored.height:
            key = scored.select(
                ids.normalize("player_name").alias("_norm"),
                pl.col("player_id").alias("gsis_id"),
            ).unique(subset=["_norm"], keep="first")
            mine = (
                match.with_columns(ids.normalize("name").alias("_norm"))
                .join(key, on="_norm", how="left")
                .get_column("gsis_id")
            )
            if mine.len() and mine[0] is not None:
                near = ar.neighbors(mine[0], scored, feats, n=8, season=season)
                if near.height:
                    ncols = [c for c in ("player_name", "team", "distance", "ppg",
                                         "pos_rank", "games") if c in near.columns]
                    ctx["neighbors"] = table_ctx(near.select(ncols))
            else:
                ctx["no_quality"] = True

    # Where he sits in his own position on the board.
    pos_rows = players.filter(pl.col("position") == row["position"]).sort(
        "board_rank", nulls_last=True
    )
    ctx["position_context"] = table_ctx(
        pos_rows.select(
            [c for c in ("board_rank", "name", "team", "adp", "par_env", "block", "need")
             if c in pos_rows.columns]
        ).head(30)
    )
    return _page(request, "player.html", ctx)


# --- Board (who the market has wrong) ---------------------------------------

# Why the sportsbook line is worth more at some positions than others. Verbatim
# from `app.py`, because each sentence is a measured limitation of the feed
# rather than a hedge, and the quarterback one inverts the obvious reading.
_VEGAS_TRUST: dict[str, str] = {
    "WR": "<strong>Strongest here.</strong> Receiving yards is the dominant input to a "
          "receiver's scoring, so the line and fantasy value point the same way. Still "
          "missing touchdowns and receptions — no season-long market exists for either "
          "— so this is a view on volume, not on points.",
    "TE": "<strong>Strong here</strong>, same as receiver: receiving yards drives the "
          "scoring. Only nine tight ends are priced, so the percentile is cut against a "
          "thin field.",
    "RB": "<strong>Weaker here.</strong> The line is rushing yards, and 23 of 25 backs "
          "have no receiving market at all — so in a half-PPR league this is blind to "
          "exactly what separates a three-down back from an early-down one.",
    "QB": "<strong>Weakest here, and it looks most exciting here.</strong> The line is "
          "passing yards, but fantasy quarterback value concentrates in <em>rushing</em> "
          "— <code>rush_share</code> is the stickiest metric measured anywhere in this "
          "project at 0.82 — and FanDuel posts a rushing line for 5 of 24 quarterbacks. "
          "High-volume pocket passers therefore rise to the top of this list for the "
          "exact reason they are mediocre fantasy quarterbacks. Read it as a statement "
          "about passing volume, not about points.",
}


@router.get("/board", response_class=HTMLResponse)
def board_page(
    request: Request,
    profile: str | None = None,
    pos: str | None = None,
    dark: int = 1,
) -> HTMLResponse:
    """Quality against price, the sportsbook against price, and what repeats.

    **This is `valuation.py`, and it is deliberately not the draft board.**
    The draft board ranks across positions on PAR; this asks the other
    question — inside one position, who is rated above what he costs. An edit
    meant for `src/board.py` belongs on Draft Day, not here.
    """
    board = wd.valuation()
    ctx: dict[str, Any] = {"active": "value", "profile": profile, "empty": not board.height}
    if not board.height:
        return _page(request, "board_value.html", ctx)

    counts = val.summary(board)
    ctx["counts"] = {
        verdict: int(counts.filter(pl.col("verdict") == verdict).get_column("n").sum())
        for verdict in ("undervalued", "fairly priced", "overvalued")
    }

    available = [p for p in FANTASY_POSITIONS if board.filter(pl.col("position") == p).height]
    if not available:
        return _page(request, "board_value.html", ctx)
    which = pos if pos in available else available[0]
    view = board.filter(pl.col("position") == which)
    ctx.update(positions=available, position=which, n=view.height)

    ctx["quality_spec"] = charts.quality_scatter(view, bool(dark))
    ctx["thin_qb"] = which == "QB"

    # --- the sportsbook ------------------------------------------------------
    priced = (
        view.filter(pl.col("line_pct").is_not_null())
        if "line_pct" in view.columns else pl.DataFrame()
    )
    vegas: dict[str, Any] = {"n": priced.height, "total": view.height,
                             "trust": _VEGAS_TRUST.get(which, "")}
    if priced.height:
        vegas["spec"] = charts.vegas_scatter(priced, bool(dark))
        # Where the two independent opinions agree is the actual output here.
        both = priced.filter(
            pl.col("value_gap").is_not_null()
            & (
                ((pl.col("value_gap") >= 15) & (pl.col("vegas_gap") >= 10))
                | ((pl.col("value_gap") <= -15) & (pl.col("vegas_gap") <= -10))
            )
        )
        if both.height:
            bcols = [c for c in ("name", "team", "adp", "price_pct_priced", "quality_pct",
                                 "value_gap", "line", "line_pct", "vegas_gap")
                     if c in both.columns]
            vegas["agree"] = table_ctx(
                both.select(bcols).sort("vegas_gap", descending=True, nulls_last=True),
                pretty=False,
            )
        pcols = [c for c in ("name", "team", "adp", "market", "line", "line_pct",
                             "price_pct_priced", "vegas_gap", "value_gap")
                 if c in priced.columns]
        vegas["table"] = table_ctx(
            priced.select(pcols).sort("vegas_gap", descending=True, nulls_last=True),
            pretty=False,
        )
    ctx["vegas"] = vegas

    # --- what repeats, against what it costs --------------------------------
    try:
        rep = wd.stability_table()
        sticky = stab.sticky_features(rep, which, top_n=6) if rep.height else pl.DataFrame()
    except Exception as err:  # noqa: BLE001
        ctx["sticky_error"] = f"{type(err).__name__}: {err}"
        sticky = pl.DataFrame()
    if sticky.height:
        wanted = sticky.get_column("metric").to_list()
        r_by = {r["metric"]: r["r_yoy"] for r in sticky.iter_rows(named=True)}
        # Said out loud rather than dropped. This panel once silently rendered
        # one of two QB metrics and two of six at RB, because the columns were
        # not on the board — and a chart missing two thirds of its subject
        # looks exactly like a chart that is finished.
        ctx["sticky"] = {
            "spec": charts.sticky_against_price(view, wanted, r_by, bool(dark)),
            "absent": [m for m in wanted if m not in view.columns],
            "table": table_ctx(sticky),
        }
    return _page(request, "board_value.html", ctx)


# --- Research ---------------------------------------------------------------


@router.get("/research", response_class=HTMLResponse)
def research(
    request: Request,
    profile: str | None = None,
    regpos: list[str] = Query(default=[]),
    days: int = 7,
    snappos: str = "RB",
    shift_cut: int = 60,
    dark: int = 1,
) -> HTMLResponse:
    """The work behind the boards, including the parts that did not pan out.

    **Nothing measured-null is hidden here.** `breakout` and `projection` are
    kept as results, because the house rule is that a negative result is a
    result and the honest intervals are the most credible thing in the repo.
    """
    regpos = _clean(regpos)
    try:
        data = wd.board(profile)
    except KeyError as err:
        return _profile_error(request, err)

    ctx: dict[str, Any] = {"active": "research", "profile": profile, "season": wd.LAST_PLAYED}

    # --- screens ------------------------------------------------------------
    screens = wd.screens(wd.LAST_PLAYED)
    reg = screens["regression"]
    if reg.height:
        options = sorted(reg.get_column("position").unique().to_list())
        view = reg.filter(pl.col("position").is_in(regpos)) if regpos else reg
        ctx["regression"] = {
            "options": options, "selected": regpos,
            "table": table_ctx(view.sort("pts_over_exp", nulls_last=True).head(20)),
        }
    dis = screens["disagreement"]
    if dis.height:
        ctx["disagreement"] = table_ctx(dis.head(20))

    # The market comes from the board's own profile, never resolved again — a
    # bare `adp.movement` reads ppr/12 and would render another league's drift.
    prof = data.get("profile")
    scoring, teams = getattr(prof, "adp_scoring", None), getattr(prof, "adp_teams", None)
    if scoring and teams:
        days = max(3, min(21, int(days)))
        mv = wd.adp_movement(scoring, teams, days)
        ctx["movement"] = {
            "market": f"{scoring}/{teams}", "days": days,
            "rising": table_ctx(mv.sort("adp_change").head(12)) if mv.height else None,
            "falling": table_ctx(
                mv.sort("adp_change", descending=True, nulls_last=True).head(12)
            ) if mv.height else None,
        }

    snaps = wd.snap_trend(wd.LAST_PLAYED, snappos)
    ctx["snaps"] = {
        "position": snappos,
        "options": list(FANTASY_POSITIONS),
        "table": table_ctx(snaps.head(20)) if snaps.height else None,
    }

    # --- where the Footballers disagree with the curve ----------------------
    # A failed section says so on screen rather than disappearing. A section
    # that silently vanishes is indistinguishable from one that was never
    # built, which is how the stability panel went missing for an afternoon.
    shifts = pl.DataFrame()
    try:
        shifts = bd.compare_footballers(data)
    except Exception as err:  # noqa: BLE001 — a missing layer costs a panel
        ctx["footballers_error"] = f"{type(err).__name__}: {err}"
    if shifts.height:
        cut = max(20, min(200, int(shift_cut)))
        if "board_rank" in shifts.columns:
            shifts = shifts.filter(pl.col("board_rank") <= cut)
        ctx["footballers"] = {
            "cut": cut,
            "table": table_ctx(shifts.head(25)),
            "panel": wd.footballers_panel(),
        }

    # --- does a metric repeat? ---------------------------------------------
    try:
        rep = wd.stability_table()
        if rep.height:
            ctx["stability"] = {
                "table": table_ctx(
                    rep.sort("r_yoy", descending=True, nulls_last=True)
                ),
                "top": rep.sort("r_yoy", descending=True, nulls_last=True).row(
                    0, named=True
                ),
            }
    except Exception as err:  # noqa: BLE001
        ctx["stability_error"] = f"{type(err).__name__}: {err}"

    return _page(request, "research.html", ctx)


# --- Reference --------------------------------------------------------------


@router.get("/reference", response_class=HTMLResponse)
def reference(
    request: Request, profile: str | None = None, q: str | None = None
) -> HTMLResponse:
    """Every metric, what it is computed from, and what it does not mean."""
    needle = (q or "").lower().strip()
    groups: list[dict[str, Any]] = []
    for group, terms in glossary.groups().items():
        hits = [
            {"key": key, "label": term.label, "long": term.long or term.short}
            for key, term in terms
            if not needle
            or needle in key.lower()
            or needle in term.label.lower()
            or needle in (term.long or "").lower()
        ]
        if hits:
            groups.append({"name": group, "terms": hits})
    return _page(
        request, "reference.html",
        {"active": "reference", "profile": profile, "groups": groups, "q": q or "",
         "n": sum(len(g["terms"]) for g in groups)},
    )
