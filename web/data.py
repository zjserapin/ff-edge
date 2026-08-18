"""Everything a page reads, assembled once and memoized.

This is app.py's cache-boundary section ported off Streamlit: the same
composition, in the same deliberate order, keyed by profile name instead of
implicit session state.

The chain in `board()` is the contract, and it is order-dependent throughout —
environment before `apply_env_weight` (which needs `env_swing`), quality before
`rank_board` (which breaks ties on it), `indistinguishable` recomputed on
`par_env` so the environment weight can move a player between blocks, and
`attach_usage` after the ranking so "usage cannot move a rank" stays a fact
about the call graph. Splitting this chain was a real bug once: two tabs
disagreeing about the top of the board eight days before a draft. One ranking,
for every page that shows one.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from src import archetypes as ar
from src import board as bd
from src import features as ft
from src import profiles as pf
from src import props as pp
from src import scoring as sc
from src import sleeper
from src import valuation as val
from src.config import LEAGUE_ID
from web.memo import memo


@memo(ttl_seconds=300)
def league_identity() -> dict[str, Any]:
    """Which league this page is actually reading, and how it got there.

    **This exists because discovery silently picks the wrong league.**
    `scoring.resolve_league_id` falls through to `sleeper.my_leagues()` and
    takes the *first* row when `FF_EDGE_LEAGUE_ID` is unset. On this account
    that row is **The Jungle** — the dynasty startup that is out of scope by
    decision — not the Shiva Bowl. The board then prices one league's rosters
    with another league's profile and looks entirely plausible doing it: no
    exception, no empty frame, just a well-formed board for a draft you are
    not having. It is the `adp.movement` wrong-market defect wearing different
    clothes, one level further up.

    So the header names the league out loud and says whether the name was
    *chosen* or *guessed*. `discovered` is the flag the template makes loud.
    """
    explicit = bool(LEAGUE_ID)
    try:
        resolved = sc.resolve_league_id()
    except Exception:  # noqa: BLE001 — offline is a supported state
        resolved = ""
    if not resolved:
        return {
            "resolved": False, "explicit": explicit, "discovered": False,
            "name": None, "others": [],
        }

    meta = None
    others: list[str] = []
    try:
        meta = sleeper.league(resolved)
        if not explicit:
            frame = sleeper.my_leagues()
            if frame.height and "name" in frame.columns:
                others = [
                    n for n, i in zip(
                        frame.get_column("name").to_list(),
                        frame.get_column("league_id").to_list(),
                    ) if i != resolved
                ]
    except Exception:  # noqa: BLE001 — a dead network costs a label, not a page
        pass

    return {
        "resolved": True,
        "explicit": explicit,
        "discovered": not explicit,
        "league_id": resolved,
        "name": (meta or {}).get("name"),
        "teams": (meta or {}).get("total_rosters"),
        "season": (meta or {}).get("season"),
        "others": others,
    }


@memo()
def features() -> pl.DataFrame:
    return ft.build()


@memo()
def scores(season: int, min_games: int = 8) -> pl.DataFrame:
    return ar.scores(season, min_games=min_games, df=features())


@memo()
def valuation() -> pl.DataFrame:
    """The quality-vs-price board with the sportsbook's view joined on.

    Composed here rather than inside `valuation.board` so the module's tests
    never require FanDuel to be reachable. A book that is down, rate-limiting,
    or renaming markets must cost the page a column, not the whole page —
    every consumer already treats `vegas_gap` as optional because it is null
    for two thirds of the board on a good day.
    """
    board_ = val.board(df=features())
    if not board_.height:
        return board_
    try:
        return pp.against_price(board_)
    except Exception:  # noqa: BLE001 — any book failure degrades to no column
        return board_


@memo(ttl_seconds=60)
def board(profile_name: str | None = None) -> dict[str, Any]:
    """The board, fully layered and ranked, for one profile.

    60-second TTL because Draft Day reads this and drafted players have to
    fall off the board while picks are being made; the expensive inputs
    (features, valuation) are memoized separately without a TTL, so a refresh
    re-runs only the league-shaped parts.

    Raises KeyError on an unknown profile name — `profiles.resolve` refuses to
    fall back, and the page layer turns that into a 404 sentence rather than a
    plausible board for the wrong league.
    """
    profile = pf.resolve(profile_name)
    out = bd.build(profile=profile)
    players = out["players"]
    if not players.height:
        return out

    players = bd.attach_quality(bd.attach_environment(players), profile=profile)
    players = bd.attach_vegas(players, valuation())
    players = bd.signal(players)
    # ECR folds into `blend_par` before the environment, so the layer order
    # stays slot value -> player -> crowd -> team.
    players = bd.blend_ecr(bd.attach_ecr(players, profile=profile))
    players = bd.apply_env_weight(players)
    players = bd.indistinguishable(players, value_col="par_env")
    players = bd.positional_drop(players)
    # Roster demand before ranking, on the same column the ranking uses: the
    # cap decides *how many* of a position the draft will absorb while
    # `par_env` still decides *which*.
    players = bd.roster_demand(players, out.get("replacement"), value_col="par_env")
    # `need` rides along on every row rather than filtering any out — a
    # position you cannot start still has bench and trade value; it just must
    # not silently rank alongside the ones that can improve your lineup.
    need = bd.roster_need(profile=profile)
    if need.height and "slots_open" in need.columns:
        players = players.join(
            need.select("position", pl.col("slots_open").alias("need")),
            on="position",
            how="left",
        )
    ranked = bd.rank_board(players, value_col="par_env")
    out["players"] = bd.attach_usage(ranked, features())
    return out


@memo(ttl_seconds=60)
def my_picks() -> pl.DataFrame:
    return bd.picks()


@memo(ttl_seconds=60)
def my_need(profile_name: str | None = None) -> pl.DataFrame:
    return bd.roster_need(profile=pf.resolve(profile_name))
