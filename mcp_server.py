"""FastMCP server — makes the ff-edge cache queryable conversationally.

Why build this rather than install one: there is no official Sleeper MCP server
and none in Anthropic's connector directory. The community servers are all tiny
(2-8 stars, several already deleted) and wrap the public Sleeper endpoints
one-to-one — none of them walk `previous_league_id` for draft history, and none
join Sleeper's roster IDs to nflverse production data. Those two things are the
entire value of this pipeline, so wrap the pipeline instead of the raw API.

Strictly read-only. Nothing here can add, drop, trade, or modify a league.

claude_desktop_config.json:

    {
      "mcpServers": {
        "ff-edge": {
          "command": "uv",
          "args": [
            "--directory", "/absolute/path/to/ff-edge",
            "run", "python", "mcp_server.py"
          ]
        }
      }
    }

Run `uv run python bootstrap.py --light` once before first use so the tools
answer from cache instead of blocking on a cold download.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
from fastmcp import FastMCP

import adp as adp_mod
import cache
import ids
import nflverse as nv
import peek
import sleeper
from config import SEASON

mcp = FastMCP("ff-edge")


def _out(df: pl.DataFrame, limit: int = 60) -> str:
    """Serialize a frame for an LLM: capped rows, plus the count that was capped.

    Row caps are not cosmetic here — `depth_charts` alone is 600k rows and would
    blow the context window several times over. Returning `rows` alongside
    `returned` lets the model know it's looking at a head, not the whole thing.
    """
    if df is None or df.height == 0:
        return json.dumps({"rows": 0, "returned": 0, "data": []})
    head = df.head(limit)
    return json.dumps(
        {
            "rows": df.height,
            "returned": head.height,
            "data": head.to_dicts(),
        },
        default=str,
    )


# --- League tools -----------------------------------------------------------


@mcp.tool
def my_leagues(season: int = SEASON) -> str:
    """List the user's Sleeper leagues for a season.

    Start here for anything league-specific — every other league tool needs a
    league_id, and this is where you get one. Requires SLEEPER_USERNAME to be set
    in config.py.
    """
    return _out(sleeper.my_leagues(season))


@mcp.tool
def league_rosters(league_id: str) -> str:
    """Who owns whom, with manager display names attached.

    Use when the question is about roster construction or trade fits: which
    manager is thin at RB, who is stockpiling a position, what's actually
    available to trade for.
    """
    rosters = sleeper.rosters(league_id)
    users = sleeper.league_users(league_id)
    if rosters.height == 0:
        return _out(rosters)
    if users.height and "user_id" in users.columns:
        rosters = rosters.join(
            users.select(["user_id", "display_name"]),
            left_on="owner_id",
            right_on="user_id",
            how="left",
        )
    return _out(rosters)


@mcp.tool
def draft_history(league_id: str, seasons_back: int = 4) -> str:
    """Every pick this league's managers have made, back through prior seasons.

    Use to answer "how does this specific league draft" — who reaches for QBs,
    who never takes a TE early, which manager always zeroes RB. This is behavior
    data about your actual opponents, which is worth more than any generic ADP
    read because you are drafting against them, not against the market.
    """
    return _out(sleeper.draft_history(league_id, seasons_back), limit=200)


@mcp.tool
def transactions(league_id: str, weeks: int = 18) -> str:
    """Season-long adds, drops, trades, and FAAB bids.

    Use to profile manager tendencies before a draft or a trade offer: FAAB
    aggression, how long people hold injured players, who churns the bottom of
    their bench weekly.
    """
    return _out(sleeper.all_transactions(league_id, weeks), limit=150)


# --- Market tools -----------------------------------------------------------


@mcp.tool
def adp(scoring: str = "ppr", teams: int = 12, position: str = "all") -> str:
    """Current consensus ADP with dispersion (stdev, high, low).

    Use for "where is this player going" and for sanity-checking whether a
    ranking disagreement is actually actionable. scoring: ppr / half-ppr /
    standard.
    """
    return _out(adp_mod.fetch(scoring=scoring, teams=teams, position=position), limit=100)


@mcp.tool
def survival(pick_no: int, scoring: str = "ppr", teams: int = 12) -> str:
    """P(each player is still on the board at a given pick).

    This is the tool for on-the-clock decisions. "Best player available" ignores
    what you can get back around; this answers the real question — who do I lose
    by waiting until my next pick. Two players at the same ADP with different
    stdev give completely different answers.
    """
    df = adp_mod.survival(adp_mod.fetch(scoring=scoring, teams=teams), pick_no)
    col = f"p_available_at_{pick_no}"
    return _out(df.sort("adp").head(80).select(["name", "position", "team", "adp", "stdev", col]), limit=80)


@mcp.tool
def adp_movement(days: int = 7) -> str:
    """Biggest ADP risers and fallers across local snapshots.

    Use to catch news the rankings haven't absorbed yet. Returns empty until at
    least two daily snapshots exist — ADP history cannot be backfilled, it only
    accumulates from when you started running `adp.snapshot()`.
    """
    return _out(adp_mod.movement(days), limit=60)


@mcp.tool
def market_disagreement(top_n: int = 250) -> str:
    """Players where expert consensus is least settled (sd, best, worst, range).

    Use to find where your own read has value. Tight consensus is already priced
    into ADP; wide dispersion means the argument is live. Note both bias
    directions documented in the underlying function — do not sort on sd or
    rel_sd alone, filter to an ECR band first.
    """
    return _out(peek.market_disagreement(top_n), limit=80)


# --- Production tools -------------------------------------------------------


@mcp.tool
def points_over_expected(
    season: int = 2025, direction: str = "under", min_games: int = 10
) -> str:
    """Players whose production diverged most from their opportunity.

    direction='under' (default) surfaces buy candidates: the volume was there,
    the points weren't, and volume is stickier than efficiency. direction='over'
    surfaces sell candidates who outran their opportunity, usually on touchdown
    variance that won't repeat.
    """
    df = peek.regression_candidates(season, min_games)
    if direction == "over":
        df = df.sort("pts_over_exp", descending=True)
    return _out(df, limit=50)


@mcp.tool
def usage_leaders(season: int = 2025, position: str = "WR", min_games: int = 8) -> str:
    """Target share and air yards share leaders at a position.

    Use to separate genuine offensive alphas from volume-dependent profiles.
    High target share with low air yards share collapses when the offense
    changes; both high is a role that survives coordinator turnover.
    """
    return _out(peek.usage_leaders(season, position, min_games), limit=50)


@mcp.tool
def snap_trend(season: int = 2025, position: str = "RB", last_n: int = 4) -> str:
    """Late-season snap share vs the full-season baseline.

    Use to catch role changes before the box score reflects them — a player has
    to be on the field before he can produce, so snaps lead production by a week
    or two. Positive delta into an offseason is a cheap signal the staff's view
    changed.
    """
    return _out(peek.snap_trend(season, position, last_n), limit=50)


@mcp.tool
def player_lookup(name: str) -> str:
    """Find a player's IDs across every source (gsis, sleeper, pfr, espn, yahoo).

    Use when a join fails or when you need to connect a Sleeper roster entry to
    nflverse production data. Matches on a normalized name, so punctuation and
    Jr./Sr. suffixes don't matter.
    """
    xw = ids.crosswalk()
    key = name.lower().replace(".", "").replace("'", "").replace("-", " ").strip()
    return _out(xw.filter(pl.col("nkey").str.contains(key, literal=True)), limit=20)


# --- Housekeeping -----------------------------------------------------------


@mcp.tool
def data_inventory() -> str:
    """What's cached locally, how big, and how stale.

    Check this first when an answer looks wrong or out of date — a table that's
    400 hours old is probably the explanation.
    """
    return _out(cache.inventory(), limit=100)


@mcp.tool
def refresh(source: str) -> str:
    """Force-refresh one cached source, bypassing its TTL.

    Valid: adp, ff_rankings, rosters, weekly_stats, ff_opportunity, snap_counts,
    crosswalk. Use when you know something changed (news broke, a game finished)
    and the cached copy predates it.
    """
    loaders: dict[str, Any] = {
        "adp": lambda: adp_mod.fetch(force=True),
        "ff_rankings": lambda: nv.ff_rankings(force=True),
        "rosters": lambda: nv.rosters(force=True),
        "weekly_stats": lambda: nv.weekly_stats(force=True),
        "ff_opportunity": lambda: nv.ff_opportunity(force=True),
        "snap_counts": lambda: nv.snap_counts(force=True),
        "crosswalk": lambda: ids.crosswalk(force=True),
    }
    if source not in loaders:
        return json.dumps({"error": f"unknown source {source!r}", "valid": sorted(loaders)})
    df = loaders[source]()
    return json.dumps({"refreshed": source, "rows": df.height})


if __name__ == "__main__":
    mcp.run()
