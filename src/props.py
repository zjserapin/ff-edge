"""FanDuel player prop lines — the market's own projection, with a price on it.

ADP tells you what a *drafter* thinks a player is worth. A prop line tells you
what a *bookmaker* thinks he will actually do, backed by money and updated far
more often than any projection site. That is a genuinely different signal, and
it is free.

Two families of market, one parser, because FanDuel shapes them identically:

  season-long  futures on a whole season ("Regular Season Rushing Yards"). Live
               now, useful for draft-season valuation, and *dead weight in
               October* — they settle in January and stop being re-priced in any
               way that helps a weekly decision.
  weekly       per-game player props, attached to an event. These are what a
               start/sit or rest-of-season tool actually wants. FanDuel does not
               post them until roughly game week, so `weekly()` returns an empty
               frame in August by design, not by failure.

What is NOT here, having checked rather than assumed (2026-08-10):

  * **No receiving-TD market exists at all**, season-long. 45 receiving-yards
    markets, zero receiving TDs. Nothing to parse around.
  * **No season-long receptions market**, for anyone. In a 0.5 PPR league that
    is a real hole in the WR/TE and pass-catching-RB picture.
  * **Not every player is priced.** 92 players season-long, top-of-board only.
    Any join to the full draft board is expected to be mostly nulls; that is the
    data, not a broken join. Use `ids.match_report` and look at it.

Traps, each found the hard way and each silent:

  * `handicap` is **0 on every one of these markets**. The line lives inside the
    runner *name* string ("Bijan Robinson Over 1150.5"). Reading `handicap` gives
    a clean frame where every line is 0.0 and nothing raises.
  * **All 97 season-long yardage markets are priced -114/-114**, exactly. There
    is no price signal to de-vig — `fair_over` is 0.500 by construction. The
    line is the entire statement. Only the TD markets carry a lean (11 and 14
    distinct price pairs across passing and rushing).
  * **`book_group` is not a position.** FanDuel files tight ends under
    WIDE_RECEIVERS — Bowers, Kittle and Loveland are all in that bucket. Joining
    on it as a position drops every TE.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import polars as pl
import requests

from src import ids
from src.cache import blob
from src.config import (
    DEFAULT_SCORING,
    FANDUEL_BASE,
    FANDUEL_KEY,
    FANDUEL_NFL_EVENT_TYPE,
    FANDUEL_REGION,
)

_session = requests.Session()
# FanDuel serves the API to anything with a browser-shaped UA. Identifying the
# project honestly in the comment rather than the header, because a non-browser
# UA gets a 403 here even though the endpoint needs no auth.
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
)

# "Ashton Jeanty Regular Season Rushing Yards 2026-27" -> player, cat, stat
_SEASON_MARKET = re.compile(
    r"^(?P<player>.+?) Regular Season "
    r"(?P<cat>Passing|Rushing|Receiving) (?P<stat>Yards|TDs) "
    r"(?P<season>\d{4}-\d{2})$"
)

# "Bijan Robinson Over 1150.5" -> side, line. See the module docstring: this is
# the only place the line actually exists.
_RUNNER = re.compile(r"^(?P<player>.+?) (?P<side>Over|Under) (?P<line>[\d.]+)$")

# market key -> the scoring key in config.DEFAULT_SCORING it feeds. Keeping this
# as a mapping rather than string surgery means a scoring change in config.py
# flows through without touching this file.
_SCORING_KEY: dict[str, str] = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",  # no such market today; here so it works if added
    "receptions": "rec",  # likewise — weekly has it, season-long does not
}

_SCHEMA: dict[str, Any] = {
    "player": pl.Utf8,
    "market": pl.Utf8,
    "line": pl.Float64,
    "over_odds": pl.Int64,
    "under_odds": pl.Int64,
    "book_group": pl.Utf8,
    "market_id": pl.Utf8,
    "event_id": pl.Int64,
    "status": pl.Utf8,
    "pulled_on": pl.Utf8,
}


def _get(path: str, params: dict[str, Any], region: str) -> dict:
    """One GET against FanDuel's sportsbook API."""
    url = f"{FANDUEL_BASE.format(region=region)}/{path}"
    resp = _session.get(
        url, params={**params, "_ak": FANDUEL_KEY, "timezone": "America/New_York"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def nfl_page(region: str = FANDUEL_REGION, force: bool = False) -> dict:
    """The whole NFL board: season-long futures, awards, and the event list."""
    return blob(
        f"fanduel_nfl_{region}",
        "weekly",  # season-long lines drift over days, not hours
        lambda: _get(
            "content-managed-page",
            {"page": "CUSTOM", "customPageId": "nfl"},
            region,
        ),
        force,
    )


def event_page(event_id: int, region: str = FANDUEL_REGION, force: bool = False) -> dict:
    """One game's markets. In-season this carries the weekly player props."""
    return blob(
        f"fanduel_event_{event_id}_{region}",
        "live",  # weekly props move on news; an hour is already generous
        lambda: _get("event-page", {"eventId": event_id}, region),
        force,
    )


def _rows_from(payload: dict, keep: Any) -> list[dict]:
    """Pull every two-way Over/Under market matching `keep` out of a payload.

    `keep` is a predicate on the raw market dict, which is what lets season-long
    and weekly share this function — they differ only in which markets to take
    and how the name parses, never in the runner shape.
    """
    markets = payload.get("attachments", {}).get("markets", {})
    today = date.today().isoformat()
    rows: list[dict] = []

    for m in markets.values():
        if not keep(m):
            continue
        parsed = _parse_market_name(m.get("marketName", ""))
        if parsed is None:
            continue
        player, market = parsed

        sides: dict[str, tuple[float, int | None]] = {}
        for r in m.get("runners", []):
            rg = _RUNNER.match(r.get("runnerName", ""))
            if rg is None:
                continue
            odds = (
                r.get("winRunnerOdds", {})
                .get("americanDisplayOdds", {})
                .get("americanOddsInt")
            )
            sides[rg["side"].lower()] = (float(rg["line"]), odds)

        # A one-sided market is a market mid-move or suspended. Skipping it beats
        # emitting a row with a line and half a price. Mismatched lines mean the
        # same thing — the book is repricing mid-pull — and are dropped for the
        # same reason: silently keeping the over's number would invent a market
        # that was never offered.
        if "over" not in sides or "under" not in sides:
            continue
        if sides["over"][0] != sides["under"][0]:
            continue

        rows.append(
            {
                "player": player,
                "market": market,
                "line": sides["over"][0],
                "over_odds": sides["over"][1],
                "under_odds": sides["under"][1],
                "book_group": m.get("marketType"),
                "market_id": str(m.get("marketId")),
                "event_id": m.get("eventId"),
                "status": m.get("marketStatus"),
                "pulled_on": today,
            }
        )
    return rows


def _parse_market_name(name: str) -> tuple[str, str] | None:
    """Market name -> (player, market key), or None if it isn't a player prop."""
    g = _SEASON_MARKET.match(name)
    if g is None:
        return None
    stat = "yards" if g["stat"] == "Yards" else "tds"
    return g["player"], f"{g['cat'].lower()}_{stat}"


def season_long(region: str = FANDUEL_REGION, force: bool = False) -> pl.DataFrame:
    """Season-long player props: one row per player per market, both prices.

    Draft-season tool. Do not reach for this in October — see the module
    docstring on why these stop being informative once games are played.
    """
    payload = nfl_page(region, force)
    rows = _rows_from(
        payload, lambda m: str(m.get("marketType", "")).startswith("REGULAR_SEASON_PROPS")
    )
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=_SCHEMA).sort(
        ["market", "line"], descending=[False, True], nulls_last=True
    )


def weekly(
    event_id: int, region: str = FANDUEL_REGION, force: bool = False
) -> pl.DataFrame:
    """Per-game player props for one event.

    Returns an empty frame (correct schema) whenever FanDuel has not posted
    player props for the game yet, which is the normal state until game week.
    The name patterns FanDuel uses for weekly markets are *not* verified here —
    none were live to check against on 2026-08-10 — so `_parse_market_name` will
    need a weekly branch once there is a real payload to read. Until then this
    returns empty rather than guessing at a format.
    """
    payload = event_page(event_id, region, force)
    rows = _rows_from(payload, lambda m: "runners" in m)
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=_SCHEMA)


def events(region: str = FANDUEL_REGION, force: bool = False) -> pl.DataFrame:
    """Scheduled NFL events with FanDuel's ids — the input to `weekly()`."""
    payload = nfl_page(region, force)
    raw = payload.get("attachments", {}).get("events", {})
    rows = [
        {
            "event_id": e.get("eventId"),
            "name": e.get("name"),
            "opens": e.get("openDate"),
        }
        for e in raw.values()
        if e.get("eventTypeId") == FANDUEL_NFL_EVENT_TYPE
    ]
    if not rows:
        return pl.DataFrame(
            schema={"event_id": pl.Int64, "name": pl.Utf8, "opens": pl.Utf8}
        )
    return pl.DataFrame(rows).sort("opens")


def _implied(odds: pl.Expr) -> pl.Expr:
    """American odds -> raw implied probability (still carrying the vig)."""
    return (
        pl.when(odds > 0)
        .then(100.0 / (odds + 100.0))
        .otherwise(-odds.cast(pl.Float64) / (-odds.cast(pl.Float64) + 100.0))
    )


def devig(df: pl.DataFrame) -> pl.DataFrame:
    """Add `fair_over` and `vig`, normalizing the two sides to sum to 1.

    Expect `fair_over` to be exactly 0.500 on every yardage market — those are
    price-pinned at -114/-114 and carry no lean. That is a property of the feed,
    not a bug here.
    """
    over, under = _implied(pl.col("over_odds")), _implied(pl.col("under_odds"))
    return df.with_columns(
        (over / (over + under)).alias("fair_over"),
        (over + under - 1.0).alias("vig"),
    )


def implied_points(
    df: pl.DataFrame, scoring: dict[str, float] | None = None
) -> pl.DataFrame:
    """Fantasy points implied by the lines, per player.

    **This is a floor, not a projection**, and the gap differs by position:
    running backs are missing their entire receiving game (no receptions market
    exists, and most RBs have no receiving-yards market either), while
    quarterbacks are missing rushing and interceptions. The `markets` column
    lists exactly which lines fed each total so the omission stays visible
    instead of being averaged into something that looks authoritative.
    """
    scoring = scoring or DEFAULT_SCORING
    weights = pl.DataFrame(
        {
            "market": list(_SCORING_KEY),
            "weight": [scoring.get(k, 0.0) for k in _SCORING_KEY.values()],
        }
    )
    return (
        df.join(weights, on="market", how="inner")
        .group_by("player")
        .agg(
            (pl.col("line") * pl.col("weight")).sum().alias("implied_points"),
            pl.col("market").sort().alias("markets"),
        )
        .sort("implied_points", descending=True, nulls_last=True)
    )


# Which positions a market can plausibly belong to. Used as an *intersection*
# across all of a player's markets, because every market describes one person:
# Lamar Jackson has passing and rushing lines, and QB ∩ {QB,RB,WR} pins him to
# QB without anyone hardcoding that he is a quarterback.
_MARKET_POSITIONS: dict[str, tuple[str, ...]] = {
    "passing": ("QB",),
    "rushing": ("QB", "RB", "WR"),
    "receiving": ("RB", "WR", "TE"),
}

# The crosswalk carries every player at every position. Restricting to skill
# positions is what stops a prop matching a defender who happens to share a name.
_SKILL: tuple[str, ...] = ("QB", "RB", "WR", "TE")


def _allowed_positions(markets: list[str]) -> list[str]:
    """Positions consistent with *every* market a player is priced in."""
    sets = [set(_MARKET_POSITIONS[m.split("_")[0]]) for m in markets]
    hit = set.intersection(*sets) if sets else set(_SKILL)
    # An empty intersection would mean contradictory markets on one name. Fall
    # back to the full skill set rather than silently resolving to nobody.
    return sorted(hit or set(_SKILL))


def resolve_players(df: pl.DataFrame) -> pl.DataFrame:
    """One identity per prop player: name -> gsis_id, sleeper_id, position.

    FanDuel ships no player id, so this is a name join — the shape that quietly
    loses rows everywhere else in this project. Three real collisions exist in
    the current board and each needs a different guard:

      Lamar Jackson     QB (BAL) vs CB (CAR)      -> skill-position filter
      Justin Jefferson  WR (MIN) vs LB (CLE, '26) -> skill-position filter
      Michael Pittman   WR (Jr, 2020) vs RB (Sr, 1998, no gsis_id)

    The third is the nasty one: `ids.normalize` strips generational suffixes, so
    father and son collapse to the same key and both are offensive players. The
    tiebreak is a real gsis_id first, then the later draft class.

    Resolving per *player* rather than per market row is deliberate — it is what
    makes `attach_ids` structurally row-count preserving instead of quietly
    fanning 145 prop rows out to 151.
    """
    per = (
        df.group_by("player")
        .agg(pl.col("market").unique().alias("markets"))
        .with_columns(ids.normalize("player"))
    )
    per = per.with_columns(
        pl.Series("pos_ok", [_allowed_positions(m) for m in per["markets"].to_list()])
    )

    xw = (
        ids.crosswalk()
        .filter(pl.col("position").is_in(_SKILL))
        .with_columns(pl.col("gsis_id").is_null().alias("_no_id"))
    )

    hit = (
        # `_allowed_positions` guarantees a non-empty list, so the Polars 2.0
        # empty-list behaviour is moot here — pinned anyway so the default flip
        # cannot change what this join does.
        per.explode("pos_ok", empty_as_null=False)
        .join(xw, left_on=["nkey", "pos_ok"], right_on=["nkey", "position"], how="inner")
        .sort(["_no_id", "draft_year"], descending=[False, True], nulls_last=True)
        .unique(subset=["player"], keep="first", maintain_order=True)
        .select(
            "player",
            "gsis_id",
            "sleeper_id",
            pl.col("pos_ok").alias("position"),
            pl.col("name").alias("nflverse_name"),
        )
    )
    return per.select("player").join(hit, on="player", how="left")


def attach_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Props with gsis_id / sleeper_id attached, one row in, one row out.

    Always print `ids.match_report(result, 'gsis_id')` — a name-only join is the
    classic way this project goes quietly wrong, and the point of `match_report`
    is that the rate is something you see rather than assume.
    """
    return df.join(resolve_players(df), on="player", how="left")
