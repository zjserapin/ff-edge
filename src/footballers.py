"""The Fantasy Footballers' projections — three analysts, scored in your league.

Every other ranking source ships you a finished ordering. This one does not, and
that turns out to be the reason it is worth having: the UDK pages carry the
**underlying projected stat lines** for Andy, Jason and Mike separately, and the
site computes its own visible ranking by scoring them at render time. There is
no stored rank to scrape. So rather than importing somebody's ordering built for
somebody else's format, this module scores their stat lines through the same
`scoring._weighted_sum` primitive that prices everything else in this project,
under **this league's** settings. A 0.5-PPR superflex board built from their
numbers is a thing they do not publish and could not be scraped from them.

Three analysts also means a **spread**, which is a free uncertainty signal no
consensus-of-one source can give you. `ffb_spread` is the range across the panel
in league points; a player the three of them disagree about by 60 points is a
genuinely different bet from one they agree on to within 10, at identical price.

**Where it comes from.** Any of the four free position pages
(`{season}-{position}-rankings-draft`) embeds the entire payload — all
positions, all analysts — in a `window.udk.data = {...}` script literal. Verified
byte-identical between the QB and RB pages. So this fetches exactly one page and
ignores which one it nominally is. There is a REST endpoint behind the same data
(`udk/projections`) but it wants a WordPress nonce minted into the page, which
means fetching the page anyway; parsing the literal is strictly less machinery.
The UDK's own *tools* are paywalled — this payload is not, and no login, key or
subscription is involved in anything here.

---

### The traps, all of which return a number rather than an error

**The panel is not evenly stale, and this is the big one.** `updated_at` is per
analyst per player, and in August 2026 the three of them were in completely
different places: Jason had refreshed 195 of 313 players that month, Andy 123 of
306, and Mike **33 of 305 — with 179 still sitting at his May upload**. A plain
mean across the panel therefore blends a fresh August opinion with a
three-month-old one and prints a single confident consensus number. Every
consensus row here carries `stalest_days` and `freshest_days`, and
`panel_report()` exists to make the skew something you look at rather than
assume away.

**Coverage is uneven too.** 302 players carry all three analysts, 7 carry two
and 4 carry one. Averaging over "whoever happened to project him" silently
changes the panel composition from row to row, so two players' consensus numbers
are not comparable when one of them is Jason-only. `n_analysts` is on every row
and is the first thing to filter on.

**Kicker rows are empty shells.** The 40 K rows carry a name and a headshot and
not one stat — no field goals, no extra points, all nulls. Scored naively every
kicker in the league ties at exactly 0.0. They are dropped in `projections()`.

**Their ADP is not FFC's ADP.** The payload carries `adp`, `adp_ppr`,
`adp_half_ppr` and `adp_2qb`, and they are sparse — Aaron Rodgers has an
`adp_2qb` and nulls in the other three. They are namespaced to `ffb_adp_*` here
so that nothing downstream can confuse them with the FFC board the rest of this
project prices against. Do not mix the two in one comparison.

**Numbers arrive as strings, inconsistently.** `receiving_targets` is `"90.00"`
and `adp` is `"1.20"` while `rushing_yards` is a bare int. Cast everything.

**Names carry generational suffixes and the slug does not** — the payload has
`"Marvin Mims Jr."` against `"slug": "marvin-mims"`. Joins here go through
`ids.normalize` on both sides, and team codes through `ids.normalize_team` on
both sides, for the reason the project CLAUDE.md gives at length.

---

### In-season weekly rankings: the hooks exist, the source does not yet

Every row carries `season_type`, and every row in the draft payload is `"1"`.
Their page JS also carries a `getSeasonWeeks(season)` helper that no draft-season
page calls. Both strongly imply a `season_type` of 2 carrying week-level
projections, on the same schema, once the season starts.

**That could not be verified on 2026-08-10 and nothing here should pretend
otherwise.** Both plausible weekly URLs 404 in the preseason
(`/fantasy-football-rankings/`, `/2026-ultimate-draft-kit/udk-weekly-rankings/`),
so the page that will serve it does not exist yet. `season_type` is deliberately
carried through `projections()` unfiltered rather than being dropped as a
constant, so when the weekly board does appear the work is a URL, a filter, and
a week column — not a rewrite. Check it in September rather than trusting this
paragraph.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Mapping

import polars as pl
import requests

from src import ids
from src.cache import frame
from src.config import DATA_DIR, DEFAULT_SCORING, FANTASY_POSITIONS, SEASON
from src.scoring import _weighted_sum

# Any position page carries the whole payload; RB is chosen only because it is
# the largest board and therefore the one most likely to survive a site redesign
# that trims the smaller ones.
_PAGE = "https://www.thefantasyfootballers.com/{season}-running-back-rankings-draft/"

# A browser-shaped UA. The page is public, but a bare python-requests UA gets a
# challenge page rather than HTML, which parses to zero rows instead of failing.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_session = requests.Session()
_session.headers.update(_HEADERS)

# Their stat names against the scoring keys in `config.DEFAULT_SCORING`, so the
# league's own settings price the projection. Deliberately the same key space as
# `scoring.SCORING_COLUMNS` and `scoring.EXPECTED_COLUMNS` — a third namespace
# reading the same league scoring dict.
#
# What they do not project, and what that costs at this league's weights:
#   pass_2pt / rush_2pt / rec_2pt   no two-point columns exist. ~1-2 pts/season.
#   st_td                           no return touchdowns. Worth 6 when it lands.
#   kicking                         K rows carry no stats at all; see module doc.
# All three are omissions in the *projection*, not in the scoring — the keys
# simply have nowhere to read from, exactly like `scoring.unmapped_keys`.
PROJECTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "pass_yd": ("passing_yards",),
    "pass_td": ("passing_touchdowns",),
    "pass_int": ("interceptions_thrown",),
    "rush_yd": ("rushing_yards",),
    "rush_td": ("rushing_touchdowns",),
    "rec": ("receptions",),
    "rec_yd": ("receiving_yards",),
    "rec_td": ("receiving_touchdowns",),
    "fum_lost": ("fumbles_lost",),
}

# Every numeric field, including the ones that arrive quoted. Cast on load so no
# downstream caller has to know which is which.
_NUMERIC = (
    "passing_attempts", "passing_yards", "passing_touchdowns", "passing_completions",
    "rushing_attempts", "rushing_yards", "rushing_yards_per_attempt",
    "rushing_touchdowns", "receptions", "receiving_yards",
    "receiving_yards_per_reception", "receiving_touchdowns", "receiving_targets",
    "interceptions_thrown", "fumbles_lost", "risk", "upside",
)

_ADP_FIELDS = ("adp", "adp_ppr", "adp_half_ppr", "adp_2qb")


def _extract(html: str) -> list[dict[str, Any]]:
    """Pull the projections list out of the page's `window.udk.data` literal.

    The page assigns `window.udk.data` **twice**: once as an empty `{}` when the
    UDK namespace is set up, and again with the real payload. Taking the first
    match returns zero rows and no error, so this walks every assignment and
    keeps the last one that actually decodes with a `projections` key.

    `raw_decode` rather than a regex for the closing brace: the payload is 350KB
    of nested JSON containing escaped braces inside string values, and no regex
    finds its end correctly.
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []

    for match in re.finditer(r"window\.udk\.data\s*=\s*", html):
        brace = html.find("{", match.end() - 1)
        if brace == -1:
            continue
        try:
            payload, _ = decoder.raw_decode(html[brace:])
        except json.JSONDecodeError:
            continue
        rows = payload.get("projections") if isinstance(payload, dict) else None
        if rows:
            found = rows

    return found


def fetch(season: int = SEASON, force: bool = False) -> pl.DataFrame:
    """The raw projections payload, one row per analyst per player.

    Cached at the `live` TTL (1h) — the analysts revise continuously through
    August and then weekly in season, and this is the hook the "keep it fresh
    until kickoff" behaviour hangs on. A daily `bootstrap` run is always past the
    TTL and pulls fresh; twenty re-runs while you work stay entirely offline.
    """

    def _load() -> pl.DataFrame:
        resp = _session.get(_PAGE.format(season=season), timeout=30)
        resp.raise_for_status()
        rows = _extract(resp.text)
        if not rows:
            # An empty frame rather than a raise: the caller decides whether a
            # missing second opinion is fatal, and `board.build` already has a
            # warnings channel for exactly this.
            return pl.DataFrame()
        return pl.DataFrame(rows, infer_schema_length=None, strict=False)

    return frame(f"footballers_projections_{season}", "live", _load, force)


def projections(season: int = SEASON, force: bool = False) -> pl.DataFrame:
    """Typed, cleaned, skill-positions-only. One row per analyst per player.

    Kickers are dropped here (their rows carry no stats — see the module
    docstring) and so is anything outside `config.FANTASY_POSITIONS`, which is
    the set the rest of the project can actually price.
    """
    raw = fetch(season, force=force)
    if not raw.height:
        return raw

    numeric = [c for c in _NUMERIC if c in raw.columns]
    adp = [c for c in _ADP_FIELDS if c in raw.columns]

    df = raw.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric]
        # Namespaced on the way in so no downstream join can mistake these for
        # the FFC board this project actually prices against.
        + [pl.col(c).cast(pl.Float64, strict=False).alias(f"ffb_{c}") for c in adp]
        + [
            pl.col("fantasy_position").cast(pl.Utf8).str.to_uppercase().alias("position"),
            pl.col("updated_at").str.to_datetime(strict=False).alias("updated"),
        ]
    ).drop(adp)

    df = df.with_columns(ids.normalize_team("team"))
    return df.filter(pl.col("position").is_in(list(FANTASY_POSITIONS)))


def scored(
    scoring: Mapping[str, float] | None = None,
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Each analyst's projection, priced in this league's scoring.

    This is the whole point of the module. Their site renders one of five preset
    scoring systems; this league is 0.5 PPR with a superflex slot and a -2
    interception, and nobody publishes that board. Passing
    `scoring.league_settings()["scoring"]` prices the same stat lines under the
    league's live rules, so `ffb_points` is directly comparable with every other
    point estimate in this project.

    Adds `ffb_points`. Row grain is unchanged — one per analyst per player.
    """
    df = projections(season, force=force)
    if not df.height:
        return df

    scoring = scoring or DEFAULT_SCORING
    return df.with_columns(
        _weighted_sum(scoring, PROJECTION_COLUMNS).round(1).alias("ffb_points")
    )


def consensus(
    scoring: Mapping[str, float] | None = None,
    season: int = SEASON,
    min_analysts: int = 1,
    force: bool = False,
) -> pl.DataFrame:
    """One row per player: the panel's consensus, its spread, and how stale it is.

    **Median, not mean.** Three analysts is exactly the panel size where one
    outlier drags a mean a third of the way to itself, and the disagreements here
    are not symmetric — they are usually one analyst out on a limb about a
    backfield split. The mean is kept alongside as `ffb_points_mean` so the
    choice stays checkable rather than assumed.

    **`ffb_spread` is a feature, not a diagnostic.** Max minus min across the
    panel, in league points. Two players at the same ADP with spreads of 12 and
    68 are not the same decision, and this is the only source in the project that
    can tell you so.

    `stalest_days` is the age of the *oldest* opinion in the consensus. Read it
    before trusting a row: see the module docstring for how far apart the three
    of them drift.

    `min_analysts=3` restricts to the full panel, which is the honest comparison
    when ranking players against each other. The default of 1 keeps everybody and
    makes you look at `n_analysts` yourself.
    """
    df = scored(scoring, season, force=force)
    if not df.height:
        return df

    now = datetime.now()
    grouped = (
        df.group_by("player_id")
        .agg(
            pl.col("name").first(),
            pl.col("position").first(),
            pl.col("team").first(),
            pl.col("bye_week").first(),
            pl.col("ffb_points").median().round(1).alias("ffb_points"),
            pl.col("ffb_points").mean().round(1).alias("ffb_points_mean"),
            (pl.col("ffb_points").max() - pl.col("ffb_points").min())
            .round(1)
            .alias("ffb_spread"),
            pl.col("ffb_points").len().cast(pl.Int32).alias("n_analysts"),
            pl.col("risk").mean().round(1).alias("ffb_risk"),
            pl.col("upside").mean().round(1).alias("ffb_upside"),
            pl.col("updated").min().alias("_oldest"),
            pl.col("updated").max().alias("_newest"),
            *[pl.col(f"ffb_{c}").first() for c in _ADP_FIELDS],
        )
        .with_columns(
            ((pl.lit(now) - pl.col("_oldest")).dt.total_hours() / 24)
            .round(0)
            .cast(pl.Int32)
            .alias("stalest_days"),
            ((pl.lit(now) - pl.col("_newest")).dt.total_hours() / 24)
            .round(0)
            .cast(pl.Int32)
            .alias("freshest_days"),
        )
        .drop("_oldest", "_newest")
        .filter(pl.col("n_analysts") >= min_analysts)
    )

    # nulls_last on every descending sort in this repo — polars puts nulls first
    # by default, so a "best N" query otherwise returns the N unscoreable rows.
    return grouped.with_columns(
        pl.col("ffb_points")
        .rank("ordinal", descending=True)
        .cast(pl.Int32)
        .alias("ffb_rank"),
        pl.col("ffb_points")
        .rank("ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("ffb_pos_rank"),
    ).sort("ffb_points", descending=True, nulls_last=True)


def snapshot(
    scoring: Mapping[str, float] | None = None,
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Append today's consensus to the rolling history file. Idempotent per day.

    The same argument as `adp.snapshot`, for the same reason: the page serves
    *today's* projections and nothing else, nobody sells the history, and it
    cannot be backfilled. Start early or don't have it.

    What it buys is the one thing their own site cannot tell you — **which way
    an analyst is moving**. A player Jason has quietly walked down 40 points
    across three revisions in August is a different proposition from one he has
    held flat, and at draft time the direction of a revision is often worth more
    than its level. `movement()` reads this file.
    """
    today = consensus(scoring, season, force=force)
    if not today.height:
        return today

    stamp = date.today().isoformat()
    today = today.with_columns(pl.lit(stamp).alias("pulled_on"))
    path = DATA_DIR / f"footballers_history_{season}.parquet"

    if path.exists():
        prior = pl.read_parquet(path).filter(pl.col("pulled_on") != stamp)
        history = pl.concat([prior, today], how="diagonal_relaxed")
    else:
        history = today

    history.write_parquet(path)
    return history


def movement(days: int = 7, season: int = SEASON) -> pl.DataFrame:
    """Who the panel has moved most between the oldest and newest snapshot.

    `points_change` is in league points and signed the intuitive way — positive
    means the consensus went **up**. Note that this is the opposite convention
    from `adp.movement`, where a falling number means a rising player, because
    ADP is a rank and this is a score. Both say so out loud rather than relying
    on the reader to remember.

    Returns empty until two days of snapshots exist.
    """
    path = DATA_DIR / f"footballers_history_{season}.parquet"
    if not path.exists():
        return pl.DataFrame()

    history = pl.read_parquet(path)
    dates = sorted(history.get_column("pulled_on").unique().to_list())
    if len(dates) < 2:
        return pl.DataFrame()

    newest = dates[-1]
    oldest = dates[max(0, len(dates) - 1 - days)]

    cols = ["player_id", "name", "position", "team", "ffb_points", "ffb_rank"]
    old = history.filter(pl.col("pulled_on") == oldest).select(cols)
    new = history.filter(pl.col("pulled_on") == newest).select(cols)

    return (
        new.join(old, on=["player_id", "name", "position"], how="inner", suffix="_prior")
        .with_columns(
            (pl.col("ffb_points") - pl.col("ffb_points_prior"))
            .round(1)
            .alias("points_change"),
            (pl.col("ffb_rank_prior") - pl.col("ffb_rank"))
            .cast(pl.Int32)
            .alias("rank_change"),
        )
        .with_columns(
            pl.when(pl.col("points_change") > 0)
            .then(pl.lit("rising"))
            .when(pl.col("points_change") < 0)
            .then(pl.lit("falling"))
            .otherwise(pl.lit("flat"))
            .alias("direction")
        )
        .sort(pl.col("points_change").abs(), descending=True, nulls_last=True)
    )


def panel_report(season: int = SEASON, force: bool = False) -> dict[str, Any]:
    """Coverage and staleness per analyst — print this before trusting a consensus.

    The equivalent of `ids.match_report`: the number you should see rather than
    assume. `days_since_median` is the one to watch. In August 2026 it separated
    Mike from Jason by roughly two months, which is not a detail when their two
    numbers are being averaged into one.
    """
    df = projections(season, force=force)
    if not df.height:
        return {"rows": 0, "players": 0, "analysts": [], "coverage": []}

    now = datetime.now()
    per = (
        df.group_by("analyst_name")
        .agg(
            pl.col("player_id").len().alias("players"),
            pl.col("updated").median().alias("_median"),
            pl.col("updated").max().alias("_newest"),
        )
        .with_columns(
            ((pl.lit(now) - pl.col("_median")).dt.total_hours() / 24)
            .round(0)
            .cast(pl.Int32)
            .alias("days_since_median"),
            ((pl.lit(now) - pl.col("_newest")).dt.total_hours() / 24)
            .round(0)
            .cast(pl.Int32)
            .alias("days_since_newest"),
        )
        .drop("_median", "_newest")
        .sort("analyst_name")
    )

    coverage = (
        df.group_by("player_id")
        .agg(pl.col("analyst_name").n_unique().alias("n"))
        .group_by("n")
        .agg(pl.col("player_id").len().alias("players"))
        .sort("n", descending=True)
    )

    return {
        "rows": df.height,
        "players": df.get_column("player_id").n_unique(),
        "analysts": per.to_dicts(),
        "coverage": coverage.to_dicts(),
    }


def attach(
    players: pl.DataFrame,
    scoring: Mapping[str, float] | None = None,
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Left-join the consensus onto a board frame by normalized name + position.

    Name-only sources join on a name, and this project has been burned by that
    twice — once by generational suffixes collapsing a father onto his son, once
    by a defender sharing a name with a skill player. Position is the tiebreaker
    that separates both, and `ids.normalize` runs on **both** sides rather than
    the source side only, which is the half-a-join failure that passed tests for
    months.

    Unmatched players survive as nulls rather than being dropped: a board that
    silently shrinks because a ranking source has not published somebody yet is
    the worst possible outcome here.
    """
    if not players.height:
        return players

    cols = [
        "ffb_points", "ffb_points_mean", "ffb_spread", "ffb_rank", "ffb_pos_rank",
        "n_analysts", "ffb_risk", "ffb_upside", "stalest_days", "freshest_days",
    ]

    panel = consensus(scoring, season, force=force)
    if not panel.height:
        return players.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in cols]
        )

    right = panel.select(
        ids.normalize("name").alias("_norm"),
        pl.col("position").alias("_pos"),
        *[pl.col(c) for c in cols if c in panel.columns],
    ).unique(subset=["_norm", "_pos"], keep="first")

    return (
        players.with_columns(
            ids.normalize("name").alias("_norm"),
            pl.col("position").cast(pl.Utf8).str.to_uppercase().alias("_pos"),
        )
        .join(right, on=["_norm", "_pos"], how="left")
        .drop("_norm", "_pos")
    )
