"""Cached wrappers over nflreadpy — the free half of the data layer.

Why a wrapper file at all: nflreadpy hits the network on every call and returns
full-history tables. Left unwrapped, an exploratory notebook re-downloads 100MB
of snap counts because you re-ran a cell. Every function here is a one-liner
over `cache.frame`, with the season range baked into the cache key so different
windows don't clobber each other.

A note on the two tables that carry the most signal — `ff_opportunity` and
`ff_rankings` — is in their docstrings. Read those before anything else.

nflreadpy returns polars natively; nothing here converts to pandas.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from src.cache import frame
from src.config import FTN_SEASONS, HISTORY_SEASONS, PBP_SEASONS, SEASON


def _key(seasons: list[int] | int) -> str:
    """Cache-key fragment for a season range: [2022,2023,2025] -> '2022-2025'."""
    if isinstance(seasons, int):
        return str(seasons)
    return f"{min(seasons)}-{max(seasons)}" if seasons else "all"


def _played(seasons: list[int]) -> list[int]:
    """Drop seasons whose games have not been played yet.

    **Asking nflreadpy for a season that has not started raises, and that killed
    the whole app on 2026-08-13.** Nine of the loaders wrapped in this file
    validate their season range — `ValueError: Season must be between 2006 and
    2025` — and two more 404 looking for a file the nflverse has not published.
    An unhandled exception inside a Streamlit tab takes the entire page, not the
    section, so a single call for `[SEASON]` in the preseason is fatal.

    The live one was `promotion.weekly_trust(2026)`, reached from the claims
    ledger. Note how it hid: `weekly_trust` already guards with
    `if not rush_raw.height`, and `claims.resolve` already treats an empty week
    table as "pending, not failed". **The intended behaviour was fully built.**
    It simply sat one line after the call that raised, because the author
    reasonably assumed a future season came back empty rather than throwing. And
    it stayed dormant until the ledger had its first row — `resolve` returns
    early on an empty ledger — so the bug shipped the day bootstrap first pulled
    claims, which was nine days before the draft.

    So this converts "raises" into "returns empty", which is what every caller in
    this repo already handles. Seasons are filtered *before* the cache key is
    built, so a request for 2018-2026 is stored as 2018-2025 — the honest name
    for what was actually fetched, and it means nothing has to be invalidated in
    September when the games start.

    **`get_current_season()` is the games-based answer and that is the one we
    want.** Its `roster=True` variant flips to the new year on March 15 and would
    report 2026 today. That is the same disagreement `CLAUDE.md` documents for
    team codes, where the 2026 roster feed alone spells Arizona `AZ`: roster data
    rolls forward months before game data does.
    """
    return [s for s in seasons if s <= nfl.get_current_season()]


# --- Production & opportunity ----------------------------------------------


def weekly_stats(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """Per-player, per-week box score production."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"weekly_stats_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
        force,
    )


def season_stats(
    seasons: list[int] | None = None, level: str = "reg", force: bool = False
) -> pl.DataFrame:
    """Season totals. `level` is one of week/reg/post/reg+post — there is no
    'season' value; passing one raises ValueError inside nflreadpy."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"season_stats_{level}_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level=level),
        force,
    )


def team_stats(
    seasons: list[int] | None = None, level: str = "reg", force: bool = False
) -> pl.DataFrame:
    """Team-level production — the denominator for target/carry share work."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"team_stats_{level}_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_team_stats(seasons=seasons, summary_level=level),
        force,
    )


def ff_opportunity(
    seasons: list[int] | None = None, stat_type: str = "weekly", force: bool = False
) -> pl.DataFrame:
    """Expected fantasy points given the opportunity a player actually received.

    This is the highest-value table in the project. For each player-week it
    carries `*_exp` (what the touches were worth in expectation, given down,
    distance, field position, and air yards) alongside actual production, plus
    `*_diff` = actual minus expected.

    That split is the thing most public projections blur together. A player who
    scored 12 PPG on 16 expected and one who scored 12 PPG on 8 expected are not
    the same asset: the first has volume that isn't converting, the second has
    efficiency that probably won't repeat. Volume is far stickier year to year
    than efficiency, so the sign of `*_diff` tells you which side of that you're
    buying.

    `stat_type` is weekly / pbp_pass / pbp_rush only — 'season' raises.
    """
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"ff_opportunity_{stat_type}_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_ff_opportunity(seasons=seasons, stat_type=stat_type),
        force,
    )


def snap_counts(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """Offensive/defensive snap participation. Role change shows here first."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"snap_counts_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_snap_counts(seasons=seasons),
        force,
    )


def injuries(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """Weekly injury reports — practice status and game designation."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"injuries_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_injuries(seasons=seasons),
        force,
    )


def participation(
    seasons: list[int] | None = None, force: bool = False
) -> pl.DataFrame:
    """Play-level personnel groupings (who was on the field for what)."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"participation_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_participation(seasons=seasons),
        force,
    )


def ftn_charting(
    seasons: list[int] | None = None, force: bool = False
) -> pl.DataFrame:
    """FTN manual charting — play action, motion, blitz, coverage.

    Defaults to FTN_SEASONS rather than HISTORY_SEASONS because 2022 is a real
    floor here and nflreadpy raises rather than returning empty. This is the one
    table in the feature set that genuinely cannot reach back as far as the
    analysis window, which is why it stays a current-year display column and
    never becomes a training feature.
    """
    seasons = _played(seasons or FTN_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"ftn_charting_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_ftn_charting(seasons=seasons),
        force,
    )


def nextgen(
    stat_type: str = "passing", seasons: list[int] | None = None, force: bool = False
) -> pl.DataFrame:
    """NGS tracking aggregates: passing / receiving / rushing."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"nextgen_{stat_type}_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type),
        force,
    )


def pfr_advstats(
    stat_type: str = "rec",
    seasons: list[int] | None = None,
    level: str = "week",
    force: bool = False,
) -> pl.DataFrame:
    """Pro-Football-Reference advanced stats: pass / rush / rec / def."""
    seasons = _played(seasons or HISTORY_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"pfr_{stat_type}_{level}_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_pfr_advstats(
            seasons=seasons, stat_type=stat_type, summary_level=level
        ),
        force,
    )


# --- Context ----------------------------------------------------------------


def schedules(seasons: list[int] | int | None = None, force: bool = False) -> pl.DataFrame:
    """Game schedule + results. Source of bye weeks and strength-of-schedule."""
    seasons = seasons if seasons is not None else SEASON
    return frame(
        f"schedules_{_key(seasons)}",
        "season",
        lambda: nfl.load_schedules(seasons=seasons),
        force,
    )


def rosters(seasons: list[int] | int | None = None, force: bool = False) -> pl.DataFrame:
    """Season rosters — who is actually on which team."""
    seasons = seasons if seasons is not None else SEASON
    return frame(
        f"rosters_{_key(seasons)}",
        "season",
        lambda: nfl.load_rosters(seasons=seasons),
        force,
    )


def depth_charts(
    seasons: list[int] | int | None = None, force: bool = False
) -> pl.DataFrame:
    """Weekly depth charts. Noisy (teams lie), but movement is informative."""
    seasons = seasons if seasons is not None else HISTORY_SEASONS
    return frame(
        f"depth_charts_{_key(seasons)}",
        "weekly",
        lambda: nfl.load_depth_charts(seasons=seasons),
        force,
    )


def players(force: bool = False) -> pl.DataFrame:
    """Every player nflverse knows about, with the full ID set."""
    return frame("players", "static", nfl.load_players, force)


def teams(force: bool = False) -> pl.DataFrame:
    """Team metadata: abbreviations, colors, conference/division."""
    return frame("teams", "static", nfl.load_teams, force)


def draft_picks(force: bool = False) -> pl.DataFrame:
    """NFL draft history — draft capital is a durable prior on opportunity."""
    return frame("draft_picks", "static", nfl.load_draft_picks, force)


def contracts(force: bool = False) -> pl.DataFrame:
    """OverTheCap contract data. Guaranteed money predicts snaps."""
    return frame("contracts", "season", nfl.load_contracts, force)


def combine(force: bool = False) -> pl.DataFrame:
    """Combine testing results."""
    return frame("combine", "static", nfl.load_combine, force)


# --- Fantasy ----------------------------------------------------------------


def ff_rankings(type: str = "draft", force: bool = False) -> pl.DataFrame:
    """FantasyPros expert consensus rankings — and, critically, its dispersion.

    Most consensus products give you the mean and stop. This file ships `sd`,
    `best`, and `worst` alongside `ecr`, which is a free read on where the
    expert market actually disagrees. Consensus with tight `sd` is priced;
    consensus with wide `sd` is an argument still in progress, and that's where
    an independent read is worth something.

    `type` is draft / week / all.
    """
    return frame(f"ff_rankings_{type}", "live", lambda: nfl.load_ff_rankings(type=type), force)


def ff_playerids(force: bool = False) -> pl.DataFrame:
    """The crosswalk: gsis / sleeper / pfr / espn / yahoo / fantasypros IDs.

    Use `ids.crosswalk()` rather than this directly — the raw file has ID columns
    typed as int64, which silently fails to join against Sleeper's string IDs.
    """
    return frame("ff_playerids", "static", nfl.load_ff_playerids, force)


# --- Heavy ------------------------------------------------------------------


def pbp(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """Full play-by-play. Large — expect a few minutes and ~1GB on a cold pull."""
    seasons = _played(seasons or PBP_SEASONS)
    if not seasons:
        return pl.DataFrame()
    return frame(
        f"pbp_{_key(seasons)}", "static", lambda: nfl.load_pbp(seasons=seasons), force
    )
