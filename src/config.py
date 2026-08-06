"""Single source of truth for seasons, paths, and cache freshness.

Everything else in ff-edge imports from here rather than hardcoding a year or a
directory. That matters most in August: rolling the project to a new season
should be a one-line edit, not a grep.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Season -----------------------------------------------------------------

SEASON = 2026

# --- Sleeper ----------------------------------------------------------------

# Your Sleeper *display name* (what leaguemates see), not the email you log in
# with. Sleeper's public API has no auth and resolves users by username only.
#
# Read from the shell first so a personal handle never lands in a public repo:
#   export FF_EDGE_SLEEPER_USER=yourname
SLEEPER_USERNAME = os.environ.get("FF_EDGE_SLEEPER_USER", "CHANGE_ME")

# Populate to skip league discovery entirely (useful if you're in leagues under
# a different account, or want to pin the pipeline to one league).
LEAGUE_IDS: list[str] = []

# --- Paths ------------------------------------------------------------------

# parents[1], not parent: this file lives in src/, and data/ + output/ sit beside
# src/ at the repo root. Using .parent would silently repoint DATA_DIR at
# src/data, create it, and orphan the entire cache without raising anything.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Cache TTLs (hours) -----------------------------------------------------

# Keyed by how fast the underlying thing actually changes. Player bios don't
# move; ADP moves every day of August; live scoring moves every Sunday.
TTL = {
    "static": 720,  # 30d — players, teams, combine, historical seasons
    "season": 168,  # 7d  — rosters, schedules, contracts
    "weekly": 12,  # in-season tables that refresh after games
    "live": 1,  # matchups, transactions, nfl_state
}

# --- Season ranges ----------------------------------------------------------

# The analysis window. Two facts set it, and neither is the one this file used
# to claim (that ff_opportunity starts in 2022 — it does not; its floor is 2006
# and every season back to then downloads fine):
#
#   1. Features lag labels by a season. A player's 2023 usage predicts his 2024
#      finish, so N label seasons need N+1 seasons of features.
#   2. A label season needs FFC ADP for that year, and FFC has none for 2025 at
#      any format. 2018-2024 is what exists.
#
# 2018-2025 features therefore yield 2019-2024 labels — six seasons. The window
# started at 2020 to get the pipeline running, and widening was the single
# highest-leverage change available: season-forward validation spends the first
# two label years on the initial training window, so four label seasons bought
# exactly two test folds and 284 out-of-sample rows. At that size no result is
# distinguishable from any other, which is why the first backtest could only
# report "roughly chance" and stop. Six label seasons give four folds and 540
# out-of-sample rows, which roughly halved every interval in `breakout.py`.
#
# 2018 is the floor, and ADP sets it: FFC publishes nothing before it at
# half-PPR/10-team. Every other source in the feature set reaches back further.
FEATURE_SEASONS: list[int] = list(range(2018, 2026))
LABEL_SEASONS: list[int] = list(range(2019, 2025))

# Newest season with complete production. Current-state features and clustering
# read from here; it is deliberately not SEASON, which is the season being
# drafted for and has no games played yet.
CURRENT_SEASON = 2025

HISTORY_SEASONS = FEATURE_SEASONS

# FTN charting genuinely starts in 2022 — the one table in the feature set with
# a floor above the window. nflreadpy raises rather than returning empty, so it
# needs its own constant instead of inheriting HISTORY_SEASONS.
FTN_SEASONS: list[int] = [s for s in FEATURE_SEASONS if s >= 2022]

# Play-by-play is ~50MB/season parsed; three years is plenty for route/target
# context without making bootstrap painful.
PBP_SEASONS = [2023, 2024, 2025]

# --- League (Shiva Bowl) ----------------------------------------------------

# Read from the shell, like SLEEPER_USERNAME, and for the same reason: a league
# id is not a credential, but Sleeper's API is public and unauthenticated, so
# anyone holding it can read the league name, every manager's display name, and
# the full draft and transaction history. That is nine other people's data in a
# public repo, so it stays out of the file.
#
#   export FF_EDGE_LEAGUE_ID=1234567890123456789
#
# Left unset, sleeper.my_leagues() discovers it from FF_EDGE_SLEEPER_USER.
LEAGUE_ID = os.environ.get("FF_EDGE_LEAGUE_ID", "")

# Defaults, not truth. scoring.league_settings() pulls these live from Sleeper;
# these exist so the app starts with no network and so the analysis modules have
# something to be called with directly.
DEFAULT_TEAMS = 10
DEFAULT_ROSTER_POSITIONS: list[str] = [
    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF",
] + ["BN"] * 6

# Weeks 1-14 decide the league; 15-17 are a 6-team playoff with byes for the top
# two seeds. Positional finish is measured over the regular season because those
# are the weeks a drafted roster is actually competing in.
REGULAR_SEASON_WEEKS = 14
PLAYOFF_WEEKS: list[int] = [15, 16, 17]
PLAYOFF_TEAMS = 6

FANTASY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")
FLEX_ELIGIBLE: tuple[str, ...] = ("RB", "WR", "TE")

# Verbatim from the 2026 league, used when Sleeper is unreachable. Note the
# league changed in 2024: 2023 was full PPR with one FLEX and pass_int -1.
DEFAULT_SCORING: dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "pass_2pt": 2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0, "fum": 0.0, "fum_rec": 2.0, "fum_rec_td": 6.0,
    "st_td": 6.0, "st_ff": 1.0, "st_fum_rec": 1.0,
    "xpm": 1.0, "xpmiss": -1.0, "fgmiss": -1.0,
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0,
    "fgm_40_49": 4.0, "fgm_50p": 5.0,
    "def_td": 6.0, "def_st_td": 6.0, "def_st_ff": 1.0, "def_st_fum_rec": 1.0,
    "int": 2.0, "sack": 1.0, "ff": 1.0, "safe": 2.0, "blk_kick": 2.0,
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0,
    "pts_allow_14_20": 1.0, "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0,
    "pts_allow_35p": -4.0,
}

# --- ADP defaults -----------------------------------------------------------

# The rolling daily snapshot. Do NOT repoint these — adp_history_ppr_12_2026
# has been accumulating under this key and ADP history cannot be backfilled.
ADP_SCORING = "ppr"
ADP_TEAMS = 12
ADP_YEAR = SEASON

# What the analysis layer uses: the league's actual format. Snapshotted
# separately so both histories accumulate.
LEAGUE_ADP_SCORING = "half-ppr"
LEAGUE_ADP_TEAMS = 10

# The league added a superflex slot for 2026, and a 1QB ADP misprices
# quarterbacks in that format by a round or more. FFC publishes 2QB ADP as its
# own format string; snapshotted like the others so the history accumulates.
# It is a *reference* board for QB decisions, not the analysis default —
# half-PPR/10 stays the modelling market because the backtest window is priced
# in it.
SUPERFLEX_ADP_SCORING = "2qb"

# FFC returns zero rows for 2025 at every scoring x team-count combination.
# Not a bug in the fetch — the data does not exist, so 2025 cannot be a label
# season no matter how much production data we have for it.
ADP_MISSING_YEARS: list[int] = [2025]

# --- Simulation -------------------------------------------------------------

# weekly_stats has no team-defense rows at all, so DST cannot be scored from
# this data layer. The sim draws it from a fixed distribution instead, identical
# for all ten teams — it washes out in head-to-head expectation while still
# injecting the variance that dilutes strategy edges, which is realistic.
DST_WEEKLY_MEAN = 7.0
DST_WEEKLY_SD = 6.0
