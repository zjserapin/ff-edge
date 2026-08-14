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
#   2. A label season needs FFC ADP for that year.
#
# 2018-2025 features therefore yield 2019-2025 labels — seven seasons. The
# window started at 2020 to get the pipeline running, and widening it has been
# the highest-leverage change available each time: season-forward validation
# spends the first two label years on the initial training window, so four label
# seasons bought two test folds and 284 out-of-sample rows. At that size no
# result is distinguishable from any other, which is why the first backtest
# could only report "roughly chance" and stop. Six gave four folds and 540 rows,
# which roughly halved every interval in `breakout.py`. Seven gives five folds
# and 666 rows — a 23% larger test set that tightened the continuous-target
# interval by about a sixth and moved no conclusion at all.
#
# **2025 was added on 2026-08-09.** It was excluded because FFC genuinely
# published nothing for it at any format — see `ADP_MISSING_YEARS` — and that
# stopped being true: 156 rows at half-PPR/10, 144 of them matching a skill
# player, which sits between 2022's 115 and 2024's 157 rather than being an
# outlier. Production for the year is complete. Nothing about the exclusion was
# wrong when it was written; the upstream data changed, so re-check it rather
# than trusting this comment.
#
# 2018 is the floor, and ADP sets it: FFC publishes nothing before it at
# half-PPR/10-team. Every other source in the feature set reaches back further.
FEATURE_SEASONS: list[int] = list(range(2018, 2026))
LABEL_SEASONS: list[int] = list(range(2019, 2026))

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

# The 2026 roster. The league changed again for 2026: one of the two FLEX slots
# became a SUPER_FLEX, which is the single largest rules change in its history
# for valuation purposes. A superflex slot roughly doubles league-wide QB
# demand — replacement quarterback moves from about QB11 to about QB21 — and
# every PAR number in the project moves with it. See `scoring.starter_demand`.
DEFAULT_ROSTER_POSITIONS: list[str] = [
    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF",
] + ["BN"] * 6

# Weeks 1-14 decide the league; 15-17 are a 6-team playoff with byes for the top
# two seeds. Positional finish is measured over the regular season because those
# are the weeks a drafted roster is actually competing in.
# The roster the draft simulator compares strategies under, deliberately NOT
# DEFAULT_ROSTER_POSITIONS. A strategy comparison is only meaningful when the
# draft market and the positional templates match the format being simulated,
# and neither does for superflex yet: the board is priced from 1QB ADP, and
# every template in `simulate.STRATEGIES` was written for a two-FLEX league
# (`elite_qb` drafts a single quarterback, which cannot fill a superflex slot).
#
# Run those templates against a 1QB market under a superflex roster and the
# sim reports that quarterbacks are nearly free *and* start twice — an artifact
# of the mismatched market, not a finding. So the strategy sim stays pinned to
# the 2024-2025 format it can actually model, and says so in the app.
#
# The unlock is real and cheap when it is worth doing: FFC publishes 2QB ADP
# for past seasons too (verified back to 2020), so a superflex strategy study
# needs that board plus QB-heavier templates, not new machinery. The lineup
# optimizer already handles superflex and is brute-force tested against it.
SIM_ROSTER_POSITIONS: list[str] = [
    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF",
] + ["BN"] * 6

REGULAR_SEASON_WEEKS = 14
PLAYOFF_WEEKS: list[int] = [15, 16, 17]
PLAYOFF_TEAMS = 6

FANTASY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Every multi-position slot Sleeper can put on a roster, and what each accepts.
# Sleeper's own slot names, so a roster pulled live from the API maps onto this
# without translation.
#
# The *size* of each eligibility set is load-bearing, not decorative: both the
# replacement math and the simulator's lineup optimizer fill the most
# restrictive slot first, which is what makes a greedy allocation exactly
# optimal when the sets nest — and dedicated ⊂ FLEX ⊂ SUPER_FLEX is precisely
# this league's shape. REC_FLEX and WRRB_FLEX do not nest with each other, so a
# roster carrying both would make the allocation a heuristic; that combination
# does not exist here and both functions say so out loud.
FLEX_SLOTS: dict[str, tuple[str, ...]] = {
    "REC_FLEX": ("WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
    "FLEX": ("RB", "WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
}

# Slots that start nobody, so they demand nobody.
NON_STARTING_SLOTS: tuple[str, ...] = ("BN", "IR", "TAXI")

FLEX_ELIGIBLE: tuple[str, ...] = FLEX_SLOTS["FLEX"]

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

# --- Team environment weight ------------------------------------------------

# How much of `env_swing` enters the board's ordering. **This number is asserted,
# not measured, and it is the only number in this file of which that is true.**
#
# The case for it being above zero: `env_swing` is large. It spans -31.5 to +47.0
# on the 2026 board against a PAR range of -65 to +72.6, so more than half the
# board's total spread comes from a column that carried no weight at all until
# 2026-08-14. De'Von Achane leads Puka Nacua by 12.3 PAR and trails him by 67.5
# points of offensive environment. Leaving it at zero is as much a choice as any
# other value, and it was never a deliberate one.
#
# The case for it being below one: `expected.py` maps *positional ADP rank* to
# points, so the curve is blind to team — but ADP itself is not. A good player on
# a bad offence is drafted later, so part of the environment discount is already
# in the price and `env_swing` cannot tell how much. Adding it whole double
# counts. `board.attach_environment` says this out loud and calls the number "the
# size of the argument" rather than a correction to subtract.
#
# So the honest reading is: the right weight is strictly between 0 and 1 and
# nobody here has measured where. 0.35 leans toward the conservative half of that
# interval on the reasoning that ADP is efficient and has probably priced most of
# what is knowable about an offence.
#
# **To measure it properly**, run env_swing as a feature against next-season
# points controlling for ADP, season-forward, and report the interval — the M1-M4
# treatment in RESEARCH_SPEC.md §5.3. It may well come back null like the last
# two, in which case this returns to 0.0 and that is a result.
ENV_WEIGHT = 0.35

# Seasons FFC publishes no board for at any scoring x team-count combination,
# and which therefore cannot be label seasons no matter how much production data
# exists for them.
#
# Empty as of 2026-08-09. This held `[2025]` and that was correct when written —
# FFC returned zero rows for 2025 everywhere. They have since backfilled it
# (half-PPR/10: 156 rows; 2QB: 215; PPR: 249), so the exclusion was costing a
# label season for a reason that had expired. The list stays because the failure
# it guards against is real and recurring: FFC suppresses a format until it has
# collected enough drafts, so a *current* season reads as missing for months
# before it fills in. Check rather than assume when a season looks empty.
ADP_MISSING_YEARS: list[int] = []

# --- Sportsbook props (FanDuel) ---------------------------------------------

# FanDuel's sportsbook API is genuinely public — no account, no cookie. The `_ak`
# key ships in their own public JS bundle and is a client identifier, not a
# credential, which is why it can live in this file when a league id cannot.
#
# DraftKings is deliberately absent. Every DK API path 403s behind Akamai bot
# management regardless of headers (verified 2026-08-10 with full browser
# headers, sec-ch-ua and fetch-metadata included); the block keys on TLS
# fingerprint, not on anything a request can set. Adding DK means a headless
# browser or a TLS-impersonating client, and neither survives as a daily job.
FANDUEL_BASE = "https://sbapi.{region}.sportsbook.fanduel.com/api"
FANDUEL_KEY = "FhMFpcPWXMeyZxOx"

# Any state where FanDuel operates serves the same board. Checked nj/pa/mi/az/co
# /tn — all 145 season-long markets; `va` lagged at 142, so it is not the default.
FANDUEL_REGION = os.environ.get("FF_EDGE_FANDUEL_REGION", "nj")

# FanDuel's own id for the NFL, carried as `eventTypeId` throughout their payloads.
FANDUEL_NFL_EVENT_TYPE = 6423

# --- Claims ledger / LLM -----------------------------------------------------

# Provider swap happens here and in src/llm.py only — logic files never touch a
# model client directly. "anthropic" for personal/PoC, "bedrock" for the Ally
# pattern; same interface either way.
LLM_PROVIDER = os.environ.get("FF_EDGE_LLM_PROVIDER", "anthropic")

# Haiku-class on purpose (CLAIMS_SPEC.md): extraction is high-volume structured
# output with a verbatim-quote contract, not deep reasoning. One env var to
# change if that stops being true.
LLM_MODEL = os.environ.get("FF_EDGE_LLM_MODEL", "claude-haiku-4-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-haiku-4-5")

# The ledger is append-only and lives in the gitignored data/ directory. Claims
# contain no league secrets, but data/ stays out of the public repo regardless.
CLAIMS_PATH = DATA_DIR / "claims.parquet"

# Recency half-life for claim scores, in days. August camp news goes stale fast;
# a two-week half-life means a report from a month ago carries ~25% weight.
CLAIM_HALF_LIFE_DAYS = 14.0

# A claim is "novel" if no claim with the same (player, type, direction) exists
# within this many days — 40 aggregators quoting one beat report count once.
NOVELTY_WINDOW_DAYS = 7

# A resolvable claim is checked against weekly usage this many weeks after it
# lands; before that it is simply pending.
RESOLUTION_WEEKS = 3

# --- Simulation -------------------------------------------------------------

# weekly_stats has no team-defense rows at all, so DST cannot be scored from
# this data layer. The sim draws it from a fixed distribution instead, identical
# for all ten teams — it washes out in head-to-head expectation while still
# injecting the variance that dilutes strategy edges, which is realistic.
DST_WEEKLY_MEAN = 7.0
DST_WEEKLY_SD = 6.0
