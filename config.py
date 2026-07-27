"""Single source of truth for seasons, paths, and cache freshness.

Everything else in ff-edge imports from here rather than hardcoding a year or a
directory. That matters most in August: rolling the project to a new season
should be a one-line edit, not a grep.
"""

from __future__ import annotations

from pathlib import Path

# --- Season -----------------------------------------------------------------

SEASON = 2026

# --- Sleeper ----------------------------------------------------------------

# Your Sleeper *display name* (what leaguemates see), not the email you log in
# with. Sleeper's public API has no auth and resolves users by username only.
SLEEPER_USERNAME = "CHANGE_ME"

# Populate to skip league discovery entirely (useful if you're in leagues under
# a different account, or want to pin the pipeline to one league).
LEAGUE_IDS: list[str] = []

# --- Paths ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
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

# ff_opportunity coverage starts in 2022, so the history window is bounded by
# the weakest table rather than by how far nflverse goes back.
HISTORY_SEASONS = [2022, 2023, 2024, 2025]

# Play-by-play is ~50MB/season parsed; three years is plenty for route/target
# context without making bootstrap painful.
PBP_SEASONS = [2023, 2024, 2025]

# --- ADP defaults -----------------------------------------------------------

ADP_SCORING = "ppr"
ADP_TEAMS = 12
ADP_YEAR = SEASON
