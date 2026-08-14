"""The preseason guard: a season with no games must return empty, never raise.

This file exists for a bug that killed the entire app on 2026-08-13.
`promotion.weekly_trust(2026)` asked nflverse for a season that had not kicked
off, nflreadpy raised `ValueError: Season must be between 2006 and 2025`, and an
unhandled exception inside a Streamlit tab takes the whole page rather than the
section. Every tab went with it, nine days before the draft.

**Three properties made it invisible, and the tests below are organised around
them.**

*The handling was already written.* `weekly_trust` guards with
`if not rush_raw.height` and `claims.resolve` already treats an empty week table
as pending-not-failed — both sitting one line after the call that raised. So the
end-to-end test at the bottom drives that exact path rather than trusting it.

*It was dormant until data arrived.* `claims.resolve` returns early on an empty
ledger, so nothing reached the raising call until `bootstrap` pulled the first
150 claims. A test that used the real ledger would therefore be **vacuous
whenever the cache happened to be empty**, which is precisely the state the bug
hid behind. The ledger here is synthetic for that reason.

*The guard must not be applied everywhere.* Forward-looking feeds have real rows
for a season that has not started — depth charts especially, which `claims.pull`
depends on. Guarding those would stop the ledger filling and produce no error at
all, so that exemption gets a test of its own.

Every "future" season is computed from `get_current_season()` rather than
hardcoded, so these keep testing the same thing after the season starts instead
of quietly going vacuous in September.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl
import pytest

from src import nflverse as nv
from src.config import SEASON


def _future() -> int:
    """A season whose games certainly have not been played."""
    return nfl.get_current_season() + 1


# --- the filter itself ------------------------------------------------------


def test_played_drops_seasons_that_have_not_happened() -> None:
    current = nfl.get_current_season()

    assert nv._played([current + 1]) == []
    assert nv._played([current, current + 1, current + 5]) == [current]


def test_played_keeps_every_season_that_has_been_played() -> None:
    """The guard must not quietly shorten a legitimate historical request."""
    window = [2020, 2021, 2022, 2023]

    assert nv._played(window) == window


def test_played_uses_the_games_answer_not_the_roster_one() -> None:
    """`get_current_season(roster=True)` flips on March 15 and runs ahead.

    From mid-March to kickoff the two disagree, and **that gap is the entire
    window this guard exists for** — swapping in the roster answer would let the
    raising call straight through for five months of every year and pass every
    other test in this file. Same roster-runs-ahead-of-games disagreement
    `CLAUDE.md` documents for team codes, where the 2026 roster feed alone spells
    Arizona `AZ`.

    Skips rather than asserting vacuously once the season starts and the two
    agree; there is nothing to distinguish then.
    """
    roster_year = nfl.get_current_season(roster=True)
    if roster_year == nfl.get_current_season():
        pytest.skip("season has started — the two answers agree, nothing to tell apart")

    assert nv._played([roster_year]) == []


# --- the wrappers that validate their season range --------------------------


# Season order differs between these signatures — `nextgen` and `pfr_advstats`
# both take `stat_type` first — so they are wrapped rather than called by name.
GUARDED = [
    ("weekly_stats", lambda s: nv.weekly_stats(s)),
    ("season_stats", lambda s: nv.season_stats(s)),
    ("team_stats", lambda s: nv.team_stats(s)),
    ("ff_opportunity", lambda s: nv.ff_opportunity(s, stat_type="pbp_rush")),
    ("snap_counts", lambda s: nv.snap_counts(s)),
    ("injuries", lambda s: nv.injuries(s)),
    ("participation", lambda s: nv.participation(s)),
    ("ftn_charting", lambda s: nv.ftn_charting(s)),
    ("nextgen", lambda s: nv.nextgen(seasons=s)),
    ("pfr_advstats", lambda s: nv.pfr_advstats(seasons=s)),
    ("pbp", lambda s: nv.pbp(s)),
]


@pytest.mark.parametrize("name,call", GUARDED, ids=[n for n, _ in GUARDED])
def test_a_future_season_returns_empty_instead_of_raising(name, call) -> None:
    """The bug, at the wrapper. Offline — the guard short-circuits the loader.

    Eleven wrappers, because the failure is one line of copy-paste away from
    coming back on the twelfth. A new wrapper over a game-data loader belongs in
    this list.
    """
    out = call([_future()])

    assert isinstance(out, pl.DataFrame)
    assert out.height == 0


def test_the_guard_does_not_eat_a_real_request() -> None:
    """The other half: an empty frame must mean "not played yet", not "guarded".

    A guard that returned empty for everything would pass every test above and
    silently empty the entire feature set.
    """
    out = nv.ff_opportunity([2024, 2025], stat_type="pbp_rush")

    assert out.height > 0


# --- the feeds that must stay exempt ----------------------------------------


@pytest.mark.parametrize(
    "name,call",
    [
        ("schedules", lambda s: nv.schedules(s)),
        ("rosters", lambda s: nv.rosters(s)),
        ("depth_charts", lambda s: nv.depth_charts(s)),
    ],
)
def test_forward_looking_feeds_still_read_the_current_season(name, call) -> None:
    """Rosters, schedules and depth charts have real rows before kickoff.

    Applying `_played` to these would look like a tidy generalisation and would
    stop `claims.pull` finding any depth chart — the ledger would simply stop
    filling, with no error anywhere. `CLAUDE.md` says to keep them exempt; this
    is the test that notices if someone doesn't.
    """
    assert call([SEASON]).height > 0


# --- the path that actually broke -------------------------------------------


def test_resolving_claims_before_the_season_starts_returns_pending() -> None:
    """End to end, through the two guards that were already there.

    A synthetic ledger on purpose: the real one is empty on a cold clone, and
    `resolve` returns early when it is — so a test using the real ledger would
    pass for the wrong reason in exactly the state the bug hid behind.

    "Pending" is the documented intent. `claims.resolve`'s own docstring says
    off-season claims stay pending until the games exist, and that a pending
    claim is a state rather than a failure.
    """
    from src import claims as cm

    ledger = pl.DataFrame(
        [
            {
                "claim_type": "role", "claimed_on": "2026-08-01",
                "direction": "growing", "player_name": "Some Player",
                "quote": "getting first-team reps", "source": "beat",
                "source_tier": 1, "specificity": "specific",
                "team": "ATL", "url": "https://example.invalid",
            }
        ]
    )

    out = cm.resolve(ledger=ledger, season=_future())

    assert out.height == 1, "the claim was dropped rather than left pending"
    assert out.get_column("resolved_hit").item() is None


def test_weekly_trust_is_empty_rather_than_fatal_before_kickoff() -> None:
    """The call that raised, directly."""
    from src import promotion as pm

    assert pm.weekly_trust(_future()).height == 0
