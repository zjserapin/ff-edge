"""The four screens, and the market they are read in.

`peek` and `adp` were finished, tested, and imported by nothing — `app.py` pulled
in sixteen modules and neither was among them. Surfacing them is display work,
so most of what can go wrong here is not in the rendering:

**The season was a literal.** All four `peek` entry points defaulted to
`season=2025` while `config.SEASON` said 2026. Correct today, silently wrong the
first August nobody re-reads them, and invisible either way because a screen
against the wrong year returns a full, plausible table.

**The market is not the default.** `adp.movement` defaults to
`config.ADP_SCORING`/`ADP_TEAMS` — ppr/12 — which prices the 828 league, not the
Shiva Bowl's 2qb/10. Reading movement in the wrong market renders a well-formed
table of somebody else's drift. This is the `profiles.py` lesson: a roster format
and the market that prices it travel together or they disagree in silence.
"""

from __future__ import annotations

import inspect

import pytest

from src import adp as adp_mod
from src import config
from src import peek
from src import profiles as pf


# --- the season is derived, not typed ---------------------------------------


def test_peek_reads_the_season_from_config() -> None:
    """`LAST_PLAYED` trails `config.SEASON`, so rolling the year moves it."""
    assert peek.LAST_PLAYED == config.SEASON - 1


@pytest.mark.parametrize(
    "fn", [peek.regression_candidates, peek.snap_trend, peek.usage_leaders]
)
def test_no_screen_defaults_to_a_hardcoded_season(fn) -> None:
    """The defect this file was written for, pinned per function.

    Asserted against `LAST_PLAYED` rather than a number, so the test cannot rot
    into the same literal it exists to forbid.
    """
    default = inspect.signature(fn).parameters["season"].default
    assert default == peek.LAST_PLAYED
    assert default == config.SEASON - 1


# --- the market -------------------------------------------------------------


def test_the_profiles_market_is_not_the_module_default() -> None:
    """If these ever coincide, the test below stops proving anything.

    Not a preference — it is the precondition that makes passing the market
    explicitly a load-bearing act rather than a no-op.
    """
    profile = pf.resolve("shiva_bowl")
    assert (profile.adp_scoring, profile.adp_teams) != (
        config.ADP_SCORING,
        config.ADP_TEAMS,
    ), "shiva_bowl must differ from the module default for this to be a real test"


def test_movement_reads_a_different_market_when_given_one() -> None:
    """Behavioural rather than mocked: the two markets disagree on real data.

    Asserting on the frames instead of on a patched path means this still fails
    if `movement` ever ignores its arguments *and* keeps the same file-naming
    scheme — which a mock of the path would not catch.
    """
    profile = pf.resolve("shiva_bowl")
    league = adp_mod.movement(scoring=profile.adp_scoring, teams=profile.adp_teams)
    default = adp_mod.movement()

    if not league.height or not default.height:
        pytest.skip("need two snapshots in both markets")

    assert not league.equals(default), (
        "the profile's market and the module default returned identical frames; "
        "movement is ignoring the market it was handed"
    )


def test_movement_is_empty_rather_than_raising_without_history() -> None:
    """One snapshot is the normal state early, and it must not look like a crash."""
    out = adp_mod.movement(scoring="nonexistent", teams=99)
    assert out.is_empty()


# --- the screens return something shaped like a screen ----------------------


@pytest.mark.parametrize(
    "name, columns",
    [
        ("regression_candidates", {"pts_over_exp", "exp_pts", "act_pts"}),
        ("market_disagreement", {"ecr", "sd"}),
    ],
)
def test_screen_columns_are_stable(name, columns) -> None:
    """The app renders these by name, so a rename upstream is a blank table."""
    out = getattr(peek, name)()
    if not out.height:
        pytest.skip(f"{name}: nothing cached")
    assert columns <= set(out.columns)


def test_negative_points_over_expected_is_the_buy_low_end() -> None:
    """Direction check. The screen is worthless if the sign is read backwards.

    `pts_over_exp` is actual minus expected, so the buy-low candidates sort to
    the *front* ascending. The app's caption depends on that being true.
    """
    out = peek.regression_candidates()
    if out.height < 2:
        pytest.skip("nothing cached")
    ordered = out.sort("pts_over_exp", nulls_last=True).get_column("pts_over_exp")
    assert ordered[0] <= ordered[-1]
    assert ordered.min() < 0, "a screen with no negative side has nothing to buy"


def test_movement_direction_matches_its_sign() -> None:
    """Negative `adp_change` means he is going EARLIER.

    The column is signed the opposite way to how it reads in English, which is
    exactly the kind of thing the UI gets backwards — so the label that ships
    beside it is pinned here rather than trusted.
    """
    profile = pf.resolve("shiva_bowl")
    mv = adp_mod.movement(scoring=profile.adp_scoring, teams=profile.adp_teams)
    if not mv.height or "direction" not in mv.columns:
        pytest.skip("no movement history yet")

    risers = mv.filter(adp_mod.pl.col("direction") == "rising")
    if risers.height:
        assert risers.get_column("adp_change").max() <= 0, (
            "a 'rising' player whose ADP number went up means the label and the "
            "sign disagree"
        )
