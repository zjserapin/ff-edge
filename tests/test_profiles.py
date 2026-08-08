"""A profile has to keep a format and its market together, or it is just a dict.

The bug these tests exist for is the one that already happened once: a roster
format changed, the ADP market did not, and every quarterback on the board was
mispriced by a round while nothing raised. So the properties worth asserting are
about the *pairing* — that selecting a format selects its market, that a typo
cannot quietly fall back to a different league's settings, and that a profile
with no keepers turns the entire keeper layer into an exact no-op rather than a
branch that happens to be skipped.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import board as bd
from src import profiles as pf
from src.config import FLEX_SLOTS, NON_STARTING_SLOTS


def test_every_profile_pairs_a_format_with_a_market() -> None:
    """No profile may declare a roster without declaring what prices it."""
    for name, profile in pf.PROFILES.items():
        assert profile.name == name, "key and name must agree or resolve() lies"
        assert profile.adp_scoring, f"{name} has no ADP market"
        assert profile.teams > 0
        assert profile.roster_positions, f"{name} has no roster"
        assert profile.scoring.get("rec") is not None


def test_superflex_profiles_use_a_superflex_market() -> None:
    """The 2026 bug, as an assertion.

    A roster carrying SUPER_FLEX priced off a 1QB board reports that
    quarterbacks are nearly free *and* start twice. Any profile with a superflex
    slot must be paired with a market that drafts two quarterbacks per team.
    """
    for profile in pf.PROFILES.values():
        has_superflex = "SUPER_FLEX" in profile.roster_positions
        if has_superflex:
            assert profile.adp_scoring == "2qb", (
                f"{profile.name} starts a superflex but prices off "
                f"{profile.adp_scoring!r}"
            )


def test_unknown_profile_raises_rather_than_falling_back() -> None:
    """A typo must not silently price one league as another."""
    with pytest.raises(KeyError, match="unknown profile"):
        pf.resolve("stanadrd_12")


def test_resolve_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("FF_EDGE_PROFILE", "standard_12")
    assert pf.resolve().name == "standard_12"
    # An explicit argument still wins over the environment.
    assert pf.resolve("shiva_bowl").name == "shiva_bowl"


def test_customize_does_not_mutate_the_builtin() -> None:
    base = pf.resolve("standard_12")
    two_flex = pf.customize("standard_12", roster_positions=base.roster_positions + ["FLEX"])
    assert two_flex.roster_positions.count("FLEX") == 2
    assert pf.resolve("standard_12").roster_positions.count("FLEX") == 1


def test_as_settings_matches_the_league_settings_shape() -> None:
    """Synthetic profiles must flow through code written for Sleeper's shape."""
    settings = pf.as_settings(pf.resolve("standard_12"))
    for key in ("season", "teams", "scoring", "roster_positions",
                "playoff_week_start", "playoff_teams", "source"):
        assert key in settings
    assert settings["source"].startswith("profile:")


def test_standard_scoring_differs_from_the_league_only_in_receptions() -> None:
    """A standard profile is half-PPR minus the reception point, nothing else.

    Written as a delta on purpose — a second hand-typed scoring dict would drift
    the moment a non-reception rule changed.
    """
    shiva = pf.resolve("shiva_bowl").scoring
    standard = pf.resolve("standard_12").scoring
    differing = {k for k in shiva if shiva[k] != standard.get(k)}
    assert differing == {"rec"}
    assert standard["rec"] == 0.0


def test_every_profile_is_redraft() -> None:
    """Scope guard, not a style check.

    This project prices a player against a market for one season. Dynasty asks
    which young players are worth building on, which turns on age curves rather
    than on this season's price, and nothing here estimates one. A dynasty
    market would therefore answer the redraft question under a dynasty label.
    """
    for profile in pf.PROFILES.values():
        assert profile.adp_scoring != "dynasty", (
            f"{profile.name} prices off a dynasty market, but the board has no "
            "age curve to make that mean anything"
        )


def test_a_keeperless_profile_makes_the_keeper_layer_a_no_op() -> None:
    """No keepers means adjusted ADP is ADP exactly, not approximately."""
    board = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
            "adp": [1.5, 20.0, 44.2],
            "kept": [False, False, False],
        }
    )
    out = bd.keeper_adjusted_adp(board, slots=[], teams=10)
    assert out.get_column("adj_adp").to_list() == [1.5, 20.0, 44.2]
    assert out.get_column("exp_pick").to_list() == [1.5, 20.0, 44.2]
    assert out.get_column("adp_shift").to_list() == [0.0, 0.0, 0.0]


def test_keeper_shift_is_the_count_of_keepers_priced_ahead() -> None:
    """The removal effect, stated as arithmetic rather than a magic constant."""
    board = pl.DataFrame(
        {
            "name": ["kept1", "kept2", "target", "deep"],
            "adp": [5.0, 10.0, 26.0, 90.0],
            "kept": [True, True, False, False],
        }
    )
    out = bd.keeper_adjusted_adp(board, slots=[], teams=10)
    shifts = dict(zip(out.get_column("name"), out.get_column("adp_shift")))
    # Two keepers priced ahead of both, so both move up exactly two.
    assert shifts["target"] == -2.0
    assert shifts["deep"] == -2.0
    # A keeper priced ahead of the other keeper moves up one; the first, none.
    assert shifts["kept1"] == 0.0
    assert shifts["kept2"] == -1.0


def test_consumed_picks_push_the_expected_pick_later_than_the_selection_index() -> None:
    """The two numbering systems, and why the board carries both.

    A keeper does not only leave the pool, he also spends a pick. So the player
    who is the 4th *selection* does not go at pick 4 if keepers sit on picks 1
    and 2 — he goes at pick 6. Comparing a selection index against a pick number
    from `picks()` would count the keeper adjustment twice.
    """
    board = pl.DataFrame(
        {
            "name": ["k1", "k2", "x"],
            "adp": [1.0, 2.0, 6.0],
            "kept": [True, True, False],
        }
    )
    out = bd.keeper_adjusted_adp(board, slots=[1, 2], teams=10)
    row = out.filter(pl.col("name") == "x").row(0, named=True)
    # Two keepers ahead: 6th in the market becomes the 4th selection made.
    assert row["adj_adp"] == 4.0
    # Picks 1 and 2 are spent, so the 4th selection happens at pick 6.
    assert row["exp_pick"] == 6.0
    assert row["exp_pick"] > row["adj_adp"]


def test_adjusted_adp_never_reorders_the_board() -> None:
    """Removal shifts everyone up; it must not overtake anyone.

    A player cannot pass another by having keepers removed from in front of both
    of them, so the adjusted ordering has to match the original one.
    """
    board = pl.DataFrame(
        {
            "name": [f"p{i}" for i in range(12)],
            "adp": [1.0, 3.0, 7.0, 8.5, 12.0, 19.0, 24.0, 30.0, 41.0, 55.0, 78.0, 99.0],
            "kept": [i % 3 == 0 for i in range(12)],
        }
    )
    out = bd.keeper_adjusted_adp(board, slots=[2, 5, 9], teams=10).sort("adp")
    adj = out.get_column("adj_adp").to_list()
    assert adj == sorted(adj), "adjustment must preserve draft order"
    assert all(a >= 1.0 for a in adj), "nobody is drafted before the first pick"


def test_demand_scales_with_team_count() -> None:
    """A 12-team league starts more players than a 10-team one at every position.

    The one thing team count genuinely changes. FFC pools its boards across team
    counts for most formats, so price does *not* move with it — demand does, and
    that is where the profile's `teams` has to reach.
    """
    from src import scoring as sc

    roster = pf.resolve("standard_12").roster_positions
    ten = sc.starter_demand(roster, teams=10)
    twelve = sc.starter_demand(roster, teams=12)
    for position in ("QB", "RB", "WR", "TE"):
        assert twelve[position] > ten[position], position


def test_depth_slots_add_no_demand() -> None:
    """Bench, IR and taxi are cosmetic for valuation.

    Worth asserting because it is what lets a profile describe a real roster —
    including a 20-deep bench — without any of that depth leaking into
    replacement level.
    """
    from src import scoring as sc

    starters = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    padded = starters + ["BN"] * 10 + ["TAXI"] * 4 + ["IR"] * 2
    assert set(NON_STARTING_SLOTS) >= {"BN", "TAXI", "IR"}
    assert sc.starter_demand(padded, teams=10) == sc.starter_demand(starters, teams=10)
