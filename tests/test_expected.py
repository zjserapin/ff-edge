"""The expected-points curve, and the properties that make it safe to draft from.

The dangerous failure here is not a crash, it is a plausible-looking board that
encodes noise as an ordering. So most of these tests are about the shape of the
estimate rather than its plumbing: that the monotone fit is actually monotone,
that the raw curve it corrects genuinely is not, and that a tier means what the
docstring claims it means.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import expected as ex


@pytest.fixture(scope="module")
def labeled() -> pl.DataFrame:
    df = ex._labeled()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def curve(labeled: pl.DataFrame) -> pl.DataFrame:
    return ex.adp_curve(labeled=labeled)


# --- the curve --------------------------------------------------------------


def test_missed_seasons_count_as_zero_not_dropped(labeled: pl.DataFrame) -> None:
    """Excluding players who never posted a season would condition the whole
    board on health and inflate every expectation on it."""
    assert (labeled.get_column("fantasy_points") == 0.0).any()
    assert labeled.get_column("fantasy_points").null_count() == 0


def test_monotone_fit_is_monotone(curve: pl.DataFrame) -> None:
    """The property the isotonic step exists to guarantee: a player drafted
    earlier is never assigned a lower expectation than one drafted later."""
    for position in curve.get_column("position").unique().to_list():
        vals = (
            curve.filter(pl.col("position") == position)
            .sort("adp_pos_rank")
            .get_column("exp_points")
            .to_list()
        )
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), position


def test_raw_curve_is_not_monotone(curve: pl.DataFrame) -> None:
    """Documents *why* the isotonic step is there rather than trusting it.

    Left raw, the window mean makes a later-drafted running back look better
    than an earlier one — an inversion inside one standard error, which is
    exactly the kind of noise a draft board must not encode as an ordering.
    """
    rb = curve.filter(pl.col("position") == "RB").sort("adp_pos_rank")
    raw = rb.get_column("raw_points").to_list()
    assert any(b > a + 1e-9 for a, b in zip(raw, raw[1:])), (
        "raw curve is monotone here, so the isotonic step is untested by this data"
    )


def test_inversions_are_inside_the_noise(curve: pl.DataFrame) -> None:
    """The justification for flattening rather than believing the inversions:
    every one of them is smaller than the standard error at that rank."""
    rb = curve.filter(pl.col("position") == "RB").sort("adp_pos_rank")
    raw = rb.get_column("raw_points").to_list()
    se = rb.get_column("se").to_list()
    for i, (a, b) in enumerate(zip(raw, raw[1:])):
        if b > a:
            assert (b - a) < 2 * max(se[i], se[i + 1]), (
                f"rank {i + 1}->{i + 2} inversion of {b - a:.1f} exceeds its noise"
            )


def test_pooled_ranks_report_their_sample_size(curve: pl.DataFrame) -> None:
    """Every estimate ships its denominator, like everything else here."""
    assert {"n", "sd", "se"} <= set(curve.columns)
    assert (curve.get_column("n") >= 5).all()
    assert (curve.get_column("se") > 0).all()


def test_uncertainty_dwarfs_the_gaps(curve: pl.DataFrame) -> None:
    """The empirical case for tiering: the spread around the curve is far
    larger than the step between neighbouring ranks, so a rank-ordered board
    claims precision the data does not contain."""
    rb = curve.filter(pl.col("position") == "RB").sort("adp_pos_rank").head(12)
    steps = np.abs(np.diff(rb.get_column("exp_points").to_numpy()))
    typical_step = float(np.median(steps)) if len(steps) else 0.0
    typical_sd = float(rb.get_column("sd").median())
    assert typical_sd > 5 * typical_step


# --- tiers ------------------------------------------------------------------


def test_tiers_respect_the_gap(curve: pl.DataFrame) -> None:
    """Within a tier, nobody may be more than `gap` below that tier's leader."""
    scored = ex.expected_points(curve=curve)
    if not scored.height:
        pytest.skip("no current-season ADP board")
    gap = 7.0
    tiered = ex.tiers(scored, gap=gap)
    for (position, tier), sub in tiered.group_by(["position", "tier"]):
        vals = sub.get_column("exp_points").to_list()
        assert max(vals) - min(vals) <= gap + 1e-9, (position, tier)


def test_tiers_are_ordered_and_contiguous(curve: pl.DataFrame) -> None:
    """Tier numbers must increase as value falls, with no gaps in numbering."""
    scored = ex.expected_points(curve=curve)
    if not scored.height:
        pytest.skip("no current-season ADP board")
    tiered = ex.tiers(scored)
    for position in tiered.get_column("position").unique().to_list():
        sub = tiered.filter(pl.col("position") == position).sort(
            "exp_points", descending=True
        )
        tiers = sub.get_column("tier").to_list()
        assert tiers == sorted(tiers)
        assert set(tiers) == set(range(1, max(tiers) + 1))


def test_a_flat_curve_becomes_one_tier() -> None:
    """The behaviour the running back board depends on: when the fit cannot
    separate players, they land in a single tier rather than a false ordering."""
    flat = pl.DataFrame(
        {
            "position": ["RB"] * 5,
            "name": list("abcde"),
            "exp_points": [170.0, 170.0, 170.0, 169.5, 169.0],
        }
    )
    assert ex.tiers(flat).get_column("tier").n_unique() == 1


# --- Vegas layer ------------------------------------------------------------


def test_implied_totals_reconstruct_the_game_total() -> None:
    """Both sides of a game must add back to the posted total — the check that
    the spread sign convention is right rather than inverted."""
    env = ex.team_environment([2024])
    if not env.height:
        pytest.skip("cold cache")
    per_game = env.group_by(["season", "week", "game_total"]).agg(
        pl.col("implied_total").sum().alias("both_sides"),
        pl.len().alias("teams"),
    ).filter(pl.col("teams") == 2)
    assert per_game.height > 100
    diff = (
        per_game.get_column("both_sides") - per_game.get_column("game_total")
    ).abs()
    assert diff.max() < 1e-6


def test_favourite_is_priced_above_the_underdog() -> None:
    """A negative spread means favoured, and a favourite must carry the higher
    implied total. Catches a sign flip that would invert every ranking built
    on top of this."""
    env = ex.team_environment([2024])
    if not env.height:
        pytest.skip("cold cache")
    favourites = env.filter(pl.col("spread") < 0)
    dogs = env.filter(pl.col("spread") > 0)
    assert favourites.get_column("implied_total").mean() > (
        dogs.get_column("implied_total").mean()
    )


def test_line_coverage_is_visible(labeled: pl.DataFrame) -> None:
    """Preseason the sportsbooks have priced only the first few weeks, and the
    tool has to say so rather than quietly ranking on a half-priced slate."""
    cov = ex.line_coverage()
    if not cov.height:
        pytest.skip("cold cache")
    assert {"week", "games", "lined", "pct_lined"} <= set(cov.columns)
    assert cov.get_column("lined").sum() < cov.get_column("games").sum()
