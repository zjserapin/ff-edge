"""Landscape metrics — shape guarantees, not value assertions.

The numbers here change every season, so asserting them would make the suite a
liability. What must hold is structural: curves that claim to be curves are
monotone, shares are shares, and the tidy frames have the columns the chart
specs encode against.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import landscape as ls
from src import scoring as sc


@pytest.fixture(scope="module")
def season_points() -> pl.DataFrame:
    df = sc.score_season()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.mark.parametrize("basis", ["ppg", "total"])
def test_scarcity_curve_is_monotone(season_points: pl.DataFrame, basis: str) -> None:
    """Rank and value must agree, or the "curve" is a sawtooth.

    This is the regression guard for the bug where ranking on season total while
    plotting per-game let an injured player at rank 27 outscore rank 24, making
    every position's dropoff look jagged and unreadable.
    """
    curve = ls.scarcity_curve(basis=basis, season_points=season_points, max_rank=36)
    col = "ppg" if basis == "ppg" else "fantasy_points"

    for (season, position), grp in curve.group_by(["season", "position"]):
        vals = grp.sort("pos_rank").get_column(col).to_list()
        assert vals == sorted(vals, reverse=True), f"{position} {season} not monotone"


def test_scarcity_curve_rejects_unknown_basis(season_points: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="basis must be"):
        ls.scarcity_curve(basis="mean", season_points=season_points)


def test_scarcity_ranks_restart_at_one(season_points: pl.DataFrame) -> None:
    """Re-ranking on ppg must produce a dense rank, not a filtered sparse one."""
    curve = ls.scarcity_curve(basis="ppg", season_points=season_points, max_rank=36)
    firsts = curve.group_by(["season", "position"]).agg(pl.col("pos_rank").min())
    assert (firsts.get_column("pos_rank") == 1).all()


def test_concentration_shares_are_bounded_and_ordered(season_points: pl.DataFrame) -> None:
    """Shares sit in (0, 1], and a bigger top-N never holds a smaller share."""
    conc = ls.concentration(season_points=season_points, shares=(5, 15, 30))
    assert conc.height > 0
    shares = conc.get_column("share")
    assert shares.min() > 0 and shares.max() <= 1.0

    for (season, position), grp in conc.group_by(["season", "position"]):
        ordered = grp.sort("top_n").get_column("share").to_list()
        assert ordered == sorted(ordered), f"{position} {season} shares not increasing"


def test_concentration_pool_is_capped(season_points: pl.DataFrame) -> None:
    """The denominator must be the capped pool, not every player who touched a ball.

    Without the cap this metric measures roster churn rather than star
    concentration.
    """
    conc = ls.concentration(season_points=season_points)
    for row in conc.iter_rows(named=True):
        assert row["pool_size"] <= ls.DEFAULT_POOL_SIZE[row["position"]]


def test_positional_mix_counts_sum_to_the_cutoff(season_points: pl.DataFrame) -> None:
    """Every one of the top N slots is attributed to exactly one position."""
    cross = ls.cross_positional_value(season_points=season_points)
    mix = ls.positional_mix(cross, cutoffs=(12, 24, 48))

    totals = mix.group_by(["season", "cutoff"]).agg(pl.col("n").sum().alias("total"))
    for row in totals.iter_rows(named=True):
        assert row["total"] == row["cutoff"]


def test_cross_positional_rank_is_unique_per_season(season_points: pl.DataFrame) -> None:
    """One player per overall rank — an ordinal rank, not a tie-sharing one."""
    cross = ls.cross_positional_value(season_points=season_points)
    per_season = cross.group_by("season").agg(
        pl.col("overall_par_rank").n_unique().alias("uniq"), pl.len().alias("n")
    )
    assert (per_season.get_column("uniq") == per_season.get_column("n")).all()


def test_par_by_position_has_the_columns_the_charts_encode(
    season_points: pl.DataFrame,
) -> None:
    par = ls.par_by_position(season_points=season_points)
    assert {
        "season",
        "position",
        "demand",
        "replacement_rank",
        "replacement_ppg",
        "par_total",
        "par_mean_starter",
    } <= set(par.columns)
    # PAR summed over starters is value the position adds; it cannot be negative.
    assert par.get_column("par_total").min() >= 0


def test_crossover_table_is_symmetric_in_coverage(season_points: pl.DataFrame) -> None:
    """Each anchor reports an equivalent rank at every other position."""
    cross = ls.cross_positional_value(season_points=season_points)
    table = ls.crossover_table(cross, anchors={"RB": 24, "WR": 36})
    assert table.height > 0
    assert set(table.get_column("anchor_position").unique()) == {"RB", "WR"}
    for anchor in ("RB", "WR"):
        others = table.filter(pl.col("anchor_position") == anchor)
        assert anchor not in others.get_column("other_position").to_list()


# --- Tier breaks on the scarcity curve --------------------------------------


def test_tier_breaks_are_contiguous_and_ordered() -> None:
    """Tiers must partition the ranks, in order, with no gaps or overlaps.

    A boundary chart that skipped a rank or ran backwards would be drawn
    without complaint and read as a cliff that is not there.
    """
    curve = ls.scarcity_curve(seasons=[2023, 2024, 2025], max_rank=36, basis="ppg")
    if not curve.height:
        pytest.skip("no curve available")
    breaks = ls.tier_breaks(curve, basis="ppg", gap=2.0)
    assert breaks.height, "a real curve must produce tiers"

    for position in breaks.get_column("position").unique().to_list():
        sub = breaks.filter(pl.col("position") == position).sort("tier")
        firsts = sub.get_column("first_rank").to_list()
        lasts = sub.get_column("last_rank").to_list()
        assert firsts[0] == 1, f"{position} must start at rank 1"
        assert firsts == sorted(firsts), f"{position} tiers must run in order"
        for prev_last, next_first in zip(lasts, firsts[1:]):
            assert next_first == prev_last + 1, (
                f"{position} has a gap or overlap at rank {prev_last}"
            )


def test_tier_values_only_fall() -> None:
    """Each tier's best player must be worth less than the previous tier's.

    The curve is monotone by construction, so a rising tier would mean the
    banding lost track of which end it was walking from.
    """
    curve = ls.scarcity_curve(seasons=[2024, 2025], max_rank=36, basis="ppg")
    if not curve.height:
        pytest.skip("no curve available")
    breaks = ls.tier_breaks(curve, basis="ppg", gap=2.0)
    for position in breaks.get_column("position").unique().to_list():
        tops = (
            breaks.filter(pl.col("position") == position)
            .sort("tier")
            .get_column("top_value")
            .to_list()
        )
        assert tops == sorted(tops, reverse=True), f"{position} tiers rise"


def test_a_wider_gap_never_makes_more_tiers() -> None:
    """The slider has to behave monotonically or it is not a width control."""
    curve = ls.scarcity_curve(seasons=[2024, 2025], max_rank=36, basis="ppg")
    if not curve.height:
        pytest.skip("no curve available")
    counts = [
        ls.tier_breaks(curve, basis="ppg", gap=g).height for g in (0.5, 1.5, 3.0)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_tier_breaks_respect_the_basis() -> None:
    """Season totals and per-game are different scales and need different gaps.

    Passing the per-game threshold to a season-total curve would cut a tier at
    almost every rank; the defaults exist so that cannot happen by omission.
    """
    assert ls.TIER_GAP_TOTAL > ls.TIER_GAP_PPG * 10
    curve = ls.scarcity_curve(seasons=[2025], max_rank=24, basis="total")
    if not curve.height:
        pytest.skip("no curve available")
    default = ls.tier_breaks(curve, basis="total")
    assert default.height, "season-total curves must tier too"
    # Using the per-game gap on a season-total curve shatters it, which is the
    # failure the separate constants prevent.
    shattered = ls.tier_breaks(curve, basis="total", gap=ls.TIER_GAP_PPG)
    assert shattered.height > default.height


def test_tier_breaks_is_empty_on_an_empty_curve() -> None:
    assert not ls.tier_breaks(pl.DataFrame(), basis="ppg").height
