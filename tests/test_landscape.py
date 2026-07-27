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
