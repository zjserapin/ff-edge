"""Scoring correctness, checked against data rather than against itself.

The valuable test here is the identity one. nflverse computes
`fantasy_points_ppr` independently of this project, using the same
passing/rushing/receiving/2pt weights the Shiva Bowl uses, so half-PPR has a
closed form: ppr minus half a point per reception. If our compiled expression
agrees with that across every cached player-week, the Sleeper->nflverse column
mapping is right — including the parts that are easy to get wrong silently,
like which of the three fumble columns to sum.

These read the local cache and skip cleanly when it is cold.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import scoring
from src.config import DEFAULT_SCORING, FANTASY_POSITIONS


@pytest.fixture(scope="module")
def weekly() -> pl.DataFrame:
    df = scoring.score_weekly()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def season_points() -> pl.DataFrame:
    return scoring.score_season()


def test_half_ppr_matches_nflverse_identity(weekly: pl.DataFrame) -> None:
    """half-PPR == fantasy_points_ppr - 0.5 * receptions, for every skill week.

    The one documented exception is a skill player recording a kicking stat.
    Sleeper scores by stat line, so a running back who makes a field goal gets
    the points; nflverse only scores kicking for players listed at K. There is
    exactly one such row in the cache (Dare Ogunbowale, 2023 week 9), and it is
    excluded here rather than papered over.
    """
    from src import nflverse as nv

    raw = nv.weekly_stats().filter(
        pl.col("position").is_in(list(FANTASY_POSITIONS))
        & (pl.col("season_type") == "REG")
        # skill players who attempted a kick — see docstring
        & (pl.col("fg_att").fill_null(0) == 0)
        & (pl.col("pat_att").fill_null(0) == 0)
    )

    joined = raw.select(
        "player_id",
        "season",
        "week",
        (
            pl.col("fantasy_points_ppr").fill_null(0)
            - 0.5 * pl.col("receptions").fill_null(0)
        ).alias("identity"),
        # Selected down to four columns deliberately: weekly_stats has its own
        # `fantasy_points` (standard scoring), and joining the full frame would
        # suffix ours and silently compare nflverse against itself.
    ).join(
        weekly.select("player_id", "season", "week", "fantasy_points"),
        on=["player_id", "season", "week"],
        how="inner",
    )

    assert joined.height > 30_000, "identity check needs the full cache to mean anything"
    gap = (joined.get_column("fantasy_points") - joined.get_column("identity")).abs()
    assert gap.max() < 1e-6, f"max gap {gap.max()} over {joined.height} player-weeks"


def test_fumble_columns_are_the_three_components() -> None:
    """`fumbles_lost_total` is not interchangeable with the three components.

    Guards the specific mapping choice in SCORING_COLUMNS. If someone later
    "simplifies" fum_lost to the total column, the identity test above breaks —
    this one says why.
    """
    from src import nflverse as nv

    raw = nv.weekly_stats().filter(pl.col("position").is_in(list(FANTASY_POSITIONS)))
    disagree = raw.filter(
        pl.col("fumbles_lost_total").fill_null(0)
        != (
            pl.col("sack_fumbles_lost").fill_null(0)
            + pl.col("rushing_fumbles_lost").fill_null(0)
            + pl.col("receiving_fumbles_lost").fill_null(0)
        )
    )
    assert disagree.height > 0, "if these ever agree, the mapping comment is stale"


def test_unmapped_keys_are_exactly_the_uncomputable_ones() -> None:
    """Every live scoring key is either mapped or explicitly surfaced as not.

    The unmapped set must be all-DST/IDP. Anything else appearing here is a
    skill-position stat we are silently failing to score.
    """
    unmapped = set(scoring.unmapped_keys(DEFAULT_SCORING))
    expected = {
        "blk_kick", "def_st_ff", "def_st_fum_rec", "def_st_td", "def_td", "ff",
        "fum_rec", "fum_rec_td", "int", "sack", "safe", "st_ff", "st_fum_rec",
        "pts_allow_0", "pts_allow_1_6", "pts_allow_7_13", "pts_allow_14_20",
        "pts_allow_28_34", "pts_allow_35p",
    }
    assert unmapped == expected


def test_zero_weight_keys_contribute_nothing() -> None:
    """A key set to 0.0 must not add a term. `fum` is 0.0 in this league."""
    with_fum = scoring.points_expr({"rec": 1.0, "fum": 0.0})
    without = scoring.points_expr({"rec": 1.0})
    df = pl.DataFrame({"receptions": [3, 0], "fumbles_total": [2, 1]})
    assert df.select(with_fum).equals(df.select(without))


TWO_FLEX = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"]
SUPER_FLEX = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"]


def test_starter_demand_conserves_slots(season_points: pl.DataFrame) -> None:
    """Every slot on the roster must start exactly one player, league-wide.

    10 teams x 8 skill slots = 80 starters, however the flex slots get split.
    A conservation failure means a slot was double-counted or dropped, which is
    exactly what SUPER_FLEX used to do — it was silently skipped, so the league
    started 70 players from an 80-slot roster.
    """
    year = season_points.filter(pl.col("season") == season_points["season"].max())
    for roster in (TWO_FLEX, SUPER_FLEX):
        demand = scoring.starter_demand(roster, season_points=year, teams=10)
        total = sum(demand[p] for p in ("QB", "RB", "WR", "TE"))
        assert total == pytest.approx(80.0), f"{roster}: {demand}"
        # No flex allocation may take from a dedicated slot.
        assert demand["QB"] >= 10 and demand["TE"] >= 10
        assert demand["RB"] >= 20 and demand["WR"] >= 20


def test_superflex_roughly_doubles_quarterback_demand(
    season_points: pl.DataFrame,
) -> None:
    """The finding this whole change exists for.

    A QB-eligible slot puts quarterbacks into the marginal-starter comparison,
    and in half-PPR the QB20 on offer beats the RB21 or WR21 it is weighed
    against essentially every time — so the slot goes to a quarterback and
    league-wide QB demand goes from one per team to two.
    """
    year = season_points.filter(pl.col("season") == season_points["season"].max())
    before = scoring.starter_demand(TWO_FLEX, season_points=year, teams=10)
    after = scoring.starter_demand(SUPER_FLEX, season_points=year, teams=10)

    assert before["QB"] == 10
    assert after["QB"] >= 19, f"superflex barely moved QB demand: {after}"
    # The slots came from somewhere: one FLEX became a SUPER_FLEX, so the
    # RB/WR/TE pool loses ten starters.
    assert sum(after[p] for p in ("RB", "WR", "TE")) == pytest.approx(
        sum(before[p] for p in ("RB", "WR", "TE")) - 10.0
    )


def test_superflex_moves_replacement_quarterback(season_points: pl.DataFrame) -> None:
    """Replacement QB should fall about ten positional ranks, and the baseline
    with it — that repricing is the entire point of the change."""
    year = season_points.filter(pl.col("season") == season_points["season"].max())
    before = scoring.replacement_level(year, TWO_FLEX, teams=10)
    after = scoring.replacement_level(year, SUPER_FLEX, teams=10)

    qb_before = before.filter(pl.col("position") == "QB").row(0, named=True)
    qb_after = after.filter(pl.col("position") == "QB").row(0, named=True)

    assert qb_before["replacement_rank"] <= 12
    assert qb_after["replacement_rank"] >= 19
    # A lower baseline means every quarterback's PAR rises.
    assert qb_after["replacement_points"] < qb_before["replacement_points"]


def test_flex_split_governs_only_the_slots_it_can_describe(
    season_points: pl.DataFrame,
) -> None:
    """The sidebar slider names RB/WR/TE shares. That is enough to express a
    FLEX and not enough to express a SUPER_FLEX, whose whole question is
    whether the slot goes to a quarterback — so the split must not silently
    decide it."""
    year = season_points.filter(pl.col("season") == season_points["season"].max())
    split = {"RB": 0.5, "WR": 0.5, "TE": 0.0}

    # With only FLEX slots the split governs everything, proportionally.
    demand = scoring.starter_demand(TWO_FLEX, teams=10, flex_split=split)
    assert demand["RB"] == pytest.approx(30.0)
    assert demand["WR"] == pytest.approx(30.0)
    assert demand["TE"] == pytest.approx(10.0)

    # With a SUPER_FLEX the split still governs the FLEX, but the superflex is
    # computed — and lands on quarterbacks.
    demand = scoring.starter_demand(
        SUPER_FLEX, teams=10, season_points=year, flex_split=split
    )
    assert demand["RB"] == pytest.approx(25.0)  # 20 dedicated + half of one FLEX
    assert demand["WR"] == pytest.approx(25.0)
    assert demand["QB"] >= 19, "flex_split silently suppressed the superflex"


def test_replacement_is_past_the_starting_pool(season_points: pl.DataFrame) -> None:
    """Replacement rank must sit at or beyond demand, never inside the starters."""
    repl = scoring.replacement_level(season_points, teams=10)
    assert repl.height > 0
    for row in repl.iter_rows(named=True):
        assert row["replacement_rank"] >= row["demand"] - 1


def test_par_is_zero_at_replacement(season_points: pl.DataFrame) -> None:
    """The replacement player's PAR is 0 by construction — proves the join lands."""
    par = scoring.points_above_replacement(season_points, teams=10)
    at_repl = par.filter(pl.col("pos_rank") == pl.col("replacement_rank"))
    assert at_repl.height > 0
    assert at_repl.get_column("par").abs().max() < 1e-6


def test_replacement_ppg_is_not_distorted_by_part_seasons(
    season_points: pl.DataFrame,
) -> None:
    """The per-game baseline must be stable year to year.

    Ranking by season total and reading that player's ppg let a part-season
    outlier set the baseline: in 2025 Tucker Kraft was TE11 on total points in 8
    games at 12.65 ppg, roughly 60% above the real streaming level, which would
    have suppressed every tight end's PAR. Guarding the range catches a
    regression to that behavior.
    """
    repl = scoring.replacement_level(season_points, teams=10)
    te = repl.filter(pl.col("position") == "TE").get_column("replacement_ppg")
    assert te.max() - te.min() < 3.0, f"TE replacement ppg unstable: {te.to_list()}"

    # And every replacement player must have been startable often enough to be
    # a genuine alternative, not a 3-game cameo.
    for pos in ("RB", "WR", "TE"):
        vals = repl.filter(pl.col("position") == pos).get_column("replacement_ppg")
        assert vals.max() < 15.0, f"{pos} replacement ppg implausibly high"


def test_positional_ranks_are_dense_and_per_season(season_points: pl.DataFrame) -> None:
    """pos_rank restarts at 1 for each season/position and has no duplicates."""
    firsts = season_points.group_by(["season", "position"]).agg(
        pl.col("pos_rank").min().alias("lo"),
        pl.col("pos_rank").n_unique().alias("uniq"),
        pl.len().alias("n"),
    )
    assert (firsts.get_column("lo") == 1).all()
    assert (firsts.get_column("uniq") == firsts.get_column("n")).all()
