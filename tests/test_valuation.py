"""Valuation — the module that names disagreements with the draft market.

The tests here are mostly about direction. A percentile that runs backwards
still produces a plausible-looking board, and the failure is invisible until you
read the names: the first version of this ranked Christian McCaffrey and Justin
Jefferson as the most undervalued players in the league, because `descending`
had been passed the wrong way round on both quality and price. So the primary
guard is a set of players whose verdict is knowable in advance.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import features as ft
from src import valuation as val


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    df = ft.build()
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def board(features: pl.DataFrame) -> pl.DataFrame:
    df = val.board(df=features)
    if not df.height:
        pytest.skip("no ADP board available")
    return df


# --- direction --------------------------------------------------------------


def test_percentile_directions_are_not_inverted(board: pl.DataFrame) -> None:
    """The most expensive player must score ~100 on price, not ~0.

    This is the regression guard for the bug that made elite players look free.
    """
    for position in board.get_column("position").unique().to_list():
        sub = board.filter(pl.col("position") == position)
        priciest = sub.sort("adp_pos_rank").row(0, named=True)
        cheapest = sub.sort("adp_pos_rank", descending=True).row(0, named=True)
        assert priciest["market_pct"] > cheapest["market_pct"], (
            f"{position}: price percentile is inverted"
        )

        best = sub.sort("quality_score", descending=True).row(0, named=True)
        worst = sub.sort("quality_score").row(0, named=True)
        assert best["quality_pct"] > worst["quality_pct"], (
            f"{position}: quality percentile is inverted"
        )


def test_elite_expensive_players_are_not_undervalued(board: pl.DataFrame) -> None:
    """A top-5-pick player cannot be a bargain — his price already says he's good.

    Named players rather than a statistical property, because this is exactly
    the check that would have caught the original inversion in one glance.
    """
    top = board.filter(pl.col("adp") <= 6.0)
    assert top.height >= 2, "expected a few top-6 ADP players on the board"
    assert not (top.get_column("verdict") == "undervalued").any(), (
        "a top-6 ADP player is flagged undervalued — percentiles are inverted"
    )


def test_value_gap_is_the_difference_of_its_parts(board: pl.DataFrame) -> None:
    gap = board.get_column("quality_pct") - board.get_column("market_pct")
    assert (gap - board.get_column("value_gap")).abs().max() < 0.05


def test_percentiles_are_within_position(board: pl.DataFrame) -> None:
    """Percentiles must be computed per position, or a TE is ranked against WRs."""
    for position in board.get_column("position").unique().to_list():
        sub = board.filter(pl.col("position") == position)
        if sub.height < 5:
            continue
        assert sub.get_column("quality_pct").max() > 95
        assert sub.get_column("market_pct").max() > 95


# --- construction -----------------------------------------------------------


def test_quality_excludes_volume() -> None:
    """The central design claim: nothing in the quality axis scales with usage.

    If a volume column leaks in, the board rediscovers ADP and the whole module
    is a slower way to read a draft ranking.
    """
    volume = {
        "target_share", "snap_pct", "air_yards_share", "tgt_per_game",
        "carry_per_game", "route_share", "rush_share", "exp_pts_share",
        "targets", "carries", "routes", "receptions",
    }
    for position in ("WR", "RB", "TE"):
        quality = set(ft.quality_features(position))
        assert not (quality & volume), f"{position} quality set contains volume: {quality & volume}"
        # And the two axes must not overlap at all.
        assert not (quality & set(ft.opportunity_features(position)))


def test_quality_excludes_fantasy_points() -> None:
    """Points are the outcome. Clustering on them makes the result circular."""
    banned = {"ppg", "fantasy_points", "exp_ppg", "pts_over_exp_per_game", "pos_rank"}
    for position in ("WR", "RB", "TE", "QB"):
        assert not (set(ft.quality_features(position)) & banned)


def test_board_covers_the_valued_positions(board: pl.DataFrame) -> None:
    """Quarterbacks are scored as of 2026-08-10, and their absence was a bug.

    This asserted `"QB" not in ...` for as long as the board excluded them. The
    exclusion was justified on the receiving metrics having no QB analogue, which
    is true of those metrics and false of the position — `archetypes.scores` has
    always produced a QB quality score and this module discarded it. In a
    superflex league that blanked the board at the position the format makes
    scarce. The assertion is inverted here on purpose so the old behaviour cannot
    come back unnoticed.
    """
    seen = set(board.get_column("position").unique().to_list())
    assert seen <= set(val.VALUED_POSITIONS)
    assert "QB" in seen, "quarterbacks are scored — see MIN_VOLUME and VALUED_POSITIONS"


def test_volume_floor_is_applied_per_position(board: pl.DataFrame) -> None:
    """Each position gated on its own denominator, checked on its own denominator.

    The trap this pins: **a quarterback's `routes` value is dropbacks**, so a
    single `routes >= 100` check passes every quarterback while measuring nothing
    about him. Taysom Hill cleared that check on 6 pass attempts and scored at
    the 25th quality percentile. Testing the floor the same way it is written
    would reproduce the bug rather than catch it, so each position is checked
    against the column that actually gates it.
    """
    for position, (volume_col, floor) in val.MIN_VOLUME.items():
        sub = board.filter(pl.col("position") == position)
        if not sub.height or volume_col not in sub.columns:
            continue
        low = sub.filter(pl.col(volume_col).fill_null(0) < floor)
        assert not low.height, (
            f"{position} rows below the {volume_col} floor of {floor:g}: "
            f"{low.get_column('name').to_list()[:5]}"
        )


def test_quarterbacks_have_no_path_score_and_are_not_filtered_by_it(
    board: pl.DataFrame,
) -> None:
    """A null `path_score` must not silently delete QBs from the undervalued list.

    `path_score` is null at QB by construction — its terms are about earning
    targets, which is not a quarterback's route to volume. A null fails `>=`
    silently in polars, so without the bypass in `undervalued` every underpriced
    quarterback would vanish for a reason unrelated to his price.
    """
    qbs = board.filter(pl.col("position") == "QB")
    if not qbs.height:
        pytest.skip("no quarterbacks on the board")
    assert qbs.get_column("path_score").is_null().all()

    cheap = board.filter(
        (pl.col("position") == "QB") & (pl.col("verdict") == "undervalued")
    )
    if cheap.height:
        surfaced = val.undervalued(board).filter(pl.col("position") == "QB")
        assert surfaced.height, "undervalued QBs were dropped by the path filter"


# --- the lists --------------------------------------------------------------


def test_undervalued_requires_both_gap_and_path(board: pl.DataFrame) -> None:
    """Quality alone is not a buy signal if he has nowhere to gain volume."""
    picks = val.undervalued(board, min_path=50.0)
    if not picks.height:
        pytest.skip("no undervalued players at these thresholds")
    assert (picks.get_column("value_gap") >= val.GAP_THRESHOLD).all()
    assert (picks.get_column("path_score") >= 50.0).all()


def test_overvalued_is_the_mirror(board: pl.DataFrame) -> None:
    picks = val.overvalued(board)
    if picks.height:
        assert (picks.get_column("value_gap") <= -val.GAP_THRESHOLD).all()


def test_verdicts_partition_the_board(board: pl.DataFrame) -> None:
    counts = val.summary(board)
    assert counts.get_column("n").sum() == board.height
    assert set(board.get_column("verdict").unique().to_list()) <= {
        "undervalued", "fairly priced", "overvalued"
    }


def test_comparables_are_same_position_and_ranked(
    board: pl.DataFrame, features: pl.DataFrame
) -> None:
    """The 'who does he look like' output, which is the module's headline use."""
    target = board.filter(pl.col("position") == "WR").sort("value_gap", descending=True)
    pid = target.get_column("gsis_id")[0]

    comps = val.comparables(board, pid, n=6, df=features)
    assert comps.height == 6
    assert set(comps.get_column("position").unique().to_list()) == {"WR"}
    assert pid not in comps.get_column("gsis_id").to_list()
    distances = comps.get_column("distance").to_list()
    assert distances == sorted(distances)


def test_room_to_grow_is_smaller_for_a_team_alpha(board: pl.DataFrame) -> None:
    """A team's alpha already has the volume, so he has less of it left to win.

    Guards the sign on the term that carries the idea — a situation score that
    rewarded already having the volume would recommend exactly the players who
    are already priced.

    **Asserted on `opportunity_pct`, not on `path_score`, and the difference is
    the point.** `path_score` averages room-to-grow against a reversed
    `teammate_top_share`, and both are true of an alpha at once: he has no room
    left *and* nobody standing in front of him. The two terms cancel, so the
    composite sits near 50 for everyone (alpha 50.7 vs other 50.4) and carries
    almost no signal about room at all.

    This test used to assert the composite and passed on a fluke: the direction
    came from a single running back, while WR and TE already leaned the other
    way. A change of ADP market swapped two players and flipped it. The
    underlying invariant is real and large — roughly 33 percentile points — so
    it is asserted where it actually lives.
    """
    if "is_team_alpha" not in board.columns:
        pytest.skip("alpha flag unavailable")
    alphas = board.filter(pl.col("is_team_alpha"))
    others = board.filter(~pl.col("is_team_alpha").fill_null(False))
    if alphas.height < 5 or others.height < 5:
        pytest.skip("not enough of each group")

    alpha_opp = alphas.get_column("opportunity_pct").mean()
    other_opp = others.get_column("opportunity_pct").mean()
    # A real margin, not a coin flip on two group means. The measured gap is
    # ~33 points; anything under 10 means the alpha flag stopped meaning volume.
    assert alpha_opp - other_opp > 10.0, (
        f"team alphas should hold far more of the opportunity: "
        f"{alpha_opp:.1f} vs {other_opp:.1f}"
    )


def test_path_score_does_not_claim_to_measure_room(board: pl.DataFrame) -> None:
    """Pin the known limitation above so it cannot be forgotten or assumed away.

    If `path_score` is ever reweighted so room-to-grow dominates, this test
    fails and the docstring in `valuation.board` needs rewriting with it. That
    is the intended outcome — the failure is the notification, not a bug.
    """
    if "is_team_alpha" not in board.columns:
        pytest.skip("alpha flag unavailable")
    alphas = board.filter(pl.col("is_team_alpha"))
    others = board.filter(~pl.col("is_team_alpha").fill_null(False))
    if alphas.height < 5 or others.height < 5:
        pytest.skip("not enough of each group")

    gap = abs(
        alphas.get_column("path_score").mean()
        - others.get_column("path_score").mean()
    )
    assert gap < 10.0, (
        "path_score now separates alphas from the field. That may be an "
        "improvement, but valuation.board's docstring and this test both "
        "describe it as a composite where the terms cancel — update them."
    )
