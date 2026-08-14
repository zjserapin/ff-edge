"""FantasyPros ECR: the page it reads, and how it enters the blend.

Two failures live here and neither raises.

**The wrong page.** FantasyPros publishes a superflex board (`redraft-op`) and a
1QB board (`redraft-overall`). Quarterbacks sit tens of ranks apart between them,
so reading the wrong one produces a well-formed column of plausible numbers with
every quarterback mispriced — the 2026 superflex bug wearing different clothes.
The page is derived from the profile's roster rather than passed in, and these
tests pin that derivation.

**The wrong scale.** ECR is an ordering, not a projection. Pushing it through the
expected-points curve would collapse it exactly where it was wanted, since that
curve pools the top six backs into one number. It is blended on the rank scale
instead, and a rank keeps one value per player all the way to the top.

The blend also has to be inert where ECR is missing: a player FantasyPros never
ranked must keep his blended value rather than being pushed to the bottom, which
is what a naive fill would do.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import board as bd
from src import profiles as pf
from src.config import ECR_PAGE_STANDARD, ECR_PAGE_SUPERFLEX, ECR_WEIGHT


def _blendable(rows: list[tuple[str, float, float | None]]) -> pl.DataFrame:
    """(name, blend_par, ecr) -> a frame `blend_ecr` will act on."""
    return pl.DataFrame(
        [{"name": n, "blend_par": b, "ecr": e} for n, b, e in rows],
        schema={"name": pl.Utf8, "blend_par": pl.Float64, "ecr": pl.Float64},
    )


# --- the blend --------------------------------------------------------------


def test_a_better_expert_rank_raises_a_player_relative_to_his_peers() -> None:
    """Lower ECR is better, so the sign has to be flipped on the way in.

    Getting this backwards produces a board that is confidently upside down and
    raises nothing, which is why it is the first test in the file.
    """
    board = _blendable(
        [("Loved", 50.0, 1.0), ("Middling", 50.0, 40.0), ("Disliked", 50.0, 90.0)]
        + [(f"Filler{i}", float(40 + i), float(20 + i)) for i in range(6)]
    )

    out = bd.blend_ecr(board, weight=0.5)
    got = dict(zip(out.get_column("name"), out.get_column("blend_par")))

    assert got["Loved"] > got["Middling"] > got["Disliked"]


def test_a_weight_of_zero_leaves_the_base_untouched() -> None:
    """ECR must be removable without residue if it turns out to add nothing."""
    board = _blendable(
        [(f"P{i}", float(50 - i), float(i + 1)) for i in range(8)]
    )

    out = bd.blend_ecr(board, weight=0.0)

    assert out.get_column("blend_par").to_list() == pytest.approx(
        board.get_column("blend_par").to_list()
    )


def test_an_unranked_player_keeps_his_value_rather_than_sinking() -> None:
    """Null ECR is "FantasyPros has no row", not "ranked last".

    A fill-then-blend would quietly push every unranked player to the bottom of
    the board, and unranked skews toward rookies and post-hype players — exactly
    the population a draft board should not silently bury.
    """
    board = _blendable(
        [("Unranked", 50.0, None)]
        + [(f"P{i}", float(50 - i), float(i + 1)) for i in range(8)]
    )

    out = bd.blend_ecr(board, weight=0.5)
    got = dict(zip(out.get_column("name"), out.get_column("blend_par")))

    assert got["Unranked"] == pytest.approx(50.0)


def test_a_degenerate_spread_falls_back_rather_than_dividing_by_zero() -> None:
    """Every player at the same ECR makes the standardization undefined."""
    board = _blendable([(f"P{i}", float(50 - i), 5.0) for i in range(8)])

    out = bd.blend_ecr(board, weight=0.5)

    assert out.get_column("blend_par").to_list() == pytest.approx(
        board.get_column("blend_par").to_list()
    )


def test_too_few_ranked_players_leaves_the_board_alone() -> None:
    board = _blendable([("A", 50.0, 1.0), ("B", 40.0, 2.0)])

    out = bd.blend_ecr(board, weight=0.5)

    assert out.get_column("blend_par").to_list() == [50.0, 40.0]


def test_blend_ecr_is_inert_without_an_ecr_column() -> None:
    board = pl.DataFrame({"name": ["A"], "blend_par": [50.0]})

    out = bd.blend_ecr(board)

    assert out.get_column("blend_par").to_list() == [50.0]


def test_ecr_is_weighted_far_below_the_footballers() -> None:
    """The asymmetry is a stated position, not an accident of tuning.

    A consensus of orderings largely read off the same market the board is
    already priced against earns less of a vote than an independent projection.
    If these ever converge, that is a decision someone should have to make on
    purpose.
    """
    from src.config import FOOTBALLERS_WEIGHT

    assert ECR_WEIGHT < FOOTBALLERS_WEIGHT / 2


# --- the page, which is the trap --------------------------------------------


def test_a_superflex_roster_reads_the_superflex_page() -> None:
    """Derived from the roster, never passed in.

    `redraft-op` and `redraft-overall` disagree about quarterbacks by tens of
    ranks. A loose argument would let the page drift from the format the way the
    2026 board drifted from its market.
    """
    assert "SUPER_FLEX" in set(pf.resolve("shiva_bowl").roster_positions)
    assert ECR_PAGE_SUPERFLEX != ECR_PAGE_STANDARD


def test_a_one_quarterback_roster_reads_the_standard_page() -> None:
    assert "SUPER_FLEX" not in set(pf.resolve("standard_12").roster_positions)


def test_attaching_ecr_never_changes_the_row_count() -> None:
    """The join fans out or it does not. It does not raise either way.

    `ids.normalize` strips generational suffixes, so the dedupe before the join
    is what keeps this true — the props join went 145 rows to 151 without it.
    """
    players = pl.DataFrame(
        {
            "name": ["Michael Pittman Jr.", "Some Player"],
            "position": ["WR", "RB"],
        }
    )

    out = bd.attach_ecr(players, profile=pf.resolve("shiva_bowl"))

    assert out.height == 2
    for column in ("ecr", "ecr_sd"):
        assert column in out.columns
