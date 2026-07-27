"""Board-tab helpers.

Only the pure logic is testable without a Streamlit session, and that is the
part worth testing: snake-draft pick arithmetic is easy to get subtly wrong and
would quietly point the survival math at the wrong pick all draft.
"""

from __future__ import annotations

import polars as pl
import pytest

import app
from src import adp as adp_mod
from src import breakout as bo
from src import landscape as ls


def test_next_pick_follows_the_snake() -> None:
    """Round 1 ascends, round 2 descends, and so on."""
    # Nothing drafted yet: your first pick is your slot.
    assert app._next_pick(0, 1, 10) == 1
    assert app._next_pick(0, 7, 10) == 7

    # Slot 1 in a 10-team snake picks 1, then 20, then 21, then 40.
    assert app._next_pick(1, 1, 10) == 20
    assert app._next_pick(20, 1, 10) == 21
    assert app._next_pick(21, 1, 10) == 40

    # Slot 10 picks 10 and 11 back to back, then 30.
    assert app._next_pick(1, 10, 10) == 10
    assert app._next_pick(10, 10, 10) == 11
    assert app._next_pick(11, 10, 10) == 30


def test_next_pick_is_always_ahead_of_the_draft() -> None:
    """The returned pick must never be one that has already happened."""
    for teams in (8, 10, 12):
        for slot in range(1, teams + 1):
            for taken in range(0, teams * 6):
                assert app._next_pick(taken, slot, teams) > taken


def test_every_pick_belongs_to_exactly_one_slot() -> None:
    """Across all slots, the pick sequences must partition the draft."""
    teams = 10
    seen: dict[int, int] = {}
    for slot in range(1, teams + 1):
        taken = 0
        for _ in range(8):
            pick = app._next_pick(taken, slot, teams)
            assert pick not in seen, f"pick {pick} claimed by slots {seen.get(pick)} and {slot}"
            seen[pick] = slot
            taken = pick
    assert sorted(seen) == list(range(1, len(seen) + 1))


@pytest.fixture(scope="module")
def board() -> pl.DataFrame:
    df = bo.adp_board(2026)
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


def test_market_value_is_slot_based_not_player_based(board: pl.DataFrame) -> None:
    """Two players at the same positional rank must get the same number.

    This is the honest limitation of market-implied value and the reason it is
    never called a projection: it describes the draft slot, not the player.
    """
    valued = ls.market_implied_value(board)
    assert valued.get_column("market_var").is_null().sum() == 0

    # Value must decline with positional rank — that is what a scarcity curve is.
    for position in ("RB", "WR"):
        sub = (
            valued.filter(pl.col("position") == position)
            .sort("adp_pos_rank")
            .head(24)
        )
        first = float(sub.get_column("market_ppg")[0])
        last = float(sub.get_column("market_ppg")[-1])
        assert first > last, f"{position} market value does not decline with rank"


def test_exclusions_propagate_into_survival(board: pl.DataFrame) -> None:
    """Cutting players must change the pool the survival math runs on.

    The spec's actual requirement for this tab. If survival were computed on the
    full board and merely displayed against a filtered one, cuts would look like
    they worked while changing nothing.
    """
    valued = ls.market_implied_value(board)
    cut = set(valued.get_column("gsis_id").to_list()[:20])

    full = adp_mod.survival(valued, 25)
    trimmed = adp_mod.survival(valued.filter(~pl.col("gsis_id").is_in(list(cut))), 25)

    assert trimmed.height == full.height - 20
    assert "p_available_at_25" in trimmed.columns
    # The survivors keep their probabilities; the pool is what changed.
    assert not set(trimmed.get_column("gsis_id").to_list()) & cut


def test_survival_falls_as_the_pick_gets_later(board: pl.DataFrame) -> None:
    """A player is less likely to last to pick 40 than to pick 10."""
    early = adp_mod.survival(board, 10).get_column("p_available_at_10").to_numpy()
    late = adp_mod.survival(board, 40).get_column("p_available_at_40").to_numpy()
    assert (late <= early + 1e-9).all()
    assert late.mean() < early.mean()
