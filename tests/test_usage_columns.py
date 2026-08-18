"""`attach_usage`: the join that must not move anything.

`context.py` computes the opportunity metrics this project measured as the
*persistent* half — at RB the opportunity axis runs 0.52-0.65 year over year
against a quality axis topping out at 0.402 — and until 2026-08-17 none of them
reached the screen while `quality_pct`, the weaker half, was a display column
*and* the within-block sort key.

Surfacing them is display work, so the risks are the ones display work always
has here and none of them raise:

**The join fans out.** Names are the key, `ids.normalize` strips generational
suffixes, and a left join without a dedupe multiplies rows rather than erroring
— the props join went 145 -> 151 exactly that way.

**The join changes the board.** These columns are attached after `rank_board`
precisely so they cannot, and that ordering is an invariant nothing else
enforces.

**A null becomes a zero.** A rookie has no prior season and a receiver has no
`rz_carry_share`. Filling either would read as "no red-zone role" when it means
"not a rushing position".
"""

from __future__ import annotations

import polars as pl
import pytest

from src import board as bd


def _ranked() -> pl.DataFrame:
    """A small board-shaped frame that has already been ranked."""
    return pl.DataFrame(
        {
            "name": ["Ja'Marr Chase", "Bijan Robinson", "Nobody At All"],
            "position": ["WR", "RB", "TE"],
            "board_rank": [1, 2, 3],
            "block": [1, 1, 2],
            "par_env": [50.0, 40.0, 10.0],
        }
    )


def _feats(rows: list[tuple[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2025,
                "player_name": n,
                "age": a,
                "snap_pct": 0.5,
                "target_share": 0.2,
                "exp_td_share": 0.1,
                "neutral_target_share": 0.2,
                "rz_target_share": 0.3,
                "rz_carry_share": None,
            }
            for n, a in rows
        ],
        schema_overrides={"rz_carry_share": pl.Float64},
    )


def test_every_usage_column_is_added_even_when_absent_upstream() -> None:
    """The app selects these by name; a missing one silently narrows the table."""
    out = bd.attach_usage(_ranked(), pl.DataFrame(), season=2025)
    for col in bd._USAGE_COLUMNS:
        assert col in out.columns


def test_the_join_preserves_the_row_count() -> None:
    """The failure mode is fan-out, and it does not raise."""
    board = _ranked()
    out = bd.attach_usage(board, _feats([("Ja'Marr Chase", 26.0)]), season=2025)
    assert out.height == board.height


def test_a_duplicated_name_upstream_still_yields_one_row() -> None:
    """`ids.normalize` collapses Jr./Sr., so the dedupe is load-bearing.

    Two feature rows normalizing to one key must not double the board row.
    """
    dupes = _feats([("Ja'Marr Chase", 26.0), ("Ja'Marr Chase Jr.", 99.0)])
    out = bd.attach_usage(_ranked(), dupes, season=2025)
    assert out.height == 3
    assert out.filter(pl.col("name") == "Ja'Marr Chase").height == 1


def test_attaching_usage_never_reorders_or_reranks_the_board() -> None:
    """**The property the whole design rests on.**

    These columns are appended downstream of `rank_board` so that "usage cannot
    move a rank" is a fact about the call graph. If someone later moves the call
    upstream, this is what fails.
    """
    board = _ranked()
    out = bd.attach_usage(board, _feats([("Bijan Robinson", 24.0)]), season=2025)

    for col in ("name", "board_rank", "block", "par_env"):
        assert out.get_column(col).to_list() == board.get_column(col).to_list()


def test_an_unmatched_player_keeps_null_usage_rather_than_zero() -> None:
    """Null means not measured. A zero would read as "no role"."""
    out = bd.attach_usage(_ranked(), _feats([("Ja'Marr Chase", 26.0)]), season=2025)
    row = out.filter(pl.col("name") == "Nobody At All")
    assert row.height == 1
    for col in bd._USAGE_COLUMNS:
        assert row.get_column(col)[0] is None, f"{col} was filled instead of left null"


def test_a_position_without_the_metric_stays_null() -> None:
    """`rz_carry_share` is a rushing metric; a receiver correctly has none."""
    out = bd.attach_usage(_ranked(), _feats([("Ja'Marr Chase", 26.0)]), season=2025)
    chase = out.filter(pl.col("name") == "Ja'Marr Chase")
    assert chase.get_column("rz_carry_share")[0] is None
    assert chase.get_column("rz_target_share")[0] == pytest.approx(0.3)


def test_only_the_requested_season_is_joined() -> None:
    """A prior year must not leak in and be read as last season's role."""
    feats = pl.concat(
        [
            _feats([("Ja'Marr Chase", 26.0)]),
            _feats([("Ja'Marr Chase", 99.0)]).with_columns(
                # Cast explicitly: `pl.lit(2024)` is Int32 and the column built
                # by `_feats` is Int64, and `concat` raises on the mismatch.
                pl.lit(2024).cast(pl.Int64).alias("season")
            ),
        ]
    )
    out = bd.attach_usage(_ranked(), feats, season=2025)
    assert out.filter(pl.col("name") == "Ja'Marr Chase").get_column("age")[0] == 26.0


def test_an_empty_board_is_returned_untouched() -> None:
    empty = pl.DataFrame({"name": [], "position": []})
    assert bd.attach_usage(empty, _feats([("x", 1.0)]), season=2025).height == 0


def test_no_usage_column_is_a_quality_metric() -> None:
    """The selection rule, pinned.

    Every column here is opportunity or age. `quality_pct` is already on the
    board and is already the within-block sort key; adding more of the weaker
    axis would be the opposite of the point.
    """
    quality = {"ypc", "yprr", "tprr", "yac_per_rec", "rush_efficiency", "quality_pct"}
    assert not (set(bd._USAGE_COLUMNS) & quality)
