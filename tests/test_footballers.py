"""The Fantasy Footballers panel, and the blend that puts it on the board.

Four failure modes drive every test here, and all four return a plausible number
rather than raising:

  1. `_extract` finding the page's *empty* `window.udk.data = {}` instead of the
     real payload — zero rows, no error, a board that silently loses its second
     opinion.
  2. A name join that normalizes one side only. This repo has been burned by
     that twice; the fixtures deliberately disagree about suffixes and team
     codes across the two sides so a half-normalized join fails loudly.
  3. Scoring the projection with the wrong scoring dict, which is invisible
     because every number stays the right order of magnitude.
  4. Blending two point columns whose *scales* differ, which silently hands one
     source more weight than the caller asked for. `test_blend_is_invariant_to_
     the_scale_of_either_source` is the one that pins this, and it is the test
     that caught the first implementation.

The network is never touched: `projections` is monkeypatched with frames whose
right answer is known by construction.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from src import board as bd
from src import footballers as ffb
from src.config import DEFAULT_SCORING


# --- payload extraction -----------------------------------------------------


def _page(*payloads: str) -> str:
    """A page with N `window.udk.data` assignments, in order."""
    blocks = "\n".join(f"<script>window.udk.data = {p};</script>" for p in payloads)
    return f"<html><body>{blocks}</body></html>"


def test_extract_skips_the_empty_assignment_the_page_makes_first() -> None:
    """The real page sets up the namespace with `{}` before assigning the data.

    Taking the first match returns zero rows and no error at all, which is the
    single easiest way for this module to go quietly dead.
    """
    real = json.dumps({"projections": [{"player_id": "1", "name": "A Back"}]})
    html = _page("{}", real)

    rows = ffb._extract(html)

    assert len(rows) == 1
    assert rows[0]["name"] == "A Back"


def test_extract_survives_a_payload_it_cannot_decode() -> None:
    real = json.dumps({"projections": [{"player_id": "1", "name": "A Back"}]})
    html = _page("{", real)

    assert len(ffb._extract(html)) == 1


def test_extract_returns_empty_rather_than_raising_on_a_changed_page() -> None:
    """A site redesign must degrade to "no second opinion", not to a traceback."""
    assert ffb._extract("<html><body>no udk here</body></html>") == []


def test_extract_ignores_a_payload_with_no_projections_key() -> None:
    html = _page(json.dumps({"tiers": [], "essentials": []}))
    assert ffb._extract(html) == []


# --- scoring ----------------------------------------------------------------


def _raw(rows: list[dict]) -> pl.DataFrame:
    """A payload-shaped frame, with the string-typed quirks the real feed has."""
    base = {
        "passing_yards": 0, "passing_touchdowns": 0, "interceptions_thrown": 0,
        "rushing_yards": 0, "rushing_touchdowns": 0, "receptions": 0,
        "receiving_yards": 0, "receiving_touchdowns": 0, "fumbles_lost": 0,
        "risk": 5, "upside": 5, "team": "ARI", "bye_week": "9",
        "updated_at": "2026-08-01 10:00:00", "analyst_name": "Andy",
        "adp": None, "adp_ppr": None, "adp_half_ppr": None, "adp_2qb": None,
    }
    return pl.DataFrame([{**base, **r} for r in rows], infer_schema_length=None)


def _patch(monkeypatch, df: pl.DataFrame) -> None:
    monkeypatch.setattr(ffb, "fetch", lambda season=2026, force=False: df)


def test_a_projection_scores_exactly_what_the_league_scoring_says(monkeypatch) -> None:
    """Hand-computed, not asserted against the function's own output."""
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "Some Back", "fantasy_position": "RB",
                    "rushing_yards": 1000, "rushing_touchdowns": 10,
                    "receptions": 50, "receiving_yards": 400,
                    "receiving_touchdowns": 2, "fumbles_lost": 1,
                }
            ]
        ),
    )

    # 1000*0.1 + 10*6 + 50*0.5 + 400*0.1 + 2*6 + 1*(-2)
    expected = 100.0 + 60.0 + 25.0 + 40.0 + 12.0 - 2.0

    got = ffb.scored(DEFAULT_SCORING).get_column("ffb_points")[0]

    assert got == pytest.approx(expected)


def test_half_ppr_and_full_ppr_disagree_by_exactly_the_receptions(monkeypatch) -> None:
    """The reason this module scores rather than importing a finished ranking."""
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "A Receiver", "fantasy_position": "WR",
                    "receptions": 100, "receiving_yards": 1200,
                }
            ]
        ),
    )

    half = ffb.scored({**DEFAULT_SCORING, "rec": 0.5}).get_column("ffb_points")[0]
    full = ffb.scored({**DEFAULT_SCORING, "rec": 1.0}).get_column("ffb_points")[0]

    assert full - half == pytest.approx(50.0)


def test_interceptions_are_read_from_their_own_column_name(monkeypatch) -> None:
    """Their feed says `interceptions_thrown`; every other source says otherwise.

    A missing key in the mapping is skipped silently by `_weighted_sum`, so a
    typo costs a quarterback 20 points and raises nothing.
    """
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "A Passer", "fantasy_position": "QB",
                    "passing_yards": 4000, "passing_touchdowns": 30,
                    "interceptions_thrown": 10,
                }
            ]
        ),
    )

    # 4000*0.04 + 30*4 + 10*(-2)
    assert ffb.scored(DEFAULT_SCORING).get_column("ffb_points")[0] == pytest.approx(
        160.0 + 120.0 - 20.0
    )


def test_kickers_are_dropped_rather_than_scored_as_zero(monkeypatch) -> None:
    """Their K rows carry no stats at all, so every kicker would tie at 0.0."""
    _patch(
        monkeypatch,
        _raw(
            [
                {"player_id": "1", "name": "A Kicker", "fantasy_position": "K"},
                {"player_id": "2", "name": "A Back", "fantasy_position": "RB"},
            ]
        ),
    )

    names = ffb.projections().get_column("name").to_list()

    assert names == ["A Back"]


# --- consensus --------------------------------------------------------------


def _panel(points_by_analyst: dict[str, float]) -> pl.DataFrame:
    """One player, projected by several analysts at given receiving yardages."""
    return _raw(
        [
            {
                "player_id": "1", "name": "A Receiver", "fantasy_position": "WR",
                "analyst_name": analyst, "receiving_yards": yards,
            }
            for analyst, yards in points_by_analyst.items()
        ]
    )


def test_consensus_uses_the_median_so_one_outlier_cannot_carry_it(monkeypatch) -> None:
    """Two analysts at 1000 yards, one at 200. The median must ignore the limb."""
    _patch(monkeypatch, _panel({"Andy": 1000, "Jason": 1000, "Mike": 200}))

    row = ffb.consensus(DEFAULT_SCORING).row(0, named=True)

    assert row["ffb_points"] == pytest.approx(100.0)
    assert row["ffb_points_mean"] == pytest.approx(73.3, abs=0.1)


def test_spread_is_the_full_range_across_the_panel(monkeypatch) -> None:
    _patch(monkeypatch, _panel({"Andy": 1000, "Jason": 600, "Mike": 200}))

    row = ffb.consensus(DEFAULT_SCORING).row(0, named=True)

    assert row["ffb_spread"] == pytest.approx(80.0)
    assert row["n_analysts"] == 3


def test_a_thin_panel_is_reported_not_hidden(monkeypatch) -> None:
    """`n_analysts` is what makes two consensus numbers comparable or not."""
    _patch(monkeypatch, _panel({"Jason": 1000}))

    assert ffb.consensus(DEFAULT_SCORING).row(0, named=True)["n_analysts"] == 1
    assert ffb.consensus(DEFAULT_SCORING, min_analysts=3).height == 0


def test_staleness_tracks_the_oldest_opinion_in_the_consensus(monkeypatch) -> None:
    """A fresh analyst must not make a three-month-old panel look current."""
    df = _panel({"Andy": 1000, "Jason": 1000})
    df = df.with_columns(
        pl.Series("updated_at", ["2026-01-01 10:00:00", "2026-08-01 10:00:00"])
    )
    _patch(monkeypatch, df)

    row = ffb.consensus(DEFAULT_SCORING).row(0, named=True)

    assert row["stalest_days"] > row["freshest_days"]


# --- the join ---------------------------------------------------------------


def test_attach_matches_across_a_suffix_disagreement(monkeypatch) -> None:
    """Their feed says "Marvin Mims Jr."; nflverse says "Marvin Mims".

    Normalizing the source side only leaves this unmatched, and an unmatched
    player is a null rather than an error.
    """
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "Marvin Mims Jr.",
                    "fantasy_position": "WR", "receiving_yards": 800,
                }
            ]
        ),
    )
    players = pl.DataFrame([{"name": "Marvin Mims", "position": "WR"}])

    out = ffb.attach(players, DEFAULT_SCORING)

    assert out.get_column("ffb_points")[0] == pytest.approx(80.0)


def test_attach_uses_position_to_separate_players_who_share_a_name(
    monkeypatch,
) -> None:
    """There are genuinely two Josh Allens, and position is what tells them apart."""
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "Josh Allen", "fantasy_position": "QB",
                    "passing_yards": 5000,
                }
            ]
        ),
    )
    players = pl.DataFrame(
        [
            {"name": "Josh Allen", "position": "QB"},
            {"name": "Josh Allen", "position": "WR"},
        ]
    )

    out = ffb.attach(players, DEFAULT_SCORING)
    points = out.get_column("ffb_points").to_list()

    assert points[0] == pytest.approx(200.0)
    assert points[1] is None


def test_attach_preserves_the_row_count_and_nulls_the_unmatched(
    monkeypatch,
) -> None:
    """A board must never shrink because a ranking source has not published."""
    _patch(
        monkeypatch,
        _raw(
            [
                {
                    "player_id": "1", "name": "A Back", "fantasy_position": "RB",
                    "rushing_yards": 1000,
                }
            ]
        ),
    )
    players = pl.DataFrame(
        [
            {"name": "A Back", "position": "RB"},
            {"name": "Nobody Projected", "position": "RB"},
        ]
    )

    out = ffb.attach(players, DEFAULT_SCORING)

    assert out.height == 2
    assert out.get_column("ffb_points").to_list()[1] is None


def test_attach_survives_an_empty_panel(monkeypatch) -> None:
    _patch(monkeypatch, pl.DataFrame())
    players = pl.DataFrame([{"name": "A Back", "position": "RB"}])

    out = ffb.attach(players, DEFAULT_SCORING)

    assert out.height == 1
    assert "ffb_points" in out.columns


# --- the blend --------------------------------------------------------------


def _blendable(pars: list[float], ffb_pars: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": [f"P{i}" for i in range(len(pars))],
            "position": ["RB"] * len(pars),
            "par": pars,
            "ffb_par": ffb_pars,
        }
    )


def _ranked(df: pl.DataFrame) -> list[str]:
    return (
        df.sort("blend_par", descending=True, nulls_last=True)
        .get_column("name")
        .to_list()
    )


def test_weight_zero_reproduces_the_boards_own_ordering() -> None:
    """The control. If this ever fails the blend is not a blend."""
    df = _blendable([50.0, 40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 30.0, 40.0, 50.0])

    out = bd.blend_par(df, weight=0.0)

    assert _ranked(out) == ["P0", "P1", "P2", "P3", "P4"]


def test_weight_one_reproduces_the_footballers_ordering() -> None:
    df = _blendable([50.0, 40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 30.0, 40.0, 50.0])

    out = bd.blend_par(df, weight=1.0)

    assert _ranked(out) == ["P4", "P3", "P2", "P1", "P0"]


def test_blend_is_invariant_to_the_scale_of_either_source() -> None:
    """The test that caught the first implementation.

    Both columns are league points above their own replacement level, which makes
    them look directly averageable. They are not — on the real 2026 board
    `ffb_par` carries 1.55x the dispersion of `par` (by IQR), so a raw average at
    weight 0.5 gives the Footballers about 60% of the say, and more than that at
    quarterback. Standardizing before blending is what makes the weight mean what
    it says, and multiplying one source by a constant is the cleanest way to
    assert it: a pure change of units must not reorder the board.
    """
    par = [50.0, 40.0, 30.0, 20.0, 10.0]
    ffb_par = [10.0, 45.0, 25.0, 50.0, 5.0]

    base = _ranked(bd.blend_par(_blendable(par, ffb_par), weight=0.5))
    scaled = _ranked(
        bd.blend_par(_blendable(par, [v * 7.0 for v in ffb_par]), weight=0.5)
    )
    shifted = _ranked(
        bd.blend_par(_blendable(par, [v + 500.0 for v in ffb_par]), weight=0.5)
    )

    assert base == scaled
    assert base == shifted


def test_an_unprojected_player_falls_back_to_his_own_par() -> None:
    """Never blended against a zero, which would rank him at replacement level."""
    df = _blendable([50.0, 40.0, 30.0, 20.0], [10.0, 45.0, 25.0, None])

    out = bd.blend_par(df, weight=0.5)
    row = out.filter(pl.col("name") == "P3").row(0, named=True)

    assert row["blend_par"] == pytest.approx(20.0)


def test_blend_output_stays_on_the_par_scale() -> None:
    """`blend_par` is printed next to `par`, so it has to mean the same thing."""
    par = [50.0, 40.0, 30.0, 20.0, 10.0]
    df = _blendable(par, [80.0, 60.0, 40.0, 20.0, 0.0])

    out = bd.blend_par(df, weight=0.5)
    blended = out.get_column("blend_par")

    assert blended.median() == pytest.approx(30.0, abs=1.0)


def test_blend_degrades_to_par_when_a_source_has_no_spread() -> None:
    df = _blendable([50.0, 40.0, 30.0, 20.0], [7.0, 7.0, 7.0, 7.0])

    out = bd.blend_par(df, weight=0.5)

    assert out.get_column("blend_par").to_list() == pytest.approx(
        [50.0, 40.0, 30.0, 20.0]
    )


def test_blend_is_a_no_op_without_the_footballers_column() -> None:
    df = pl.DataFrame({"name": ["A"], "position": ["RB"], "par": [10.0]})

    assert "blend_par" not in bd.blend_par(df).columns


def test_an_exact_value_survives_the_rounding_used_for_display() -> None:
    """Two players a rounding-tenth apart must still order by their real values.

    Standardizing compresses the Footballers' scale, so players 0.1 apart on
    `ffb_par` can land inside the same displayed tenth. Ranking the rounded
    column breaks that tie on row order, which reversed two real players against
    their own projections at weight 1.0. `blend_par_exact` is what the rank is
    taken on; the tenth is a display convention.
    """
    df = _blendable(
        [40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0],
        [400.0, 300.0, -34.5, -34.6, 0.0, -100.0, -200.0, -300.0],
    )

    out = bd.blend_par(df, weight=1.0)
    exact = dict(zip(out.get_column("name"), out.get_column("blend_par_exact")))
    shown = dict(zip(out.get_column("name"), out.get_column("blend_par")))

    # P2 and P3 are 0.1 apart on ffb_par, which the wider Footballers scale
    # compresses to 0.03 here — inside one displayed tenth.
    assert shown["P2"] == shown["P3"]
    # The column the rank is taken on still separates them, in the right order.
    assert exact["P2"] > exact["P3"]


def test_blend_orders_by_the_footballers_exactly_at_weight_one() -> None:
    """The end-to-end control, on the column `build` actually ranks."""
    par = [40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
    ffb_par = [400.0, 300.0, -34.5, -34.6, 0.0, -100.0, -200.0, -300.0]
    df = _blendable(par, ffb_par)

    out = bd.blend_par(df, weight=1.0)

    by_ffb = df.sort("ffb_par", descending=True).get_column("name").to_list()
    by_blend = (
        out.sort("blend_par_exact", descending=True).get_column("name").to_list()
    )

    assert by_ffb == by_blend
