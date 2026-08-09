"""Rookie model — mostly join integrity, because that is what breaks.

The model itself is a ridge regression and does what ridge regressions do. What
nearly cost it a quarter of its signal was two joins that returned plausible
frames with silently missing data.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import ids
from src import nflverse as nv
from src import rookies as rk


@pytest.fixture(scope="module")
def board_2026() -> pl.DataFrame:
    df = rk.board(2026)
    if not df.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return df


@pytest.fixture(scope="module")
def preds() -> pl.DataFrame:
    return rk.fit()


def test_team_codes_are_normalized() -> None:
    """draft_picks is a PFR feed; eight of its team codes differ from nflverse's.

    Left unmapped, the landing-spot join drops a quarter of the league to null
    and those rookies get a league-median opportunity imputed — the failure is
    invisible in every output except the one that matters.
    """
    picks = nv.draft_picks().filter(pl.col("season") >= 2020)
    raw = set(picks.get_column("team").unique().to_list())
    assert {"GNB", "KAN", "LVR", "NOR", "NWE", "SFO", "TAM", "LAR"} <= raw

    mapped = set(
        picks.select(ids.normalize_team("team")).get_column("team").unique().to_list()
    )
    assert not (mapped & {"GNB", "KAN", "LVR", "NOR", "NWE", "SFO", "TAM", "LAR"})

    # Both sides normalized, which is the rule this function exists to enforce.
    # The 2026 roster feed is itself non-standard — it spells Arizona `AZ` where
    # every other nflverse table says `ARI` — so comparing mapped codes against
    # a *raw* roster was testing half the join and passed only for as long as
    # nflverse happened to agree with itself.
    roster_teams = set(
        nv.rosters(2026)
        .select(ids.normalize_team("team"))
        .get_column("team")
        .unique()
        .to_list()
    )
    assert mapped <= roster_teams, f"still unmatched: {sorted(mapped - roster_teams)}"


def test_landing_spot_data_is_complete(board_2026: pl.DataFrame) -> None:
    """Every rookie on the board must have real vacated opportunity, not an imputed one."""
    assert board_2026.height > 50
    for col in ("vacated_target_share", "vacated_carry_share"):
        assert board_2026.get_column(col).is_null().sum() == 0, f"{col} has nulls"


def test_combine_joins_for_the_incoming_class() -> None:
    """The incoming class has a null draft_year, so the combine joins on season.

    Filtering on draft_year returns zero rows for the class being evaluated —
    the athletic features vanish for exactly the players the board is about.
    """
    combine = nv.combine()
    assert combine.filter(pl.col("draft_year") == 2026).height == 0
    assert combine.filter(pl.col("season") == 2026).height > 200

    cls = rk.rookie_class(2026)
    assert "wt" in cls.columns
    assert cls.get_column("wt").is_not_null().sum() > 20


def test_model_beats_predicting_the_mean(preds: pl.DataFrame) -> None:
    """The bar any model must clear. Below it, draft capital explains nothing."""
    perf = rk.performance(preds)
    overall = perf.filter(pl.col("scope") == "overall")
    assert overall.height == 1
    assert overall.get_column("mae")[0] < overall.get_column("baseline_mae")[0]
    assert overall.get_column("corr")[0] > 0.3


def test_predictions_are_out_of_sample(preds: pl.DataFrame) -> None:
    """Leave-one-season-out: every rookie scored by a model blind to his class."""
    assert preds.height > 300
    assert preds.get_column("season").n_unique() >= 4


def test_draft_capital_dominates() -> None:
    """Reported so the app does not imply the combine numbers carry the model."""
    coefs = rk.coefficients()
    assert coefs.height > 0
    assert coefs.get_column("feature")[0] == "draft_ovr"


def test_players_who_never_appeared_are_kept() -> None:
    """Never playing is the most common rookie outcome and must count as zero.

    Note the target is not floored at zero for players who *did* appear: a rookie
    quarterback can post a negative per-game average under -2 per interception,
    and that is a real outcome rather than a data error.
    """
    frame = rk._train_frame()
    assert frame.height > 300

    never = frame.filter(pl.col("games") == 0)
    assert never.height > 20
    assert (never.get_column("ppg") == 0.0).all()
    assert (never.get_column("fantasy_points") == 0.0).all()

    # Negatives only ever come from players who actually played.
    assert (frame.filter(pl.col("ppg") < 0).get_column("games") > 0).all()


def test_rookies_are_not_in_the_veteran_feature_table() -> None:
    """Kept visibly separate, per the design. A rookie has no prior-season usage."""
    from src import features as ft

    veterans = ft.build()
    rookies_2025 = rk.rookie_class(2025).get_column("gsis_id").to_list()
    overlap = veterans.filter(
        (pl.col("season") == 2024) & pl.col("player_id").is_in(rookies_2025)
    )
    assert overlap.height == 0, "a 2025 rookie has a 2024 feature row"
