"""Feature table and clustering.

Weighted toward the four silent-failure joins. Each of them returns a
plausible-looking frame when it goes wrong — a dtype mismatch that matches
nothing, a filter on a value that doesn't exist, an id column that lives in a
different namespace — so the tests assert the join actually landed rather than
that the code ran.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import archetypes as ar
from src import features as ft


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    out = ft.player_seasons()
    if not out.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return out


# --- the four join landmines ------------------------------------------------


def test_opportunity_join_landed(df: pl.DataFrame) -> None:
    """ff_opportunity's season is a String and week a Float64.

    Cast either wrong and the join matches nothing, leaving a frame with the
    right columns and no rows — or worse, a partial match that looks fine.
    """
    assert df.height > 2_000
    seasons = set(df.get_column("season").unique().to_list())
    assert len(seasons) >= 5, f"only got seasons {sorted(seasons)}"
    # Usage shares are the whole point; they must be populated, not null.
    assert df.get_column("target_share").is_not_null().mean() > 0.99


def test_no_unattributed_team_rows(df: pl.DataFrame) -> None:
    """7% of ff_opportunity rows have a null player_id and must never survive."""
    assert df.get_column("player_id").is_null().sum() == 0
    assert df.get_column("player_name").is_null().sum() == 0


def test_playoff_weeks_excluded() -> None:
    """ff_opportunity carries weeks 19-22 and has no season_type column.

    A player's season usage must not include playoff games, or the players on
    good teams get systematically inflated volume.
    """
    from src import nflverse as nv

    raw = nv.ff_opportunity(stat_type="weekly")
    assert raw.get_column("week").max() > 18, "source no longer has playoff weeks"

    opp = ft._opportunity([2024], {"rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0})
    # 18 regular-season weeks minus a bye is the ceiling for any single player.
    assert opp.get_column("opp_games").max() <= 17


def test_snap_bridge_landed(df: pl.DataFrame) -> None:
    """snap_counts has no gsis_id — only pfr_player_id — and game_type is REG/WC/DIV/CON/SB.

    A break in the crosswalk bridge, or filtering on the nonexistent "POST",
    shows up here as a coverage collapse rather than an error.
    """
    coverage = df.get_column("snap_pct").is_not_null().mean()
    assert coverage > 0.90, f"snap bridge coverage fell to {coverage:.1%}"
    assert df.get_column("snap_pct").max() <= 100.0


def test_nextgen_uses_season_aggregates(df: pl.DataFrame) -> None:
    """NGS coverage is qualified receivers only, so partial by nature.

    Asserting a band rather than a floor: near zero means the week==0 join
    broke, near one means weekly rows leaked in and got averaged.
    """
    wr = df.filter(pl.col("position") == "WR")
    coverage = wr.get_column("avg_separation").is_not_null().mean()
    assert 0.30 < coverage < 0.95, f"NGS coverage {coverage:.1%} is out of band"


# --- feature semantics ------------------------------------------------------


def test_volume_shares_are_bounded(df: pl.DataFrame) -> None:
    """Counting shares — targets, carries — are strictly within [0, 1]."""
    for col in ("target_share", "rush_share"):
        vals = df.get_column(col).drop_nulls()
        assert vals.min() >= 0.0, f"{col} went negative"
        assert vals.max() <= 1.0, f"{col} exceeded 1.0 — denominator is wrong"


def test_expected_points_share_may_be_slightly_negative(df: pl.DataFrame) -> None:
    """Expected points are signed, because interceptions and fumbles are.

    A backup quarterback with a handful of attempts and interception risk has
    negative expected points under -2/INT scoring, so his share of his own
    offense is negative. Three player-seasons in the window (Tim Boyle, Logan
    Woodside, Kyle Allen). Bounded tightly so a genuine denominator bug — which
    would produce large negatives or values above 1 — still fails.
    """
    vals = df.get_column("exp_pts_share").drop_nulls()
    assert vals.min() > -0.05, "negative beyond what interception risk explains"
    assert vals.max() <= 1.0


def test_air_yards_share_is_signed_by_design(df: pl.DataFrame) -> None:
    """Air yards are measured from the line of scrimmage, so they can be negative.

    A checkdown back finishes the season with negative air yards and therefore a
    negative share. This is the one share that must NOT be clipped at zero —
    doing so would make a screen-game specialist indistinguishable from a player
    who is never targeted. Bounded on both sides so a denominator bug still
    fails: no player commands more than half his team's air yards.
    """
    vals = df.get_column("air_yards_share").drop_nulls()
    assert vals.min() < 0.0, "expected some negative shares (checkdown backs)"
    assert vals.min() > -0.5, "negative beyond plausible — check the denominator"
    assert vals.max() <= 1.0

    negatives = df.filter(pl.col("air_yards_share") < 0)
    assert (negatives.get_column("adot") < 0).all(), "negative share without negative aDOT"


def test_features_are_rates_not_totals(df: pl.DataFrame) -> None:
    """No feature may correlate with games played the way a total would.

    A model fed totals learns "stayed healthy" and reports it as insight. Rates
    should be close to uncorrelated with availability.
    """
    for col in ("target_share", "snap_pct", "adot", "catch_rate"):
        sub = df.select(["games", col]).drop_nulls()
        r = sub.select(pl.corr("games", col)).item()
        assert abs(r) < 0.65, f"{col} correlates {r:.2f} with games — looks like a total"


def test_undrafted_is_encoded_not_null(df: pl.DataFrame) -> None:
    """Going undrafted is signal. Left null it gets imputed to a mid-round median.

    Verified against real UDFAs (Taysom Hill, James Robinson, J.D. McKissic).
    """
    assert df.get_column("undrafted").sum() > 100
    undrafted = df.filter(pl.col("undrafted"))
    assert (undrafted.get_column("draft_round") == 8).all()
    assert (undrafted.get_column("draft_pick") == 262).all()
    # And genuinely-drafted players keep their real capital.
    drafted = df.filter(~pl.col("undrafted") & pl.col("draft_round").is_not_null())
    assert drafted.get_column("draft_round").min() == 1


def test_outcome_columns_are_not_features() -> None:
    """fantasy_points and pos_rank are the target. Feeding them back in is a leak."""
    for scope in (None, "QB", "RB", "WR", "TE"):
        cols = set(ft.feature_columns(scope)) | set(ft.cluster_feature_columns(scope))
        assert not (cols & set(ft.OUTCOME_COLUMNS))


def test_cluster_features_exclude_production_and_booleans() -> None:
    """Clustering on points produces scoring tiers, not usage archetypes.

    And a standardized boolean spikes the distance metric: with `undrafted` in
    the set, QB's first split was Taysom Hill alone against the other 31.
    """
    for scope in (None, "QB", "RB", "WR", "TE"):
        cols = set(ft.cluster_feature_columns(scope))
        assert "exp_ppg" not in cols
        assert "pts_over_exp_per_game" not in cols
        assert "undrafted" not in cols
        assert "ppg" not in cols


# --- clustering -------------------------------------------------------------


@pytest.fixture(scope="module")
def clusters(df: pl.DataFrame) -> pl.DataFrame:
    return ar.cluster(2025, df=df)


def test_clustering_is_reproducible(df: pl.DataFrame, clusters: pl.DataFrame) -> None:
    """Same seed, same labels. Otherwise the app shows a different board on reload."""
    again = ar.cluster(2025, df=df)
    assert clusters.equals(again)


def test_no_singleton_clusters(clusters: pl.DataFrame) -> None:
    """A cluster of one is k-means quarantining an outlier, not an archetype.

    Guards the viability rule in choose_k. Without it, QB selects k=3 on a
    higher silhouette whose smallest cluster has a single member.
    """
    sizes = clusters.group_by(["position", "cluster"]).agg(pl.len().alias("n"))
    assert sizes.get_column("n").min() >= 4


def test_choose_k_flags_unviable_solutions(df: pl.DataFrame) -> None:
    """The viability flag must actually fire on this data, or it is untested."""
    pool = df.filter((pl.col("season") == 2025) & (pl.col("games") >= 8))
    qb = pool.filter(pl.col("position") == "QB")
    x, used = ar._matrix(qb, ft.cluster_feature_columns("QB"))
    assert used
    scores = ar.choose_k(x, (2, 6))
    assert not scores.get_column("viable").all(), "expected some k to be unviable at QB"
    # And the winner among viable ones must be viable.
    chosen_k = int(clusters_k(df))
    assert scores.filter(pl.col("k") == chosen_k).get_column("viable").all()


def clusters_k(df: pl.DataFrame) -> int:
    out = ar.cluster(2025, positions=("QB",), df=df)
    return int(out.get_column("k")[0])


def test_k_override_is_respected(df: pl.DataFrame) -> None:
    out = ar.cluster(2025, df=df, k=4)
    assert set(out.get_column("k").unique().to_list()) == {4}
    per_pos = out.group_by("position").agg(pl.col("cluster").n_unique().alias("n"))
    assert (per_pos.get_column("n") == 4).all()


def test_neighbors_are_same_position_and_exclude_self(
    df: pl.DataFrame, clusters: pl.DataFrame
) -> None:
    target = clusters.filter(pl.col("position") == "WR").sort("pos_rank")
    pid = target.get_column("player_id")[0]
    nb = ar.neighbors(pid, clusters, df, n=6, season=2025)

    assert nb.height == 6
    assert set(nb.get_column("position").unique().to_list()) == {"WR"}
    assert pid not in nb.get_column("player_id").to_list()
    # Distances are sorted ascending — nearest first is the whole contract.
    dists = nb.get_column("distance").to_list()
    assert dists == sorted(dists)


def test_cluster_profiles_label_every_cluster(
    df: pl.DataFrame, clusters: pl.DataFrame
) -> None:
    prof = ar.cluster_profiles(clusters, df, season=2025)
    expected = clusters.select("position", "cluster").unique().height
    assert prof.height == expected
    assert prof.get_column("label").is_not_null().all()
    assert (prof.get_column("n") > 0).all()
