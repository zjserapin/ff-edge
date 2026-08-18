"""Four worked examples over the cached data.

These are not models and they aren't rankings. They exist to (a) prove the joins
actually hold on real data and (b) give you something to poke at on day one.
Each returns a polars frame, so any of them is a starting point for a notebook
rather than a dead end.

    uv run python -m src.peek
"""

from __future__ import annotations

import polars as pl

from src import nflverse as nv
from src.config import OUTPUT_DIR, SEASON

# The last season with completed games. Every screen here reads *finished*
# production, so it trails `SEASON` by one — and it is derived rather than
# typed, because these four defaults said `2025` while `config.SEASON` said
# 2026 and nothing would have raised in 2027 either. `config.py` is the
# source of truth for a season; a module-level literal is how that stops
# being true one August at a time.
LAST_PLAYED = SEASON - 1


def regression_candidates(season: int = LAST_PLAYED, min_games: int = 10) -> pl.DataFrame:
    """Players whose actual production diverged most from their opportunity.

    Large *negative* pts_over_exp: the volume was there and the points weren't.
    Usually the better buy, because volume is much stickier season to season than
    efficiency — the touches tend to come back, and the finishing luck reverts.

    Large *positive*: the player outran their opportunity, typically on touchdown
    variance. That's real production you already paid for and probably won't get
    again at the same rate.

    Treat this as a question generator, not an answer. A back who underperformed
    expectation because he's genuinely bad at football will keep doing that; the
    table can't tell you which is which, it can only tell you where to look.
    """
    opp = nv.ff_opportunity(stat_type="weekly").filter(
        (pl.col("season").cast(pl.Int64) == season)
        # 7% of rows are unattributed team plays with a null player_id. They all
        # collapse into a single group and land at the top of the sort — the
        # "player" with 423 games and -1274 points over expected.
        & pl.col("player_id").is_not_null()
        # This table has no season_type column and does carry weeks 19-22.
        # Without the cap, playoff production folds into the season total.
        & (pl.col("week") <= 18)
    )

    return (
        opp.group_by(["player_id", "full_name", "position"])
        .agg(
            pl.len().alias("games"),
            pl.col("total_fantasy_points_exp").sum().round(1).alias("exp_pts"),
            pl.col("total_fantasy_points").sum().round(1).alias("act_pts"),
        )
        .filter(pl.col("games") >= min_games)
        .with_columns(
            (pl.col("act_pts") - pl.col("exp_pts")).round(1).alias("pts_over_exp"),
            (pl.col("exp_pts") / pl.col("games")).round(2).alias("exp_ppg"),
        )
        .sort("pts_over_exp")
    )


def usage_leaders(
    season: int = LAST_PLAYED, position: str = "WR", min_games: int = 8
) -> pl.DataFrame:
    """Who commands their offense — target share and air yards share.

    Target share is the piece of the passing game a player owns; air yards share
    is how far downfield that ownership sits. High target share with low air
    yards share is a volume-dependent profile (screens, checkdowns) that collapses
    if the offense changes. Both high is a genuine alpha.
    """
    wk = nv.weekly_stats().filter(
        (pl.col("season") == season)
        & (pl.col("season_type") == "REG")
        & (pl.col("position") == position)
    )

    return (
        wk.group_by(["player_id", "player_display_name", "team"])
        .agg(
            pl.len().alias("games"),
            pl.col("targets").sum().alias("targets"),
            pl.col("target_share").mean().round(3).alias("tgt_share"),
            pl.col("air_yards_share").mean().round(3).alias("ay_share"),
            pl.col("fantasy_points_ppr").mean().round(2).alias("ppg"),
        )
        .filter(pl.col("games") >= min_games)
        .sort("tgt_share", descending=True)
    )


def snap_trend(season: int = LAST_PLAYED, position: str = "RB", last_n: int = 4) -> pl.DataFrame:
    """Late-season snap share minus full-season baseline.

    This is the shape of the in-season signal worth tracking weekly: a role
    change shows up in snaps a week or two before it shows up in the box score,
    because a player has to be on the field before he can be productive. Positive
    delta going into an offseason is the cheapest version of "the coaches like
    him now" you can get.
    """
    snaps = nv.snap_counts().filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("position") == position)
    )
    if snaps.height == 0:
        return snaps

    cutoff = snaps.get_column("week").max() - last_n

    baseline = snaps.group_by(["pfr_player_id", "player", "team"]).agg(
        pl.len().alias("games"),
        pl.col("offense_pct").mean().round(3).alias("season_snap_pct"),
    )
    recent = (
        snaps.filter(pl.col("week") > cutoff)
        .group_by("pfr_player_id")
        .agg(pl.col("offense_pct").mean().round(3).alias(f"last{last_n}_snap_pct"))
    )

    return (
        baseline.join(recent, on="pfr_player_id", how="inner")
        .with_columns(
            (pl.col(f"last{last_n}_snap_pct") - pl.col("season_snap_pct"))
            .round(3)
            .alias("snap_delta")
        )
        .filter(pl.col("games") >= 6)
        .sort("snap_delta", descending=True)
    )


def market_disagreement(top_n: int = 250) -> pl.DataFrame:
    """Where the expert consensus is least settled.

    FantasyPros ships `sd`, `best`, and `worst` next to `ecr`, so disagreement is
    free to measure. Wide dispersion means the argument is still in progress,
    which is where your own read is actually worth something — a tight consensus
    is already priced into ADP.

    Two biases, and you have to correct for both — neither column is safe alone:

      - Raw `sd` scales with rank. A player ranked 180th can carry an sd of 40
        simply because there's more room to be wrong down there. Sorting on `sd`
        surfaces deep-league fliers, not names you'd actually draft.
      - `rel_sd` (sd / ecr) over-corrects in the other direction. Dividing by a
        tiny ecr inflates the top of the board, so a naive rel_sd sort puts
        Ja'Marr Chase at #1 on an sd of ~1. Verified on the real 2026 file.

    Neither sort is the answer. Filter to an ECR band you'd actually draft in
    (say 25-100), or rank sd *within tier*, and read the residual. The default
    sort here is rel_sd because it makes the bias visible rather than hiding it.
    """
    ecr = nv.ff_rankings()

    return (
        ecr.filter(pl.col("ecr_type") == "ro")  # redraft, overall board
        .with_columns(
            # pos arrives as WR1 / RB2 — the rank is already in `ecr`.
            pl.col("pos").str.replace_all(r"\d+", "").alias("pos")
        )
        .filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
        .filter(pl.col("ecr") <= top_n)
        .with_columns(
            (pl.col("worst") - pl.col("best")).alias("range"),
            (pl.col("sd") / pl.col("ecr")).round(3).alias("rel_sd"),
        )
        .select(["player", "pos", "team", "ecr", "sd", "best", "worst", "range", "rel_sd", "bye"])
        .sort("rel_sd", descending=True)
    )


if __name__ == "__main__":
    pl.Config.set_tbl_rows(15)

    print(f"\n=== Negative regression candidates ({LAST_PLAYED}, volume > points) ===")
    reg = regression_candidates()
    print(reg.head(15))
    print("\n--- other direction (outran their opportunity) ---")
    print(reg.tail(10))

    print(f"\n=== WR usage leaders ({LAST_PLAYED}) ===")
    usage = usage_leaders()
    print(usage.head(15))

    print(f"\n=== RB snap trend (last 4 weeks vs season, {LAST_PLAYED}) ===")
    print(snap_trend().head(15))

    print("\n=== Market disagreement (2026 redraft ECR) ===")
    disagree = market_disagreement()
    print(disagree.head(15))

    reg.write_csv(OUTPUT_DIR / "regression_candidates.csv")
    disagree.write_csv(OUTPUT_DIR / "market_disagreement.csv")
    print(f"\nwrote {OUTPUT_DIR}/regression_candidates.csv and market_disagreement.csv")
