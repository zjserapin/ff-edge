"""Does a metric mean the same thing next year?

The check that should gate every feature in the project, and the one it was
missing. A metric with no year-over-year persistence is describing what happened,
not what a player is — and a board built on it ranks last season's variance.

The test is deliberately blunt: rank every qualified player within his position
and season, pair each player-season with his own next one, and correlate the two
percentiles. No model, no label, no outcome column. That last part matters —
selecting features by this number is not the "search until something scores well"
trap that `breakout.model_features` warns about, because `beat_adp` never enters.
A feature can be chosen here and still fail out of sample, which is the point.

Percentiles rather than raw values, because the league moves. Passing volume,
route rates and yards per attempt all drift year to year, and a raw correlation
would mix "this player stayed good" with "the league got more pass-happy". Rank
within season removes the league trend and leaves the player.

**What it found, and it reorganized the feature sets.**

Opportunity is stickier than quality at every position — the premise the project
was built on, now with a number attached. The quality column below is as it stood
*before* this check pruned the feature sets; re-running now returns higher
numbers (WR 0.442, RB 0.278, TE 0.400) precisely because the columns that dragged
it down were removed. Both are worth seeing: the first is what motivated the
change, the second is what shipped.

    position   opportunity   quality (before pruning)
    WR            0.549       0.382
    RB            0.526       0.244
    TE            0.510       0.296
    QB            0.472       0.394

Several columns that had been sitting in the quality sets turned out to be noise
dressed as skill:

    contested_catch_rate   WR  0.061   TE -0.118   dropped
    drop_rate              WR  0.096   TE  0.183   dropped
    catch_rate             RB  0.046               dropped at RB
    ryoe_per_att           RB  0.202               kept, demoted

`ryoe_per_att` is the interesting failure. Rush yards over expected is the metric
public analysis reaches for as the running back equivalent of yards per route
run, and it was added here on that reasoning. It persists at 0.20 — better than
yards per carry's 0.28 would suggest is possible for a rushing-efficiency
measure, but well below the receiving columns at the same position (`tprr` 0.40,
`yprr` 0.36). The honest reading is that what persists about a running back is
mostly *whether his offense throws to him*, not what he does with a handoff.

`contested_catch_rate` is the cleanest negative in the project. It is charted by
hand, it sounds like exactly the sort of thing that separates good receivers from
average ones, and it has a year-over-year correlation of 0.06 at receiver and a
*negative* one at tight end on 101 pairs. Contested-catch rate is a coin flip
with a small sample, and adding it to a distance metric adds noise with a
respectable name.
"""

from __future__ import annotations

import polars as pl

from src import features as ft
from src.config import FANTASY_POSITIONS

# Volume floors for inclusion, per position. A rate on twelve targets is not a
# measurement, and pairing two of them measures nothing twice.
MIN_VOLUME: dict[str, tuple[str, int]] = {
    "QB": ("pass_attempts", 150),
    "RB": ("carries", 60),
    "WR": ("targets", 40),
    "TE": ("targets", 30),
}

# Below this, a column is describing the season rather than the player. Not a
# law of nature — it is roughly where `drop_rate` and `contested_catch_rate` sit,
# and those are the two the eye test also rejects.
NOISE_FLOOR = 0.20

# Fewer pairs than this and the correlation itself is the noisy measurement.
MIN_PAIRS = 30


def _percentile(column: str) -> pl.Expr:
    """Within-season rank as a 0-1 percentile."""
    return pl.col(column).rank("average").over("season") / pl.len().over("season")


def year_over_year(
    df: pl.DataFrame | None = None,
    positions: tuple[str, ...] = FANTASY_POSITIONS,
    columns: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Percentile correlation between consecutive seasons, per position and metric.

    Pairs are formed on the player, not the row: a player-season joins to *his
    own* next season, so a player who missed a year contributes no pair rather
    than a spurious one across the gap.

    Returns: position, axis, metric, n_pairs, r_yoy, verdict.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for position in positions:
        volume_col, floor = MIN_VOLUME.get(position, ("games", 8))
        if volume_col not in base.columns:
            continue
        pool = base.filter(
            (pl.col("position") == position) & (pl.col(volume_col) >= floor)
        )
        if pool.height < MIN_PAIRS:
            continue

        quality = ft.quality_features(position)
        opportunity = ft.opportunity_features(position)
        axis = {c: "quality" for c in quality}
        axis.update({c: "opportunity" for c in opportunity})
        outcomes = ["ppg", "exp_ppg", "pts_over_exp_per_game"]

        wanted = (
            columns.get(position, []) if columns else quality + opportunity + outcomes
        )
        cols = [c for c in wanted if c in pool.columns]
        if not cols:
            continue

        ranked = pool.with_columns([_percentile(c).alias(f"p_{c}") for c in cols])
        # Shifting the *season* on the copy rather than using a window shift is
        # what makes the join land on the same player's next year regardless of
        # how the frame is sorted or whether he changed teams.
        following = ranked.select(
            ["player_id", (pl.col("season") - 1).alias("season")]
            + [pl.col(f"p_{c}").alias(f"n_{c}") for c in cols]
        )
        paired = ranked.join(following, on=["player_id", "season"], how="inner")

        for col in cols:
            sub = paired.select(f"p_{col}", f"n_{col}").drop_nulls()
            if sub.height < MIN_PAIRS:
                continue
            r = sub.select(pl.corr(f"p_{col}", f"n_{col}")).item()
            if r is None:
                continue
            rows.append(
                {
                    "position": position,
                    "axis": axis.get(col, "outcome"),
                    "metric": col,
                    "n_pairs": sub.height,
                    "r_yoy": round(float(r), 3),
                    "verdict": (
                        "sticky" if r >= 0.45
                        else "usable" if r >= NOISE_FLOOR
                        else "noise"
                    ),
                }
            )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["position", "r_yoy"], descending=[False, True])


def axis_summary(stability: pl.DataFrame) -> pl.DataFrame:
    """Mean persistence per position and axis — the one-line version of the finding.

    Returns: position, axis, metrics, mean_r.
    """
    if not stability.height:
        return pl.DataFrame()
    return (
        stability.group_by(["position", "axis"])
        .agg(
            pl.len().alias("metrics"),
            pl.col("r_yoy").mean().round(3).alias("mean_r"),
        )
        .sort(["axis", "mean_r"], descending=[False, True])
    )


def noisy_features(stability: pl.DataFrame) -> pl.DataFrame:
    """Columns below the noise floor, which should not be in any distance metric.

    Kept as a function rather than a hardcoded list so that re-running on a wider
    window can promote or demote a column without anyone editing a constant and
    forgetting why it was there.

    Returns: position, axis, metric, n_pairs, r_yoy.
    """
    if not stability.height:
        return pl.DataFrame()
    return stability.filter(pl.col("r_yoy") < NOISE_FLOOR).select(
        "position", "axis", "metric", "n_pairs", "r_yoy"
    )
