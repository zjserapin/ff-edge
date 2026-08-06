"""The promotion screen: grade a player whose role is growing, by position rules.

Division of labor, and it is the load-bearing design decision: **nothing in the
data predicts who gets promoted.** Correlation with next-season role growth:
vacated targets -0.04, vacated carries -0.03, teammate's target share 0.07,
prior quality 0.02. The only nonzero term is prior opportunity at -0.33, which
is mean reversion, not foresight. So the user supplies the names — from camp
reporting, a depth chart, a departure — and this module does the half the data
can do: grade the player against what actually predicted production among
historically promoted players at his position.

Those criteria are position-specific, and at running back they are not what
intuition says. Among backups whose role then grew (below-median opportunity,
+10 percentile points or more the next season):

    predictor of next-season points   RB (n~72)   WR (n~122)   TE (n~67)
    snap share                          0.43        0.35         0.28
    red-zone carry share                0.41        —            —
    TD-equity share                     0.37        0.22         0.16
    yards per route run                -0.00        0.26         0.38
    yards per carry                    -0.02        —            —
    yards after contact / att           0.02        —            —
    broken tackles / att                0.04        —            —

Five independent efficiency metrics land within 0.04 of zero at RB. What
predicts a promoted back is whether the staff already gave him the *valuable*
touches — snaps, red-zone and goal-line carries. A backup with a shiny YPC
usually earned it in the fourth quarter against a light box; a backup with
red-zone carries is one the coaches already trusted. At WR and TE the
efficiency logic holds and yards per route run is the criterion.

Quality is a filter, not a picker. Promoted backups by prior quality tercile
hit a top-quartile finish at 19.7% (top 30%), 14.1% (middle), 4.8% (bottom
30%) — a monotonic 4x spread that is much better at ruling players out than
ranking them in. The screen therefore reports a base rate with its interval,
never a point projection.

Every number above is reproduced from the repo by `validate()` and
`quality_tiers()`, and the tests assert the shape of the finding (RB efficiency
null, WR/TE efficiency positive, monotone tiers) rather than trusting this
docstring to stay true.
"""

from __future__ import annotations

import polars as pl

from src import archetypes as ar
from src import breakout as bo
from src import features as ft
from src import ids
from src import nflverse as nv
from src import uncertainty as unc
from src.config import CURRENT_SEASON, FEATURE_SEASONS
from src.context import _scrimmage

# Cohort definition, from the exploratory session that found the criteria:
# below-median role, then a role at least this much larger the next season.
BACKUP_CEILING_PCT = 50.0
PROMOTION_PTS = 10.0

# A hit is a top-quartile points-per-game finish within position. Primary
# because it is defined for every promoted player; "returned ADP value" is
# carried alongside where an ADP exists, and is null for the many promoted
# players the market never drafted — dropping those rows would silently remove
# the most interesting ones.
HIT_PCT = 75.0

# What the screen grades, per position. Chosen from the measured table in the
# module docstring, never by searching this cohort for what scores well.
CRITERIA: dict[str, list[str]] = {
    "RB": ["snap_pct", "rz_carry_share", "gz_carry_share", "exp_td_share"],
    "WR": ["yprr", "snap_pct", "exp_td_share"],
    "TE": ["yprr", "snap_pct", "exp_td_share"],
}

# What `validate()` reports, criteria plus the efficiency metrics whose null
# result is the reason the RB criteria look the way they do. The null is part
# of the finding and stays visible next to the positives.
CANDIDATES: dict[str, list[str]] = {
    "RB": CRITERIA["RB"]
    + ["ypc", "ryoe_per_att", "yards_after_contact_per_att",
       "rush_broken_tackles_per_att", "yprr"],
    "WR": CRITERIA["WR"] + ["tprr", "rz_target_share", "ypt", "avg_separation"],
    "TE": CRITERIA["TE"] + ["tprr", "rz_target_share", "ypt"],
}

QUALITY_TIER_CUTS = (30.0, 70.0)  # bottom 30% / middle / top 30%


def history(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Quality/opportunity percentiles for every qualified player-season.

    `archetypes.scores` per season over the whole window, with the stability
    weights derived once from the full history rather than once per season —
    how much a metric repeats is a property of the metric, not of the year.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    weights = ar.stability_weights(base)
    parts = [
        ar.scores(season, df=base, weights=weights) for season in FEATURE_SEASONS
    ]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def cohort(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every historical promoted player-season, with prior features and outcome.

    Pairs each qualified player-season with his own next one; keeps rows where
    the prior role was below median and the next role grew by 10+ percentile
    points. The pairing itself conditions on qualifying in both seasons — a
    promotion cannot be measured for a player with no next season — so this
    answers "given a promotion, what predicts production", not "who survives".

    `next_ppg_pct` is the percentile of next-season points per game within
    position among that season's qualified pool; `hit` is the top quartile of
    it. `beat_adp` is carried where the next season had a draft price and is
    null where it did not.

    Returns one row per promoted player-season: identity, prior-season criteria
    and candidate metrics, prior quality/opportunity percentiles, next-season
    outcome columns.
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame()

    hist = history(base)
    if not hist.height:
        return pl.DataFrame()

    hist = hist.with_columns(
        (
            pl.col("ppg").rank("average").over(["season", "position"])
            / pl.len().over(["season", "position"]) * 100
        ).round(1).alias("ppg_pct")
    )

    nxt = hist.select(
        (pl.col("season") - 1).alias("season"),
        "player_id",
        pl.col("opportunity_pct").alias("next_opportunity_pct"),
        pl.col("ppg_pct").alias("next_ppg_pct"),
        pl.col("ppg").alias("next_ppg"),
        pl.col("season").alias("next_season"),
    )

    paired = hist.join(nxt, on=["season", "player_id"], how="inner").filter(
        (pl.col("opportunity_pct") < BACKUP_CEILING_PCT)
        & ((pl.col("next_opportunity_pct") - pl.col("opportunity_pct")) >= PROMOTION_PTS)
        & pl.col("position").is_in(list(CRITERIA))
    )
    if not paired.height:
        return pl.DataFrame()

    metric_cols = sorted({c for cols in CANDIDATES.values() for c in cols})
    feats = base.select(
        ["season", "player_id"]
        + [c for c in metric_cols + ["targets", "carries"] if c in base.columns]
    )

    labeled = bo.labels()
    price = (
        labeled.select(
            pl.col("season").alias("next_season"),
            pl.col("gsis_id").alias("player_id"),
            "beat_adp",
            "adp_pos_rank",
        )
        if labeled.height
        else None
    )

    out = paired.join(feats, on=["season", "player_id"], how="left").with_columns(
        (pl.col("next_ppg_pct") >= HIT_PCT).alias("hit"),
        (pl.col("next_opportunity_pct") - pl.col("opportunity_pct")).alias("opp_change"),
        pl.when(pl.col("quality_pct") < QUALITY_TIER_CUTS[0])
        .then(pl.lit("bottom 30%"))
        .when(pl.col("quality_pct") < QUALITY_TIER_CUTS[1])
        .then(pl.lit("middle"))
        .otherwise(pl.lit("top 30%"))
        .alias("quality_tier"),
    )
    if price is not None:
        out = out.join(price, on=["next_season", "player_id"], how="left")
    return out.sort(["position", "season", "player_name"])


def validate(coh: pl.DataFrame | None = None) -> pl.DataFrame:
    """The criteria table, recomputed — the docstring's numbers, not trusted.

    Rank correlation of each prior-season candidate metric against next-season
    points percentile, per position, n beside every value. The RB efficiency
    rows are the point: five metrics near zero is why the RB criteria are trust
    markers. `n` varies by metric because coverage does (NGS thresholds, FTN's
    2022 floor).
    """
    coh = coh if coh is not None else cohort()
    if not coh.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for position, cols in CANDIDATES.items():
        sub = coh.filter(pl.col("position") == position)
        for col in cols:
            if col not in sub.columns:
                continue
            pair = sub.select(col, "next_ppg_pct").drop_nulls()
            if pair.height < 20:
                continue
            r = pair.select(
                pl.corr(
                    pl.col(col).rank("average"),
                    pl.col("next_ppg_pct").rank("average"),
                )
            ).item()
            rows.append(
                {
                    "position": position,
                    "metric": col,
                    "r": round(float(r), 3) if r is not None else None,
                    "n": pair.height,
                    "criterion": col in CRITERIA[position],
                }
            )

    return pl.DataFrame(rows).sort(
        ["position", "criterion", pl.col("r").abs()], descending=[False, True, True]
    )


def quality_tiers(coh: pl.DataFrame | None = None) -> pl.DataFrame:
    """Hit rate by prior-quality tercile — the filter the screen leans on.

    Overall and per position, Wilson intervals throughout. The per-position
    cells are thin (a tercile of an n=72 cohort is ~24 players) and the n
    column says so; the overall rows are the ones with enough sample to act on.
    """
    coh = coh if coh is not None else cohort()
    if not coh.height:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []

    def add(scope: str, tier: str, sub: pl.DataFrame) -> None:
        n = sub.height
        hits = int(sub.get_column("hit").sum())
        lo, hi = unc.wilson_interval(hits, n)
        rows.append(
            {
                "scope": scope, "quality_tier": tier, "n": n, "hits": hits,
                "hit_rate": round(hits / n, 3) if n else 0.0,
                "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            }
        )

    tiers = ["bottom 30%", "middle", "top 30%"]
    for tier in tiers:
        add("all", tier, coh.filter(pl.col("quality_tier") == tier))
    for position in sorted(coh.get_column("position").unique().to_list()):
        for tier in tiers:
            add(
                position,
                tier,
                coh.filter(
                    (pl.col("position") == position) & (pl.col("quality_tier") == tier)
                ),
            )
    return pl.DataFrame(rows)


def _resolve(names: list[str], pool: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Match free-typed names against the pool. Exact normalized first, then contains."""
    wanted = pl.DataFrame({"raw": [n.strip() for n in names if n.strip()]}).with_columns(
        ids.normalize("raw").alias("_norm")
    )
    pooled = pool.with_columns(ids.normalize("player_name").alias("_norm"))

    exact = wanted.join(pooled, on="_norm", how="inner")
    missing = wanted.filter(
        ~pl.col("_norm").is_in(exact.get_column("_norm").implode())
    ).get_column("raw").to_list()

    parts = [exact.drop("_norm")]
    still_missing: list[str] = []
    for raw in missing:
        got = pooled.filter(
            pl.col("player_name").str.to_lowercase().str.contains(raw.lower(), literal=True)
        )
        if got.height:
            parts.append(got.drop("_norm").with_columns(pl.lit(raw).alias("raw")))
        else:
            still_missing.append(raw)

    matched = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    return matched, still_missing


def screen(
    names: list[str],
    season: int = CURRENT_SEASON,
    df: pl.DataFrame | None = None,
    coh: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Grade the players the user says are getting a bigger role.

    For each resolved name: his percentile on each of his position's criteria
    (within position, this season, qualified players), the mean of those
    percentiles as `screen_pct`, his quality tercile, and the base rate that
    tercile earned among historical promoted players — with its interval and
    its n, because the base rate *is* the answer. A point projection would
    claim precision the cohort cannot support.

    Efficiency percentiles are included for RBs even though they are not
    criteria, labeled by `criteria_only` columns downstream — the app shows
    them greyed as "reference", because hiding them entirely invites re-adding
    them from memory.

    Returns (grades, unmatched_names).
    """
    base = df if df is not None else ft.build()
    if not base.height:
        return pl.DataFrame(), list(names)

    pool = base.filter(
        (pl.col("season") == season)
        & pl.col("position").is_in(list(CRITERIA))
        & (pl.col("games") >= 4)
    )
    metric_cols = sorted({c for cols in CANDIDATES.values() for c in cols})
    pool = pool.with_columns(
        [
            (
                pl.col(c).rank("average").over("position")
                / pl.col(c).is_not_null().sum().over("position") * 100
            ).round(0).alias(f"{c}_pct")
            for c in metric_cols
            if c in pool.columns
        ]
    )

    hist = history(base).filter(pl.col("season") == season).select(
        "player_id", "quality_pct", "opportunity_pct"
    )
    pool = pool.join(hist, on="player_id", how="left")

    matched, unmatched = _resolve(names, pool)
    if not matched.height:
        return pl.DataFrame(), unmatched

    tiers = quality_tiers(coh if coh is not None else cohort())

    rows: list[dict[str, object]] = []
    for r in matched.iter_rows(named=True):
        position = r["position"]
        crit_pcts = [
            r.get(f"{c}_pct") for c in CRITERIA[position] if r.get(f"{c}_pct") is not None
        ]
        quality_pct = r.get("quality_pct")
        tier = (
            None if quality_pct is None
            else "bottom 30%" if quality_pct < QUALITY_TIER_CUTS[0]
            else "middle" if quality_pct < QUALITY_TIER_CUTS[1]
            else "top 30%"
        )
        base_rate = (
            tiers.filter((pl.col("scope") == "all") & (pl.col("quality_tier") == tier))
            if tier and tiers.height
            else pl.DataFrame()
        )

        row: dict[str, object] = {
            "player_name": r["player_name"],
            "position": position,
            "team": r.get("team"),
            "games": r.get("games"),
            "screen_pct": round(sum(crit_pcts) / len(crit_pcts), 1) if crit_pcts else None,
            "quality_pct": quality_pct,
            "opportunity_pct": r.get("opportunity_pct"),
            "quality_tier": tier,
            "tier_hit_rate": base_rate.get_column("hit_rate")[0] if base_rate.height else None,
            "tier_ci_lo": base_rate.get_column("ci_lo")[0] if base_rate.height else None,
            "tier_ci_hi": base_rate.get_column("ci_hi")[0] if base_rate.height else None,
            "tier_n": base_rate.get_column("n")[0] if base_rate.height else None,
        }
        for c in CANDIDATES[position]:
            row[f"{c}_pct"] = r.get(f"{c}_pct")
        rows.append(row)

    return (
        pl.DataFrame(rows).sort("screen_pct", descending=True, nulls_last=True),
        unmatched,
    )


def archetype_split(coh: pl.DataFrame | None = None) -> pl.DataFrame:
    """Do light, pass-catching backs hit less often when promoted? Measured.

    The decay-rate hypothesis, reduced to the one split this cohort can
    support: promoted RBs above vs below the median prior pass-touch mix
    (targets over targets-plus-carries). A median split of ~72 players is two
    cells of ~36 — anything finer is astrology, which is why this is one split
    and not a feature.

    Returns: group, n, hits, hit_rate, ci_lo, ci_hi, mean_next_ppg_pct.
    """
    coh = coh if coh is not None else cohort()
    if not coh.height:
        return pl.DataFrame()

    rb = coh.filter(
        (pl.col("position") == "RB")
        & pl.col("targets").is_not_null()
        & pl.col("carries").is_not_null()
        & ((pl.col("targets") + pl.col("carries")) > 0)
    ).with_columns(
        (pl.col("targets") / (pl.col("targets") + pl.col("carries"))).alias("pass_mix")
    )
    if rb.height < 20:
        return pl.DataFrame()

    median = float(rb.get_column("pass_mix").median())
    rows = []
    for group, sub in (
        ("receiving-profile (above-median pass mix)", rb.filter(pl.col("pass_mix") > median)),
        ("rushing-profile (at/below-median pass mix)", rb.filter(pl.col("pass_mix") <= median)),
    ):
        hits = int(sub.get_column("hit").sum())
        lo, hi = unc.wilson_interval(hits, sub.height)
        rows.append(
            {
                "group": group,
                "n": sub.height,
                "hits": hits,
                "hit_rate": round(hits / sub.height, 3) if sub.height else 0.0,
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "mean_next_ppg_pct": round(
                    float(sub.get_column("next_ppg_pct").mean()), 1
                ),
            }
        )
    return pl.DataFrame(rows)


def weekly_trust(season: int = CURRENT_SEASON, force: bool = False) -> pl.DataFrame:
    """Week-by-week trust markers, so a December role change is not averaged away.

    The season-aggregate features hide within-season trend: a back whose
    red-zone share doubled after Thanksgiving looks identical to one who held a
    flat middling share all year. This is the frame the screen plots to tell
    them apart — per player-week, his share of that week's team carries,
    valuable (red-zone + goal-line) carries, and targets.

    Week-level shares are noisy by construction (a team may have two red-zone
    carries in a week); the app draws them with a rolling mean and this
    function stays raw so the smoothing choice is visible where it is made.

    Returns: season, week, player_id, carry_share_wk, rz_carry_share_wk,
    target_share_wk — null where the team had no such plays that week.
    """
    from src.context import RED_ZONE

    def _weekly(
        raw: pl.DataFrame, player_key: str, parts: list[pl.Expr]
    ) -> pl.DataFrame:
        plays = _scrimmage(raw).filter(pl.col(player_key).is_not_null())
        if not plays.height:
            return pl.DataFrame()
        names = [e.meta.output_name() for e in parts]
        per = plays.group_by(["season", "week", "posteam", player_key]).agg(parts)
        team = (
            plays.group_by(["season", "week", "posteam"])
            .agg(parts)
            .rename({n: f"team_{n}" for n in names})
        )
        return (
            per.join(team, on=["season", "week", "posteam"], how="left")
            .rename({player_key: "player_id"})
        )

    rush_raw = nv.ff_opportunity([season], stat_type="pbp_rush", force=force)
    pass_raw = nv.ff_opportunity([season], stat_type="pbp_pass", force=force)
    if not rush_raw.height and not pass_raw.height:
        return pl.DataFrame()

    out: pl.DataFrame | None = None
    if rush_raw.height:
        rush = _weekly(
            rush_raw,
            "rusher_player_id",
            [
                pl.len().cast(pl.Float64).alias("att"),
                (pl.col("yardline_100") <= RED_ZONE).sum().cast(pl.Float64).alias("rz_att"),
            ],
        ).select(
            "season", "week", "player_id",
            (pl.col("att") / pl.col("team_att")).alias("carry_share_wk"),
            pl.when(pl.col("team_rz_att") > 0)
            .then(pl.col("rz_att") / pl.col("team_rz_att"))
            .otherwise(None)
            .alias("rz_carry_share_wk"),
        )
        out = rush

    if pass_raw.height:
        tgt = _weekly(
            pass_raw,
            "receiver_player_id",
            [pl.len().cast(pl.Float64).alias("tgt")],
        ).select(
            "season", "week", "player_id",
            (pl.col("tgt") / pl.col("team_tgt")).alias("target_share_wk"),
        )
        out = tgt if out is None else out.join(
            tgt, on=["season", "week", "player_id"], how="full", coalesce=True
        )

    return out.sort(["player_id", "week"]) if out is not None else pl.DataFrame()
