"""Rookies — a separate model, deliberately kept apart from the veterans.

A rookie has no prior-season usage, which is the entire input to every other
model in this project. Imputing it would mean inventing the one thing that
matters; putting rookies in the veteran clusters would mean grouping them by
imputed values and then reading meaning into the grouping. So they get their own
model, their own features, and their own section of the app.

What is actually available before a rookie plays a down:

  draft capital     where the NFL took him, which is the league's own aggregated
                    scouting opinion and historically the strongest single
                    predictor of fantasy production
  combine           athletic testing, partially observed — plenty of players
                    skip drills, and that is not missing at random
  landing spot      the opportunity his new team just vacated, which is the
                    closest thing to a usage projection that exists in advance
  age               a 21-year-old declaring early is a different prospect from a
                    23-year-old senior with the same tape

n is small — roughly 40-50 drafted skill rookies a season — so this is a ridge
regression with leave-one-season-out validation and nothing fancier.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import ids
from src import nflverse as nv
from src import scoring as sc
from src.config import (
    DEFAULT_SCORING,
    FANTASY_POSITIONS,
    FEATURE_SEASONS,
    REGULAR_SEASON_WEEKS,
    SEASON,
)

ROOKIE_FEATURES = [
    "draft_ovr",
    "draft_round",
    "age_at_draft",
    "wt",
    "forty",
    "vacated_target_share",
    "vacated_carry_share",
    "vacated_exp_points",
]


def rookie_class(season: int, force: bool = False) -> pl.DataFrame:
    """Drafted skill players entering the league in `season`, with combine data.

    `draft_picks` carries gsis_id directly and `combine` carries pfr_id, and
    draft_picks has both — so these join to each other without going through the
    crosswalk at all.

    Returns: season, gsis_id, name, position, team, draft_round, draft_pick,
    draft_ovr, age_at_draft, ht, wt, forty, vertical, broad_jump, cone, shuttle.
    """
    picks = nv.draft_picks(force=force).filter(
        (pl.col("season") == season) & pl.col("position").is_in(list(FANTASY_POSITIONS))
    )
    if not picks.height:
        return pl.DataFrame()

    # Join on the combine's `season`, not its `draft_year`. The combine is held
    # in February of the draft year, so they agree for every past class — but
    # for the incoming class `draft_year` is null (the file is scraped before
    # the draft), and filtering on it silently returns zero rows and drops every
    # athletic feature for exactly the class you are trying to evaluate.
    combine = nv.combine(force=force).filter(pl.col("season") == season)
    combine_cols = [
        c for c in ("pfr_id", "ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle")
        if c in combine.columns
    ]

    out = picks.select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        "gsis_id",
        pl.col("pfr_player_name").alias("name"),
        "position",
        # draft_picks is a PFR feed and uses PFR team codes. Eight of them differ
        # from the ones rosters and weekly stats use, so without this the landing
        # -spot join drops a quarter of the league to null.
        ids.normalize_team("team"),
        pl.col("round").cast(pl.Int32).alias("draft_round"),
        pl.col("pick").cast(pl.Int32).alias("draft_ovr"),
        pl.col("age").cast(pl.Float64).alias("age_at_draft"),
        "pfr_player_id",
    )

    if combine_cols and combine.height:
        out = out.join(
            combine.select(combine_cols).unique(subset=["pfr_id"], keep="first"),
            left_on="pfr_player_id",
            right_on="pfr_id",
            how="left",
        )

    return out.drop_nulls("gsis_id").unique(subset=["gsis_id"], keep="first")


def vacated_opportunity(
    season: int,
    scoring: Mapping[str, float] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Volume a team's departed players leave behind, by position group.

    Prior-season production from players who are no longer on the roster. This
    is the only forward-looking opportunity signal available for someone who has
    never played, and it is why a third-round back landing on a team that just
    lost 300 carries is a different bet from the same player landing behind an
    incumbent.

    Documented caveat: a historical roster file is an end-of-season snapshot, so
    "who is still here" is approximate for past seasons. For the live season it
    is the current roster, which is exactly right.

    Returns: season, team, vacated_targets, vacated_carries, vacated_exp_points,
    vacated_target_share, vacated_carry_share.
    """
    scoring = scoring or DEFAULT_SCORING
    prior = season - 1

    weekly = sc.score_weekly([prior], scoring, force=force)
    if not weekly.height:
        return pl.DataFrame()

    raw = nv.weekly_stats([prior], force=force).filter(
        pl.col("position").is_in(list(FANTASY_POSITIONS)) & (pl.col("season_type") == "REG")
    )
    produced = raw.group_by(["team", "player_id"]).agg(
        pl.col("targets").fill_null(0).sum().alias("targets"),
        pl.col("carries").fill_null(0).sum().alias("carries"),
    )
    points = weekly.group_by(["team", "player_id"]).agg(
        pl.col("fantasy_points").sum().alias("points")
    )
    produced = produced.join(points, on=["team", "player_id"], how="left")

    roster = nv.rosters(season, force=force)
    if not roster.height:
        return pl.DataFrame()
    staying = roster.select(pl.col("gsis_id").alias("player_id"), pl.col("team").alias("new_team")).drop_nulls()

    tagged = produced.join(staying, on="player_id", how="left").with_columns(
        (pl.col("new_team") != pl.col("team")).fill_null(True).alias("gone")
    )

    return (
        tagged.group_by("team")
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.col("carries").sum().alias("team_carries"),
            pl.col("targets").filter(pl.col("gone")).sum().alias("vacated_targets"),
            pl.col("carries").filter(pl.col("gone")).sum().alias("vacated_carries"),
            pl.col("points").filter(pl.col("gone")).sum().alias("vacated_exp_points"),
        )
        .with_columns(
            pl.lit(season, dtype=pl.Int32).alias("season"),
            (pl.col("vacated_targets") / pl.col("team_targets")).alias("vacated_target_share"),
            (pl.col("vacated_carries") / pl.col("team_carries")).alias("vacated_carry_share"),
        )
    )


def rookie_features(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Draft class joined to the opportunity waiting on the other end."""
    seasons = seasons or [s for s in FEATURE_SEASONS]
    parts = []
    for season in seasons:
        cls = rookie_class(season, force=force)
        if not cls.height:
            continue
        vac = vacated_opportunity(season, scoring, force=force)
        parts.append(
            cls.join(vac, on=["season", "team"], how="left") if vac.height else cls
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def rookie_outcomes(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """What each rookie actually did in his first year.

    Players who never appeared get 0 points and 0 games rather than being
    dropped — never playing is the most common rookie outcome and excluding it
    would train the model only on rookies who made it onto the field.

    Returns: season, gsis_id, games, fantasy_points, ppg, pos_rank, top24_hit.
    """
    seasons = seasons or FEATURE_SEASONS
    finishes = sc.score_season(
        seasons, scoring, weeks=(1, REGULAR_SEASON_WEEKS), force=force
    ).select(
        "season",
        pl.col("player_id").alias("gsis_id"),
        "games",
        "fantasy_points",
        "ppg",
        "pos_rank",
    )
    return finishes.with_columns((pl.col("pos_rank") <= 24).alias("top24_hit"))


def _design(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    usable = [c for c in cols if c in df.columns and df.get_column(c).is_not_null().any()]
    if not usable:
        return np.empty((df.height, 0)), []
    filled = df.select(
        [pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).median()) for c in usable]
    )
    keep = [c for c in usable if filled.get_column(c).is_not_null().all()]
    return (filled.select(keep).to_numpy() if keep else np.empty((df.height, 0))), keep


def _train_frame(
    seasons: list[int] | None = None, scoring: Mapping[str, float] | None = None
) -> pl.DataFrame:
    feats = rookie_features(seasons, scoring)
    if not feats.height:
        return pl.DataFrame()
    outcomes = rookie_outcomes(seasons, scoring)
    return feats.join(outcomes, on=["season", "gsis_id"], how="left").with_columns(
        pl.col("games").fill_null(0),
        pl.col("fantasy_points").fill_null(0.0),
        pl.col("ppg").fill_null(0.0),
        pl.col("top24_hit").fill_null(False),
    )


def fit(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    target: str = "ppg",
    alpha: float = 10.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Leave-one-season-out predictions. Each rookie scored by a model blind to his class.

    Leave-one-season-out rather than season-forward here because the sample is
    small enough that a two-fold expanding window would train on ~90 players. It
    is a weaker guarantee — the model sees later seasons when predicting earlier
    ones — and it is stated rather than glossed. The purpose is to measure
    whether draft capital plus landing spot explains anything at all, not to
    simulate a live draft.

    Returns: season, gsis_id, name, position, team, draft_ovr, actual, predicted.
    """
    df = _train_frame(seasons, scoring)
    if not df.height:
        return pl.DataFrame()

    out = []
    for season in sorted(df.get_column("season").unique().to_list()):
        train = df.filter(pl.col("season") != season)
        test = df.filter(pl.col("season") == season)
        if train.height < 30 or not test.height:
            continue

        x_tr, used = _design(train, ROOKIE_FEATURES)
        if not used:
            continue
        x_te, _ = _design(test.select(used), used)
        y = train.get_column(target).to_numpy().astype(float)

        scaler = StandardScaler().fit(x_tr)
        model = Ridge(alpha=alpha, random_state=seed).fit(scaler.transform(x_tr), y)
        pred = model.predict(scaler.transform(x_te))

        out.append(
            test.select("season", "gsis_id", "name", "position", "team", "draft_ovr").with_columns(
                pl.Series("actual", test.get_column(target).to_numpy()).round(3),
                pl.Series("predicted", pred).round(3),
            )
        )

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def performance(preds: pl.DataFrame) -> pl.DataFrame:
    """Out-of-sample correlation and error, overall and by position.

    Returns: scope, n, corr, mae, baseline_mae.
    """
    if not preds.height:
        return pl.DataFrame()

    rows = []

    def add(scope: str, sub: pl.DataFrame) -> None:
        if sub.height < 5:
            return
        a = sub.get_column("actual").to_numpy()
        p = sub.get_column("predicted").to_numpy()
        corr = float(np.corrcoef(a, p)[0, 1]) if np.std(p) > 0 else float("nan")
        rows.append(
            {
                "scope": scope,
                "n": sub.height,
                "corr": round(corr, 4),
                "mae": round(float(np.mean(np.abs(a - p))), 3),
                # Predicting the mean for everyone — the bar any model must clear.
                "baseline_mae": round(float(np.mean(np.abs(a - a.mean()))), 3),
            }
        )

    add("overall", preds)
    for position in preds.get_column("position").unique().sort().to_list():
        add(position, preds.filter(pl.col("position") == position))

    return pl.DataFrame(rows)


def coefficients(
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    target: str = "ppg",
    alpha: float = 10.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Standardized coefficients on a fit over every season.

    Exists so the app can show that draft capital is doing the work, rather than
    implying that eight features each contribute.
    """
    df = _train_frame(seasons, scoring)
    if not df.height:
        return pl.DataFrame()

    x, used = _design(df, ROOKIE_FEATURES)
    if not used:
        return pl.DataFrame()
    y = df.get_column(target).to_numpy().astype(float)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, random_state=seed).fit(scaler.transform(x), y)

    return pl.DataFrame(
        {"feature": used, "coef": [round(float(c), 4) for c in model.coef_]}
    ).sort(pl.col("coef").abs(), descending=True)


def board(
    season: int = SEASON,
    scoring: Mapping[str, float] | None = None,
    target: str = "ppg",
    alpha: float = 10.0,
    seed: int = 0,
) -> pl.DataFrame:
    """This year's rookie class, scored by a model fit on prior classes.

    Returns: gsis_id, name, position, team, draft_round, draft_ovr,
    vacated_target_share, vacated_carry_share, predicted.
    """
    history = _train_frame(None, scoring)
    incoming = rookie_features([season], scoring)
    if not history.height or not incoming.height:
        return pl.DataFrame()

    x_tr, used = _design(history, ROOKIE_FEATURES)
    if not used:
        return pl.DataFrame()

    # Add any column the incoming class lacks entirely as nulls, so _design
    # imputes it from the historical median rather than the whole board coming
    # back empty because one drill wasn't run this year.
    incoming = incoming.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in used if c not in incoming.columns]
    )
    x_new, _ = _design(incoming.select(used), used)
    if x_new.shape[1] != len(used):
        return pl.DataFrame()
    # Impute from the training distribution, not the incoming one — a class with
    # only five recorded forty times should not have its own median define
    # "average speed".
    for j, col in enumerate(used):
        missing = incoming.get_column(col).is_null().to_numpy()
        if missing.any():
            x_new[missing, j] = float(np.median(x_tr[:, j]))

    y = history.get_column(target).to_numpy().astype(float)
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=alpha, random_state=seed).fit(scaler.transform(x_tr), y)

    keep = [
        c for c in (
            "gsis_id", "name", "position", "team", "draft_round", "draft_ovr",
            "vacated_target_share", "vacated_carry_share",
        )
        if c in incoming.columns
    ]
    return (
        incoming.select(keep)
        .with_columns(pl.Series("predicted", model.predict(scaler.transform(x_new))).round(3))
        .sort("predicted", descending=True)
    )
