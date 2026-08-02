"""Where a player's touches came from, not just how many he got.

`features.py` answers "how much work" and "how good per touch". This module
answers the question sitting between them: *what kind* of work. Two receivers
with identical target shares are not the same asset if one of them gets eight
end-zone looks a year and the other gets none, and no share, rate, or efficiency
column in the feature table can tell them apart.

Four sources, all already on disk, none of them previously read by a model:

  ff_opportunity pbp_pass/pbp_rush   play-level, with field position, game state,
                                     and per-play expected touchdowns
  nextgen rushing                    rush yards over expected, box counts
  ftn_charting                       manual charting: catchable, contested, screen

Three things they buy, in descending order of how much they should matter:

**Touchdown equity.** `pass_touchdown_exp` is the modeled probability that a
play ends in a touchdown, given where the ball was snapped and how far it was
thrown. Summed over a season and divided by his team's, it says what fraction of
the offense's scoring chances run through him. That is the largest thing target
share cannot see: a slot receiver on 25% of targets and a big-slot on 18% with
every goal-line fade can finish miles apart, and only one of those gaps is
priced.

**Neutral game script.** A receiver on a bad team accumulates targets down three
scores, when the defense has conceded the short middle and stopped caring. Those
targets count the same in the raw share and are worth much less going forward,
because next season's team may not trail by three scores. `vegas_wp` between 0.2
and 0.8 keeps the snaps where the game was live. Win probability rather than
score differential, because it already knows about time: down seven in the first
quarter is a normal football game, down seven with two minutes left is not, and
the score column alone calls them the same thing.

**Quarterback decoupling.** Catch rate is charged entirely to the receiver and
is not entirely his. FTN charts whether each throw was catchable, which splits it
in two: `catchable_rate` is a property of the man throwing to him, and
`catch_rate_on_catchable` is a property of his hands. A receiver punished in
`catch_rate` for a bad quarterback shows up here as exactly that.

Traps, all found by looking:

  weeks 19-22   Both pbp tables carry the playoffs with no season_type column,
                same as ff_opportunity weekly.
  two-point     776 pass rows have `pass_attempt == 1` and `two_point_attempt
                == 1`. Every one is snapped from the two, so leaving them in
                inflates red-zone share for whoever happened to be targeted.
  duplicate     The rush table has `rushing_td_exp` *and* `rush_touchdown_exp`,
  expectations  and `rushing_yards_exp` *and* `rush_yards_exp`. They disagree on
                4,558 rows, and the disagreement is not noise: the short-named
                pair carries sentinels (0.0, -1.0) where the long-named pair has
                a real modeled value. The `rushing_*` columns are the model
                output and the ones used here.
  play_id       Float64 in ff_opportunity, Int32 in FTN. Joining without the
                cast raises; joining after casting matches 99.9%.
  null receiver 4% of pass plays have no `receiver_player_id` — throwaways and
                spikes. Dropped from both numerator and denominator, since a
                throwaway is not a target anyone competed for.
"""

from __future__ import annotations

import polars as pl

from src import nflverse as nv
from src.config import FEATURE_SEASONS, FTN_SEASONS

# Field-position thresholds, in yards from the opponent's goal line.
RED_ZONE = 20
GREEN_ZONE = 5

# Win-probability band that counts as a live game. Wide enough to keep most of
# the season (roughly 70% of plays) and narrow enough to drop the garbage time
# that inflates a losing team's target shares.
NEUTRAL_WP = (0.2, 0.8)

CONTEXT_COLUMNS = [
    "rz_target_share",
    "ez_target_share",
    "neutral_target_share",
    "rz_carry_share",
    "gz_carry_share",
    "neutral_rush_share",
    "exp_td_share",
    "exp_td_per_touch",
    "ryoe_per_att",
    "rush_efficiency",
    "stacked_box_rate",
    "time_to_los",
    "catchable_rate",
    "catch_rate_on_catchable",
    "contested_rate",
    "contested_catch_rate",
    "screen_target_rate",
]


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Ratio that yields null rather than inf when the denominator is zero."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def _flag(column: str) -> pl.Expr:
    """Boolean from a 0/1 column, whatever dtype it arrived as.

    ff_opportunity's pbp tables store some indicators as Float64 and others as
    Categorical — `complete_pass`, `goal_to_go`, and `down` are all categories of
    the strings "0"/"1". Comparing a Categorical to an integer raises rather than
    coercing, so every indicator goes through here.
    """
    return pl.col(column).cast(pl.Utf8).cast(pl.Float64, strict=False) == 1


def _scrimmage(df: pl.DataFrame) -> pl.DataFrame:
    """Regular-season plays from scrimmage, with the shared columns normalized."""
    return df.filter(
        (pl.col("week") <= 18) & (pl.col("two_point_attempt") != 1)
    ).with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("vegas_wp")
        .is_between(NEUTRAL_WP[0], NEUTRAL_WP[1])
        .alias("neutral"),
    )


def _share(
    plays: pl.DataFrame, player_key: str, parts: list[pl.Expr]
) -> pl.DataFrame:
    """Player totals over his own weeks, divided by his team's over the same weeks.

    The "same weeks" part is the whole reason this is not a simple season-over-
    season division. A back who played nine games and a back who played all
    seventeen should be compared on the share of the offense they commanded while
    they were on it — dividing a nine-game numerator by a seventeen-game team
    denominator reports the healthy player as twice the talent, which is a
    measure of availability wearing a usage costume. `games` is carried
    separately for exactly this reason.
    """
    per_week = plays.group_by(["season", "week", "posteam", player_key]).agg(parts)
    team_week = (
        plays.group_by(["season", "week", "posteam"])
        .agg(parts)
        .rename({e.meta.output_name(): f"team_{e.meta.output_name()}" for e in parts})
    )
    joined = per_week.join(team_week, on=["season", "week", "posteam"], how="left")

    names = [e.meta.output_name() for e in parts]
    return (
        joined.group_by(["season", player_key])
        .agg([pl.col(c).sum() for c in names] + [pl.col(f"team_{c}").sum() for c in names])
        .rename({player_key: "player_id"})
    )


def _pass_context(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Red-zone, end-zone, neutral-script target shares and passing TD equity.

    `relative_to_endzone` is `air_yards - yardline_100`, verified exactly on all
    114k rows. Zero or above means the ball was thrown to or past the goal line,
    which is what "end-zone target" means — and unlike a red-zone target it does
    not require the offense to already be inside the twenty, so it catches the
    forty-yard post that is also a scoring chance.

    Returns: season, player_id, rz_target_share, ez_target_share,
    neutral_target_share, and the raw pass TD-equity totals for combining.
    """
    raw = nv.ff_opportunity(seasons, stat_type="pbp_pass", force=force)
    if not raw.height:
        return pl.DataFrame()

    plays = _scrimmage(raw).filter(pl.col("receiver_player_id").is_not_null())
    if not plays.height:
        return pl.DataFrame()

    counted = _share(
        plays,
        "receiver_player_id",
        [
            pl.len().cast(pl.Float64).alias("tgt"),
            (pl.col("yardline_100") <= RED_ZONE).sum().cast(pl.Float64).alias("rz_tgt"),
            (pl.col("relative_to_endzone") >= 0).sum().cast(pl.Float64).alias("ez_tgt"),
            pl.col("neutral").sum().cast(pl.Float64).alias("neutral_tgt"),
            pl.col("pass_touchdown_exp").sum().alias("pass_td_exp"),
        ],
    )

    return counted.select(
        "season",
        "player_id",
        _safe_div(pl.col("rz_tgt"), pl.col("team_rz_tgt")).alias("rz_target_share"),
        _safe_div(pl.col("ez_tgt"), pl.col("team_ez_tgt")).alias("ez_target_share"),
        _safe_div(pl.col("neutral_tgt"), pl.col("team_neutral_tgt")).alias(
            "neutral_target_share"
        ),
        pl.col("tgt"),
        pl.col("pass_td_exp"),
        pl.col("team_pass_td_exp"),
    )


def _rush_context(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Red-zone, goal-line, neutral-script carry shares and rushing TD equity.

    The goal-line share is the one that moves fantasy scoring most. A back with
    18% of his team's carries and 60% of the carries inside the five is a
    completely different asset from one with the same 18% and none of them, and
    `rush_share` reports them identically.

    Returns: season, player_id, rz_carry_share, gz_carry_share,
    neutral_rush_share, and the raw rush TD-equity totals for combining.
    """
    raw = nv.ff_opportunity(seasons, stat_type="pbp_rush", force=force)
    if not raw.height:
        return pl.DataFrame()

    plays = _scrimmage(raw).filter(pl.col("rusher_player_id").is_not_null())
    if not plays.height:
        return pl.DataFrame()

    counted = _share(
        plays,
        "rusher_player_id",
        [
            pl.len().cast(pl.Float64).alias("att"),
            (pl.col("yardline_100") <= RED_ZONE).sum().cast(pl.Float64).alias("rz_att"),
            (pl.col("yardline_100") <= GREEN_ZONE).sum().cast(pl.Float64).alias("gz_att"),
            pl.col("neutral").sum().cast(pl.Float64).alias("neutral_att"),
            pl.col("rushing_td_exp").sum().alias("rush_td_exp"),
        ],
    )

    return counted.select(
        "season",
        "player_id",
        _safe_div(pl.col("rz_att"), pl.col("team_rz_att")).alias("rz_carry_share"),
        _safe_div(pl.col("gz_att"), pl.col("team_gz_att")).alias("gz_carry_share"),
        _safe_div(pl.col("neutral_att"), pl.col("team_neutral_att")).alias(
            "neutral_rush_share"
        ),
        pl.col("att"),
        pl.col("rush_td_exp"),
        pl.col("team_rush_td_exp"),
    )


def _ngs_rushing(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """Tracking-derived rushing quality: yards over expected, and the box he ran into.

    Rush yards over expected per attempt is the closest thing running backs have
    to yards per route run. It models what an average back gains given the
    blocking, the box count, and where every defender actually was, then charges
    the back only with the difference. Yards per carry — currently the position's
    only quality feature outside the PFR charting — is famously unstable year to
    year precisely because it credits the offensive line to the runner.

    `efficiency` is distance travelled per yard gained, so **lower is better**;
    it is negated here to `rush_directness` so that every quality column in the
    project points the same way and the percentile helpers in `valuation.py` do
    not need a per-column direction flag.

    Uses `week == 0`, which is NGS's own season aggregate, matching
    `features._nextgen`. Coverage is qualified rushers only.

    Returns: season, player_id, ryoe_per_att, rush_efficiency, stacked_box_rate,
    time_to_los.
    """
    ng = nv.nextgen("rushing", seasons, force=force).filter(
        (pl.col("week") == 0) & (pl.col("season_type") == "REG")
    )
    if not ng.height:
        return pl.DataFrame()

    return (
        ng.with_columns(pl.col("season").cast(pl.Int32))
        .group_by(["season", "player_gsis_id"])
        .agg(
            pl.col("rush_yards_over_expected_per_att").mean().alias("ryoe_per_att"),
            (-pl.col("efficiency")).mean().alias("rush_efficiency"),
            pl.col("percent_attempts_gte_eight_defenders").mean().alias("stacked_box_rate"),
            pl.col("avg_time_to_los").mean().alias("time_to_los"),
        )
        .rename({"player_gsis_id": "player_id"})
        .drop_nulls("player_id")
    )


def _charting(seasons: list[int], force: bool = False) -> pl.DataFrame:
    """FTN manual charting, joined to targets: catchable, contested, screen.

    This is the only source in the project that separates a receiver from his
    quarterback. `catch_rate` charges a receiver for every uncatchable ball
    thrown at him; splitting on `is_catchable_ball` gives one column describing
    the passer (`catchable_rate`) and one describing the receiver
    (`catch_rate_on_catchable`).

    `screen_target_rate` is here as a discount, not a virtue. Screen targets are
    manufactured by the play-caller rather than earned against coverage, so a
    receiver whose yards-per-route-run rests on a heavy screen diet is a worse
    bet to repeat it under a new coordinator than one who earned the same number
    downfield.

    FTN begins in 2022 — a genuine floor, not a gap. Seasons before it come back
    null and must be handled as unavailable rather than imputed to the median,
    which is why `features.context_features` keeps these out of the backtest and
    exposes them for the current-season board instead.

    Returns: season, player_id, catchable_rate, catch_rate_on_catchable,
    contested_rate, contested_catch_rate, screen_target_rate.
    """
    charted = [s for s in seasons if s in FTN_SEASONS]
    if not charted:
        return pl.DataFrame()

    ftn = nv.ftn_charting(charted, force=force)
    passes = nv.ff_opportunity(seasons, stat_type="pbp_pass", force=force)
    if not ftn.height or not passes.height:
        return pl.DataFrame()

    plays = (
        _scrimmage(passes)
        .filter(pl.col("season").is_in(charted) & pl.col("receiver_player_id").is_not_null())
        # Float64 here, Int32 in FTN. Without the cast the join raises outright,
        # which is the good failure mode; it is called out because the same
        # mismatch on a string key would have silently matched nothing.
        .with_columns(pl.col("play_id").cast(pl.Int32))
        .join(
            ftn.select(
                "nflverse_game_id",
                "nflverse_play_id",
                "is_catchable_ball",
                "is_contested_ball",
                "is_screen_pass",
            ),
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "nflverse_play_id"],
            how="inner",
        )
    )
    if not plays.height:
        return pl.DataFrame()

    agg = plays.group_by(["season", "receiver_player_id"]).agg(
        pl.len().alias("tgt"),
        pl.col("is_catchable_ball").sum().alias("catchable"),
        pl.col("is_contested_ball").sum().alias("contested"),
        pl.col("is_screen_pass").sum().alias("screen"),
        (pl.col("is_catchable_ball") & _flag("complete_pass")).sum().alias("caught_catchable"),
        (pl.col("is_contested_ball") & _flag("complete_pass")).sum().alias("caught_contested"),
    )

    return agg.select(
        "season",
        pl.col("receiver_player_id").alias("player_id"),
        _safe_div(pl.col("catchable"), pl.col("tgt")).alias("catchable_rate"),
        _safe_div(pl.col("caught_catchable"), pl.col("catchable")).alias(
            "catch_rate_on_catchable"
        ),
        _safe_div(pl.col("contested"), pl.col("tgt")).alias("contested_rate"),
        _safe_div(pl.col("caught_contested"), pl.col("contested")).alias(
            "contested_catch_rate"
        ),
        _safe_div(pl.col("screen"), pl.col("tgt")).alias("screen_target_rate"),
    )


def play_context(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """One row per player-season with every context column. Joined by `features`.

    Touchdown equity is combined across pass and rush rather than reported twice,
    because for a running back both halves are the same question — what share of
    his offense's scoring chances does he get — and splitting it hands the model
    two columns that each answer half of it. The denominator is his team's total
    expected touchdowns from scrimmage over the weeks he played.

    Returns: season, player_id, plus `CONTEXT_COLUMNS`.
    """
    seasons = seasons or FEATURE_SEASONS

    passing = _pass_context(seasons, force=force)
    rushing = _rush_context(seasons, force=force)
    if not passing.height and not rushing.height:
        return pl.DataFrame()

    frames = [f for f in (passing, rushing) if f.height]
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.join(extra, on=["season", "player_id"], how="full", coalesce=True)

    for col in ("tgt", "att", "pass_td_exp", "team_pass_td_exp", "rush_td_exp", "team_rush_td_exp"):
        if col not in merged.columns:
            merged = merged.with_columns(pl.lit(0.0).alias(col))

    merged = merged.with_columns(
        (pl.col("pass_td_exp").fill_null(0) + pl.col("rush_td_exp").fill_null(0)).alias(
            "_exp_td"
        ),
        (
            pl.col("team_pass_td_exp").fill_null(0) + pl.col("team_rush_td_exp").fill_null(0)
        ).alias("_team_exp_td"),
        (pl.col("tgt").fill_null(0) + pl.col("att").fill_null(0)).alias("_touches"),
    ).with_columns(
        _safe_div(pl.col("_exp_td"), pl.col("_team_exp_td")).alias("exp_td_share"),
        _safe_div(pl.col("_exp_td"), pl.col("_touches")).alias("exp_td_per_touch"),
    )

    for extra in (_ngs_rushing(seasons, force=force), _charting(seasons, force=force)):
        if extra.height:
            merged = merged.join(extra, on=["season", "player_id"], how="left")

    keep = ["season", "player_id"] + [c for c in CONTEXT_COLUMNS if c in merged.columns]
    return merged.select(keep)
