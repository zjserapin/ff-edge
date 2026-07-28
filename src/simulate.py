"""Monte Carlo over Shiva Bowl rules: how much is a draft strategy actually worth?

Each simulation replays one real historical season. Nine opponents draft from
that season's ADP with noise scaled by each player's observed dispersion, your
team follows a named strategy, and then everyone plays a 14-week schedule with
the players' *actual* weekly scores from that year, sets optimal lineups, and
runs a 6-team playoff.

Replaying a real season rather than mixing years is the point. ADP and outcome
have to come from the same season or the joint distribution — which players were
overpriced, which were bargains, who got hurt — is destroyed, and that joint
distribution is the entire quantity being measured.

**Two limitations, both structural, both stated in the app rather than buried.**

*This sims drafting, not managing.* No waivers, no trades, no streaming, no
in-season roster decisions beyond setting the best lineup. Those are a large
share of real outcomes. So the claim here is narrow: how much a draft strategy
is worth holding management constant. That is not "what wins leagues."

*Sims within a season are not independent.* They share one realized set of
player outcomes. Five thousand simulations over four seasons are closer to four
observations than to five thousand, and the intervals reported reflect that —
see `summarize`.
"""

from __future__ import annotations

import argparse
from typing import Mapping

import numpy as np
import polars as pl

from src import adp as adp_mod
from src import breakout as bo
from src import scoring as sc
from src import uncertainty as unc
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    DST_WEEKLY_MEAN,
    DST_WEEKLY_SD,
    FLEX_ELIGIBLE,
    LABEL_SEASONS,
    LEAGUE_ADP_SCORING,
    LEAGUE_ADP_TEAMS,
    OUTPUT_DIR,
    PLAYOFF_TEAMS,
    REGULAR_SEASON_WEEKS,
)

# Position codes as small ints — the draft loop indexes arrays by them.
POS_CODES = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
N_POS = 4

# Sixteen-round positional templates. A slot names the position that pick must
# take; the best available at that position by sampled draft order is chosen.
#
# Rounds 15-16 are K and DEF for every team including the opponents, because
# spending two picks on them is a real cost that has to be in the model.
_K, _D = "K", "DEF"
STRATEGIES: dict[str, tuple[str, ...]] = {
    # The control, and the most important entry here. It follows ADP exactly.
    # Without it you cannot tell whether any template beats simply drafting the
    # best player available, which is the actual question.
    "bpa": ("BPA",) * 14 + (_K, _D),
    "balanced": ("RB", "WR", "RB", "WR", "WR", "TE", "RB", "QB", "WR", "RB", "WR", "TE", "QB", "RB", _K, _D),
    "zero_rb": ("WR", "WR", "TE", "WR", "WR", "QB", "RB", "RB", "RB", "WR", "RB", "RB", "QB", "TE", _K, _D),
    "hero_rb": ("RB", "WR", "WR", "WR", "TE", "WR", "QB", "RB", "WR", "RB", "RB", "WR", "QB", "TE", _K, _D),
    "robust_rb": ("RB", "RB", "RB", "WR", "WR", "TE", "WR", "QB", "RB", "WR", "WR", "RB", "QB", "TE", _K, _D),
    "early_te": ("RB", "TE", "WR", "RB", "WR", "WR", "QB", "RB", "WR", "RB", "WR", "TE", "QB", "RB", _K, _D),
    "late_qb": ("RB", "WR", "RB", "WR", "WR", "TE", "RB", "WR", "RB", "WR", "QB", "TE", "QB", "RB", _K, _D),
    "elite_qb": ("QB", "RB", "WR", "RB", "WR", "WR", "TE", "RB", "WR", "RB", "WR", "TE", "RB", "WR", _K, _D),
}

# Roster caps for the nine ADP-following opponents, so nobody drafts six
# quarterbacks just because they fell.
OPPONENT_CAPS: dict[str, int] = {"QB": 2, "RB": 6, "WR": 7, "TE": 2}

# Every team must be able to fill its starting lineup. Enforced as a reservation
# on the final rounds rather than as a preference, so no roster reaches week 1
# without a quarterback.
MIN_ROSTER: dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

# Smallest edge over the control worth calling nonzero. Guards against reporting
# floating-point residue as a win. A tenth of a percentage point on title rate
# is far below what four seasons could resolve anyway.
_EDGE_TOL = 1e-6


def score_matrix(
    season: int,
    scoring: Mapping[str, float] | None = None,
    n_weeks: int = 17,
) -> tuple[pl.DataFrame, np.ndarray]:
    """The draftable board and every player's weekly scores for one season.

    Zeros where a player did not play. Byes and injuries are the risk being
    simulated, so they must stay in the data rather than be imputed away — a
    strategy that concentrates value in three players should be penalised when
    one of them misses six weeks, because that is what actually happens.

    Returns (board, scores) where board has gsis_id, name, position, pos_code,
    adp, stdev, and scores is float32 of shape (players, n_weeks).
    """
    scoring = scoring or DEFAULT_SCORING

    board = bo.adp_board(season, LEAGUE_ADP_SCORING, LEAGUE_ADP_TEAMS)
    if not board.height:
        return pl.DataFrame(), np.empty((0, n_weeks), dtype=np.float32)

    board = (
        board.filter(pl.col("position").is_in(list(POS_CODES)))
        .unique(subset=["gsis_id"], keep="first")
        .sort("adp")
        .with_columns(
            pl.col("position").replace_strict(POS_CODES, return_dtype=pl.Int32).alias("pos_code")
        )
    )

    weekly = sc.score_weekly([season], scoring).filter(pl.col("week") <= n_weeks)
    index = {pid: i for i, pid in enumerate(board.get_column("gsis_id").to_list())}

    scores = np.zeros((board.height, n_weeks), dtype=np.float32)
    for pid, week, points in zip(
        weekly.get_column("player_id").to_list(),
        weekly.get_column("week").to_list(),
        weekly.get_column("fantasy_points").to_list(),
    ):
        i = index.get(pid)
        if i is not None:
            scores[i, int(week) - 1] = points

    return board, scores


def draft_orders(board: pl.DataFrame, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a full draft board per simulation, in one vectorized draw.

    A player's slot is Normal(adp, slot_scale(stdev)) — the same distribution
    `adp.survival()` integrates to answer "will he last until my next pick", so
    the board you plan against and the board you are tested on agree.

    Returns (n_sims, n_players): player indices in sampled draft order.
    """
    adp = board.get_column("adp").to_numpy().astype(float)
    sd = adp_mod.slot_scale(board.get_column("stdev").to_numpy())
    draws = rng.normal(adp[None, :], sd[None, :], size=(n_sims, adp.size))
    return np.argsort(draws, axis=1)


def run_draft(
    board: pl.DataFrame,
    orders: np.ndarray,
    strategy: str,
    my_seat: np.ndarray,
    teams: int = DEFAULT_TEAMS,
    rounds: int = 16,
) -> np.ndarray:
    """Snake draft, vectorized across simulations.

    The loop is over the 160 picks, not over simulations: at each pick every
    simulation resolves its selection in a handful of numpy operations. A pick is
    "the earliest player in this sim's sampled order who is still available and
    eligible for this slot".

    `my_seat` is per-simulation so draft position varies across runs without a
    second loop — no strategy should get credit for always picking first.

    Returns (n_sims, teams, rounds) of indices into `board`, with -1 for the
    K and DEF slots, which are not drafted from this pool.
    """
    n_sims = orders.shape[0]
    n_players = orders.shape[1]
    pos_code = board.get_column("pos_code").to_numpy().astype(np.int32)

    # Position of each player *in each sim's order*, so availability can be
    # tracked in order-space and argmax picks the earliest survivor directly.
    pos_in_order = pos_code[orders]                       # (n_sims, n_players)
    available = np.ones((n_sims, n_players), dtype=bool)

    picks = np.full((n_sims, teams, rounds), -1, dtype=np.int64)
    counts = np.zeros((n_sims, teams, N_POS), dtype=np.int32)
    sim_idx = np.arange(n_sims)

    template = STRATEGIES[strategy]
    skill_rounds = [r for r in range(rounds) if (template[r] if r < len(template) else "BPA") not in (_K, _D)]
    caps = np.array([OPPONENT_CAPS[p] for p in POS_CODES], dtype=np.int32)
    minimums = np.array([MIN_ROSTER[p] for p in POS_CODES], dtype=np.int32)

    for position, rnd in enumerate(skill_rounds):
        order = range(teams) if rnd % 2 == 0 else range(teams - 1, -1, -1)
        rounds_left = len(skill_rounds) - position

        for seat in order:
            mine = my_seat == seat
            slot = template[rnd] if rnd < len(template) else "BPA"

            eligible = available.copy()

            # Positional caps apply to everyone, including your team on a BPA
            # pick. Without them the control strategy drafts eight receivers,
            # cannot fill its quarterback slot, and loses to the capped
            # opponents — which would make every template look good by
            # comparison rather than because it works.
            at_cap = counts[:, seat, :] >= caps                       # (n_sims, N_POS)
            for code in range(N_POS):
                blocked = at_cap[:, code]
                if blocked.any():
                    eligible[blocked] &= pos_in_order[blocked] != code

            # Reserve the last picks for unfilled mandatory slots, so nobody
            # forfeits a starting spot by chasing value to the end of the draft.
            shortfall = np.clip(minimums - counts[:, seat, :], 0, None)  # (n_sims, N_POS)
            must_fill = shortfall.sum(axis=1) >= rounds_left
            if must_fill.any():
                rows = np.flatnonzero(must_fill)
                needed = shortfall[rows] > 0
                mask = needed[np.arange(rows.size)[:, None], pos_in_order[rows]]
                eligible[rows] &= mask

            # Independent of the reservation above, not an else: these are
            # per-simulation conditions, and only the simulations actually out
            # of rounds should lose their template. Gating the whole batch on
            # `must_fill.any()` silently disables every strategy from the first
            # round in which a single simulation runs short.
            if slot != "BPA":
                take = mine & ~must_fill
                if take.any():
                    want = POS_CODES[slot]
                    eligible[take] &= pos_in_order[take] == want

            # Fall back to best available if the requested position is gone.
            empty = ~eligible.any(axis=1)
            if empty.any():
                eligible[empty] = available[empty]

            choice = np.argmax(eligible, axis=1)              # index in order-space
            chosen = orders[sim_idx, choice]                  # index into board

            picks[:, seat, rnd] = chosen
            available[sim_idx, choice] = False
            counts[sim_idx, seat, pos_code[chosen]] += 1

    return picks


def replacement_kicker_dst(
    season: int,
    scoring: Mapping[str, float] | None,
    n_sims: int,
    teams: int,
    n_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weekly points from the K and DEF slots, drawn rather than drafted.

    Kickers come from the real distribution — `weekly_stats` carries every field
    goal bucket, so this costs nothing. Defenses come from a documented constant,
    because `weekly_stats` has no team-defense rows at all and this data layer
    genuinely cannot score them.

    Both are drawn identically for all ten teams, so they wash out in head-to-head
    expectation while still injecting the variance that dilutes strategy edges.
    That dilution is realistic: two roster spots really are noise everyone
    carries, and pretending otherwise would overstate how much drafting matters.
    """
    baseline = sc.kicker_baseline([season], scoring)
    if baseline.height:
        k_mean = float(baseline.get_column("mean_weekly")[0])
        k_sd = float(baseline.get_column("sd_weekly")[0])
    else:
        k_mean, k_sd = 7.5, 5.0

    kicker = rng.normal(k_mean, k_sd, size=(n_sims, teams, n_weeks))
    defense = rng.normal(DST_WEEKLY_MEAN, DST_WEEKLY_SD, size=(n_sims, teams, n_weeks))
    return np.clip(kicker, 0, None) + defense


def weekly_lineup_points(
    picks: np.ndarray,
    board: pl.DataFrame,
    scores: np.ndarray,
    k_dst: np.ndarray,
    roster_positions: list[str] | None = None,
    flex_eligible: tuple[str, ...] = FLEX_ELIGIBLE,
) -> np.ndarray:
    """Best legal lineup every week, for every team, in every simulation.

    Greedy is provably optimal for this slot structure, which is why there is no
    solver here: the dedicated slots are disjoint by position and FLEX accepts
    exactly RB union WR union TE, so filling each dedicated slot with that
    position's highest scorers and then taking the best remaining two across the
    flex-eligible positions cannot be improved on. With a superflex or any slot
    that overlapped a dedicated one, this would no longer hold.

    Returns (n_sims, teams, n_weeks).
    """
    roster_positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    slots: dict[str, int] = {}
    n_flex = 0
    for slot in roster_positions:
        if slot == "FLEX":
            n_flex += 1
        elif slot in POS_CODES:
            slots[slot] = slots.get(slot, 0) + 1

    n_sims, teams, rounds = picks.shape
    n_weeks = scores.shape[1]
    pos_code = board.get_column("pos_code").to_numpy().astype(np.int32)

    drafted = picks.copy()
    empty = drafted < 0
    drafted[empty] = 0

    # (n_sims, teams, rounds, n_weeks)
    pts = scores[drafted]
    pts[empty] = -np.inf  # K/DEF slots contribute via k_dst, not from the pool
    codes = pos_code[drafted]
    codes[empty] = -1

    total = np.zeros((n_sims, teams, n_weeks), dtype=np.float32)
    leftovers = []

    for pos, count in slots.items():
        code = POS_CODES[pos]
        masked = np.where(codes[..., None] == code, pts, -np.inf)
        ordered = -np.sort(-masked, axis=2)
        started = ordered[:, :, :count, :]
        total += np.where(np.isfinite(started), started, 0).sum(axis=2)
        if pos in flex_eligible:
            leftovers.append(ordered[:, :, count:, :])

    if n_flex and leftovers:
        bench = np.concatenate(leftovers, axis=2)
        best = -np.sort(-bench, axis=2)[:, :, :n_flex, :]
        total += np.where(np.isfinite(best), best, 0).sum(axis=2)

    return total + k_dst[:, :, :n_weeks]


def league_schedule(teams: int = DEFAULT_TEAMS, weeks: int = REGULAR_SEASON_WEEKS) -> np.ndarray:
    """Round-robin by the circle method, repeating once the rotation completes.

    Ten teams gives nine unique rounds, so a 14-week season plays the first five
    again — which is what a real ten-team league does.

    Returns (weeks, teams) of opponent indices.
    """
    rotation = list(range(teams))
    rounds = []
    for _ in range(teams - 1):
        pairs = np.empty(teams, dtype=np.int64)
        for i in range(teams // 2):
            a, b = rotation[i], rotation[teams - 1 - i]
            pairs[a], pairs[b] = b, a
        rounds.append(pairs)
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]

    return np.array([rounds[w % len(rounds)] for w in range(weeks)])


def play_season(
    points: np.ndarray,
    schedule: np.ndarray,
    playoff_teams: int = PLAYOFF_TEAMS,
    regular_weeks: int = REGULAR_SEASON_WEEKS,
) -> pl.DataFrame:
    """Head-to-head record, seeding, and a 6-team / 3-week playoff.

    Weeks 15-17 with byes for the top two seeds, which is the live league's
    actual format. Seeding is wins, then points for.

    Returns one row per (sim, team): wins, points_for, seed, made_playoffs,
    won_title.
    """
    n_sims, teams, _ = points.shape
    reg = points[:, :, :regular_weeks]

    opponents = schedule[:regular_weeks]                      # (weeks, teams)
    opp_points = reg[:, opponents.T, np.arange(regular_weeks)]
    wins = (reg > opp_points).sum(axis=2)
    points_for = reg.sum(axis=2)

    # Seed on wins, break ties on points for, via a single composite key.
    key = wins * 1e6 + points_for
    seed = np.argsort(np.argsort(-key, axis=1), axis=1) + 1

    made = seed <= playoff_teams
    champion = np.zeros((n_sims, teams), dtype=bool)

    order = np.argsort(-key, axis=1)                          # seeds, best first
    sims = np.arange(n_sims)

    def team_at(rank: int) -> np.ndarray:
        return order[:, rank]

    def beats(a: np.ndarray, b: np.ndarray, week: int) -> np.ndarray:
        pa = points[sims, a, week]
        pb = points[sims, b, week]
        return np.where(pa >= pb, a, b)

    if playoff_teams >= 6 and points.shape[2] > regular_weeks + 2:
        w1, w2, w3 = regular_weeks, regular_weeks + 1, regular_weeks + 2
        # Week 15: 3v6 and 4v5; seeds 1 and 2 rest.
        r1a = beats(team_at(2), team_at(5), w1)
        r1b = beats(team_at(3), team_at(4), w1)
        # Week 16: 1 plays the lower surviving seed, 2 plays the other.
        lower = np.where(seed[sims, r1a] > seed[sims, r1b], r1a, r1b)
        higher = np.where(seed[sims, r1a] > seed[sims, r1b], r1b, r1a)
        s1 = beats(team_at(0), lower, w2)
        s2 = beats(team_at(1), higher, w2)
        winner = beats(s1, s2, w3)
        champion[sims, winner] = True

    return pl.DataFrame(
        {
            "sim": np.repeat(np.arange(n_sims), teams),
            "team": np.tile(np.arange(teams), n_sims),
            "wins": wins.ravel(),
            "points_for": np.round(points_for.ravel(), 2),
            "seed": seed.ravel(),
            "made_playoffs": made.ravel(),
            "won_title": champion.ravel(),
        }
    )


def run(
    strategy: str,
    seasons: list[int] | None = None,
    n_sims: int = 4000,
    scoring: Mapping[str, float] | None = None,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulate `n_sims` seasons of one strategy, split evenly across seasons.

    Only your team's rows are returned — the nine ADP-following opponents exist
    to make the draft realistic, not to be measured.

    Returns: strategy, season, sim, wins, points_for, seed, made_playoffs,
    won_title.
    """
    seasons = seasons or LABEL_SEASONS
    scoring = scoring or DEFAULT_SCORING
    per_season = max(1, n_sims // len(seasons))
    rounds = len([s for s in (roster_positions or DEFAULT_ROSTER_POSITIONS) if s != "IR"])

    out: list[pl.DataFrame] = []
    for i, season in enumerate(seasons):
        board, scores = score_matrix(season, scoring)
        if not board.height:
            continue

        rng = np.random.default_rng(seed * 1000 + i)
        orders = draft_orders(board, per_season, rng)
        my_seat = rng.integers(0, teams, size=per_season)

        picks = run_draft(board, orders, strategy, my_seat, teams, rounds)
        k_dst = replacement_kicker_dst(season, scoring, per_season, teams, scores.shape[1], rng)
        points = weekly_lineup_points(picks, board, scores, k_dst, roster_positions)

        results = play_season(points, league_schedule(teams))
        mine = results.filter(
            pl.col("team") == pl.Series("seat", np.repeat(my_seat, teams))
        )
        out.append(
            mine.with_columns(
                pl.lit(strategy).alias("strategy"),
                pl.lit(season, dtype=pl.Int32).alias("season"),
            )
        )

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def summarize(runs: pl.DataFrame, n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Rates per strategy, with both kinds of interval.

    The two interval columns disagree by roughly an order of magnitude and the
    wider one is the honest one. Simulations inside a season share a single
    realized set of player outcomes, so treating 4,000 sims as 4,000 independent
    draws reports a +/-1 point interval on a title rate that is simply false. The
    season-clustered interval resamples whole seasons and answers the question
    actually being asked: how much would this change if 2026 turns out like some
    other year?

    With four seasons those intervals are wide. That is the sample size showing
    up where it belongs, and the app leads with it.

    Returns: strategy, n_sims, n_seasons, win_rate, playoff_rate, title_rate,
    each with *_mc_lo/_mc_hi and *_lo/_hi (season-clustered).
    """
    if not runs.height:
        return pl.DataFrame()

    rows = []
    for strategy in runs.get_column("strategy").unique().sort().to_list():
        sub = runs.filter(pl.col("strategy") == strategy)
        seasons = sub.get_column("season").to_numpy()
        row: dict[str, object] = {
            "strategy": strategy,
            "n_sims": sub.height,
            "n_seasons": int(np.unique(seasons).size),
            "mean_wins": round(float(sub.get_column("wins").mean()), 3),
            "mean_points_for": round(float(sub.get_column("points_for").mean()), 1),
        }

        for name, col in (
            ("playoff_rate", "made_playoffs"),
            ("title_rate", "won_title"),
        ):
            values = sub.get_column(col).to_numpy().astype(float)
            rate = float(values.mean())
            mc_lo, mc_hi = unc.wilson_interval(int(values.sum()), values.size)
            cl_lo, cl_hi = unc.cluster_bootstrap_ci(
                values, seasons, np.mean, n_boot=n_boot, seed=seed
            )
            row[name] = round(rate, 4)
            row[f"{name}_mc_lo"] = round(mc_lo, 4)
            row[f"{name}_mc_hi"] = round(mc_hi, 4)
            row[f"{name}_lo"] = round(cl_lo, 4)
            row[f"{name}_hi"] = round(cl_hi, 4)

        wins = sub.get_column("wins").to_numpy().astype(float)
        row["win_rate"] = round(float(wins.mean() / REGULAR_SEASON_WEEKS), 4)
        w_lo, w_hi = unc.cluster_bootstrap_ci(wins, seasons, np.mean, n_boot=n_boot, seed=seed)
        row["win_rate_lo"] = round(w_lo / REGULAR_SEASON_WEEKS, 4)
        row["win_rate_hi"] = round(w_hi / REGULAR_SEASON_WEEKS, 4)
        rows.append(row)

    return pl.DataFrame(rows, infer_schema_length=None).sort("title_rate", descending=True)


def compare_to_control(
    runs: pl.DataFrame,
    control: str = "bpa",
    metric: str = "won_title",
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Each strategy's edge over the control, with a season-clustered interval.

    This is the comparison that answers the question, and it is not the same as
    checking whether two intervals overlap. Overlap is a crude proxy: the
    control's own interval happens to be narrow — following ADP performs about
    the same in every season — so a strategy can clear it on a technicality while
    its actual edge is indistinguishable from zero.

    Bootstrapping the *difference* over shared seasons also pairs the two
    strategies on the same resampled years, which removes the season effect that
    dominates everything here. Season variation is the largest term in this
    simulation; comparing unpaired estimates spends all the power on it.

    `bonferroni_lo/hi` widen the interval for the number of strategies compared.
    Picking the best of eight and reporting its uncorrected interval is how you
    manufacture a finding out of noise.

    Returns: strategy, rate, control_rate, edge, edge_lo, edge_hi,
    bonferroni_lo, bonferroni_hi, beats_control.
    """
    if not runs.height:
        return pl.DataFrame()

    strategies = [s for s in runs.get_column("strategy").unique().sort().to_list()]
    if control not in strategies:
        return pl.DataFrame()

    others = [s for s in strategies if s != control]
    alpha_adj = 0.05 / max(1, len(others))

    ctrl = runs.filter(pl.col("strategy") == control)
    ctrl_by_season = {
        int(s): float(g.get_column(metric).mean())
        for (s,), g in ctrl.group_by(["season"])
    }
    control_rate = float(ctrl.get_column(metric).mean())

    rows = []
    for strategy in others:
        sub = runs.filter(pl.col("strategy") == strategy)
        seasons = sub.get_column("season").to_numpy()
        values = sub.get_column(metric).to_numpy().astype(float)
        # Per observation, the control's rate in that same season — so a
        # resampled season carries both sides with it.
        paired = values - np.array([ctrl_by_season.get(int(s), control_rate) for s in seasons])

        lo, hi = unc.cluster_bootstrap_ci(paired, seasons, np.mean, n_boot=n_boot, seed=seed)
        b_lo, b_hi = unc.cluster_bootstrap_ci(
            paired, seasons, np.mean, n_boot=n_boot, seed=seed, alpha=alpha_adj
        )
        rows.append(
            {
                "strategy": strategy,
                "rate": round(float(values.mean()), 4),
                "control_rate": round(control_rate, 4),
                "edge": round(float(paired.mean()), 4),
                "edge_lo": round(lo, 4),
                "edge_hi": round(hi, 4),
                "bonferroni_lo": round(b_lo, 4),
                "bonferroni_hi": round(b_hi, 4),
                # Tolerance, not a bare sign test. A strategy identical to the
                # control produces paired differences that are zero up to
                # floating point, and `b_lo > 0` on a quantile of values like
                # 1e-18 would report it as a winner.
                "beats_control": bool(b_lo > _EDGE_TOL),
            }
        )

    return pl.DataFrame(rows).sort("edge", descending=True)


def run_all(
    strategies: tuple[str, ...] | None = None,
    n_sims: int = 4000,
    seasons: list[int] | None = None,
    scoring: Mapping[str, float] | None = None,
    roster_positions: list[str] | None = None,
    teams: int = DEFAULT_TEAMS,
    seed: int = 0,
) -> pl.DataFrame:
    """Every strategy, same seasons, same seed."""
    strategies = strategies or tuple(STRATEGIES)
    parts = [
        run(s, seasons, n_sims, scoring, roster_positions, teams, seed) for s in strategies
    ]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def baseline(n_sims: int = 4000, force: bool = False) -> pl.DataFrame:
    """Read the precomputed artifact, building it if missing."""
    path = OUTPUT_DIR / "simulation_baseline.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)

    runs = run_all(n_sims=n_sims)
    if runs.height:
        runs.write_parquet(path)
    return runs


def _write_report(runs: pl.DataFrame) -> None:
    summary = summarize(runs)
    lines = [
        "# Draft strategy simulation",
        "",
        f"- simulations per strategy: **{runs.height // summary.height:,}**",
        f"- seasons replayed: **{sorted(runs.get_column('season').unique().to_list())}**",
        "",
        "Intervals below are season-clustered: they resample whole seasons rather",
        "than individual simulations, because simulations within one season share",
        "a single realized set of player outcomes. The narrower Monte Carlo",
        "intervals are also in the parquet and should not be quoted.",
        "",
        "**This simulates drafting, not managing.** No waivers, no trades, no",
        "streaming. The claim is how much a draft strategy is worth holding",
        "in-season management constant.",
        "",
        "| strategy | title rate | 95% CI | playoff rate | 95% CI | mean wins |",
        "|---|---:|---|---:|---|---:|",
    ]
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {r['strategy']} | {r['title_rate']:.1%} | "
            f"[{r['title_rate_lo']:.1%}, {r['title_rate_hi']:.1%}] | "
            f"{r['playoff_rate']:.1%} | "
            f"[{r['playoff_rate_lo']:.1%}, {r['playoff_rate_hi']:.1%}] | "
            f"{r['mean_wins']:.2f} |"
        )

    path = OUTPUT_DIR / "simulation_summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the draft strategy simulation.")
    parser.add_argument("--sims", type=int, default=4000, help="simulations per strategy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import time

    started = time.time()
    runs = run_all(n_sims=args.sims, seed=args.seed)
    if not runs.height:
        raise SystemExit("no simulations produced — is the cache hydrated?")

    out = OUTPUT_DIR / "simulation_baseline.parquet"
    runs.write_parquet(out)
    print(f"wrote {out}  ({runs.height:,} rows, {time.time() - started:.1f}s)")
    _write_report(runs)

    pl.Config.set_tbl_rows(20)
    print(summarize(runs).select(
        "strategy", "title_rate", "title_rate_lo", "title_rate_hi", "playoff_rate", "mean_wins"
    ))
