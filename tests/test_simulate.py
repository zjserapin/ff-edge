"""Simulation mechanics.

Structural invariants first, because a draft simulator that is subtly wrong
still produces a leaderboard, and a leaderboard is exactly the kind of output
people believe. The strongest tests here are the ones with an answer known in
advance from the league format itself: six of ten teams make the playoffs, one
of ten wins, ten teams average seven wins in a fourteen-week season.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import simulate as sim


@pytest.fixture(scope="module")
def board_scores():
    board, scores = sim.score_matrix(2024)
    if not board.height:
        pytest.skip("cold cache — run `uv run python -m src.bootstrap --light`")
    return board, scores


@pytest.fixture(scope="module")
def drafted(board_scores):
    board, _ = board_scores
    rng = np.random.default_rng(0)
    orders = sim.draft_orders(board, 200, rng)
    seats = rng.integers(0, 10, 200)
    return board, sim.run_draft(board, orders, "bpa", seats, 10, 16)


# --- draft mechanics --------------------------------------------------------


def test_draft_orders_respect_adp_and_dispersion(board_scores) -> None:
    """Sampled draft slots must centre on ADP and spread by the player's stdev."""
    board, _ = board_scores
    rng = np.random.default_rng(0)
    orders = sim.draft_orders(board, 500, rng)

    assert orders.shape == (500, board.height)
    # Every sim is a permutation of the whole pool.
    assert (np.sort(orders, axis=1) == np.arange(board.height)).all()

    # The consensus first pick should usually go first, but not always.
    first = orders[:, 0]
    top = int(np.bincount(first, minlength=board.height).argmax())
    assert top < 5, "the most-frequent first pick is not near the top of ADP"
    assert np.unique(first).size > 1, "draft order is deterministic — stdev is being ignored"


def test_no_player_drafted_twice(drafted) -> None:
    """The single most damaging possible bug, and invisible in any summary."""
    _, picks = drafted
    for sim_idx in range(picks.shape[0]):
        taken = picks[sim_idx][picks[sim_idx] >= 0]
        assert taken.size == np.unique(taken).size, f"duplicate pick in sim {sim_idx}"


def test_every_roster_can_field_a_legal_lineup(drafted) -> None:
    """A roster that cannot start a QB forfeits a slot every week.

    Without the reservation rule, a best-available team drafts receivers all day
    and quietly loses to the capped opponents — which would make every template
    look effective by comparison.
    """
    board, picks = drafted
    pos = board.get_column("pos_code").to_numpy()
    safe = np.where(picks < 0, 0, picks)
    codes = np.where(picks < 0, -1, pos[safe])

    for name, code in sim.POS_CODES.items():
        per_roster = (codes == code).sum(axis=2)
        assert per_roster.min() >= sim.MIN_ROSTER[name], f"a roster has no {name}"
        assert per_roster.max() <= sim.OPPONENT_CAPS[name], f"{name} cap exceeded"


def test_strategy_templates_are_followed(board_scores) -> None:
    """A named strategy must draft the positions it claims, in the order it claims.

    Measured as draft capital, not roster count. Every template here rosters the
    same number of backs — what separates zero-RB from robust-RB is *when* the
    picks are spent, so counting them finds no difference and would pass even if
    the templates were being ignored entirely.
    """
    board, _ = board_scores
    rng = np.random.default_rng(1)
    orders = sim.draft_orders(board, 150, rng)
    seats = rng.integers(0, 10, 150)
    pos = board.get_column("pos_code").to_numpy()
    rows = np.arange(150)

    def mean_rb_round(strategy: str) -> float:
        picks = sim.run_draft(board, orders, strategy, seats, 10, 16)
        mine = picks[rows, seats]
        safe = np.where(mine < 0, 0, mine)
        codes = np.where(mine < 0, -1, pos[safe])
        is_rb = codes == sim.POS_CODES["RB"]
        rounds = np.tile(np.arange(mine.shape[1]), (mine.shape[0], 1))
        return float(np.where(is_rb, rounds, np.nan).mean(axis=1, where=is_rb).mean())

    # Robust-RB front-loads its backs; zero-RB defers them by several rounds.
    assert mean_rb_round("robust_rb") + 2.0 < mean_rb_round("zero_rb")

    # And the first pick of each strategy matches its template.
    for strategy, want in (("robust_rb", "RB"), ("zero_rb", "WR"), ("elite_qb", "QB")):
        picks = sim.run_draft(board, orders, strategy, seats, 10, 16)
        first = pos[picks[rows, seats][:, 0]]
        assert (first == sim.POS_CODES[want]).mean() > 0.95, f"{strategy} round 1"


def test_all_strategies_have_sixteen_rounds() -> None:
    for name, template in sim.STRATEGIES.items():
        assert len(template) == 16, f"{name} has {len(template)} rounds"
        assert template[-2:] == ("K", "DEF"), f"{name} does not end with K/DEF"


# --- season mechanics -------------------------------------------------------


def test_schedule_is_a_valid_round_robin() -> None:
    """Pairings must be symmetric and nobody plays themselves."""
    schedule = sim.league_schedule(10, 14)
    assert schedule.shape == (14, 10)
    for week in schedule:
        assert (week != np.arange(10)).all(), "a team is scheduled against itself"
        for team, opponent in enumerate(week):
            assert week[opponent] == team, "pairing is not symmetric"


TWO_FLEX = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"]
SUPER_FLEX = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF"]


def _brute_force_best(pts, codes, dedicated, flex_sets) -> float:
    """Highest-scoring legal lineup, by exhaustive search over every subset.

    The reference implementation the greedy is checked against: it tries every
    combination of starters and every way to assign them to slots, so it cannot
    inherit a bug from the thing it is testing.
    """
    from itertools import combinations, permutations

    n_start = len(dedicated) + len(flex_sets)
    best = 0.0
    for chosen in combinations(range(len(codes)), n_start):
        remaining = [codes[i] for i in chosen]
        ok = True
        for want in dedicated:
            code = sim.POS_CODES[want]
            if code in remaining:
                remaining.remove(code)
            else:
                ok = False
                break
        if not ok:
            continue
        # Any assignment of the leftovers to the flex slots will do.
        if not any(
            all(
                code in {sim.POS_CODES[p] for p in eligible}
                for code, eligible in zip(order, flex_sets)
            )
            for order in permutations(remaining)
        ):
            continue
        best = max(best, float(sum(pts[i] for i in chosen)))
    return best


@pytest.mark.parametrize(
    "roster,flex_sets",
    [
        (TWO_FLEX, [("RB", "WR", "TE"), ("RB", "WR", "TE")]),
        (SUPER_FLEX, [("RB", "WR", "TE"), ("QB", "RB", "WR", "TE")]),
    ],
    ids=["two_flex", "super_flex"],
)
def test_lineup_selection_is_optimal(board_scores, roster, flex_sets) -> None:
    """Greedy must match brute force on the actual slot structure.

    The superflex case is the one worth having. The old implementation could
    not fill a QB-eligible slot at all, and the module docstring said a
    superflex would break the greedy — it does not, provided the slot types are
    filled most-restrictive-first, which is what this proves.
    """
    board, scores = board_scores
    rng = np.random.default_rng(3)
    orders = sim.draft_orders(board, 6, rng)
    seats = rng.integers(0, 10, 6)
    picks = sim.run_draft(board, orders, "balanced", seats, 10, 16)
    k_dst = np.zeros((6, 10, scores.shape[1]), dtype=np.float32)

    got = sim.weekly_lineup_points(picks, board, scores, k_dst, roster)
    pos = board.get_column("pos_code").to_numpy()
    dedicated = [s for s in roster if s in sim.POS_CODES]

    for s in range(2):
        for t in range(3):
            members = picks[s, t]
            members = members[members >= 0]
            week = 0
            best = _brute_force_best(
                scores[members, week], pos[members], dedicated, flex_sets
            )
            assert abs(got[s, t, week] - best) < 1e-3, (
                f"lineup suboptimal: {got[s, t, week]} vs {best}"
            )


def test_superflex_slot_starts_a_second_quarterback(board_scores) -> None:
    """The behavioural consequence: the same roster scores more under a
    superflex, because a bench QB becomes a starter."""
    board, scores = board_scores
    rng = np.random.default_rng(11)
    orders = sim.draft_orders(board, 8, rng)
    seats = rng.integers(0, 10, 8)
    picks = sim.run_draft(board, orders, "balanced", seats, 10, 16)
    k_dst = np.zeros((8, 10, scores.shape[1]), dtype=np.float32)

    two_flex = sim.weekly_lineup_points(picks, board, scores, k_dst, TWO_FLEX)
    super_flex = sim.weekly_lineup_points(picks, board, scores, k_dst, SUPER_FLEX)

    # Same eight slots either way, but the superflex can reach a position the
    # two-FLEX roster cannot, so it is never worse and usually better.
    assert (super_flex >= two_flex - 1e-3).all()
    assert super_flex.mean() > two_flex.mean()


def test_playoff_structure_matches_the_league(board_scores) -> None:
    """Six of ten make it, exactly one wins, and champions are playoff teams."""
    board, scores = board_scores
    rng = np.random.default_rng(4)
    orders = sim.draft_orders(board, 300, rng)
    seats = rng.integers(0, 10, 300)
    picks = sim.run_draft(board, orders, "bpa", seats, 10, 16)
    k_dst = sim.replacement_kicker_dst(2024, None, 300, 10, scores.shape[1], rng)
    points = sim.weekly_lineup_points(picks, board, scores, k_dst)
    results = sim.play_season(points, sim.league_schedule(10))

    per_sim = results.group_by("sim").agg(
        pl.col("made_playoffs").sum().alias("playoff_teams"),
        pl.col("won_title").sum().alias("champions"),
        pl.col("wins").sum().alias("total_wins"),
    )
    assert (per_sim.get_column("playoff_teams") == 6).all()
    assert (per_sim.get_column("champions") == 1).all()
    # Ten teams, fourteen weeks, one winner per game: 70 wins total.
    assert per_sim.get_column("total_wins").max() <= 70

    champs = results.filter(pl.col("won_title"))
    assert champs.get_column("made_playoffs").all(), "a non-playoff team won the title"
    assert champs.get_column("seed").max() <= 6


def test_bpa_control_lands_on_the_structural_baseline() -> None:
    """The control follows ADP, so it must be average by construction.

    Six of ten teams make the playoffs and one of ten wins, so a strategy with
    no edge should land there. If BPA sits meaningfully below, some asymmetry is
    penalising your seat and every template will look good by comparison.
    """
    runs = sim.run("bpa", n_sims=1500, seed=7)
    if not runs.height:
        pytest.skip("cold cache")
    assert abs(float(runs.get_column("made_playoffs").mean()) - 0.60) < 0.06
    assert abs(float(runs.get_column("won_title").mean()) - 0.10) < 0.04
    assert abs(float(runs.get_column("wins").mean()) - 7.0) < 0.5


def test_adp_actually_reaches_the_lineup(board_scores) -> None:
    """Early ADP must buy more production than late ADP.

    Note what this deliberately does *not* test: shuffling the shared board and
    comparing your own total. That comparison is near-flat by construction —
    all ten teams draft from the same board, so whether it is sorted by ADP or
    random, your sixteen picks are about a tenth of the same pool either way.
    The property that actually matters is that the market ordering carries real
    information into the scores the simulation uses.
    """
    board, scores = board_scores
    weekly = scores[:, :14].sum(axis=1)

    early = weekly[: board.height // 4].mean()
    late = weekly[-board.height // 4 :].mean()
    assert early > late * 1.5, f"ADP is not predictive: early {early:.0f} vs late {late:.0f}"


# --- reporting --------------------------------------------------------------


def test_summary_reports_both_intervals() -> None:
    runs = sim.run("bpa", n_sims=800, seed=2)
    if not runs.height:
        pytest.skip("cold cache")
    summary = sim.summarize(runs, n_boot=200)

    for col in (
        "title_rate", "title_rate_lo", "title_rate_hi",
        "title_rate_mc_lo", "title_rate_mc_hi",
        "playoff_rate", "playoff_rate_lo", "playoff_rate_hi",
    ):
        assert col in summary.columns


def test_season_clustered_interval_is_wider_than_monte_carlo() -> None:
    """The whole reason uncertainty.cluster_bootstrap_ci exists.

    Simulations within a season share one realized set of player outcomes, so
    the naive interval understates the real uncertainty. If these ever came out
    the same width, the app would be quoting error bars an order of magnitude
    too tight.
    """
    runs = sim.run("zero_rb", n_sims=1600, seed=3)
    if not runs.height or runs.get_column("season").n_unique() < 3:
        pytest.skip("need several seasons")

    s = sim.summarize(runs, n_boot=400).row(0, named=True)
    mc_width = s["title_rate_mc_hi"] - s["title_rate_mc_lo"]
    clustered = s["title_rate_hi"] - s["title_rate_lo"]
    assert clustered > mc_width, f"clustered {clustered:.4f} not wider than mc {mc_width:.4f}"


def test_control_comparison_is_paired_and_corrected() -> None:
    """The edge over the control must be paired by season and multiplicity-corrected.

    Two failure modes this guards. Unpaired comparison spends all its power on
    season variation, which is the largest term in the whole simulation. And
    comparing seven strategies then quoting the winner's uncorrected interval
    finds a result whether or not one exists.
    """
    runs = sim.run_all(n_sims=800, seed=5)
    if not runs.height:
        pytest.skip("cold cache")

    edges = sim.compare_to_control(runs, n_boot=400)
    assert edges.height == len(sim.STRATEGIES) - 1
    assert "bpa" not in edges.get_column("strategy").to_list()

    for row in edges.iter_rows(named=True):
        # The point estimate sits inside its own interval.
        assert row["edge_lo"] <= row["edge"] <= row["edge_hi"]
        # Correction only ever widens.
        assert row["bonferroni_lo"] <= row["edge_lo"]
        assert row["bonferroni_hi"] >= row["edge_hi"]
        # A claim of beating the control requires the corrected bound.
        if row["beats_control"]:
            assert row["bonferroni_lo"] > 0


def test_control_compared_to_itself_has_no_edge() -> None:
    """Sanity floor: the control cannot beat itself.

    Catches a sign error or a mispaired join, which would otherwise show up as
    a strategy 'edge' that is really an artifact of how the two sides were
    lined up.
    """
    runs = sim.run_all(strategies=("bpa", "balanced"), n_sims=600, seed=6)
    if not runs.height:
        pytest.skip("cold cache")

    doubled = pl.concat(
        [runs, runs.filter(pl.col("strategy") == "bpa").with_columns(
            pl.lit("bpa_copy").alias("strategy")
        )],
        how="diagonal_relaxed",
    )
    edges = sim.compare_to_control(doubled, n_boot=300)
    copy = edges.filter(pl.col("strategy") == "bpa_copy")
    assert copy.height == 1
    assert abs(float(copy.get_column("edge")[0])) < 1e-9
    assert not bool(copy.get_column("beats_control")[0])
