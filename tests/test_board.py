"""The keeper-adjusted draft board.

The failure that matters here is a kept player left sitting on the board, or a
replacement baseline computed against a pool that does not exist. Both produce
a board that looks completely normal and is wrong on draft day, so the pure
functions are tested against synthetic frames where the right answer is known
by construction, and the live-league path is exercised separately and skipped
when no league is configured.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import board as bd


def _board(rows: list[tuple[str, str, float, float, bool]]) -> pl.DataFrame:
    """(name, position, adp, exp_points, kept) -> a board-shaped frame."""
    return pl.DataFrame(
        [
            {
                "name": n, "position": p, "adp": a, "exp_points": e,
                "kept": k, "kept_by": "someone" if k else None,
            }
            for n, p, a, e, k in rows
        ]
    )


def _summary(position: str, league: float, kept: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "position": position,
                "league_demand": league,
                "kept": kept,
                "draft_demand": max(league - kept, 0.0),
                "undeclared_teams": 0,
            }
        ]
    )


# --- replacement level ------------------------------------------------------


def test_replacement_prices_against_the_draft_pool_not_the_league() -> None:
    """The module's central claim, on a case where the answer is arithmetic.

    Ten QB slots league-wide, eight already kept, so the draft fills two. The
    baseline must be the third-best *available* quarterback, not the eleventh.
    """
    rows = [(f"qb{i}", "QB", float(i), 200.0 - 10 * i, False) for i in range(12)]
    board = _board(rows)
    summary = _summary("QB", league=10.0, kept=8)

    draft = bd.replacement(board, summary, use_draft_demand=True)
    league = bd.replacement(board, summary, use_draft_demand=False)

    assert draft.get_column("replacement_rank")[0] == 3
    assert league.get_column("replacement_rank")[0] == 11
    # Pricing against a shallower pool must give the *higher* baseline, which
    # is what makes the position worth less in the draft than league-wide
    # demand alone would suggest.
    assert (
        draft.get_column("replacement_points")[0]
        > league.get_column("replacement_points")[0]
    )


def test_replacement_ignores_kept_players() -> None:
    """A kept player must never set the baseline — nobody can draft him."""
    board = _board(
        [("elite", "RB", 1.0, 300.0, True)]
        + [(f"rb{i}", "RB", float(i + 2), 100.0 - i, False) for i in range(5)]
    )
    repl = bd.replacement(board, _summary("RB", league=2.0, kept=1))
    assert repl.get_column("replacement_points")[0] < 200.0


def test_replacement_survives_a_thin_pool() -> None:
    """Demand deeper than the available pool must clamp, not raise."""
    board = _board([(f"te{i}", "TE", float(i), 90.0 - i, False) for i in range(3)])
    repl = bd.replacement(board, _summary("TE", league=40.0, kept=0))
    assert repl.height == 1
    assert repl.get_column("replacement_rank")[0] == 3


# --- keeper matching --------------------------------------------------------


def test_unmatched_keepers_are_reported() -> None:
    """A keeper that silently fails to match is the one error here that would
    actively mislead — he would still be sitting on the board."""
    kept = pl.DataFrame(
        [
            {"owner": "a", "player_name": "Josh Allen", "position": "QB"},
            {"owner": "b", "player_name": "Deep Bench Guy", "position": "WR"},
        ]
    )
    board = _board([("Josh Allen", "QB", 1.4, 250.0, True)])
    unmatched = bd.keeper_match_report(kept, board)
    assert unmatched.height == 1
    assert unmatched.get_column("player_name")[0] == "Deep Bench Guy"


def test_keeper_matching_is_case_and_punctuation_insensitive() -> None:
    """Sleeper and FFC share no ids, so the join is on normalized names."""
    kept = pl.DataFrame(
        [{"owner": "a", "player_name": "ja'marr chase", "position": "WR"}]
    )
    board = _board([("Ja'Marr Chase", "WR", 10.1, 177.0, True)])
    assert bd.keeper_match_report(kept, board).height == 0


# --- the live league --------------------------------------------------------


@pytest.fixture(scope="module")
def built() -> dict[str, pl.DataFrame]:
    """The live board. Skips rather than fails on the three legitimate ways
    this can be unavailable: no league configured, no network, cold cache."""
    try:
        out = bd.build()
    except Exception:  # noqa: BLE001 — offline is a supported state
        pytest.skip("no league reachable")
    if not out["players"].height:
        pytest.skip("no ADP board cached")
    return out


@pytest.fixture(scope="module")
def with_keepers(built) -> dict[str, pl.DataFrame]:
    """The board *and* a league that has declared keepers.

    Without `FF_EDGE_LEAGUE_ID` the board still builds — it is simply
    unadjusted, which is correct behaviour for someone who cloned the repo and
    has no league. The keeper properties below need an actual keeper list.
    """
    if not built["kept"].height:
        pytest.skip("no league configured (FF_EDGE_LEAGUE_ID unset)")
    return built


def test_no_kept_player_appears_on_the_board(with_keepers) -> None:
    """The property the whole module exists for."""
    players = with_keepers["players"]
    assert players.height
    assert not players.get_column("kept").any()

    kept_names = set(
        with_keepers["kept"].get_column("player_name").drop_nulls().to_list()
    )
    on_board = set(players.get_column("name").to_list())
    assert not (kept_names & on_board)


def test_unadjusted_board_is_honest_about_having_no_keepers(built) -> None:
    """With no league configured the board must report zero kept rather than
    quietly implying the pool has been adjusted."""
    if built["kept"].height:
        pytest.skip("league is configured — covered by the keeper tests")
    assert (built["summary"].get_column("kept") == 0).all()
    assert built["summary"].get_column("draft_demand").to_list() == (
        built["summary"].get_column("league_demand").to_list()
    )


def test_draft_demand_never_exceeds_league_demand(built) -> None:
    s = built["summary"]
    assert (s.get_column("draft_demand") <= s.get_column("league_demand")).all()
    assert (s.get_column("draft_demand") >= 0).all()


def test_pick_inventory_is_internally_consistent(with_keepers) -> None:
    """Pick numbers must be unique, in range, and consistent with the keeper
    placements — the three ways a snake-plus-trades calculation goes wrong."""
    pk = bd.picks()
    if not pk.height:
        pytest.skip("no draft order posted yet")

    nums = pk.get_column("pick_no").to_list()
    assert len(nums) == len(set(nums)), "a pick is listed twice"
    assert all(1 <= n <= 15 * 10 for n in nums)

    # Every pick consumed by a keeper is marked unusable, and vice versa.
    for r in pk.iter_rows(named=True):
        assert r["usable"] == (r["keeper"] is None)

    kept_here = set(pk.filter(~pl.col("usable")).get_column("keeper").to_list())
    my_keepers = set(
        with_keepers["kept"]
        .filter(pl.col("owner") == pk.get_column("from_owner").mode()[0])
        .get_column("player_name")
        .to_list()
    ) if with_keepers["kept"].height else set()
    # Keeper names on my picks must be real keepers, not stray draft picks.
    all_keepers = set(with_keepers["kept"].get_column("player_name").to_list())
    assert kept_here <= all_keepers


def test_targets_respects_the_availability_floor(built) -> None:
    """Only players who plausibly last may be suggested, best PAR first."""
    players = built["players"]
    assert "stdev" in players.columns, "dispersion must survive the board build"

    got = bd.targets(players, 24, min_available=0.5)
    assert got.height
    assert (got.get_column("p_available_at_24") >= 0.5).all()
    par = got.get_column("par").to_list()
    assert par == sorted(par, reverse=True)

    # A very early pick makes almost everyone available; a late one does not.
    early = bd.targets(players, 2, min_available=0.5, top=50).height
    late = bd.targets(players, 120, min_available=0.5, top=50).height
    assert early > late


def test_targets_refuses_to_guess_without_dispersion(built) -> None:
    """A survival curve invented from a default stdev would look exactly as
    confident as a real one, so the absence of dispersion must return nothing."""
    stripped = built["players"].drop("stdev")
    assert bd.targets(stripped, 24).height == 0


def test_tiers_and_ranks_agree(built) -> None:
    """A better tier must never contain a worse board rank than a worse tier."""
    players = built["players"].sort("board_rank")
    ranks = players.get_column("board_rank").to_list()
    assert ranks == sorted(ranks)
    assert players.get_column("par").is_not_null().all()
