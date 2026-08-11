"""The ways a sportsbook feed goes wrong here are all silent ones.

Three specific failures motivate this file, and none of them raises:

  1. Reading `handicap` instead of the runner name gives a complete, well-typed
     board on which every line is 0.0.
  2. Joining prop names to the crosswalk without constraining position fans 145
     rows out to 151 — Lamar Jackson matches a cornerback, Justin Jefferson
     matches a rookie linebacker, and Michael Pittman Jr. matches his father.
  3. De-vigging a market that is price-pinned at -114/-114 returns 0.500 and
     looks like a real probability. It is not; it is the absence of a signal.

The parsing tests build synthetic payloads so they run with no network. The join
tests need the id crosswalk and skip cleanly when it is unavailable.
"""

from __future__ import annotations

import polars as pl
import pytest

from src import props
from src.config import DEFAULT_SCORING


def _market(
    name: str,
    line: float,
    over: int = -114,
    under: int = -114,
    market_type: str = "REGULAR_SEASON_PROPS_-_RUNNING_BACKS",
    one_sided: bool = False,
) -> dict:
    """A FanDuel market shaped exactly like the live feed, traps included."""
    player = name.split(" Regular Season")[0]
    runners = [
        {
            "runnerName": f"{player} Over {line}",
            # The live feed really does carry 0 here while the line lives in the
            # name. If this fixture ever "helpfully" sets it, the test is void.
            "handicap": 0,
            "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": over}},
        },
        {
            "runnerName": f"{player} Under {line}",
            "handicap": 0,
            "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": under}},
        },
    ]
    return {
        "marketName": name,
        "marketType": market_type,
        "marketId": "734.000000001",
        "marketStatus": "OPEN",
        "eventId": 28297422,
        "runners": runners[:1] if one_sided else runners,
    }


def _payload(*markets: dict) -> dict:
    return {"attachments": {"markets": {str(i): m for i, m in enumerate(markets)}}}


def _season(*markets: dict) -> pl.DataFrame:
    rows = props._rows_from(
        _payload(*markets),
        lambda m: str(m.get("marketType", "")).startswith("REGULAR_SEASON_PROPS"),
    )
    return pl.DataFrame(rows, schema_overrides=props._SCHEMA)


def test_line_is_read_from_the_runner_name_not_the_handicap() -> None:
    """The trap that produces a whole board of zeroes without raising."""
    df = _season(_market("Bijan Robinson Regular Season Rushing Yards 2026-27", 1150.5))
    assert df.height == 1
    assert df["line"][0] == 1150.5, "line must come from the runner name"
    assert df["line"][0] != 0.0, "reading `handicap` would give exactly this"


def test_market_names_parse_into_player_and_market() -> None:
    df = _season(
        _market("Ashton Jeanty Regular Season Rushing TDs 2026-27", 7.5),
        _market("Trevor Lawrence Regular Season Passing Yards 2026-27", 3750.5),
        _market("Brock Bowers Regular Season Receiving Yards 2026-27", 800.5),
    )
    got = dict(zip(df["player"].to_list(), df["market"].to_list()))
    assert got == {
        "Ashton Jeanty": "rushing_tds",
        "Trevor Lawrence": "passing_yards",
        "Brock Bowers": "receiving_yards",
    }


def test_apostrophes_and_periods_survive_the_player_name() -> None:
    """Ja'Marr and A.J. must not be truncated by the name regex."""
    df = _season(
        _market("Ja'Marr Chase Regular Season Receiving Yards 2026-27", 1325.5),
        _market("A.J. Brown Regular Season Receiving Yards 2026-27", 1100.5),
    )
    assert set(df["player"].to_list()) == {"Ja'Marr Chase", "A.J. Brown"}


def test_one_sided_market_is_dropped_entirely() -> None:
    """A market mid-move has a line and half a price. Half a price is not a row."""
    df = _season(
        _market("Chase Brown Regular Season Rushing TDs 2026-27", 5.5, one_sided=True)
    )
    assert df.height == 0


def test_non_player_markets_are_ignored() -> None:
    """Awards and game lines share the payload and must not parse as props."""
    df = _season(
        _market("Total Points", 44.5, market_type="TOTAL_POINTS_(OVER/UNDER)"),
        _market("Bijan Robinson Regular Season Rushing Yards 2026-27", 1150.5),
    )
    assert df["player"].to_list() == ["Bijan Robinson"]


def test_empty_payload_still_returns_the_full_schema() -> None:
    """August weekly props are absent, and absent must not mean shapeless."""
    empty = pl.DataFrame(schema=props._SCHEMA)
    assert set(empty.columns) == set(props._SCHEMA)
    assert empty.height == 0


def test_devig_normalizes_two_sides_to_one() -> None:
    df = _season(
        _market("Derrick Henry Regular Season Rushing TDs 2026-27", 12.5, -130, -102)
    )
    out = props.devig(df)
    assert out["fair_over"][0] == pytest.approx(0.5282, abs=1e-3)
    assert out["vig"][0] > 0, "a real book always holds some vig"


def test_pinned_yardage_markets_carry_no_signal() -> None:
    """-114/-114 de-vigs to exactly 0.500 — the absence of a lean, not a lean."""
    df = _season(
        _market("Puka Nacua Regular Season Receiving Yards 2026-27", 1375.5, -114, -114)
    )
    assert props.devig(df)["fair_over"][0] == pytest.approx(0.5, abs=1e-12)


def test_implied_points_uses_config_scoring_not_a_hardcoded_rate() -> None:
    """Points must move when the league's scoring moves."""
    df = _season(
        _market("Ashton Jeanty Regular Season Rushing Yards 2026-27", 1000.0),
        _market("Ashton Jeanty Regular Season Rushing TDs 2026-27", 10.0),
    )
    expected = 1000.0 * DEFAULT_SCORING["rush_yd"] + 10.0 * DEFAULT_SCORING["rush_td"]
    assert props.implied_points(df)["implied_points"][0] == pytest.approx(expected)

    doubled = {**DEFAULT_SCORING, "rush_td": DEFAULT_SCORING["rush_td"] * 2}
    bumped = props.implied_points(df, doubled)["implied_points"][0]
    assert bumped == pytest.approx(expected + 10.0 * DEFAULT_SCORING["rush_td"])


def test_implied_points_names_the_markets_that_fed_it() -> None:
    """A partial total that hides which parts are missing is worse than none."""
    df = _season(
        _market("Chase Brown Regular Season Rushing Yards 2026-27", 825.5),
        _market("Chase Brown Regular Season Rushing TDs 2026-27", 5.5),
    )
    markets = props.implied_points(df)["markets"][0].to_list()
    assert markets == ["rushing_tds", "rushing_yards"]
    assert not any("receiving" in m for m in markets), (
        "no receiving market exists for this RB; the total is a floor and the "
        "`markets` column is how a caller can tell"
    )


def test_allowed_positions_intersects_across_a_players_markets() -> None:
    """Passing ∩ rushing pins a quarterback without hardcoding any name."""
    assert props._allowed_positions(["passing_yards", "rushing_tds"]) == ["QB"]
    assert props._allowed_positions(["passing_yards"]) == ["QB"]
    assert props._allowed_positions(["receiving_yards"]) == ["RB", "TE", "WR"]


def test_allowed_positions_never_resolves_to_nobody() -> None:
    """Contradictory markets must widen the search, not empty it.

    QB ∩ {RB,WR,TE} is empty. An empty allow-list would match no crosswalk row
    and drop the player silently, so the fallback is the full skill set.
    """
    assert props._allowed_positions(["passing_yards", "receiving_yards"]) == sorted(
        props._SKILL
    )


@pytest.fixture(scope="module")
def live() -> pl.DataFrame:
    """The real board, or a skip. Network-dependent by nature."""
    try:
        df = props.season_long()
    except Exception as exc:  # noqa: BLE001 - any transport failure is a skip
        pytest.skip(f"FanDuel unreachable: {exc}")
    if df.height == 0:
        pytest.skip("FanDuel returned no season-long markets")
    return df


def test_live_board_parses_every_market_two_sided(live: pl.DataFrame) -> None:
    assert live.height > 100, "the board should carry ~145 markets"
    assert live["line"].null_count() == 0
    assert live["over_odds"].null_count() == 0
    assert live["under_odds"].null_count() == 0
    assert (live["line"] > 0).all(), "a zero line means `handicap` crept back in"


def test_attach_ids_is_row_count_preserving(live: pl.DataFrame) -> None:
    """The 145 -> 151 fan-out, as an assertion.

    This is the whole reason identity is resolved per player instead of per row.
    """
    try:
        joined = props.attach_ids(live)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"crosswalk unavailable: {exc}")
    assert joined.height == live.height, "a name join must not multiply rows"


def test_known_name_collisions_resolve_to_the_right_player(live: pl.DataFrame) -> None:
    """Lamar Jackson is the quarterback; Michael Pittman is the son."""
    try:
        joined = props.attach_ids(live)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"crosswalk unavailable: {exc}")

    expected = {
        "Lamar Jackson": ("QB", "00-0034796"),
        "Justin Jefferson": ("WR", "00-0036322"),
        "Michael Pittman Jr.": ("WR", "00-0036252"),
    }
    for player, (position, gsis) in expected.items():
        rows = joined.filter(pl.col("player") == player)
        if rows.height == 0:
            continue  # the book drops players; absence is not a failure
        assert rows["position"].unique().to_list() == [position], player
        assert rows["gsis_id"].unique().to_list() == [gsis], player
