"""Ledger integrity: the guards, the math, and the audit property.

The claims layer cannot be backtested (no historical claims corpus), so what
*can* be tested is tested hard: the hallucination guard actually drops
paraphrased quotes, novelty actually collapses forty echoes into one claim,
the scoring factors multiply the way the docstrings say, and every flag
decomposes back into the claims that produced it. No test here touches the
network or needs an API key — the LLM is a stub, which is the point: the
machinery around the model is where silent poisoning would happen.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src import claims as cl
from src import news
from src.config import DATA_DIR
from src.llm import LLMResponse

TODAY = dt.date.today()


def _claim(**kw) -> dict:
    base = dict(
        claimed_on=TODAY.isoformat(),
        player_name="A Back",
        team="JAX",
        claim_type="first_team_reps",
        direction="growing",
        specificity="concrete",
        source="Local Beat",
        source_tier=3,
        url="",
        quote="q",
    )
    base.update(kw)
    return base


class StubLLM:
    """Returns a canned response; records what it was asked."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def complete(self, messages, system="", max_tokens=4096) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system})
        return LLMResponse(self.content, "stub", 0, 0, "end_turn")


# --- RSS parsing ------------------------------------------------------------


RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item>
  <title>Tuten takes first-team reps at Jaguars camp - The Florida Times-Union</title>
  <link>https://example.com/a</link>
  <pubDate>Mon, 03 Aug 2026 14:00:00 GMT</pubDate>
  <source url="https://tu.com">The Florida Times-Union</source>
  <description>&lt;a href="x"&gt;Bhayshul Tuten worked with the first team&lt;/a&gt;</description>
</item>
<item>
  <title>Ten bold fantasy takes</title>
  <link>https://example.com/b</link>
  <pubDate>bad date</pubDate>
  <source url="https://agg.com">FantasyAggregator</source>
  <description>hot takes inside</description>
</item>
</channel></rss>"""


def test_rss_parse_strips_html_and_source_suffix() -> None:
    got = news._parse_rss(RSS_FIXTURE)
    assert got.height == 2
    first = got.row(0, named=True)
    assert first["title"] == "Tuten takes first-team reps at Jaguars camp"
    assert first["snippet"] == "Bhayshul Tuten worked with the first team"
    assert first["published"] == "2026-08-03"
    # An unparseable pubDate degrades to null, not a crash.
    assert got.row(1, named=True)["published"] is None


# --- Extraction guards ------------------------------------------------------


def _items() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "title": "Tuten takes first-team reps at Jaguars camp",
                "snippet": "Bhayshul Tuten worked with the first team",
                "source": "The Florida Times-Union",
                "url": "https://example.com/a",
                "published": "2026-08-03",
                "team": "JAX",
            }
        ]
    )


def test_extract_keeps_a_valid_verbatim_claim() -> None:
    stub = StubLLM(
        '[{"item_id": 0, "player_name": "Bhayshul Tuten", "team": "JAX",'
        ' "claim_type": "first_team_reps", "direction": "growing",'
        ' "specificity": "concrete", "quote": "Bhayshul Tuten worked with the first team"}]'
    )
    got, report = cl.extract(_items(), client=stub)
    assert got.height == 1 and report["kept"] == 1
    row = got.row(0, named=True)
    assert row["source_tier"] == 3  # Times-Union isn't in the starter dict yet
    assert row["claimed_on"] == "2026-08-03"
    # The prompt that ran is the versioned one, not an inline string.
    assert "verbatim" in stub.calls[0]["system"]


def test_extract_drops_hallucinated_quotes() -> None:
    """The guard: a paraphrased quote loses the claim, silently to the model
    but loudly in the report."""
    stub = StubLLM(
        '[{"item_id": 0, "player_name": "Bhayshul Tuten", "team": "JAX",'
        ' "claim_type": "first_team_reps", "direction": "growing",'
        ' "specificity": "concrete", "quote": "Tuten was named the starter"}]'
    )
    got, report = cl.extract(_items(), client=stub)
    assert got.height == 0
    assert report["bad_quote"] == 1 and report["kept"] == 0


def test_extract_drops_invalid_types_and_survives_garbage() -> None:
    stub = StubLLM(
        '[{"item_id": 0, "player_name": "X", "claim_type": "hot_take",'
        ' "direction": "growing", "specificity": "vibes", "quote": "Bhayshul"},'
        ' {"item_id": 99, "player_name": "Y", "claim_type": "coach_usage",'
        ' "direction": "growing", "specificity": "vibes", "quote": "z"}]'
    )
    got, report = cl.extract(_items(), client=stub)
    assert got.height == 0 and report["invalid"] == 2

    got, report = cl.extract(_items(), client=StubLLM("I could not find any claims."))
    assert got.height == 0 and report["raw"] == 0


def test_extract_tolerates_code_fences() -> None:
    stub = StubLLM('```json\n[]\n```')
    got, report = cl.extract(_items(), client=stub)
    assert got.height == 0 and report["raw"] == 0


# --- Depth-chart machine claims ---------------------------------------------


def _chart(rows: list[tuple[str, str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"team": t, "position": p, "player_name": n, "depth_rank": r}
            for t, p, n, r in rows
        ]
    )


def test_depth_chart_diff_directions() -> None:
    prev = _chart([("JAX", "RB", "A", 2), ("JAX", "RB", "B", 1), ("JAX", "WR", "C", 1)])
    curr = _chart([("JAX", "RB", "A", 1), ("JAX", "RB", "B", 3), ("JAX", "WR", "C", 1),
                   ("JAX", "TE", "D", 1), ("JAX", "RB", "E", 5)])
    got = cl.from_depth_charts(prev, curr)
    by_name = {r["player_name"]: r for r in got.iter_rows(named=True)}
    assert by_name["A"]["direction"] == "growing"       # 2 -> 1
    assert by_name["B"]["direction"] == "shrinking"     # 1 -> 3
    assert "C" not in by_name                            # unchanged
    assert by_name["D"]["direction"] == "growing"       # new starter
    assert "E" not in by_name                            # new fourth-stringer is churn
    assert all(r["source_tier"] == 1 for r in by_name.values())


# --- Ledger mechanics -------------------------------------------------------


def test_append_is_duplicate_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cl, "CLAIMS_PATH", tmp_path / "claims.parquet")
    first = pl.DataFrame([_claim(quote="a"), _claim(quote="b")])
    assert cl.append(first).height == 2
    again = pl.DataFrame([_claim(quote="b"), _claim(quote="c")])
    ledger = cl.append(again)
    assert ledger.height == 3  # b was not double-counted


def test_novelty_collapses_echoes() -> None:
    """Forty aggregators quoting one beat report count once."""
    rows = [_claim(source=f"Agg{i}", quote=f"q{i}") for i in range(40)]
    rows.append(_claim(claimed_on=(TODAY - dt.timedelta(days=30)).isoformat(),
                       source="Old Beat", quote="old"))
    marked = cl.novelty(pl.DataFrame(rows))
    assert int(marked.get_column("novel").sum()) == 2  # the old one + today's first
    today_rows = marked.filter(pl.col("claimed_on") == TODAY.isoformat())
    assert int(today_rows.get_column("novel").sum()) == 1


def test_score_factors_multiply_as_documented() -> None:
    two_half_lives = (TODAY - dt.timedelta(days=28)).isoformat()
    df = pl.DataFrame([
        _claim(source_tier=1, specificity="concrete"),                      # 1.0
        _claim(source_tier=2, specificity="vibes", quote="v"),              # 0.7*0.4*0.15 echo
        _claim(claimed_on=two_half_lives, source_tier=1, direction="shrinking",
               claim_type="depth_chart", quote="s"),
    ])
    scored = cl.score(df, as_of=TODAY)
    by_quote = {r["quote"]: r for r in scored.iter_rows(named=True)}
    assert by_quote["q"]["claim_score"] == pytest.approx(1.0)
    # Same (player, type, direction) day as "q" -> echo weight applies.
    assert by_quote["v"]["claim_score"] == pytest.approx(0.7 * 0.4 * 0.15, abs=1e-4)
    # 28 days at a 14-day half-life is 0.25, negative for shrinking.
    assert by_quote["s"]["claim_score"] == pytest.approx(-0.25, abs=1e-4)


def test_flags_grade_a_requires_concrete_quality() -> None:
    """Hype volume alone cannot reach A: ten vibes claims from aggregators
    outscore the floor but stay capped at B."""
    hype = [
        _claim(player_name="Hype Man", source=f"Agg{i}", quote=f"h{i}",
               claimed_on=(TODAY - dt.timedelta(days=8 * i)).isoformat(),
               specificity="vibes")
        for i in range(10)
    ]
    real = [
        _claim(player_name="Real Deal", source_tier=1, source="nflverse depth chart",
               claim_type="depth_chart", quote="r1"),
        _claim(player_name="Real Deal", source_tier=2, source="ESPN",
               claim_type="coach_usage", quote="r2",
               claimed_on=(TODAY - dt.timedelta(days=3)).isoformat()),
    ]
    got = cl.flags(pl.DataFrame(hype + real), as_of=TODAY)
    by_name = {r["player_name"]: r for r in got.iter_rows(named=True)}
    assert by_name["Real Deal"]["grade"] == "A"
    assert by_name["Hype Man"]["grade"] != "A"
    # The flag decomposes: n_claims matches what went in.
    assert by_name["Hype Man"]["n_claims"] == 10


def test_player_claims_is_the_audit_trail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cl, "CLAIMS_PATH", tmp_path / "claims.parquet")
    cl.append(pl.DataFrame([_claim(player_name="Bhayshul Tuten", quote="t1"),
                            _claim(player_name="Someone Else", quote="e1")]))
    mine = cl.player_claims("bhayshul tuten")  # case-insensitive resolve
    assert mine.height == 1
    assert {"quote", "source", "claim_score"} <= set(mine.columns)


# --- ADP-at-claim and resolution --------------------------------------------


def test_adp_at_claim_joins_the_snapshot_in_force_on_claim_day() -> None:
    path = DATA_DIR / "adp_history_half-ppr_10_1999.parquet"
    pl.DataFrame(
        [
            {"name": "A Back", "pulled_on": (TODAY - dt.timedelta(days=5)).isoformat(), "adp": 90.0},
            {"name": "A Back", "pulled_on": (TODAY + dt.timedelta(days=5)).isoformat(), "adp": 60.0},
        ]
    ).write_parquet(path)
    try:
        got = cl.adp_at_claim(pl.DataFrame([_claim()]), year=1999)
        # The claim gets the snapshot in force on its date — never the future one.
        assert got.get_column("adp_at_claim")[0] == 90.0
    finally:
        path.unlink()


def test_resolution_labels_growing_claims_by_what_usage_did() -> None:
    """A growing claim resolves true when the share actually moved."""
    players = pl.DataFrame({"player_id": ["00-X"] * 6, "week": [1, 2, 3, 4, 5, 6],
                            "carry_share_wk": [0.1, 0.1, 0.3, 0.35, 0.3, 0.3],
                            "target_share_wk": [0.0] * 6})
    season = 2026
    claim_wk3 = _claim(player_name="Bhayshul Tuten",
                       claimed_on=(dt.date(season, 9, 4) + dt.timedelta(days=15)).isoformat())
    ledger = pl.DataFrame([claim_wk3])

    # Patch the crosswalk so the test doesn't depend on the players cache.
    import src.claims as claims_mod

    real_players = claims_mod.nv.players

    def fake_players(force=False):
        return pl.DataFrame({"display_name": ["Bhayshul Tuten"], "gsis_id": ["00-X"],
                             "position": ["RB"]})

    claims_mod.nv.players = fake_players
    try:
        got = cl.resolve(ledger, weekly=players, season=season)
    finally:
        claims_mod.nv.players = real_players

    assert got.get_column("resolved_hit")[0] is True

    # Off-season claim: window not complete -> pending, not failed.
    pending = _claim(player_name="Bhayshul Tuten", claimed_on=TODAY.isoformat())
    claims_mod.nv.players = fake_players
    try:
        got = cl.resolve(pl.DataFrame([pending]), weekly=players, season=season)
    finally:
        claims_mod.nv.players = real_players
    assert got.get_column("resolved_hit")[0] is None


def test_source_grades_report_n_and_intervals() -> None:
    resolved = pl.DataFrame(
        [_claim(source="Beat A", quote=f"a{i}") for i in range(8)]
        + [_claim(source="Beat B", quote=f"b{i}") for i in range(4)]
    ).with_columns(
        pl.Series("resolved_hit", [True] * 6 + [False] * 2 + [True] + [False] * 3)
    )
    got = cl.source_grades(resolved)
    by_source = {r["source"]: r for r in got.iter_rows(named=True)}
    assert by_source["Beat A"]["n_resolved"] == 8
    assert by_source["Beat A"]["hit_rate"] == pytest.approx(0.75)
    assert by_source["Beat A"]["ci_lo"] < 0.75 < by_source["Beat A"]["ci_hi"]
