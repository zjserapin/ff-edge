"""Sleeper's own ADP — the number the draft room puts on the screen.

FFC's ADP is a good estimate of *a* market. Sleeper's is this league's market,
and the difference is causal rather than statistical: the nine other managers in
the Shiva Bowl are looking at these exact numbers while they pick. FFC's board
is seen by nobody in the league. So this feed does not merely predict the draft
better — it partly *creates* the draft it is predicting.

That distinction is why this is a source swap rather than a correction applied
on top of FFC. `LEAGUE_ADP_SPEC.md` measured a per-position bias against three
seasons of real Shiva Bowl picks and found tight ends going a median 9-20 picks
earlier than FFC said. Sleeper's board reproduces that independently: TEs sit a
median 25 picks earlier than FFC inside the top 150. Two unrelated methods
agreeing that FFC is wrong about tight ends means the honest fix is to stop
reading FFC, not to keep reading it through a patch.

### What this module does *not* replace

The historical ADP -> points curve in `expected.adp_curve` still comes from FFC.
ADP does two jobs here and they separate cleanly:

  - **Value.** "What does a market's TE1 typically score?" That is a function of
    positional *rank*, fit across `LABEL_SEASONS`. FFC covers all seven; Sleeper
    covers 2020-2026 (2019 returns no `adp_2qb` at all). Rank is rank, so a
    curve fit on FFC ranks applies to Sleeper ranks without contradiction.
  - **Cost.** "Who occupies each rank, and what will he cost me?" That is what
    was wrong, and that is what moves here.

Moving the curve to Sleeper as well is defensible and is a measurement, not an
assumption — do it by comparing boards, not by asserting it is tidier.

### Two traps in this feed

**`999.0` is the sentinel for "undrafted", and it is never null.** 8,537 of the
9,414 rows in the 2026 feed carry it. Nothing raises, and an ascending sort
looks perfectly correct because 999 sorts last exactly where you want it. But a
mean, a curve fit, or a `nulls_last` guard all read it as a real draft slot at
pick 999. It is converted to null on ingest here, once, so no caller has to
remember. Related in spirit to the `MISSED_SEASON_RANK` note in `breakout`: a
sentinel that sorts correctly is the most dangerous kind.

**Sleeper ships no dispersion.** `survival()` and the draft simulator both need
`stdev`, and `config.py` already recorded the right answer when it sketched this
swap for FantasyPros: *keep `stdev` from FFC even then*, because FFC measures it
over real drafts and is the only honest source of it here. So the price comes
from Sleeper and the spread comes from FFC, joined by normalized name.

Where FFC has no row at all — Sleeper's board runs about 3.7x deeper — the
stdev is *imputed from the dispersion typical of that round*, not floored.
A missing stdev claims "we did not measure this player's spread", not "this
player has no spread", and `slot_scale`'s 0.5 floor would turn the second
reading into near-certainty that a pick lands exactly on its ADP. Same
impute-rather-than-sink rule `board.rank_board` follows for `quality_pct`.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import requests

from src import ids
from src.cache import frame
from src.config import DATA_DIR, SEASON

BASE = "https://api.sleeper.app/projections/nfl"

# Sleeper serves every market in one payload as differently-named stat keys.
# Keyed by the *FFC* scoring name so `profile.adp_scoring` selects the market
# here and at FFC with one string, and a profile therefore cannot ask the two
# sources for different formats. `profiles.LeagueProfile` exists to stop a
# format and its market drifting apart; reusing the vocabulary stops the same
# drift appearing between two spellings of the same market.
MARKETS: dict[str, str] = {
    "2qb": "adp_2qb",
    "ppr": "adp_ppr",
    "half-ppr": "adp_half_ppr",
    "standard": "adp_std",
}

# Not a draft slot. See the module docstring.
UNDRAFTED = 999.0

# FFC spells kickers PK. Nothing downstream reads either code today — the board
# filters to `FANTASY_POSITIONS` — but a board whose position vocabulary depends
# on which source filled it is a trap waiting for the first panel that filters
# on a string.
_POSITIONS = {"K": "PK"}

_session = requests.Session()
_session.headers.update({"User-Agent": "ff-edge/0.1 (personal research)"})


def markets() -> list[str]:
    """Scoring names this source can price, in FFC's vocabulary."""
    return sorted(MARKETS)


def fetch(
    scoring: str = "2qb",
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Sleeper's ADP for one market, shaped like `adp.fetch`.

    Returns `name, position, team, adp, sleeper_id, adp_rank, scoring, source,
    pulled_on`. `stdev` is *not* attached here — see `board()`, which joins it
    from FFC. Rows with no real ADP are dropped rather than carried as nulls,
    because "undrafted in every league" is not a price.

    Raises on an unknown scoring name rather than falling back, for the reason
    `profiles.resolve` does: a typo that silently returned the 1QB market would
    price a superflex league off a board where quarterbacks are free.
    """
    if scoring not in MARKETS:
        known = ", ".join(markets())
        raise KeyError(f"unknown market {scoring!r}; Sleeper serves: {known}")
    key = MARKETS[scoring]

    def _load() -> pl.DataFrame:
        resp = _session.get(
            f"{BASE}/{season}",
            params={"season_type": "regular", "order_by": key},
            timeout=120,
        )
        resp.raise_for_status()
        rows = []
        for row in resp.json():
            value = (row.get("stats") or {}).get(key)
            if value is None or value >= UNDRAFTED:
                continue
            player = row.get("player") or {}
            first, last = player.get("first_name"), player.get("last_name")
            if not first or not last:
                continue
            rows.append(
                {
                    "sleeper_id": row.get("player_id"),
                    "name": f"{first} {last}",
                    "position": player.get("position"),
                    "team": player.get("team"),
                    "adp": float(value),
                }
            )
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(
                pl.col("position").replace(_POSITIONS),
                ids.normalize_team("team"),
                pl.lit(scoring).alias("scoring"),
                pl.lit("sleeper").alias("source"),
                pl.lit(date.today().isoformat()).alias("pulled_on"),
            )
            .sort("adp")
            .with_columns(pl.col("adp").rank("ordinal").cast(pl.Int32).alias("adp_rank"))
        )

    return frame(f"sleeper_adp_{scoring}_{season}", "live", _load, force)


def _round_of(col: str, teams: int) -> pl.Expr:
    """Which draft round a pick number falls in, 1-indexed."""
    return (((pl.col(col) - 1) // teams) + 1).cast(pl.Int32).alias("_round")


def _dispersion_by_round(
    ffc: pl.DataFrame, teams: int, max_round: int
) -> pl.DataFrame:
    """Typical FFC stdev per draft round, for imputing where FFC has no row.

    Rounds rather than a fitted curve because dispersion genuinely steps by
    round — early picks are near-consensus and late ones are nearly random —
    and a median per round says that without asserting a functional form the
    data does not support.

    **The fill happens over a complete round index, not over the board.** That
    distinction is the whole function. Forward-filling the joined board instead
    fills each gap from whichever board row happened to sort before it, so a
    round-21 player standing behind a round-1 player inherits *round 1's*
    dispersion — near-consensus, assigned to the player nobody has an opinion
    about, which is the exact inversion this imputation exists to prevent. Over
    a dense index the carry always comes from the nearest measured round below.
    """
    empty = pl.DataFrame(schema={"_round": pl.Int32, "_round_stdev": pl.Float64})
    if not ffc.height or max_round < 1:
        return empty

    measured = (
        ffc.filter(pl.col("stdev").is_not_null() & (pl.col("stdev") > 0))
        .with_columns(_round_of("adp", teams))
        .group_by("_round")
        .agg(pl.col("stdev").median().alias("_round_stdev"))
    )
    if not measured.height:
        return empty

    index = pl.DataFrame(
        {"_round": list(range(1, max_round + 1))},
        schema={"_round": pl.Int32},
    )
    return (
        index.join(measured, on="_round", how="left")
        .sort("_round")
        .with_columns(pl.col("_round_stdev").fill_null(strategy="forward"))
        # Rounds before FFC's first measured one have nothing to carry forward.
        .with_columns(pl.col("_round_stdev").fill_null(strategy="backward"))
    )


def board(
    scoring: str = "2qb",
    teams: int = 10,
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Sleeper ADP with FFC dispersion attached — a drop-in for `adp.fetch`.

    `teams` only sizes the rounds used to impute a missing stdev; Sleeper's ADP
    is not published per team-count, which is the same pooling FFC does and is
    noted as such in `profiles.LeagueProfile.adp_teams`.
    """
    from src import adp as ffc_adp

    market = fetch(scoring, season, force=force)
    if not market.height:
        return market

    ffc = ffc_adp.fetch(scoring, teams, season, force=force)
    if not ffc.height:
        # An unpriceable FFC format is a sentence, not a crash: the Sleeper
        # price is still the thing being asked for, and a null stdev degrades
        # `survival` rather than the board.
        return market.with_columns(pl.lit(None, dtype=pl.Float64).alias("stdev"))

    ffc = (
        ffc.with_columns(ids.normalize("name"))
        .filter(pl.col("stdev").is_not_null())
        # One row per identity, not per row. Generational suffixes collapse
        # father onto son and defenders share names with skill players, so a
        # bare name join fans out; keeping the earliest-drafted row per
        # (name, position) makes the row count preserved structurally.
        .sort("adp")
        .unique(subset=["nkey", "position"], keep="first")
        .select("nkey", "position", "stdev")
    )

    before = market.height
    joined = market.with_columns(ids.normalize("name")).join(
        ffc, on=["nkey", "position"], how="left"
    )
    if joined.height != before:
        raise ValueError(
            f"stdev join changed row count {before} -> {joined.height}; "
            "an FFC name resolved to more than one identity"
        )

    joined = joined.with_columns(_round_of("adp", teams))
    rounds = _dispersion_by_round(
        ffc_adp.fetch(scoring, teams, season),
        teams,
        max_round=int(joined.get_column("_round").max() or 0),
    )
    joined = (
        joined.join(rounds, on="_round", how="left")
        .with_columns(
            pl.col("stdev").fill_null(pl.col("_round_stdev")).alias("stdev"),
            # Reads the *pre-fill* stdev: both expressions in one `with_columns`
            # are evaluated against the input frame, so this flags who was
            # missing rather than who is missing after the fill (nobody).
            pl.col("stdev").is_null().alias("stdev_imputed"),
        )
        .drop("_round", "_round_stdev", "nkey")
        .sort("adp")
    )
    return joined


def snapshot(
    scoring: str = "2qb",
    season: int = SEASON,
    force: bool = False,
) -> pl.DataFrame:
    """Append today's Sleeper ADP to the rolling history. Idempotent per day.

    Same standing reason as `adp.snapshot`, and it applies here with more force:
    the API serves *today* and nobody sells the history, so a day not recorded
    is gone permanently. Sleeper's board is the one this league drafts against,
    which makes its drift the drift that actually matters — and the first
    snapshot is worth nothing until there is a second one to difference it
    against. Start it early even when the payoff is a season away.
    """
    today = board(scoring, season=season, force=force)
    if not today.height:
        return today

    path = DATA_DIR / f"sleeper_adp_history_{scoring}_{season}.parquet"
    stamp = date.today().isoformat()

    if path.exists():
        prior = pl.read_parquet(path).filter(pl.col("pulled_on") != stamp)
        history = pl.concat([prior, today], how="diagonal_relaxed")
    else:
        history = today

    history.write_parquet(path)
    return history


def match_report(
    scoring: str = "2qb",
    teams: int = 10,
    season: int = SEASON,
) -> dict:
    """How much of the board Sleeper prices, and how much of it FFC can spread.

    The `ids.match_report` of this feed. Both numbers are worth watching for the
    same reason: neither a thin market nor a fully-imputed dispersion raises.
    """
    b = board(scoring, teams, season)
    if not b.height:
        return {"season": season, "scoring": scoring, "rows": 0}
    inside = b.filter(pl.col("adp") <= teams * 15)
    return {
        "season": season,
        "scoring": scoring,
        "rows": b.height,
        "inside_draft": inside.height,
        "stdev_from_ffc": int((~b.get_column("stdev_imputed")).sum()),
        "stdev_imputed": int(b.get_column("stdev_imputed").sum()),
        "imputed_inside_draft": int(inside.get_column("stdev_imputed").sum()),
        "by_position": dict(
            sorted(
                inside.group_by("position")
                .agg(pl.len())
                .iter_rows(),
                key=lambda kv: -kv[1],
            )
        ),
    }
