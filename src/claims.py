"""The claims ledger: role-change claims, scored, auditable, append-only.

The atom is a claim, not a player. Every number the ledger produces decomposes
back into the quoted claims that made it — the moment that stops being true,
the layer is vibes with extra steps and should be deleted (CLAIMS_SPEC.md).

The pipeline, in the project's usual shape:

    ingest (news.py)  →  extract / diff (here)  →  guard (verbatim quote)
    →  append (parquet, deduped)  →  score (tier × specificity × novelty ×
    recency)  →  flags (per-player role-growing grade for the Screen tab)
    →  resolve (against promotion.weekly_trust, building source grades)

Honesty notes, up front:

- **This layer cannot be backtested.** There is no historical claims corpus.
  Season one is labeled-data collection: log everything, resolve everything,
  judge the layer next August. The score weights below are hand-set priors,
  not fitted parameters, and are named as such.
- **Source tiers are coarse on purpose.** A season yields a handful of
  resolvable claims per source; three tiers is all that sample can carry.
- **Absence of claims is not absence of signal.** Sources cover interesting
  players; a boring veteran with no claims is bullish stability, and the
  Screen tab says so rather than treating silence as data.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Literal

import polars as pl
from pydantic import BaseModel, ValidationError

from src import ids
from src import nflverse as nv
from src import prompts
from src import uncertainty as unc
from src.config import (
    CLAIM_HALF_LIFE_DAYS,
    CLAIMS_PATH,
    LEAGUE_ADP_SCORING,
    LEAGUE_ADP_TEAMS,
    NOVELTY_WINDOW_DAYS,
    RESOLUTION_WEEKS,
    SEASON,
)
from src.llm import LLMClient

ClaimType = Literal[
    "depth_chart",
    "first_team_reps",
    "coach_usage",
    "injury_teammate",
    "departure",
    "role_change_observed",
]


class Claim(BaseModel):
    """One structured role-change claim. Pydantic so bad rows fail loudly."""

    claimed_on: str  # ISO date
    player_name: str
    team: str | None = None
    claim_type: ClaimType
    direction: Literal["growing", "shrinking"]
    specificity: Literal["concrete", "vibes"]
    source: str
    source_tier: int
    url: str = ""
    quote: str


# Hand-assigned starting tiers, matched as lowercase substrings of the source
# name. 1 = beat writer / structured data, 2 = national reporter, 3 =
# aggregator (the default for anything unrecognized). Deliberately a short,
# editable dict rather than a database: `source_grades()` is what earns a
# source a better number over time, and precision beyond three tiers is fake.
SOURCE_TIERS: dict[str, int] = {
    "nflverse depth chart": 1,
    "espn": 2,
    "the athletic": 2,
    "nfl network": 2,
    "nfl.com": 2,
    "yahoo": 2,
    "cbs": 2,
}

# Score weights. Priors, not fits — season one exists to collect the data that
# could ever justify better numbers.
TIER_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.4}
SPECIFICITY_WEIGHT = {"concrete": 1.0, "vibes": 0.4}
REPEAT_WEIGHT = 0.15  # a non-novel claim is mostly an echo

# Flag grades: cumulative signed score thresholds. A is reserved for a score
# built on at least one concrete tier-1/2 claim; hype alone tops out at B.
GRADE_FLOOR = {"A": 1.5, "B": 0.75, "C": 0.3}


def tier_for(source: str) -> int:
    low = (source or "").lower()
    for pattern, tier in SOURCE_TIERS.items():
        if pattern in low:
            return tier
    return 3


# --- Extraction -------------------------------------------------------------


def _parse_llm_json(text: str) -> list[dict]:
    """The model was told 'JSON array only'; tolerate fences anyway."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        got = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return got if isinstance(got, list) else []


def extract(
    items: pl.DataFrame, client: LLMClient | None = None
) -> tuple[pl.DataFrame, dict[str, int]]:
    """News items → validated claims, with the hallucination guard applied.

    `items` needs: title, snippet, source, url, published, team (from
    `news.all_team_news`). Returns (claims frame, report) where the report
    counts what was dropped and why — silent loss is how a ledger gets
    quietly poisoned, so the counts are surfaced all the way to the app.

    The guard: a claim's `quote` must appear verbatim in the item text it
    cites. An LLM that paraphrases loses the claim, by design.
    """
    report = {"items": items.height, "raw": 0, "invalid": 0, "bad_quote": 0, "kept": 0}
    if not items.height:
        return pl.DataFrame(), report

    client = client or LLMClient()
    payload = [
        {
            "id": i,
            "source": r["source"],
            "published": r["published"],
            "text": f"{r['title']}. {r['snippet']}".strip(),
        }
        for i, r in enumerate(items.iter_rows(named=True))
    ]
    resp = client.complete(
        messages=[{"role": "user", "content": json.dumps(payload)}],
        system=prompts.CLAIM_EXTRACTION_V1,
    )

    rows: list[dict] = []
    raw_claims = _parse_llm_json(resp.content)
    report["raw"] = len(raw_claims)
    item_rows = list(items.iter_rows(named=True))

    for raw in raw_claims:
        item_id = raw.get("item_id")
        if not isinstance(item_id, int) or not (0 <= item_id < len(item_rows)):
            report["invalid"] += 1
            continue
        item = item_rows[item_id]
        text = f"{item['title']}. {item['snippet']}"
        quote = raw.get("quote") or ""
        if quote not in text:
            report["bad_quote"] += 1
            continue
        try:
            claim = Claim(
                claimed_on=item["published"] or date.today().isoformat(),
                player_name=raw.get("player_name") or "",
                team=raw.get("team") or item.get("team"),
                claim_type=raw.get("claim_type"),
                direction=raw.get("direction"),
                specificity=raw.get("specificity"),
                source=item["source"],
                source_tier=tier_for(item["source"]),
                url=item["url"],
                quote=quote,
            )
        except ValidationError:
            report["invalid"] += 1
            continue
        if not claim.player_name:
            report["invalid"] += 1
            continue
        rows.append(claim.model_dump())

    report["kept"] = len(rows)
    return (pl.DataFrame(rows) if rows else pl.DataFrame()), report


def from_depth_charts(prev: pl.DataFrame, curr: pl.DataFrame) -> pl.DataFrame:
    """Depth-chart movement → machine-generated claims. No LLM involved.

    A rank change on a structured chart is the cleanest claim the ledger will
    ever hold: concrete, tier 1, and self-quoting. Teams do lie on charts —
    which is a reason to let `source_grades()` measure the source like any
    other, not a reason to skip the data.
    """
    if not prev.height or not curr.height:
        return pl.DataFrame()

    joined = curr.join(
        prev.select("team", "position", "player_name", pl.col("depth_rank").alias("prev_rank")),
        on=["team", "position", "player_name"],
        how="left",
    ).filter(
        pl.col("prev_rank").is_null() | (pl.col("depth_rank") != pl.col("prev_rank"))
    )
    if not joined.height:
        return pl.DataFrame()

    today = date.today().isoformat()
    rows = []
    for r in joined.iter_rows(named=True):
        was = r["prev_rank"]
        now = r["depth_rank"]
        if now is None:
            continue
        direction = "growing" if (was is None and now <= 2) or (was is not None and now < was) else "shrinking"
        if was is None and now > 2:
            continue  # a new fourth-stringer is roster churn, not a role change
        rows.append(
            Claim(
                claimed_on=today,
                player_name=r["player_name"],
                team=r["team"],
                claim_type="depth_chart",
                direction=direction,
                specificity="concrete",
                source="nflverse depth chart",
                source_tier=1,
                quote=f"{r['position']} depth: {'unlisted' if was is None else was} -> {now}",
            ).model_dump()
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


# --- The ledger -------------------------------------------------------------

_KEY = ["claimed_on", "player_name", "claim_type", "direction", "source", "quote"]


def append(new: pl.DataFrame) -> pl.DataFrame:
    """Append claims to the parquet ledger, exact-duplicate-safe. Returns all.

    Append-only is the point: claims are never edited or deleted, because the
    ledger doubles as the labeled dataset future-you will judge this layer
    with. Corrections happen by newer claims, not by rewriting history.
    """
    ledger = load()
    if new.height:
        new = new.select(sorted(new.columns))
        if ledger.height:
            existing = ledger.select(_KEY).with_columns(pl.lit(True).alias("_dup"))
            new = (
                new.join(existing, on=_KEY, how="left")
                .filter(pl.col("_dup").is_null())
                .drop("_dup")
            )
            ledger = pl.concat(
                [ledger.select(sorted(ledger.columns)), new], how="diagonal_relaxed"
            )
        else:
            ledger = new
        ledger.write_parquet(CLAIMS_PATH)
    return ledger


def load() -> pl.DataFrame:
    return pl.read_parquet(CLAIMS_PATH) if CLAIMS_PATH.exists() else pl.DataFrame()


def novelty(ledger: pl.DataFrame) -> pl.DataFrame:
    """Mark each claim novel or echo: dedupe by claim, not by mention.

    Forty aggregators quoting one beat report is one claim — the first
    (player, type, direction) within the novelty window is novel; the rest
    are echoes and score at `REPEAT_WEIGHT`. This is the mechanical half of
    the volume/value fix: star hype is many mentions of few claims.
    """
    if not ledger.height:
        return ledger
    return (
        ledger.with_columns(pl.col("claimed_on").str.to_date().alias("_d"))
        .sort("_d")
        .with_columns(
            (pl.col("_d") - pl.col("_d").shift(1).over(["player_name", "claim_type", "direction"]))
            .dt.total_days()
            .alias("_gap")
        )
        .with_columns(
            (pl.col("_gap").is_null() | (pl.col("_gap") > NOVELTY_WINDOW_DAYS)).alias("novel")
        )
        .drop("_gap")
    )


def adp_at_claim(
    ledger: pl.DataFrame,
    scoring: str = LEAGUE_ADP_SCORING,
    teams: int = LEAGUE_ADP_TEAMS,
    year: int = SEASON,
) -> pl.DataFrame:
    """Attach each claim's ADP as of the day it landed, from the snapshots.

    This is the already-priced check — the biggest failure mode in the spec.
    A claim's value is only the part not yet in price, and by draft day most
    August news is priced within days. Null where the player has no ADP at
    all, which is itself informative: the market hasn't priced him, so a real
    claim is pure edge.
    """
    if not ledger.height:
        return ledger
    from src.config import DATA_DIR

    path = DATA_DIR / f"adp_history_{scoring}_{teams}_{year}.parquet"
    if not path.exists():
        return ledger.with_columns(pl.lit(None, dtype=pl.Float64).alias("adp_at_claim"))

    history = (
        pl.read_parquet(path)
        .select(
            ids.normalize("name").alias("_norm"),
            pl.col("pulled_on").str.to_date().alias("_pulled"),
            "adp",
        )
        .sort("_pulled")
    )
    out = (
        ledger.with_columns(
            ids.normalize("player_name").alias("_norm"),
            pl.col("claimed_on").str.to_date().alias("_d"),
        )
        .sort("_d")
        .join_asof(
            history,
            left_on="_d",
            right_on="_pulled",
            by="_norm",
            strategy="backward",
            # Both sides are sorted on the asof key above; polars just cannot
            # verify it within by-groups and would warn on every call.
            check_sortedness=False,
        )
        .rename({"adp": "adp_at_claim"})
        .drop("_norm", "_pulled")
    )
    return out


def score(ledger: pl.DataFrame, as_of: date | None = None) -> pl.DataFrame:
    """Per-claim signed score: tier × specificity × novelty × recency decay.

    All four factors are visible as columns so a score is never a mystery —
    the app shows the decomposition, per the auditability rule.
    """
    if not ledger.height:
        return ledger
    as_of = as_of or date.today()
    marked = novelty(ledger)
    return (
        marked.with_columns(
            pl.col("source_tier").replace_strict(TIER_WEIGHT, default=0.4).alias("w_tier"),
            pl.col("specificity").replace_strict(SPECIFICITY_WEIGHT, default=0.4).alias("w_spec"),
            pl.when(pl.col("novel")).then(1.0).otherwise(REPEAT_WEIGHT).alias("w_novel"),
            ((pl.lit(as_of) - pl.col("_d")).dt.total_days().clip(0) / CLAIM_HALF_LIFE_DAYS)
            .alias("_half_lives"),
        )
        .with_columns((0.5 ** pl.col("_half_lives")).alias("w_recency"))
        .with_columns(
            (
                pl.when(pl.col("direction") == "growing").then(1.0).otherwise(-1.0)
                * pl.col("w_tier") * pl.col("w_spec") * pl.col("w_novel") * pl.col("w_recency")
            ).round(4).alias("claim_score")
        )
        .drop("_half_lives", "_d")
    )


def flags(ledger: pl.DataFrame | None = None, as_of: date | None = None) -> pl.DataFrame:
    """Per-player role-growing flags — what the Screen tab pre-fills from.

    `grade` is A/B/C/watch by cumulative score, with A additionally requiring
    at least one concrete tier-1/2 claim: hype volume alone cannot reach A no
    matter how loud. The thresholds are hand-set priors (see module docstring)
    and the n of claims rides along, as everywhere in this project.

    Returns: player_name, team, role_score, grade, n_claims, n_novel,
    best_tier, has_concrete, latest_claim, adp_at_latest.
    """
    ledger = ledger if ledger is not None else load()
    if not ledger.height:
        return pl.DataFrame()

    scored = adp_at_claim(score(ledger, as_of))
    out = (
        scored.group_by("player_name")
        .agg(
            pl.col("team").drop_nulls().last().alias("team"),
            pl.col("claim_score").sum().round(3).alias("role_score"),
            pl.len().alias("n_claims"),
            pl.col("novel").sum().alias("n_novel"),
            pl.col("source_tier").min().alias("best_tier"),
            ((pl.col("specificity") == "concrete") & (pl.col("source_tier") <= 2))
            .any()
            .alias("has_concrete"),
            pl.col("claimed_on").max().alias("latest_claim"),
            pl.col("adp_at_claim").drop_nulls().last().alias("adp_at_latest"),
        )
        .with_columns(
            pl.when((pl.col("role_score") >= GRADE_FLOOR["A"]) & pl.col("has_concrete"))
            .then(pl.lit("A"))
            .when(pl.col("role_score") >= GRADE_FLOOR["B"])
            .then(pl.lit("B"))
            .when(pl.col("role_score") >= GRADE_FLOOR["C"])
            .then(pl.lit("C"))
            .when(pl.col("role_score") > 0)
            .then(pl.lit("watch"))
            .otherwise(pl.lit("shrinking"))
            .alias("grade")
        )
        .sort("role_score", descending=True)
    )
    return out


def player_claims(name: str, ledger: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every claim behind one player's flag — the audit trail, on demand."""
    ledger = ledger if ledger is not None else load()
    if not ledger.height:
        return pl.DataFrame()
    wanted = pl.DataFrame({"_q": [name]}).select(ids.normalize("_q")).item()
    return (
        score(ledger)
        .with_columns(ids.normalize("player_name").alias("_n"))
        .filter(pl.col("_n") == wanted)
        .drop("_n")
        .sort("claimed_on", descending=True)
    )


# --- Resolution -------------------------------------------------------------


def resolve(
    ledger: pl.DataFrame | None = None,
    weekly: pl.DataFrame | None = None,
    season: int = SEASON,
) -> pl.DataFrame:
    """Check resolvable claims against what the usage data then did.

    A growing claim resolves true if the player's carry-or-target share over
    the RESOLUTION_WEEKS after the claim beats the weeks before by 2+ points
    (shrinking, the mirror). Off-season claims stay pending until the games
    exist — pending is a state, not a failure.

    This is what turns the ledger into labeled data: `source_grades()` is
    computed from these resolutions, and next August judges the whole layer.
    """
    ledger = ledger if ledger is not None else load()
    if not ledger.height:
        return pl.DataFrame()

    if weekly is None:
        from src import promotion as pm

        weekly = pm.weekly_trust(season)

    base = ledger.with_columns(pl.col("claimed_on").str.to_date().alias("_d"))
    if not weekly.height:
        return base.with_columns(pl.lit(None, dtype=pl.Boolean).alias("resolved_hit")).drop("_d")

    players = nv.players()
    name_col = "display_name" if "display_name" in players.columns else "player_name"
    id_col = "gsis_id" if "gsis_id" in players.columns else "player_id"
    xwalk = (
        players.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(ids.normalize(name_col).alias("_norm"), pl.col(id_col).alias("player_id"))
        .unique("_norm", keep="none")  # ambiguous names resolve nothing
    )
    base = base.with_columns(ids.normalize("player_name").alias("_norm")).join(
        xwalk, on="_norm", how="left"
    )

    # Week-of-season for each claim date: week 1 ≈ the first Thursday after
    # Labor Day; good enough to bucket a claim into before/after windows.
    season_start = date(season, 9, 4)
    base = base.with_columns(
        ((pl.col("_d") - pl.lit(season_start)).dt.total_days() // 7 + 1)
        .clip(0, 18)
        .cast(pl.Int32)
        .alias("_claim_week")
    )

    share = weekly.with_columns(
        pl.max_horizontal(
            pl.col("carry_share_wk").fill_null(0), pl.col("target_share_wk").fill_null(0)
        ).alias("_share")
    )

    rows = []
    max_week = int(share.get_column("week").max())
    for r in base.iter_rows(named=True):
        pid, wk = r["player_id"], r["_claim_week"]
        if pid is None or wk is None or wk < 1 or wk + RESOLUTION_WEEKS > max_week:
            rows.append(None)  # unmatched name or window not yet complete
            continue
        mine = share.filter(pl.col("player_id") == pid)
        before = mine.filter(pl.col("week") < wk).get_column("_share")
        after = mine.filter(
            (pl.col("week") >= wk) & (pl.col("week") < wk + RESOLUTION_WEEKS)
        ).get_column("_share")
        if not len(after):
            rows.append(False if r["direction"] == "growing" else True)
            continue
        delta = (after.mean() or 0.0) - (before.mean() or 0.0 if len(before) else 0.0)
        rows.append(delta >= 0.02 if r["direction"] == "growing" else delta <= -0.02)

    return base.with_columns(pl.Series("resolved_hit", rows, dtype=pl.Boolean)).drop(
        "_d", "_norm", "_claim_week"
    )


def source_grades(resolved: pl.DataFrame) -> pl.DataFrame:
    """Per-source hit rate on resolved claims, Wilson intervals, n visible.

    This converges over seasons, not weeks — which is why `SOURCE_TIERS`
    stays a coarse hand-set dict until these n columns say otherwise.
    """
    if not resolved.height or "resolved_hit" not in resolved.columns:
        return pl.DataFrame()
    done = resolved.filter(pl.col("resolved_hit").is_not_null())
    if not done.height:
        return pl.DataFrame()

    rows = []
    for source in done.get_column("source").unique().sort().to_list():
        sub = done.filter(pl.col("source") == source)
        hits = int(sub.get_column("resolved_hit").sum())
        lo, hi = unc.wilson_interval(hits, sub.height)
        rows.append(
            {
                "source": source,
                "tier": tier_for(source),
                "n_resolved": sub.height,
                "hits": hits,
                "hit_rate": round(hits / sub.height, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
            }
        )
    return pl.DataFrame(rows).sort(["hit_rate"], descending=True)


# --- The daily pull ----------------------------------------------------------


def pull(force: bool = False, with_llm: bool = True) -> dict[str, object]:
    """One ledger update: depth charts always; news extraction when a key exists.

    Returns a report dict rather than printing — bootstrap and the app both
    surface it, and a pipeline that hides its drop counts is how hallucinated
    claims sneak in.
    """
    from src import news

    report: dict[str, object] = {}

    curr = news.depth_chart_snapshot(SEASON, force=force)
    prev_path = CLAIMS_PATH.with_name("depth_snapshot.parquet")
    prev = pl.read_parquet(prev_path) if prev_path.exists() else pl.DataFrame()
    machine = from_depth_charts(prev, curr)
    if curr.height:
        curr.write_parquet(prev_path)
    report["depth_claims"] = machine.height

    extracted = pl.DataFrame()
    if with_llm:
        from src.config import ANTHROPIC_API_KEY

        if ANTHROPIC_API_KEY:
            items = news.all_team_news(force=force)
            extracted, ex_report = extract(items)
            report["extraction"] = ex_report
        else:
            report["extraction"] = "skipped — ANTHROPIC_API_KEY not set"

    new = (
        pl.concat([machine, extracted], how="diagonal_relaxed")
        if machine.height or extracted.height
        else pl.DataFrame()
    )
    ledger = append(new)
    report["ledger_total"] = ledger.height
    return report
