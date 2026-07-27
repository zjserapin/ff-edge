"""One command to hydrate the whole cache.

Designed to be run cold on a new machine and to never die halfway. A single
failing source (a nflverse file that hasn't been published for the new season,
a host your network blocks) is recorded and stepped past, because a partial
cache is far more useful than an exception traceback and nothing on disk.

    uv run python -m src.bootstrap            # everything, including play-by-play
    uv run python -m src.bootstrap --light    # skip pbp (minutes -> seconds)
    uv run python -m src.bootstrap --sanity   # + run the ADP->nflverse match report
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Callable

import polars as pl

from src import adp, cache, ids
from src import nflverse as nv
from src import sleeper
from src.config import (
    ADP_MISSING_YEARS,
    FEATURE_SEASONS,
    FTN_SEASONS,
    HISTORY_SEASONS,
    LEAGUE_ADP_SCORING,
    LEAGUE_ADP_TEAMS,
    OUTPUT_DIR,
    PBP_SEASONS,
    SEASON,
    SLEEPER_USERNAME,
)

RESULTS: list[dict[str, Any]] = []


def step(label: str, fn: Callable[[], Any], required: bool = False) -> Any:
    """Run one pull, record the outcome, keep going."""
    try:
        out = fn()
        rows = out.height if isinstance(out, pl.DataFrame) else len(out or [])
        RESULTS.append({"step": label, "status": "ok", "rows": rows, "note": ""})
        print(f"  {label:<34} ok      {rows:>8,}")
        return out
    except Exception as exc:  # noqa: BLE001 — the whole point is not to raise
        note = f"{type(exc).__name__}: {exc}"[:160]
        RESULTS.append({"step": label, "status": "FAIL", "rows": 0, "note": note})
        print(f"  {label:<34} FAIL    {note}")
        if required:
            traceback.print_exc()
            sys.exit(1)
        return None


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def run(light: bool = False) -> None:
    _section("nflverse / ffverse — context")
    step("players", nv.players, required=True)
    step("teams", nv.teams)
    step("ff_playerids", nv.ff_playerids, required=True)
    step("crosswalk", ids.crosswalk)
    # Every season in the window, not just the current one: rookies.py diffs a
    # team's prior-season producers against its current roster to find vacated
    # opportunity, which needs both ends of every year pair.
    step(f"rosters {SEASON}", lambda: nv.rosters(SEASON))
    step("rosters (window)", lambda: nv.rosters(FEATURE_SEASONS + [SEASON]))
    step(f"schedules {SEASON}", lambda: nv.schedules(SEASON))
    step("depth_charts", lambda: nv.depth_charts(HISTORY_SEASONS))
    step("draft_picks", nv.draft_picks)
    step("contracts", nv.contracts)
    step("combine", nv.combine)
    step("ff_rankings", nv.ff_rankings)

    _section("production & opportunity")
    step("weekly_stats", nv.weekly_stats)
    step("season_stats (reg)", lambda: nv.season_stats(level="reg"))
    step("team_stats (reg)", lambda: nv.team_stats(level="reg"))
    step("ff_opportunity weekly", lambda: nv.ff_opportunity(stat_type="weekly"))
    step("ff_opportunity pbp_pass", lambda: nv.ff_opportunity(stat_type="pbp_pass"))
    step("ff_opportunity pbp_rush", lambda: nv.ff_opportunity(stat_type="pbp_rush"))
    step("snap_counts", nv.snap_counts)
    step("injuries", nv.injuries)
    step("participation", nv.participation)
    # Explicit rather than relying on the default: FTN is the one table that
    # cannot span the analysis window, and nflreadpy raises rather than
    # returning empty when asked for a season below 2022.
    step("ftn_charting", lambda: nv.ftn_charting(FTN_SEASONS))
    step("nextgen receiving", lambda: nv.nextgen("receiving"))
    step("nextgen rushing", lambda: nv.nextgen("rushing"))
    step("nextgen passing", lambda: nv.nextgen("passing"))
    step("pfr_advstats rec", lambda: nv.pfr_advstats("rec"))
    step("pfr_advstats rush", lambda: nv.pfr_advstats("rush"))
    step("pfr_advstats pass", lambda: nv.pfr_advstats("pass"))

    if light:
        print("\nplay-by-play\n------------\n  skipped (--light)")
    else:
        _section("play-by-play")
        step(f"pbp {PBP_SEASONS[0]}-{PBP_SEASONS[-1]}", lambda: nv.pbp(PBP_SEASONS))

    _section("ADP (fantasyfootballcalculator.com)")
    step("adp ppr", lambda: adp.fetch("ppr"))
    step("adp multi-format", adp.multi_format)
    step("adp snapshot ppr/12", adp.snapshot)
    # The league's actual format. Snapshotted separately from ppr/12 rather than
    # instead of it — the ppr/12 history has already been accumulating and ADP
    # history is the one thing in this project that cannot be backfilled.
    step(
        f"adp snapshot {LEAGUE_ADP_SCORING}/{LEAGUE_ADP_TEAMS}",
        lambda: adp.snapshot(LEAGUE_ADP_SCORING, LEAGUE_ADP_TEAMS),
    )

    _section("ADP history — backtest labels")
    for year in FEATURE_SEASONS + [SEASON]:
        if year in ADP_MISSING_YEARS:
            print(f"  adp {LEAGUE_ADP_SCORING}/{LEAGUE_ADP_TEAMS} {year:<15} skipped  "
                  "FFC has no rows for this year at any format")
            continue
        step(
            f"adp {LEAGUE_ADP_SCORING}/{LEAGUE_ADP_TEAMS} {year}",
            lambda y=year: adp.fetch(LEAGUE_ADP_SCORING, LEAGUE_ADP_TEAMS, y),
        )

    _section("Sleeper")
    if SLEEPER_USERNAME == "CHANGE_ME":
        print("  skipped — export FF_EDGE_SLEEPER_USER=<display name> to pull league data")
    else:
        step("nfl_state", sleeper.nfl_state)
        step("sleeper players", sleeper.players_nfl)
        leagues = step("my_leagues", lambda: sleeper.my_leagues(SEASON))

        league_ids = (
            leagues.get_column("league_id").to_list()
            if isinstance(leagues, pl.DataFrame) and "league_id" in leagues.columns
            else []
        )
        for lid in league_ids:
            step(f"users {lid}", lambda lid=lid: sleeper.league_users(lid))
            step(f"rosters {lid}", lambda lid=lid: sleeper.rosters(lid))
            step(f"draft_history {lid}", lambda lid=lid: sleeper.draft_history(lid))
            # transaction_history, not all_transactions: in the offseason the
            # current league is empty and the behavior data is in prior seasons.
            step(f"transactions {lid}", lambda lid=lid: sleeper.transaction_history(lid))


def sanity() -> None:
    """Prove the join layer works against the messiest real source we have."""
    _section("sanity — ADP name match against nflverse IDs")
    try:
        matched = ids.match_by_name(adp.fetch("ppr"), "name", "position")
        report = ids.match_report(matched, "gsis_id")
        print(f"  ADP match rate: {report['matched']}/{report['rows']} = {report['rate']:.1%}")
        misses = ids.unmatched(matched, "gsis_id")
        if misses.height:
            print("  top unmatched positions:")
            print(misses.group_by("position").len().sort("len", descending=True).head(6))
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped — ADP unavailable ({type(exc).__name__}: {exc})"[:200])

    try:
        ecr = nv.ff_rankings()
        report = ids.match_report(ids.match_by_name(ecr, "player", "pos"), "gsis_id")
        print(f"  ECR match rate: {report['matched']}/{report['rows']} = {report['rate']:.1%}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ECR check skipped ({type(exc).__name__}: {exc})"[:200])


def report() -> None:
    """Write output/inventory.md — pull results, cache contents, totals."""
    inv = cache.inventory()
    ok = sum(1 for r in RESULTS if r["status"] == "ok")
    total_rows = sum(r["rows"] for r in RESULTS)
    total_mb = round(inv.get_column("mb").sum() if inv.height else 0.0, 1)

    lines = [
        "# ff-edge data inventory",
        "",
        f"- pulls: **{ok}/{len(RESULTS)} ok**",
        f"- rows fetched this run: **{total_rows:,}**",
        f"- cache files: **{inv.height}** ({total_mb} MB)",
        "",
        "## Pull results",
        "",
        "| step | status | rows | note |",
        "|---|---|---:|---|",
    ]
    lines += [
        f"| {r['step']} | {r['status']} | {r['rows']:,} | {r['note']} |" for r in RESULTS
    ]
    lines += ["", "## Cache files", "", "| name | kind | MB | age (h) |", "|---|---|---:|---:|"]
    lines += [
        f"| {r['name']} | {r['kind']} | {r['mb']} | {r['age_hours']} |"
        for r in inv.iter_rows(named=True)
    ]

    path = OUTPUT_DIR / "inventory.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}  ({ok}/{len(RESULTS)} ok, {total_mb} MB cached)")


def analysis() -> None:
    """Materialize the derived artifacts the app reads.

    Separate from `run` because these are minutes of compute over data that is
    already local, not network pulls. Imports are deferred so a cold checkout can
    still hydrate its cache before the analysis layer exists or its dependencies
    are installed.
    """
    _section("analysis artifacts")
    from src import archetypes, features, simulate

    step("features (player-season)", features.build)
    step("archetypes (current season)", archetypes.cluster)
    step("simulation baseline", simulate.baseline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrate the ff-edge cache.")
    parser.add_argument("--light", action="store_true", help="skip play-by-play")
    parser.add_argument("--sanity", action="store_true", help="run join match report")
    parser.add_argument(
        "--analysis", action="store_true", help="also build features/clusters/sim"
    )
    args = parser.parse_args()

    run(light=args.light)
    if args.sanity:
        sanity()
    if args.analysis:
        analysis()
    report()
