# ff-edge

A free-data foundation for 2026 redraft fantasy football research. No paid
subscriptions, no scraping anything behind a login. Everything here comes from
nflverse, Sleeper's public API, and Fantasy Football Calculator's open REST
endpoint.

This is a **data layer**, not a ranking product. It gets clean, joined, cached
data onto your disk so you can ask your own questions.

## Setup

```bash
uv sync
uv run python bootstrap.py --light   # ~2 min cold, hydrates everything but pbp
uv run python peek.py                # four worked examples
```

To pull your own league data, export your Sleeper **display name** (not your
email) and re-run bootstrap. It's read from the shell rather than committed so a
personal handle doesn't end up in the repo:

```bash
export FF_EDGE_SLEEPER_USER=yourname
uv run python bootstrap.py --light
```

Without it the Sleeper section is skipped cleanly.

Drop `--light` to also pull play-by-play (~1GB, several minutes).

## Modules

| File | What it's for |
|---|---|
| `config.py` | Season, paths, cache TTLs, season ranges. One place to roll the year. |
| `cache.py` | TTL-aware parquet/JSON cache. Every network pull goes through it. |
| `nflverse.py` | Cached wrappers over `nflreadpy` — production, opportunity, context. |
| `sleeper.py` | Read-only Sleeper client, including the `previous_league_id` walk. |
| `adp.py` | FFC ADP, daily snapshots, and pick-survival probability. |
| `ids.py` | The join layer. All cross-source ID and name matching lives here. |
| `bootstrap.py` | One command to hydrate the cache. Never dies on one bad source. |
| `peek.py` | Four worked examples that prove the joins hold. |
| `mcp_server.py` | FastMCP server so the same cache is queryable in conversation. |

## MCP

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ff-edge": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/ff-edge",
        "run", "python", "mcp_server.py"
      ]
    }
  }
}
```

14 read-only tools: `my_leagues`, `league_rosters`, `draft_history`,
`transactions`, `adp`, `survival`, `adp_movement`, `points_over_expected`,
`usage_leaders`, `snap_trend`, `market_disagreement`, `player_lookup`,
`data_inventory`, `refresh`. Nothing can modify a league.

Run bootstrap once before first use so the tools answer from cache.

## What's verified

Row counts from a real run on 2026-07-27. `uv run python bootstrap.py --light`
reported **26/26 pulls ok**, 39.2 MB cached.

| table | rows |
|---|---:|
| players | 25,035 |
| ff_playerids | 12,468 |
| crosswalk (deduped) | 11,904 |
| rosters 2026 | 2,930 |
| schedules 2026 | 272 |
| depth_charts 2022-25 | 666,634 |
| draft_picks | 12,927 |
| contracts | 51,741 |
| combine | 8,968 |
| ff_rankings | 6,391 |
| weekly_stats 2022-25 | 75,876 |
| season_stats (reg) | 7,967 |
| ff_opportunity weekly | 24,178 |
| ff_opportunity pbp_pass | 75,585 |
| ff_opportunity pbp_rush | 61,636 |
| snap_counts | 106,148 |
| injuries | 23,564 |
| participation | 187,421 |
| ftn_charting | 185,215 |
| adp ppr | 228 |

Also verified by running it:

- **Warm cache**: second `bootstrap.py --light` completes in 0.4s with no
  network calls.
- **Join rates**: ECR 3,933/6,391 = **61.5%**; ADP 187/228 = **82.0%**.
- **Survival math**: ADP 40 / stdev 3 → 0.0038 at pick 48; ADP 40 / stdev 14 →
  0.2839. Same price, ~75x difference in whether you can wait.
- **MCP**: all 14 tools register and return real data under `fastmcp.Client`.

The Sleeper section is verified too, against a live account (**33/33 pulls ok**):
user resolves, 1 league, 10 managers, 10 rosters. The `previous_league_id` walk
returned **470 picks across 3 prior seasons** (2023/2024/2025), joining to
nflverse at **96.0%**. `transaction_history` returned **1,984 transactions**
across 2023-2025 (1,071 waiver, 884 free agent, 32 trade). All 14 MCP tools
return real data, including the four league tools.

Note that both multi-season pulls run off the same `sleeper.league_chain()`
traversal, and that transactions deliberately span prior seasons — during the
offseason the current league is empty, so a single-season pull returns 0 and
tells you nothing about how these managers behave.

## Notes

**`nfl_data_py` is dead — use `nflreadpy`.** The old package pins `pandas<2.0`,
which fails to build on modern Python (`No module named 'pkg_resources'` during
the wheel build). `nflreadpy` is the maintained nflverse successor and returns
**polars**, not pandas. This project is polars end to end; nothing converts
internally.

**The ID dtype trap.** `load_ff_playerids()` types `sleeper_id`, `espn_id`, and
friends as `Int64`. The Sleeper API returns those same IDs as **strings**. Joining
across that mismatch doesn't reliably error — it can just match nothing and hand
back a plausible-looking frame. `ids.crosswalk()` casts every `*_id` column to
`Utf8` once so no downstream join has to think about it. This is the single most
likely way a project like this breaks quietly, which is why `ids.match_report()`
exists: print the rate, don't assume it.

The ~62% ECR match rate is expected and fine. The misses are dominated by IDP
(DL 614, DB 549, LB 187), DSTs (169, no `gsis_id` exists), and kickers — none of
which matter for redraft skill positions.

**`ff_opportunity` is the highest-value table here.** It carries `*_exp`
(expected production given the opportunity actually received) alongside actuals,
and `*_diff` = actual − expected. That's the volume-vs-efficiency split most
public projections blur together. Volume is far stickier year over year than
efficiency, so the sign of `*_diff` tells you which one you're buying.

**ADP stdev is the underused column.** Two players at ADP 40 with stdev 3 vs 14
are 0.4% and 28% likely to survive to pick 48. Identical price, completely
different decision. `adp.survival()` reframes the draft from "who's best
available" to "who do I lose if I wait" — which is the question you actually
have on the clock.

**`market_disagreement` has bias in both directions.** Raw `sd` scales with
rank, so sorting on it surfaces deep-league fliers. `rel_sd` (sd/ecr)
over-corrects and inflates the top of the board — on the real 2026 file a naive
rel_sd sort puts Ja'Marr Chase first on an sd of ~1. Neither sort is the answer.
Filter to an ECR band you'd actually draft in, or rank sd within tier.

**`previous_league_id` is the unlock.** Sleeper links each league season to its
predecessor, so `sleeper.draft_history()` walks the chain backwards and returns
years of *your specific leaguemates'* draft behavior from one current league ID.
No paid service sells that, because it's specific to your twelve people. Combined
with `all_transactions()` (FAAB aggression, hold times, positional bias), it's
the closest thing to an actual edge in this repo.

**Scoring is league-specific.** `sleeper.league()` returns `scoring_settings` and
`roster_positions`. Derive replacement level from those rather than assuming
generic PPR — a superflex league with 6-point passing TDs has a different RB2
than every public ranking assumes.

## ADP history

ADP is a moving target through August and the FFC API only serves *today*.
Nobody sells the history, so you have to accumulate it yourself. **It cannot be
backfilled** — `adp.movement()` returns empty until you have two days of
snapshots, and every day you don't run it is a day permanently missing.

```cron
0 6 * * * cd /absolute/path/to/ff-edge && uv run python -c "import adp; adp.snapshot()"
```

Start it now, not in August.
