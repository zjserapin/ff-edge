# CLAUDE.md — ff-edge

Project-specific rules. Inherits `Projects/CLAUDE.md`; this file wins where they
conflict.

**Read this before touching any code, including in a subagent.** Most of what is
below describes ways to be *silently* wrong in this repo — bad joins that return
nulls instead of errors, sign conventions that are backwards from the obvious
one, columns that look like duplicates but aren't. None of it raises.

---

## The docs, and what each is for

| file | read it when |
|---|---|
| `HANDOFF.md` | Start of a session. Current state, findings, what's next. |
| `RESEARCH_SPEC.md` | Current product direction (2026-08-13). Supersedes both dashboard specs on layout and scope; their "what shipped" sections still stand. |
| `BIG_BOARD_SPEC.md` | The Big Board tab. Built 2026-08-13. Why there is no blended score, and why only two of the three numbers are independent of ADP. |
| `README.md` | Setup, module map, what the analysis has verified. |
| `ANALYSIS_SPEC.md` | The analytical contract — what gets measured and how. |
| `CLAIMS_SPEC.md` | Design contract for the claims ledger. Marked built. |
| `src/config.py` | Any question about seasons, paths, league format, TTLs. It is heavily commented and is the source of truth. |

---

## Running things

**Shell exports do not reach the Bash tool.** `export FF_EDGE_LEAGUE_ID=...` in
the user's terminal is invisible here. Prefix every command that needs it:

```bash
uv run pytest                                          # 318 pass, 8 skip w/o a league
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run pytest   # 324 pass, 2 skip
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_LEAGUE_ID=... uv run python -c "from src import board; print(board.build())"
uv run python -m src.bootstrap --light                 # daily cache refresh
```

Env vars that matter: `FF_EDGE_LEAGUE_ID`, `FF_EDGE_SLEEPER_USER`,
`ANTHROPIC_API_KEY` (news extraction only — depth-chart claims work without it).

**`FF_EDGE_SLEEPER_USER` gates more than it looks like it does.** Without the
handle, `board.picks` resolves no owner, the Draft Day pick selector never
renders, and `tests/test_draft_day.py` skips — so the tab's most important
control goes untested while the suite reports green. Set both vars together.

### The daily bootstrap runs itself — check it, don't re-add it

A launchd agent runs `bootstrap --light` at **08:00 and 17:00 daily**. It lives
at `~/Library/LaunchAgents/com.zjserapin.ff-edge.bootstrap.plist` — **outside
the repo, deliberately**, because it carries the Sleeper handle and this repo is
public. Nothing about it is committed, so a fresh clone has no agent and a
future session should not assume one.

```bash
launchctl print gui/$UID/com.zjserapin.ff-edge.bootstrap | grep -E "runs|last exit"
tail -40 output/bootstrap-daily.log     # gitignored; last run's full inventory
launchctl kickstart -k gui/$UID/com.zjserapin.ff-edge.bootstrap   # run it now
launchctl bootout   gui/$UID/com.zjserapin.ff-edge.bootstrap      # stop it
```

**launchd rather than cron, for one specific reason:** cron silently skips a run
if the laptop is asleep at the scheduled minute, and `adp_history_*` cannot be
backfilled — a missed day is gone permanently. launchd runs a missed calendar
job when the machine next wakes. The twice-daily schedule is insurance on top of
that; `adp.snapshot` replaces the same-day row rather than appending, so the
second run refreshes the day with fresher numbers and never double-counts.

It needs **only `FF_EDGE_SLEEPER_USER`** — `bootstrap` discovers every league
from `sleeper.my_leagues()`, so no league id is stored anywhere on disk.

A run that logs anything other than `N/N ok` is a real failure. The most likely
causes are no network and an expired Sleeper handle, and both are silent in the
app rather than loud: the cache simply goes stale.

**Never use `uv run --no-sync` to work around a dependency problem.** Plain
`uv run` re-syncs from the lock on every invocation, which is the mechanism that
keeps the pyarrow exclusion below actually enforced. `--no-sync` exists here for
one purpose: deliberately installing a known-bad version to prove a guard fires.

Worktrees: `data/` and `.venv/` are gitignored, so a git worktree has neither.
14 of 16 test files touch the cache. **Anything that reads data runs in the main
tree, not a worktree.**

---

## Non-negotiables

- **The repo is public.** `data/` is gitignored and stays that way — a league id
  exposes nine other people's Sleeper data. League ids and handles come from the
  shell, never a committed file.
- **Never push without explicit approval in the current conversation.**
- **Season-forward validation only.** Never a random split. A random split leaks
  future seasons into the training window and every result becomes fiction.
- **`src/llm.py` is the only file that touches a model client.** Provider swaps
  happen in `config.py` (`LLM_PROVIDER`). Prompts live in `src/prompts.py`,
  versioned, never inline.

---

## Data traps — every one of these fails silently

These were each found the hard way. Do not reintroduce them.

**pyarrow 25.0.0 segfaults the app, and 270 tests pass against it.** Excluded in
`pyproject.toml`; fixed upstream in 25.0.1. Every table in `app.py` is polars,
and `st.dataframe` round-trips each one through pandas and back into Arrow bytes
for the browser. On 25.0.0 that corrupts memory and takes the interpreter with
it — SIGSEGV, no traceback, no Streamlit error page, the server just gone.

Worth internalising *how* it hid, because it generalises past this one release:

- It needed **repetition**, not an input. Rendering once passed. Rendering nine
  times, changing the pick each time, died — in the picks table, whose contents
  are identical on every rerun.
- It was **order dependent**. The same fifteen picks walked in reverse never
  crashed once. That is heap layout, so any test free to choose a convenient
  order would have stayed green.
- The whole unit suite passed against a build that could not survive a draft.

`tests/test_draft_day.py` is the answer to that class and exists for this
reason: it drives the real widget, forward, the whole list. **A failure there
may not look like a failure** — pytest exits 139 with no summary. A test run
that vanishes is a red result.

**Asking nflverse for a season that has not kicked off raises, and it takes the
whole app.** Eleven of the loaders wrapped in `src/nflverse.py` reject a future
season — `ValueError: Season must be between 2006 and 2025` — and two more 404.
`src/nflverse._played` now filters those seasons out *before* the cache key is
built, so they return an empty frame, which is what every caller here already
handles. **Do not remove that guard, and add it to any new wrapper** over a
game-data loader. Forward-looking feeds are exempt and must stay exempt:
schedules, rosters, depth charts and draft picks all have real 2026 rows.

Use `nfl.get_current_season()`, the games-based answer. Its `roster=True`
variant flips on March 15 and reports 2026 today — the same roster-runs-ahead
disagreement as the `AZ`/`ARI` split below.

Worth studying *how this one hid*, because nothing about it was silent:

- **The handling was already written.** `promotion.weekly_trust` guards with
  `if not rush_raw.height`, and `claims.resolve` already treats an empty week
  table as *pending, not failed*. Both sat one line after the call that raised.
  The author assumed a future season came back empty rather than throwing.
- **It was dormant until data arrived.** `claims.resolve` returns early on an
  empty ledger, so the path was unreachable until `bootstrap` pulled the first
  150 claims. The bug shipped on the day the ledger stopped being empty — nine
  days before the draft — and nothing about that day touched the code.
- **It was invisible to a green suite**, because the app-level tests were the
  only ones that render `app.py`, and an unhandled exception in a Streamlit tab
  kills the entire page rather than the section.

The general lesson, and it is the same one `tests/test_draft_day.py` teaches
from the other direction: **a code path guarded by `if not x.height` is untested
until something makes `x` non-empty.** A cache that fills, a ledger that
accumulates, or a season that starts will each do that on their own schedule.

**`config.ROOT` uses `parents[1]`, not `.parent`.** `src/config.py` lives in
`src/`. Using `.parent` repoints `DATA_DIR` at `src/data`, creates it, and
orphans the entire cache without raising.

**`spread_line` in nflverse is positive when the *home* team is favoured** —
the opposite of betting convention. Read `spread` from
`expected.team_environment`, never `spread_line` raw.

**Team codes disagree across sources, and nflverse disagrees with itself.** FFC
says `LAR` where nflverse says `LA`. Worse, the **2026 roster feed alone spells
Arizona `AZ`** — the 2025 rosters, weekly stats, schedules and draft picks all
say `ARI`. Both are mapped in `ids._PFR_TEAMS`.

Run **both sides** through `ids.normalize_team`, **every time**, and note that
"both sides" includes a bare equality check, not just a `join`. The live bug was
`rookies.py` deciding whether a player changed teams with
`new_team != team` — roster on one side, weekly stats on the other. Every
Cardinal who stayed read as *gone*, so the whole team's production landed in the
vacated pool: Arizona's vacated target share came out 1.00, first in the league,
against a corrected 0.209 that is *below* the league mean. The analysis pointed
at the opposite of the truth.

The failure mode is a null row or a silently inverted flag, never an exception.
A test that normalizes one side and compares against a raw feed is testing half
the join and will pass right up until the day a feed changes.

**A traded pick's `roster_id` is whose pick it *originally* was**, not who holds
it now. Reading it the other way puts picks in entirely the wrong rounds.

**The nflverse teams table has 36 rows, not 32** — it carries relocated
franchises. Key team sheets off the season schedule instead.

**`rank(descending=True)` gives rank 1 to the *largest* value.** Check the
direction every time; ADP rank and points rank run opposite ways.

**`sort(descending=True)` puts nulls FIRST.** Polars defaults `nulls_last=False`,
so a "best N" query returns the N rows that could not be scored — deep players
with no ADP-curve match, every time. Pass `nulls_last=True` on any descending
sort meant to surface a top. `expected.tiers` drops null values before sorting,
which is why the real board is unaffected and an ad-hoc query is not.

**Both `ff_opportunity` pbp tables carry weeks 19-22 with no `season_type`.**
Filter weeks explicitly.

**FanDuel prop markets carry `handicap: 0` and hide the line in the runner
name.** `"Bijan Robinson Over 1150.5"` is where the number lives. Reading
`handicap` yields a full, well-typed board on which every line is 0.0. Two more
in the same feed: all 97 season-long *yardage* markets are priced -114/-114, so
de-vigging them returns exactly 0.500 — the absence of a signal, not a
probability — and `marketType` is a **bucket, not a position** (Bowers, Kittle
and Loveland are all filed under `WIDE_RECEIVERS`).

**`player_id` means two different things, and they are different types.** The
draft board's `player_id` is **FFC's, an Int64**; `archetypes.scores`,
`features.build` and the nflverse tables key on **gsis_id, a String**
(`00-0034857`). Joining one to the other without renaming leaves two columns
sharing a name, polars silently suffixes the newcomer to `player_id_right`, and
reading `player_id` back hands you the wrong namespace. It surfaces downstream as
`[(col("player_id")) == (dyn int: 5683)]` against a String column, a long way
from the join that caused it.

**Alias on the way in** — `pl.col("player_id").alias("gsis_id")` — rather than
relying on join suffixes. `board.attach_quality` sidesteps this entirely by
joining on the normalized name and never carrying an id across; that is the
pattern to copy.

**Generational suffixes collapse father onto son.** `ids.normalize` strips
`Jr./Sr.`, so Michael Pittman Jr. (WR, 2020) and Michael Pittman Sr. (RB, 1998,
no `gsis_id`) share one key. Combined with defenders who share a name — Lamar
Jackson the CB, Justin Jefferson the rookie LB — a name-only join fanned 145
prop rows out to 151 without raising. `props.resolve_players` resolves one
identity per *player* rather than per row, which is what makes the row count
preserved structurally instead of by luck.

More play-level traps (two-point plays inflating red-zone share, `play_id` type
mismatch between ff_opportunity and FTN, null receivers on 4% of pass plays) are
documented in the `src/context.py` module docstring. Read it before doing
anything play-level.

**The `ff_opportunity` *pbp* rush table has duplicate expectation columns** —
`rushing_td_exp`/`rush_touchdown_exp` and `rushing_yards_exp`/`rush_yards_exp`.
They disagree on 4,558 rows, and it is not noise: the `rush_*` pair carries
sentinels (0.0, -1.0) where the `rushing_*` pair has a real modeled value. **Use
the `rushing_*` columns.** See `src/context.py:48`.

Note this is the *pbp* table only. The **weekly** ff_opportunity table is a
different column namespace, and `scoring.EXPECTED_COLUMNS` correctly reads
`rush_touchdown_exp` there. Don't "fix" it.

---

## Codebase conventions

- **Polars, not pandas.** All 25 data modules use `polars as pl`. pandas is a
  transitive dependency only — do not introduce it into `src/`.
- **`config.py` is the source of truth.** Never hardcode a season, a path, a
  roster slot, or a scoring value. Rolling to a new season should be a one-line
  edit, not a grep.
- **Comments explain why, not what.** This codebase's existing comments are the
  house style — they record the reasoning and the traps, not the syntax. Match
  that density.
- **Tests brute-force claims rather than asserting them** where feasible. See
  `test_lineup_selection_is_optimal`, which verifies the greedy flex allocation
  against exhaustive search instead of trusting it.
- **Corrections stay checkable.** `board.compare_baselines` runs the keeper
  adjustment both ways on purpose — if a correction ever reads near zero, the
  complexity is not earning its keep and should come out.

### Formats come from `src/profiles.py`, never from loose arguments

A `LeagueProfile` carries a roster format **and the ADP market that prices it**
as one object, because the 2026 superflex bug was exactly what happens when
those two drift apart. Never set a roster format without setting the market;
`profiles.customize` exists for one-off variants.

```bash
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
```

Two built-ins: `shiva_bowl` (default, live from Sleeper) and `standard_12`.
`resolve()` raises on an unknown name rather than falling back — a typo that
silently returned the Shiva Bowl would price a standard league as a superflex
keeper league and look entirely plausible doing it.

`board.build()` returns `warnings` alongside the frames. An unpriceable profile
must produce a sentence, not an empty DataFrame that looks like a network blip.

**Redraft only, deliberately.** Everything here prices a player against a market
for one season. Dynasty is a different question — which young players are worth
building on — and it turns on age curves and quality signals rather than this
season's price. Nothing in this project estimates an age curve, so a dynasty
profile would answer the redraft question under a dynasty label. Don't add one.

### League format is superflex as of 2026

`QB/RB/RB/WR/WR/TE/FLEX/SUPER_FLEX/K/DEF` + 5 BN, 10 teams, 0.5 PPR.

Flex slots fill **most restrictive first** (`config.FLEX_SLOTS`), which is what
makes greedy allocation exactly optimal when the eligibility sets nest. This
league's sets nest; `REC_FLEX` and `WRRB_FLEX` would not, and both functions say
so out loud.

**`config.SIM_ROSTER_POSITIONS` is deliberately pinned to the old two-FLEX
format.** The strategy simulator's board is 1QB ADP and its templates are
two-FLEX shaped, so running it under superflex would report that QBs are nearly
free *and* start twice. That pin is intentional — do not "fix" it without also
swapping in 2QB ADP and rewriting the templates.

---

## Analysis standards

- **Report intervals, not point estimates.** The binding constraint on this
  project is label seasons (2.5-6.5 events per variable across 831 labeled
  player-seasons), not algorithms.
- **A negative result is a result.** `breakout.py` and `projection.py` are both
  honest nulls and stay in the repo as such. Do not quietly delete or re-run a
  measured null with more parameters until it flips.
- **Fitted models have repeatedly failed to beat ADP here.** Before proposing
  one, say where the extra data is coming from. Week-level and play-level tables
  are orders of magnitude larger than the season-level label set; that is where
  learning has a chance.
- **Don't invent an ordering the data doesn't support.** The ADP curve is not
  monotone; isotonic regression pools the offending ranks rather than forcing a
  rank order. Tiering here is an empirical result, not a presentation choice.

---

## Working style for this project

The user is building this to *learn*, not just to have it. Per
`Projects/CLAUDE.md`: explain before building, block by block, and lead with the
analogy. That applies with extra force to the contextual-scoring and ML threads,
which he named as things he wants to understand rather than receive.

Dashboard work (`app.py`, 1800+ lines, 6 tabs) is a **spec exercise before it is
a build exercise**. Write the spec to a file and get agreement before editing.

*This file evolves. Update it when the traps or the format change.*
