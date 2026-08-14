# ff-edge — handoff

**Session date:** 2026-08-13 (two sessions, this is the second)
**Branch:** `dropoff-and-landscape`, off `main`. **`main` is finally current** —
`measure-what-repeats` merged and pushed on 08-13 (fast-forward, `673f148`), so
the 07-27 gap is closed and `main` no longer carries the preseason crash.
**State:** 330 tests pass with a league, 324 pass and 8 skip without one.
**The Shiva Bowl draft is 2026-08-22 at 19:00.** Nine days. **Two more drafts
follow it** — see the calendar below.

Read `RESEARCH_SPEC.md` for direction and `BIG_BOARD_SPEC.md` for what was just
built. This file is the state around them.

---

## The one-paragraph version

The earlier session retired the app's premise and wrote `RESEARCH_SPEC.md`. This
one verified that spec against live data and **found it wrong in three places
that mattered**, then built the thing Zach actually needed. `SUPER_FLEX` is real
and still on the roster. The ADP history the spec described as "accumulating
daily" held exactly one snapshot per market. And there are **three leagues, not
one**, across 24 days. Then, while building the Big Board, an
**app-killing crash** turned up that had gone live earlier the same day.

The thing to carry forward is the ADP-independence point, because it is now
load-bearing in code: **only two of the three numbers on the board are
independent of the market.** `par` is derived from the ADP curve, so a composite
that averages `par`, `value_gap` and `vegas_gap` counts ADP twice and calls the
result a second opinion. That is why the Big Board annotates rather than blends,
and it is a correction to `RESEARCH_SPEC.md` §5.2.

---

## The calendar, which the spec did not know about

`bootstrap` reported `my_leagues -> 3`. All confirmed live from Sleeper.

| league | draft | format | status |
|---|---|---|---|
| **Shiva Bowl** | **08-22** | 10T, superflex, 2 keepers, 0.5 PPR | supported |
| **The Jungle** | **08-30** | 10T superflex **dynasty startup**, full PPR, 12 BN, no K/DEF | **out of scope, by decision** |
| **828 Omegle Chat** | **09-06** | 12T, one FLEX, 1 keeper, full PPR | not built yet |

**The Jungle stays out.** Zach's call, and it holds the line `CLAUDE.md` and
`profiles.py` both draw: nothing here estimates an age curve, and a startup
draft is the *worst* case to break that on, since startup pricing is where age
matters most. Do not quietly add a dynasty profile.

**828 is cheap and unbuilt.** It is `standard_12` with `rec: 1.0`. FFC `ppr`/12
ADP is already cached for all eight seasons (2019-2026), so both the board and
the historical curve come from a matching market. It needs the profile selector,
which is the one Phase 0 item still outstanding.

---

## What this session did

| file | what |
|---|---|
| `src/nflverse.py` | **The fix that mattered.** `_played()` filters future seasons before the cache key, turning a fatal raise into an empty frame. Applied to all 11 game-data wrappers. |
| `src/board.py` | `attach_vegas` and `signal` — the two ADP-independent reads, joined and labelled. |
| `src/glossary.py` | 8 new terms. `vegas_gap` had been displayed on the Board tab with no definition at all. |
| `app.py` | `_tab_big_board`, now first in the tab strip. |
| `tests/test_big_board.py` | **New.** 9 unit + 2 driven-tab, the latter ported from `test_draft_day.py`. |
| `tests/test_nflverse.py` | **New.** 20 tests pinning the preseason guard, verified against the unguarded code — 15 of them fail without it. |
| `BIG_BOARD_SPEC.md` | **New.** Spec, then build, then the results written back into it. |
| `CLAUDE.md` | The future-season trap, and why it was invisible. |
| `RESEARCH_SPEC.md` | Unchanged so far — **its §3 and §5.2 both need correcting**, see below. |

Then a second pass on the board, after Zach read it (`BIG_BOARD_SPEC.md` §10):

| file | what |
|---|---|
| `src/board.py` | `positional_drop` — the *shape* of the positional curve beside PAR's *level*. |
| `app.py` | `drop` column, a cost-of-waiting panel on the real pick list, and a "what this board assumes" panel that prints draft demand instead of the format label. |
| `src/glossary.py` | 7 more terms, including `drop`, `cost_of_waiting` and `draft_demand`. |
| `app.py` | **Landscape cut.** Split three ways: dropoff curves to the Big Board, PAR-per-slot and positional mix to Research, concentration-over-time deleted. Five tabs, down from six. |
| `src/landscape.py` | `concentration` marked deliberately unreachable, with its finding, so it is neither re-surfaced nor deleted as an orphan. |

Then a third pass, after Zach challenged the rankings (`BIG_BOARD_SPEC.md` §12):

| file | what |
|---|---|
| `src/board.py` | **`rank_board`.** The old tiebreak inside equal PAR was row order, i.e. ADP — so the board deferred to the market in every tie. Now: block by what the curve can resolve, then `quality_pct` within the block. Also fixed a latent `nulls_last` trap in `build`. |
| `app.py` | `block` leads the table instead of `board_rank`; `quality_pct` shown, because a board must display the number it sorted on. |

And a fourth, on comparables (`BIG_BOARD_SPEC.md` §13):

| file | what |
|---|---|
| `src/archetypes.py` | **`_distance`** — every distance was unweighted while `quality_score` built from the same matrix was stability-weighted. Now both use the weights. `neighbors` gains `restrict_to`. |
| `src/valuation.py` | `comparables` uses the weighted distance too. |
| `app.py` | Block-similarity panel on the Big Board: inside a block the curve calls one asset, which players actually look alike? |
| `tests/test_archetypes.py` | **New.** 7 tests, including that flat weights reproduce the old number exactly. |
| `CLAUDE.md` | The `player_id` namespace trap — FFC Int64 vs gsis String, same column name. |

---

## The findings, compressed

Full reasoning is in `RESEARCH_SPEC.md`. The load-bearing ones:

**Three sharp tools are built and unreachable.** `app.py` imports 16 modules and
**`peek` and `adp` are not among them.** So `peek.regression_candidates` (points
over expected — the canonical buy-low screen), `peek.market_disagreement`,
`peek.snap_trend`, and `adp.movement` (risers/fallers off the `adp_history_*`
snapshots that bootstrap has been accumulating daily) have never been visible in
the dashboard. Same failure as `board.py` sitting unreachable two weeks before
a draft. Cheapest wins available.

**`profiles.py` cannot reach the sidebar.** The multi-league pivot rides on it,
and it already does the hard part — binding a roster format to the ADP market
that prices it. But `app.py` never imports it. The sidebar reads one live league
and exposes teams / points-per-reception / passing-TD only. **Roster shape is
not selectable**, which means the thing that moves replacement level cannot be
changed in the UI.

**`board.py` is two modules.** ~500 of its 1,111 lines are keeper and
pick-ownership plumbing that is genuinely finished. The other ~600 —
`replacement`, `cost_of_waiting`, `tier_map`, `indistinguishable`,
`attach_quality` — is the format-general ranking spine and is what Rankings
would sit on. **Split it; do not delete it.**

**`par`'s within-position ordering is tautological with ADP.** It is derived
from the ADP curve. Its only non-tautological content is cross-position — what a
QB1 is worth against an RB1 at this format's replacement levels — which is
exactly what a multi-league tool needs. Keep it, stop calling it a player
ranking. `app.py:main` already says this in a comment.

**`simulate.py` is 741 lines and should come out of the product.** Pinned to a
format the league does not use, templates that do not match the board, a
headline result inside the noise, and it simulates drafting when the edge is in
managing. The finding survives as a sentence in Method.

---

## Where to pick up

### 1. Run `bootstrap` every day until 09-06. This is the only item with an expiry.

`RESEARCH_SPEC.md` §3 says the ADP history has "been accumulating in the cache
since bootstrap started running daily." **It had not.** Each of the three
history files held exactly one snapshot — 07-27, 07-27, and 08-03 — so
`adp.movement` returned an empty frame, and surfacing it would have rendered the
empty table `CLAUDE.md` warns reads as a network blip.

There are now two snapshots each, so it returns real rows, but the window is
10-17 days wide rather than the labelled 7.

**ADP history cannot be backfilled.** August drift is camp news made numeric and
the fortnight before a draft is the highest-information stretch of the year. A
day not pulled is gone.

**This is now automated.** A launchd agent runs it at 08:00 and 17:00 daily —
`~/Library/LaunchAgents/com.zjserapin.ff-edge.bootstrap.plist`, outside the repo
because it carries the Sleeper handle. Verified end to end on 08-13: 58/58 ok,
and the same-day snapshot was replaced rather than duplicated. See `CLAUDE.md`
→ *The daily bootstrap runs itself*.

**It is not committed and cannot be**, so it exists on this machine only. Check
it rather than re-adding it:

```bash
launchctl print gui/$UID/com.zjserapin.ff-edge.bootstrap | grep -E "runs|last exit"
tail -40 output/bootstrap-daily.log
uv run python -m src.bootstrap --light      # ~90s, by hand if needed
```

**Still worth a manual run on the morning of each draft**, since the 17:00 job
lands two hours before the Shiva Bowl's 19:00 start and ADP moves all day.

### 2. The superflex question is settled, and the answer has a wrinkle

**Confirmed live on 08-13:** the roster is
`QB/RB/RB/WR/WR/TE/FLEX/SUPER_FLEX/K/DEF` + 5 BN. The 2024 and 2025 rosters both
ran two FLEX; 2026 swapped one for `SUPER_FLEX` and dropped a bench slot. That
is a deliberate settings change, so `RESEARCH_SPEC.md` §0 is right: the mocks
pricing it like a 1QB league is the edge existing, not evaporating.

**The wrinkle, found while building the board: the edge is mostly already
spent.** 13 of the top quarterbacks are keepers. QB draft demand is therefore 7,
not 20, and replacement lands at **QB8 among the available** rather than QB21.
The first quarterback on the board is Brock Purdy at **rank 19**, then Mahomes
23 and Herbert 30. (A previous version of this file said "no QB until rank 30" —
that was the first `split`, not the first quarterback.)

**This is why a 1QB profile is the wrong fix and was declined on 08-13.** Measured
three ways: the board already behaves like a 1QB draft; QB draft demand is 7
slots across 10 teams, which is *less* than a 1QB league's 10, so switching would
make quarterbacks **more** valuable rather than less; and a synthetic 1QB profile
reports `kept = 0` and puts all 20 kept players — Josh Allen included — back on
the board. On the ADP market specifically, non-QB ordering correlates 0.974
between the 2qb and 1QB markets with a median shift of 4 places, and all eight
QBs whose price moves most between them are kept. The confusion was real; it was
a labelling problem, now answered by the "What this board assumes" panel.

Both things are true and they are not in tension: the format really is
superflex, and Zach's league-mates already took most of what that was worth by
keeping quarterbacks. The board computes this correctly and always has —
`board.py`'s module docstring describes exactly this case. What needed the
qualifier was the *reading* of §0, which implies an edge still lying around.

### 3. Phase 0 — what is left before 08-22

Additive, no restructure. Rebuilding a 2,900-line app in nine days is how you
arrive at a draft with a broken tool.

- ~~Confirm the Sleeper roster.~~ Done, above.
- ~~One ranked list to prepare from.~~ Done — the Big Board tab.
- **Surface the remaining `peek` screens** — `regression_candidates`,
  `market_disagreement`, `snap_trend` — in an expander on the Research tab.
  `adp.movement` should wait for a few more days of snapshots. Zero risk.
- **Add the profile selector**, which now has a concrete customer: 828 drafts
  09-06 and cannot be priced without it. `RESEARCH_SPEC.md` §7.1.
- **Sit with the Big Board and mark it up.** It exports CSV for this.
- Draft.

Everything else — the `board.py` split, the cuts, the three-surface rebuild — is
Phase 1. The four measurements (M1-M4) are Phase 2.

### 4. Corrections `RESEARCH_SPEC.md` still needs

Not yet applied to that file. All three are established:

- **§3 is wrong about ADP history** accumulating daily. It was not. See item 1.
- **§5.2 counts three independent opinions. There are two.** `par` is derived
  from the ADP curve, so a blend of all three double-counts the market. The Big
  Board resolves this by annotating rather than blending; see
  `BIG_BOARD_SPEC.md` §2.
- **§9 Q1 and Q2 are answered** — three leagues, listed above, with settings
  read live rather than asked for. Q3 (Zach's tab ideas) and Q4 (composite
  weighting) are still open, and Q4 now has a shipped answer to react to.

### 5. Still open, unchanged

- **Sit with the Draft Day tab yourself.** The machine dry run proves it does
  not crash and answers in 0.3s. It cannot tell you whether the *answer* arrives
  in ten seconds. Human test, still unrun, and it expires on the 22nd.
- **Contextual scoring on Reddit.** Unstarted, still wanted, and Zach wants to
  **learn** it — build slowly, explain first, spec before code. `claims.py` has
  the scoring spine; a 5-10 player WR subset makes resolution tractable. Season
  one is labeled-data collection; do not promise predictive validation before
  there is a corpus.
- **`app.py` module split.** Now 2,991 lines and getting worse, not better —
  violates the `Projects/CLAUDE.md` standard. Folds naturally into the Phase 1
  restructure, and the Big Board tab is the natural first thing to lift out
  since it has no dependants.
- **Decide what happens to `main`.** The branch is still the only place this
  work exists.
- **The claims ledger now has 150 rows and every one resolves *pending*** — the
  2026 games do not exist yet, which is correct and documented behaviour. It
  starts producing labels once the season begins, and next August is when the
  layer gets judged. Nothing to do now; do not mistake the pending column for a
  bug, and note that the code path behind it was untested until those rows
  arrived (see `CLAUDE.md`).

---

## What was true before this session and still is

The last session's work stands and is format-independent.

**pyarrow 25.0.0 segfaults the app**, and 270 tests passed against it. Every
table in `app.py` is polars and `st.dataframe` round-trips each through pandas
and back into Arrow; on 25.0.0 that corrupts memory inside `pandas_compat`. No
traceback, no error page — the server is simply gone. It died on the **ninth**
forward render of the pick selector, every time, and never once when the same
fifteen picks were walked in reverse. Heap layout, not data.

Excluded in `pyproject.toml` (fixed upstream in 25.0.1), with two guards both
verified against the real failure by reinstalling 25.0.0 under
`uv run --no-sync`. **Plain `uv run` re-syncs from the lock on every invocation,
which is what keeps the exclusion enforced. Never use `--no-sync` to work around
a dependency problem.**

The general lesson, which outlives the release: **rendering once is not
testing.** The crash needed repetition, not an input, and only fired in one
direction. `tests/test_draft_day.py` is the answer to that class — it drives the
real widget, forward, the whole list. That file is tied to the pick selector and
goes wherever the selector goes, but **port the pattern, do not delete it.**

---

## Things to be careful about

**Read `CLAUDE.md` first.** Every silent-failure trap lives there — team codes
that disagree across sources, `sort(descending=True)` putting nulls first,
FanDuel hiding lines in runner names, the duplicate `ff_opportunity` expectation
columns. None of them raise.

**Never push without asking.**

**`data/` is gitignored and must stay that way.** The repo is public and a league
id exposes nine other people's Sleeper data.

**Shell exports do not reach the Bash tool.** Prefix commands, and set
`FF_EDGE_SLEEPER_USER` alongside `FF_EDGE_LEAGUE_ID` — without the handle the
Draft Day pick selector never renders and its tests skip, so the tab's most
important control goes untested while the suite reports green.

**Season-forward validation only.** Never a random split.

**A negative result is a result.** `breakout.py` and `projection.py` stay.
Do not quietly re-run a measured null with more parameters until it flips — and
note that three of the four new measurements in `RESEARCH_SPEC.md` may well come
back null too. A null ships as a null.

**A vanished test run is a red result**, not a flake. If the pyarrow class of
bug returns, pytest exits 139 with no summary at all.

**Numbers from before 2026-08-09 are stale** — the label window changed to seven
seasons.

---

## Commands

```bash
uv run pytest                                                  # 324 pass, 8 skip
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run pytest    # 330 pass, 2 skip
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
uv run python -m src.peek                                      # the screens, still unreached from the UI
uv run python -m src.bootstrap --light                         # DAILY. see "where to pick up" §1
```

The claims ledger and the ADP history both fill going forward only — the daily
bootstrap is what accumulates them, and **neither can be backfilled.** As of
08-13 the ledger has 150 rows and the ADP histories have two snapshots each.

---

## The docs, and what each is for

| file | read it when |
|---|---|
| `RESEARCH_SPEC.md` | **Current plan.** The critique, the cuts, the build order. Three corrections outstanding — see "where to pick up" §4. |
| `BIG_BOARD_SPEC.md` | The Big Board tab: spec, then what shipped. Why nothing is blended. |
| `HANDOFF.md` | Start of a session. State around the plan. |
| `CLAUDE.md` | Before touching code. Every silent-failure trap. |
| `README.md` | Setup, module map, what the analysis verified. |
| `ANALYSIS_SPEC.md` | The analytical contract. Built; five assumptions corrected inline. |
| `CLAIMS_SPEC.md` | Claims ledger design. Built. |
| `DASHBOARD_SPEC.md`, `_v2.md` | **Product direction superseded** by `RESEARCH_SPEC.md`. Their "what actually shipped" sections are still authoritative and worth reading before re-proposing anything they declined. |
| `src/config.py` | Any question about seasons, paths, league format, TTLs. |
