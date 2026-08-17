# ff-edge — handoff

**Session date:** 2026-08-16. **Shiva Bowl drafts 08-22 at 19:00 — six days.**
**State:** 361 tests pass, 8 skip, without a league. 367 / 2 with both env vars.

**→ The action list is `DRAFT_CHECKLIST.md`.** This file is the state around it.
Read `CLAUDE.md` before touching code; every silent-failure trap lives there.

---

## Read this first: the working tree is behind the work

Two pieces of finished, tested work exist outside local `main`, and both are
fast-forwards.

1. **`main` is three commits behind `origin/main`.** The board in the working
   tree still ranks all 18 tight ends inside the top 100 of a league that
   rosters about 13 — fixed on the remote, not locally.
2. **Two commits are stranded in the `promote-cost-of-waiting` worktree**, on no
   merged branch and pushed nowhere. ~575 lines of tested draft-day surface.

`DRAFT_CHECKLIST.md` A1 and A2 have the commands. **Nothing else in this repo is
worth doing before those two merges**, because everything else is judged against
a board that is three changes stale.

---

## The calendar

| league | draft | format | status |
|---|---|---|---|
| **Shiva Bowl** | **08-22** | 10T, superflex, 2 keepers, 0.5 PPR | supported |
| The Jungle | 08-30 | 10T superflex dynasty startup | **out of scope, by decision** |
| **828 Omegle Chat** | **09-06** | 12T, one FLEX, 1 keeper, full PPR | **needs the profile selector** |

**The Jungle stays out.** Nothing here estimates an age curve, and a startup
draft is the worst case to break that on, since startup pricing is where age
matters most. Do not quietly add a dynasty profile.

---

## Where the board actually stands

The board layers four sources. A row reads left to right as an argument:

| layer | column | what it adds | weight |
|---|---|---|---|
| ADP curve | `par` | the draft slot — blind to player *and* team | — |
| Fantasy Footballers | `ffb_par` | **the player** | 0.5 |
| FantasyPros ECR | `ecr` | the crowd | 0.15 |
| Team environment | `env_swing` | **the team** | 0.35 |
| | `par_env` | what the board is ordered on | |

Then `board.roster_demand` cuts the board at the replacement line. Above the
line the board ranks across positions; below it the order is **ADP's**, and
`block` is null so the reader can see where the tool stops claiming.

Read the correlation honestly: whole-board Spearman against ADP is +0.963,
because 98 of 158 rows are ADP by construction. **Inside the line it is +0.820.**
The board gave up an opinion it could not support about the bottom two thirds
and kept its disagreement where picks are decided.

### The two structural facts that keep coming back

**Only two of the three original numbers are independent of ADP.** `par` is
derived from the ADP curve, so a composite averaging `par`, `value_gap` and
`vegas_gap` counts the market twice and calls it a second opinion. This is why
the board annotates rather than blends, and it is a standing correction to
`RESEARCH_SPEC.md` §5.2.

**The superflex edge is mostly already spent in this league.** 13 of the top
quarterbacks are keepers, so QB draft demand is 7 rather than 20 and replacement
lands at QB8 *among the available*. The format really is superflex **and** the
league-mates already took most of what that was worth. Both are true and they
are not in tension. A 1QB profile is the wrong fix and was declined with
evidence on 08-13 — it would raise QB demand from 7 to 10, making quarterbacks
*more* valuable, which is the reverse of the intent, while silently returning 20
undraftable keepers to the board.

---

## What is built and still unreachable from the UI

Verified by grep this session, not read from a doc. `app.py` imports 16 modules:

| module | status |
|---|---|
| `peek` | **not imported.** Three finished screens unreachable |
| `adp` | **not imported.** `adp.movement` never surfaced |
| `profiles` | **not imported.** Roster shape is not selectable in the UI |

This is the same failure that left `board.py` unreachable two weeks before a
draft. See `DRAFT_CHECKLIST.md` B1, B2 and C1.

---

## The docs, and what each is for

Reduced from ten files to seven this session. Three superseded specs moved to
`docs/archive/` — they are history, and their findings sections are still
worth reading before re-proposing anything they declined.

| file | read it when |
|---|---|
| **`DRAFT_CHECKLIST.md`** | **What to do next.** Ordered by deadline. |
| `HANDOFF.md` | Start of a session. State around the checklist. |
| `CLAUDE.md` | Before touching code. Every silent-failure trap. |
| `BIG_BOARD_SPEC.md` | The Big Board: spec, then eight numbered records of what each change found. |
| `RESEARCH_SPEC.md` | Direction and the build order. Phase 0/1/2. |
| `FOOTBALLERS_SPEC.md` | The Footballers layer. Data built, display half-built. |
| `CLAIMS_SPEC.md` | The claims ledger. Built; a 2027 asset. |
| `README.md` | Setup, module map, what the analysis verified. |
| `docs/archive/` | `ANALYSIS_SPEC.md`, `DASHBOARD_SPEC.md`, `DASHBOARD_SPEC_v2.md` — history. |
| `src/config.py` | Any question about seasons, paths, league format, TTLs. |

---

## Things to be careful about

**Read `CLAUDE.md` first.** Team codes that disagree across sources,
`sort(descending=True)` putting nulls first, FanDuel hiding lines in runner
names, the duplicate `ff_opportunity` expectation columns, `player_id` meaning
two different types. **None of them raise.**

**Never push without asking.**

**`data/` is gitignored and must stay that way.** The repo is public and a
league id exposes nine other people's Sleeper data.

**Shell exports do not reach the Bash tool.** Prefix commands, and set
`FF_EDGE_SLEEPER_USER` alongside `FF_EDGE_LEAGUE_ID` — without the handle the
Draft Day pick selector never renders and its tests skip, so the tab's most
important control goes untested while the suite reports green.

**Anything that reads data runs in the main tree, not a worktree.** `data/` and
`.venv/` are gitignored, so a worktree has neither, and 14 of 16 test files
touch the cache.

**A vanished test run is a red result**, not a flake. If the pyarrow class of
bug returns, pytest exits 139 with no summary at all. `tests/test_draft_day.py`
is the answer to that class — it drives the real widget forward, the whole list,
because rendering once passed and rendering nine times segfaulted.

**Season-forward validation only.** Never a random split.

**A negative result is a result.** `breakout.py` and `projection.py` stay. Do
not quietly re-run a measured null with more parameters until it flips.

---

## Commands

```bash
uv run pytest                                                  # 361 pass, 8 skip
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run pytest    # 367 pass, 2 skip
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
uv run python -m src.peek                                      # the screens, still unreached from the UI
uv run python -m src.bootstrap --light                         # automated twice daily; by hand on draft morning
```

The claims ledger and the ADP history both fill **going forward only** and
neither can be backfilled. The launchd agent at 08:00 and 17:00 is what
accumulates them; it is not committed and cannot be, so it exists on this
machine only. Check it rather than re-adding it — `CLAUDE.md` has the commands.
