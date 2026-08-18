# ff-edge — handoff

**Session date:** 2026-08-18. **Shiva Bowl drafts 08-22 at 19:00 — four days.**
**State:** 451 tests pass, 2 skip (with both env vars). **`main` is the only branch.**

---

## New 08-18: there is a website, and Streamlit is untouched

`web/` is a FastAPI + Jinja + htmx site over the same `src/` modules — see
`WEB_SPEC.md`. **All five pages are built**: Big Board, Draft Day, Player,
Research and Reference.

**The Player page is new and closes checklist E1** — the structural gap that
cost more than anything else in the file. Researching a player meant five
selectboxes across four tabs joined by eye; it is now one URL,
`/player?name=...`, carrying the board row, the four layers as an argument,
last season's role against his position's median, and his nearest comparables.

Five sections stay on Streamlit deliberately, all read once rather than on a
clock: the two measured nulls (`breakout`, `projection`), the rookie model, the
strategy simulator, and the claims ledger. The Research page says so on screen.

```bash
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run uvicorn web.server:app --reload
```

**`app.py` was not modified — not one line.** Draft the Shiva Bowl from
Streamlit on the 22nd. The website targets 09-06 (828), which is also when the
profile selector it already carries stops being optional. Retire Streamlit only
after the site has been used through a real draft.

Two things worth knowing about it:

- **State is the URL**, not session state. `?profile=&positions=&usage=1` —
  bookmarkable, and the five-selectboxes-across-four-tabs problem (E1) cannot
  reproduce here.
- **`st.dataframe`'s pandas↔Arrow round-trip is gone from the render path**, so
  the pyarrow 25.0.0 crash class does not exist on this surface. The exclusion
  in `pyproject.toml` stays while `app.py` lives.

**Building it found a live defect: an unset `FF_EDGE_LEAGUE_ID` boards The
Jungle, not the Shiva Bowl.** See `DRAFT_CHECKLIST.md` **A0** — it is the first
thing to read before the 22nd, and `app.py` gives you no way to notice it.

Two more found by rendering rather than by asserting, both recorded in
`WEB_SPEC.md`: a broad `except` silently deleted the whole stability section
(sections now name their failure instead of disappearing), and Vega's
`width: "container"` draws a **blank chart rather than raising** (charts are
sized in pixels, and a test forbids the string). The pattern is old news here —
Block B's season literal and wrong ADP market were the same shape — and it is
the argument for A3: **look at the thing.**

**→ The action list is `DRAFT_CHECKLIST.md`.** This file is the state around it.
Read `CLAUDE.md` before touching code; every silent-failure trap lives there.

---

## The branch cleanup is done, and one merge mattered a lot

Nine branches merged into `main` and deleted, four worktrees removed. Every
delete used `git branch -d`, which refuses on unmerged work — so nothing was
lost, verified structurally rather than by inspection.

**The stranded `promote-cost-of-waiting` commits carried `board.roster_need`,
and it fixed a board that was giving actively bad advice.** The board had only
ever modelled *league scarcity*, never *your need*. On the 2026 board quarterback
reads as 7 slots against 20 and therefore screaming scarce — true, and not your
problem, because **two of the thirteen kept quarterbacks are yours** (Jayden
Daniels, Trevor Lawrence) against a roster with exactly two QB-capable slots.

The inversion is the part to remember: **league scarcity is maximally misleading
to the manager who caused it.** Teams that make a position scarce by keeping one
are precisely the teams that must not draft one. The board was telling you to
buy the thing you already own, at pick 4.

**Two remote branches survive** and `main` is **10 commits ahead of
`origin/main`**, all unpushed — `CLAUDE.md` gates every push on explicit
approval and none was given.

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

Three superseded specs moved to `docs/archive/` on 08-16 — they are history, and
their findings sections are still worth reading before re-proposing anything
they declined. Two more docs arrived on 08-17 with the branch merges.

| file | read it when |
|---|---|
| **`DRAFT_CHECKLIST.md`** | **What to do next.** Ordered by deadline. Block E is the research-readiness list. |
| `HANDOFF.md` | Start of a session. State around the checklist. |
| `CLAUDE.md` | Before touching code. Every silent-failure trap. |
| `BIG_BOARD_SPEC.md` | The Big Board: spec, then eighteen numbered records of what each change found. |
| `RESEARCH_SPEC.md` | Direction and the build order. Phase 0/1/2. Read its correction block first. |
| `FOOTBALLERS_SPEC.md` | The Footballers layer. Data built, display half-built. |
| `CLAIMS_SPEC.md` | The claims ledger. Built; a 2027 asset. |
| `HOW_IT_WORKS.md` | **New 08-17.** The pipeline's shape. Its tab walkthrough is stale and says so at the top. |
| `FANTASYPROS_IDEAS.md` | **New 08-17.** Brainstorm. Holds the unused seven-years-of-ECR finding. |
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
