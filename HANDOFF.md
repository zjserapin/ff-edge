# ff-edge — handoff

**Session date:** 2026-08-12
**Branch:** `measure-what-repeats`, 27 commits ahead of `main`, and **pushed**.
`origin/main` is still at `5da50ec` from 07-27 — the branch is backed up, main
is not caught up. Those are different problems and the previous handoff conflated
them; see §1.
**State:** 272 tests pass with a league, 269 pass and 5 skip without one. App
renders and, as of this session, *survives being driven*.
**The draft is 2026-08-22 at 19:00.** Ten days.

Read this before doing anything else. It is written to be the whole context.

---

## The one-paragraph version

The dashboard build is finished. Both spec rounds and all six v2 decisions
shipped, plus the sportsbook layer that v2 had recommended declining. What was
left on the list was one item nobody had done: **actually drive the Draft Day
tab instead of rendering it once.** Doing that found a crash that would have
killed the app mid-draft — not a wrong number, a dead process — and it was
invisible to 270 passing tests. That is the whole session.

The thing to carry forward is the same lesson as last session, sharper:
**rendering once is not testing.** The previous session's version was "look at
the output, don't read the code." This session's is **"drive it repeatedly, in
the order a human would."** The crash needed nine renders and only fired in one
direction.

---

## What this session did

| file | what changed |
|---|---|
| `pyproject.toml` | `pyarrow != 25.0.0` — load bearing, see below. `polars >= 1.43.2`, off two yanked releases. |
| `tests/test_draft_day.py` | **New.** The dry run, as a test. Drives the pick selector through all 15 picks and times each. |
| `tests/test_display.py` | Guard that fails loudly if pyarrow 25.0.0 returns through a stale lock. |
| `CLAUDE.md` | The trap, how it hid, and the `--no-sync` / `FF_EDGE_SLEEPER_USER` notes. |

No application code changed. The bug was never in this repo.

---

## The crash, because it is the only thing here that mattered

`HANDOFF.md` asked for a draft-day dry run and said why: the tab *"has never
been driven under time pressure, only rendered … that is a different test from
renders without exceptions, and it is the only one that matters on the 22nd."*

That was right, and more literally than intended.

Walking the pick selector through the fifteen picks Zach owns — 4, 17, 24, 37 …
— **killed the interpreter on the ninth render.** SIGSEGV. No traceback, no
Streamlit error page, no log line. On draft night the tab would simply have
stopped existing, and it would have looked like a laptop problem.

Three properties made it invisible:

- **It needed repetition, not an input.** One render passed. It died in the
  *picks* table, whose contents are byte-identical on every rerun.
- **It was order dependent.** The same fifteen picks walked in reverse never
  crashed, across every attempt. That is heap layout, not data.
- **The suite was green.** 270 unit tests passed against a build that could not
  survive a draft.

**Cause: pyarrow 25.0.0.** Every table in `app.py` is polars, and `st.dataframe`
round-trips each one through pandas and back into Arrow bytes for the browser.
On 25.0.0 that round trip corrupts memory inside `pandas_compat`.

Ruled out by measurement, in this order, because the first three were the
obvious suspects and all three were wrong:

| tried | result |
|---|---|
| Hand Streamlit polars directly, skipping `to_pandas` | crash moved to `st.dataframe`, which converts internally |
| `pd.set_option("mode.string_storage", "python")` | fixed the arrow→pandas leg; crash moved to pandas→arrow |
| pandas 3.0.5 → 2.3.3 | **worse** — died on the 5th render instead of the 9th |
| polars 1.43.1 → 1.43.2 (1.43.1 is yanked) | no change, 6/6 still crashed |
| **pyarrow 25.0.0 → 25.0.1** | **0 crashes in 8 forward runs + a reverse pass** |

Against 9/9 crashes on 25.0.0. Fixed upstream, so `pyproject.toml` excludes the
one release rather than holding the floor down.

**Both guards were verified against the real failure**, by reinstalling 25.0.0
under `uv run --no-sync`: the version guard fails with its explanation, and
`tests/test_draft_day.py` segfaults pytest to exit 139. Plain `uv run` re-syncs
from the lock every time, which is what keeps the exclusion enforced in practice.

### And the tab is fast enough

The other half of the dry run. Cold start ≈10s; **every pick change resolves in
about 0.3 seconds**, all fifteen, against a budget of ten. `test_draft_day.py`
asserts the 10s bound so a moved cache boundary fails a test rather than
surfacing on the clock.

---

## Corrections to the previous handoff

It was written at `3795973` and four commits landed after it. If you read it,
read these too.

**Priority #1 "push" — still open and now larger.** 26 commits, not 11. Origin
is at 07-27, not 08-06.

**Priority #2 "draft-day dry run" — done.** It is `tests/test_draft_day.py` now.
It found the crash above.

**Priority #3 "fill `data/win_totals_2026.csv`" — strike it. It buys nothing.**
The file is 32 rows and 0 filled, and all 32 teams already resolve on `basis =
lines`. `src/expected.py:53` has the measurement: actual wins → team fantasy
points is 0.615, and the game lines already there are 0.619. Actual wins is the
*ceiling* a preseason win total could reach and the free data already matches
it. The file is inert; deleting it is optional tidying, not a task.

**"225 tests pass" — 272 now** (269 + 3 that need a league and a handle).

**The props layer is not in it at all.** Two commits after it was written, and
they reverse its "player props are not free" position — see below.

---

## What is built, as of now

Both dashboard specs are complete. `DASHBOARD_SPEC.md` (08-08) and
`DASHBOARD_SPEC_v2.md` (08-10) each carry a "what actually shipped" section
recording where the spec was wrong; those are worth reading before re-proposing
anything they declined.

Five tabs: `Draft Day | Board | Landscape | Research | Glossary`.

Shipped since the last handoff, beyond v2's six decisions:

- **Quarterbacks are scored.** `valuation.SKILL_POSITIONS` → `VALUED_POSITIONS`
  with QB added, gated on a season-forward measurement taken *before* any code:
  QB n=163, rho +0.367, CI [+0.213, +0.493]. The gate also caught a live bug —
  a QB's `routes` value is *dropbacks*, so the `routes >= 100` filter waved
  quarterbacks through on a column that means something else at their position,
  and Taysom Hill scored at the 25th percentile off six pass attempts.
- **PAR stopped printing precision it never had.** `board.indistinguishable`
  groups players whose PAR gaps sit inside the pooled standard error. On the
  live board the top five receivers are one group.
- **The board notices the draft is happening** — live picks come off it.
- **FanDuel season-long props**, 145 markets across 92 players. This reverses
  v2's D2 ("decline for 2026") on the grounds that its premise was wrong: the
  API is genuinely public. `vegas_gap` is a third opinion, and it is least
  trustworthy exactly where it looks most exciting — see the trap list.

---

## Where to pick up

### 1. Decide what happens to `main`. The push itself is done.

The previous handoff said "11 unpushed" and this one initially said 26, both
measured against `origin/main`. That was the wrong ruler: the branch has its own
upstream and was already pushed. **Nothing is at risk of being lost.**

What is actually outstanding is that `origin/main` has not moved since 07-27, so
every one of these 27 commits lives only on `measure-what-repeats`. That is a
merge or a PR, not a push, and it is a judgement call rather than a deadline —
the app runs off the branch either way. Measure against `@{u}` before claiming
anything is unpushed again.

### 2. Sit with the tab yourself

The machine dry run proves it does not crash and answers in 0.3s. It cannot
tell you whether the *answer* arrives in ten seconds — whether you can look at
Draft Day on the clock and know what to do. That is a human test and it is still
unrun.

### 3. Contextual scoring on Reddit — still unstarted, still wanted

Unchanged from the last handoff. The user wants to **learn** this, so build it
slowly and explain first, and spec it before writing code. `claims.py` has the
scoring spine; a 5-10 player WR subset makes resolution tractable. Season one is
labeled-data collection — do not promise predictive validation before there is a
corpus.

### 4. After the draft, by decision

- `app.py` module split. 2,849 lines, violates the `Projects/CLAUDE.md`
  standard, invisible to the user. Deliberately not done ten days out.
- **L3** — concentration, deeper. Pick one direction, it needs scoping.
- **L4** — concentration into rankings. Measure first, build only if the
  interval clears zero. Note this is *not* the same refusal as positional
  scarcity, which is already in `par` by construction.
- **Weekly props.** `props.weekly()` returns empty by design — FanDuel does not
  post per-game markets until game week. The parser is shared and ready.
- **Strategy → 2027.** Needs 2QB ADP and new templates.
- **Dynasty.** Out of scope: needs an age curve this project does not have.

### 5. The open research question, unchanged and still the best thread

The project measures *ranking* accuracy and is blind to *payoff asymmetry*. The
spread around the ADP curve is more than five times the step between ranks, so
which players land at the top of their tier's distribution is worth far more
than nailing the order — and nothing here has tested whether that is
predictable. Better than re-running a measured null with more parameters.

---

## Things to be careful about

**Read `CLAUDE.md` first.** Every silent-failure trap lives there, including the
new pyarrow entry and *how it hid*, which is the part that generalises.

**Never push without asking.**

**`data/` is gitignored and must stay that way.** The repo is public.

**Shell exports do not reach the Bash tool.** Prefix commands instead — and set
`FF_EDGE_SLEEPER_USER` alongside `FF_EDGE_LEAGUE_ID`. Without the handle the
Draft Day pick selector never renders and its tests skip, so the tab's most
important control goes untested while the suite reports green.

**Season-forward validation only.** Never a random split.

**Numbers from before 2026-08-09 are stale** — the label window changed to seven
seasons.

**Verify by driving, not by rendering.** Last session's rule was to look at
output rather than read code; this session sharpens it. Render once and you
learn nothing about a draft. `streamlit.testing.v1.AppTest` walking a widget
forward, repeatedly, is what caught a crash that 270 tests missed. For charts,
render to PNG and *look* — `uv add vl-convert-python`, then remove it again.

**A vanished test run is a red result**, not a flake. If the pyarrow class of
bug returns, pytest exits 139 with no summary at all.

---

## Where to pick up

```bash
uv run pytest                                                  # 269 pass, 5 skip
FF_EDGE_LEAGUE_ID=... FF_EDGE_SLEEPER_USER=... uv run pytest    # 272 pass, 2 skip
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
```

The claims ledger fills going forward only — `uv run python -m src.bootstrap
--light` daily during camp is what accumulates it.
