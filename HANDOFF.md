# ff-edge — handoff

**Session date:** 2026-08-09
**Branch:** `measure-what-repeats`, 21 commits ahead of `main`.
**11 of them unpushed** — origin is at `10eb558` from 08-06.
**State:** 225 tests pass, 1 skips cleanly without a league. App renders with
zero exceptions. Working tree clean.
**The draft is 2026-08-22 at 19:00.** Twelve days.

Read this before doing anything else. It is written to be the whole context.

---

## The one-paragraph version

The previous session built the analysis; this one built the **product** and then
spent most of its time finding out where the product was lying. A Draft Day tab
now surfaces `board.py`, which was fully built and completely unreachable two
weeks before a draft. A league-profile layer makes the format and its ADP market
inseparable, because them drifting apart is what caused the superflex bug. PAR
turned out to be a rating of the *draft slot* rather than the player, so the
board grew a second opinion that is not derived from ADP. Four real bugs were
found and fixed, three of them by tests that were themselves only testing half
of what they claimed. The seventh label season was added at the end.

The thing to carry forward: **every significant find this session came from
looking at output, not from reading code.** Rendering a chart to PNG found the
zero-baseline. Running the screen on an injured player found the windowing
artifact. A failing test nobody expected found Arizona.

---

## What is new, by module

| module | what changed |
|---|---|
| `src/profiles.py` | **New.** A format and its ADP market as one indivisible choice. `shiva_bowl`, `standard_12`. |
| `src/board.py` | `keeper_adjusted_adp`, `keeper_slots`, `attach_quality`, `cost_of_waiting`. |
| `src/adp.py` | `survival` takes `adp_col` — availability must be measured in pick numbers, not ADP. |
| `src/landscape.py` | `tier_breaks` — where the dropoff actually cliffs. |
| `src/promotion.py` | `TRUST_METRICS`, `role_shift`, receiver metrics in `weekly_trust`. |
| `src/valuation.py` | Takes a profile, so `market_pct` is priced in the league's own market. |
| `src/ids.py` | `AZ → ARI`. |
| `src/rookies.py` | Team codes normalized on **both** sides of the stayed/left comparison. |
| `app.py` | Draft Day tab; Screen retired into it; Landscape rebuilt; `width="stretch"`. |
| `CLAUDE.md` | **New.** The traps, the conventions, the non-negotiables. |
| `DASHBOARD_SPEC.md` | **New.** The agreed spec, both rounds, and item 5's scoping. |

---

## The four bugs, because they are the most useful thing here

### 1. Arizona did not exist (silent, live, worst of the four)

nflverse's **2026 roster feed alone** spells Arizona `AZ`; its own 2025 rosters,
weekly stats, schedules and draft picks all say `ARI`. `rookies.py` decided
whether a player changed teams by comparing those two codes directly — not a
`join`, just `new_team != team` — so **every Cardinal who stayed read as gone**
and the whole team's production fell into the vacated pool.

    Arizona vacated target share    1.00   (would rank 1st in the league)
    corrected                       0.209  (league mean 0.249 — below average)

The rookie opportunity analysis was pointing at the exact opposite of the truth.

**The test that should have caught it was testing half the join** — it normalized
one side and compared against a raw feed, and passed for as long as nflverse
agreed with itself.

### 2. Availability was measured against the wrong number line

`adp.survival` compared **raw ADP against a pick number**. Public ADP is priced
where nobody is kept; a real pick number is not. `board.targets` and the
Draft Day availability panel both sat on it.

    James Cook, P(available at pick 24)    0.137 raw    0.003 adjusted

Fixed by giving `survival` an `adp_col` and passing `exp_pick`. Fixing it
**changed the answer** to the feature built on top: on raw ADP the early
quarterback wait looked cheap; adjusted, it is the most expensive wait on the
board.

### 3. `path_score` never measured what its test claimed

It averages room-to-grow against a reversed teammate share, and **both are true
of a team alpha at once** — no room left, nobody in front of him. They cancel:

    opportunity_pct   alpha 76.9  vs other 43.9   gap +32.9
    path_score        alpha 50.7  vs other 50.4   gap  +0.3

The old test asserted the composite and passed on **a single running back**; WR
and TE already leaned the wrong way. The invariant is now asserted where it
lives, and a second test pins the limitation so reweighting fails loudly.
**The formula was not changed — that is a judgment call, not a bug fix.**

### 4. A four-game season reported a confident zero

`role_shift` with a four-week window on a player who played four weeks returns
the same four weeks twice and a delta of exactly `0.000` on every marker — an
artifact that reads like a finding of "no change". Windows now shrink to at most
half the observed weeks and refuse a season under four.

---

## Findings, in order of how much they should shape what you do next

### 1. PAR is a rating of the draft slot, not of the player

`exp_points` maps a player's *positional ADP rank* to what players at that rank
historically scored, so **within a position the board reproduces the market's
order exactly**.

| | 2026 ADP | pos rank | PAR | 2025 actual | 2025 finish |
|---|---|---|---|---|---|
| Nabers | 60.2 | WR21 | 14.2 | 48.1 | WR81 *(4 games)* |
| Waddle | 72.4 | WR25 | 0.0 | 148.1 | WR9 |

Waddle outscored Nabers by 100 points and carries the worse PAR, entirely
because he is drafted twelve picks later. This is `board.py` working as designed
and is the honest consequence of the measured null — but the tab presented it as
a player rating, which it is not. There is now a warning callout and a second
opinion beside it.

### 2. `value_gap` is the second opinion, and it is not derived from ADP

Quality percentile within position minus price percentile within position,
weighted by how much each metric repeats. Adams comes back at quality **7**
against a market percentile of **72**; Shakir the reverse.

**Quarterbacks are not scored** — yards per route run, separation and yards
after contact have no QB analogue, so all 23 come back null. In a superflex
league that is a hole exactly where the edge is. Blank means *not measured*.

### 3. Waiting is free at tight end and expensive at quarterback

`board.cost_of_waiting`, expected PAR of the best player of a position still
available at each pick you own:

| pos | 4→17 | 17→24 | 24→37 | 37→44 |
|---|---|---|---|---|
| QB | **20.3** | 4.4 | 3.1 | 4.5 |
| RB | 3.5 | 11.6 | **19.5** | 13.0 |
| WR | 14.4 | 15.5 | 10.5 | 0.4 |
| TE | 0.0 | 0.1 | 0.4 | 0.5 |

Tight end costs **0.3 points across all six picks**. Running back is cheap until
24 and expensive after. The 24 → 37 gap is where your draft happens.

### 4. The dropoff is a slope, not a staircase

Running back falls 1.8, 1.0 and 1.7 points per game across the first four ranks,
then settles at **0.68 per rank, range 0.50-0.98**, the whole way to 48. **RB12
and RB24 are not in the data.** Reaching pays at the very top and steadily less
after — a different instruction than "reach for the tier break".

### 5. Concentration has barely moved, and tight end moved the wrong way

Over 2018-2025 QB, RB and WR all shifted under two points. Tight end went
**down**: top five from 31.7% of the position to 23.9%, top fifteen from 64.7%
to 58.6%. The elite-tight-end premium claims a few players own the position;
over this window the position spread out. Consistent with §3.

### 6. Positional scarcity does not trend, so it was not fed into rankings

Spearman of PAR-per-starting-slot against season, bootstrapped:

| pos | ρ | 95% CI |
|---|---|---|
| QB | +0.05 | [−0.62, +0.93] |
| RB | +0.10 | [−0.62, +1.00] |
| WR | −0.26 | [−0.79, +0.86] |
| **TE** | **−0.90** | **[−0.95, −0.36]** |

Three of four are indistinguishable from flat. And the *level* of scarcity is
already in the board by construction — `par = exp_points − replacement`, and
replacement comes from `starter_demand`. Adding a second scarcity term would
count one fact twice. **Scoped and declined; see `DASHBOARD_SPEC.md`.**

### 7. Carried forward, unchanged

Superflex demand (QB 10 → 20, replacement QB11 → QB21); keeper adjustment
matters only at QB (+47.2); keepers shift everyone ~15 picks earlier; team
environment routinely outweighs board edges (45.3 points per point of implied
team total); RB promotion criteria (snap share 0.47, red-zone carry share 0.34,
prior efficiency reads as noise).

---

## The 2025 label season — decision recorded

**Taken 2026-08-09, twelve days before the draft, with the churn understood.**
FFC backfilled 2025 after the previous session recorded it as permanently
missing, so `ADP_MISSING_YEARS` was stale and `LABEL_SEASONS` was one season
short of what the data supported.

    labelled player-seasons     831 -> 957
    test folds                    4 -> 5
    out-of-sample rows          540 -> 666   (+23%)

**Every conclusion is unchanged**, which is the honest headline:

    breakout   AUC 0.528 vs 0.493 price     delta +0.035  CI [-0.014, +0.083]
    projection rho 0.497 vs 0.494 price     delta +0.002  CI [-0.008, +0.010]

The continuous interval tightened by about a sixth. The null is now measured on
666 rows rather than 540. QB remains significantly *worse* than price
(−0.066, CI [−0.152, −0.008]).

**Why it was taken now rather than after the draft:** it only moves estimates,
never the shape of any conclusion, and better estimates before drafting beat
better estimates after. The risk accepted is that every published number moved
slightly, so **any figure quoted from a session before 2026-08-09 is stale.**

**Not re-run on the new window:** the four "things that were tried and did not
help" in `breakout.py` are deltas against the old 0.513 baseline. They are
labelled as six-season results. Directions, not current figures.

---

## The league, as the code now sees it

Pulled live. **All 10 teams have now declared — 20 keepers, not 18.** The
previous handoff's "one undeclared team" is stale.

- **Shiva Bowl**, 10 teams, 0.5 PPR, `QB/RB/RB/WR/WR/TE/FLEX/SUPER_FLEX/K/DEF`.
- His keepers: **Jayden Daniels (R6) and Trevor Lawrence (R9)**. Daniels is in
  year two, so 2026 is his last.
- Draft slot 4. **17 picks owned, 15 usable.**
- The structural edge: 6 teams need 7 QB-capable starters from ~3 startable
  quarterbacks. He needs zero.

---

## Where to pick up

Nothing is half-built. Every item from both spec rounds is done or deliberately
deferred. In rough priority order:

### 1. Push. Twelve days out and unpushed since 08-06.

Eleven commits sit local, everything this session built. `Projects/CLAUDE.md`
requires explicit approval, so **ask** — but ask early rather than the night
before the draft.

### 2. Draft-day dry run

The tab has never been driven under time pressure, only rendered. Sit with it,
walk the pick selector through 4 → 17 → 24, and see whether the answer arrives
in the ten seconds a real pick allows. That is a different test from "renders
without exceptions", and it is the only one that matters on the 22nd.

### 3. Fill `data/win_totals_2026.csv`

Written and blank. The one manual input that would improve the preseason
environment layer, and the best available proxy for "will this team still be
trying in December".

### 4. Contextual scoring on Reddit — still unstarted, still wanted

The user wants to **learn** this, so build it slowly and explain first. Spec it
before writing code. `claims.py` has the whole scoring spine; Reddit's API is
the friendlier of the two free options; a 5-10 player WR subset makes the
resolution step tractable. Season one is labeled-data collection — do not
promise predictive validation before there is a corpus.

### 5. ML/DL — the constraint is unchanged and now better measured

Adding a seventh label season moved nothing, which is more evidence for the
existing conclusion: **the binding constraint is label seasons, not algorithms.**
Where the volume actually is: week-level and play-level tables, claim extraction
once the ledger fills, and change-point detection over weekly usage.

**The one genuinely open question this session raised:** the project measures
*ranking* accuracy and is blind to *payoff asymmetry*. The spread around the ADP
curve is more than five times the step between ranks, so which players land at
the top of their tier's distribution is worth far more than nailing the order —
and nothing here has tested whether that is predictable. That is a better thread
than re-running a measured null with more parameters.

### 6. Deferred by decision

- **Strategy tab → 2027.** Only useful as a live co-pilot; needs live rankings,
  roster state and league draft rates.
- **Dynasty.** Out of scope: it needs an age curve this project does not have.
- **Scarcity into rankings.** Scoped and declined — §6 above.

---

## Things to be careful about

**Read `CLAUDE.md` first.** Every silent-failure trap is there now rather than
buried at line 260 of this file. The team-code entry in particular says *both
sides, every time, including a bare equality check* — that phrasing is load
bearing and was written after Arizona.

**Never push without asking.**

**`data/` is gitignored and must stay that way.** The repo is public.

**Shell exports do not reach the Bash tool.** Prefix commands instead.

**Season-forward validation only.** Never a random split.

**Numbers from before 2026-08-09 are stale** — the label window changed.

**Verify with output, not inspection.** Use `streamlit.testing.v1.AppTest` for
the app; it catches exceptions a spec-level check will not. For charts, render
to PNG and *look* — `uv add vl-convert-python`, then remove it again; it is a
verification tool, not an app dependency. That is how the zero-baseline problem
was found, and it built fine either way.

---

## Where to pick up

```bash
uv run pytest                                     # 225 tests, ~11s
FF_EDGE_LEAGUE_ID=... uv run streamlit run app.py # Draft Day/Landscape/Players/Strategy/Board/Glossary
FF_EDGE_PROFILE=standard_12 uv run python -c "from src import board; print(board.build()['players'].head())"
```

The claims ledger fills going forward only — `uv run python -m src.bootstrap
--light` daily during camp is what accumulates it.
