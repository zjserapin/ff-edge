# The Fantasy Footballers on the dashboard — spec

**Status:** data layer built and tested. **App surface half built.**

**Written:** 2026-08-10. **Status re-checked against `app.py` on 2026-08-16.**
Draft is 2026-08-22.

> ### What shipped, and what did not — 2026-08-16
>
> The header used to read *"App surface not built."* Part of it since has been.
>
> **Built:** `ffb_par` is a display column on the Big Board and the blend is
> live at `FOOTBALLERS_WEIGHT = 0.5`. It is also **centered per position**
> rather than globally — a refinement this spec did not anticipate, because the
> two sources disagree about where a whole position sits (+8.7 at TE against
> −16.2 at QB) and a global center would land that on the blend as a shift that
> is by construction not an opinion about any player. See `BIG_BOARD_SPEC.md`
> §16.
>
> **Not built, all three from the proposed surface below:**
>
> - **§4, the staleness caption.** This spec marks it **"required, not
>   optional"** and it is the one element here that is not cosmetic.
>   `stalest_days` appears nowhere in `app.py`. With the blend now carrying half
>   the board's say, a 90-day-old May opinion is being presented as current.
> - **§2, the disagreement panel** (`board.compare_footballers`).
> - **§3, `ffb_spread` as an uncertainty read.** This is the thing no other
>   source in the project can do.
>
> Tracked as B3 in `DRAFT_CHECKLIST.md`.

---

## What already exists, and what it costs to use

`src/footballers.py` and three functions in `src/board.py`. All of it runs
offline after the first pull; the feed is public and unauthenticated.

```bash
uv run python -c "from src import footballers as f; print(f.consensus().head(20))"
uv run python -c "from src import footballers as f; print(f.panel_report())"
FF_EDGE_LEAGUE_ID=... uv run python -c "
from src import board; b=board.build(); print(board.compare_footballers(b).head(20))"
```

`board.build()` already returns `ffb_par`, `blend_par`, `blend_rank` and the
whole consensus block on `players`, and already sorts the board by `blend_par`.
So the dashboard change is **display only** — no recompute, no new call.

---

## The one number that needs a decision

`config.FOOTBALLERS_WEIGHT`, currently `0.5` — an equal say between their panel
and this project's expected-points curve.

It is a judgment call, not a measurement, and it is the only one in `config.py`.
Their historical boards are not published, so this cannot be backtested the way
everything else in the repo is. What the blend buys is independence: `exp_points`
is a function of positional ADP rank and reproduces the market's ordering within
a position exactly, while their number is built from projected touches.

At 0.5, measured on the 2026 board: **163 of 175 matched players move, mean
absolute shift 7.9 ranks, max 38.** The correction is doing real work without
swamping the board. If it ever reads near zero, the weight should go to 0.0 and
the complexity should come out — the same standard `compare_baselines` is held
to.

---

## Proposed surface

### 1. Draft Day board — three new columns

On the existing player table, after `par`:

| column | source | why it earns a column |
|---|---|---|
| `ffb_par` | `board.attach_footballers` | Their read, on the board's own scale |
| `blend_rank` | `board.build` | What the board now sorts by |
| `ffb_spread` | `footballers.consensus` | Panel disagreement, in league points |

`board_rank` stays visible next to `blend_rank`. That pairing is the whole point
— it is what makes the blend checkable rather than asserted, and it is the same
reason `compare_baselines` prints both baselines.

### 2. A disagreement panel

`board.compare_footballers(built)` sorted by `|rank_shift|`, filtered to
`board_rank <= 60`. Rows where two independent reads diverge are the only ones
worth spending time on; agreement is not information.

Positive `rank_shift` means the Footballers like him **more** than this
project's curve does.

### 3. `ffb_spread` as an uncertainty read, not a ranking

This is the part no other source in the project can do. Three analysts means a
range, and two players at the same ADP with spreads of 12 and 68 points are
different bets at identical price. Suggested framing on the player card:

> **Panel spread 66 pts** — Andy, Jason and Mike disagree about this player by
> more than a full round's worth of production. Their consensus is a midpoint,
> not an agreement.

### 4. A staleness caption — required, not optional

**This is the one display element that is not cosmetic.** Panel freshness is
wildly uneven and a consensus number hides it completely. From `panel_report()`
on 2026-08-10:

| analyst | players | median age |
|---|---|---|
| Jason | 313 | 7 days |
| Andy | 306 | 64 days |
| Mike | 305 | **90 days** |

Any view showing `ffb_points` must show `stalest_days` beside it, or it is
presenting a May opinion as a current one. A player whose `stalest_days` is 90
should be visually flagged.

### 5. What must NOT be shown

- **Their `ffb_adp_*` columns next to this project's `adp`.** Different market,
  different sample, sparse. Namespaced in the data layer specifically so they
  cannot be confused; do not undo that in the UI.
- **A kicker row.** Dropped upstream — their K rows carry no stats at all.
- **`ffb_points` as a cross-position ranking.** Raw points rank every superflex
  QB above every RB. `ffb_par` is the cross-position number; `ffb_points` is
  within-position only.

---

## Phase 2 — weekly rankings, in season

The user's stated next step. Status: **the hooks exist, the source does not
yet.**

- Every row carries `season_type`, and every draft-season row is `"1"`.
- Their page JS carries a `getSeasonWeeks(season)` helper no draft page calls.
- Both imply `season_type = 2` on the same schema once the season starts.
- **Unverified.** Both plausible weekly URLs 404 as of 2026-08-10
  (`/fantasy-football-rankings/`, `/2026-ultimate-draft-kit/udk-weekly-rankings/`).

`projections()` carries `season_type` through unfiltered rather than dropping it
as a constant, so the weekly work should be a URL, a filter and a week column
rather than a rewrite. **Re-probe in September rather than trusting this.**

The open design question for weekly, worth deciding then and not now: weekly
rankings answer "who do I start", which is a different question from "who do I
draft" and wants a different baseline (per-week replacement, not season). That
is closer to `scoring.replacement_ppg` than to anything in `board.py`.

---

## Keeping it fresh until kickoff

Already wired. `src/bootstrap.py` gained a Footballers section:

```bash
uv run python -m src.bootstrap --light    # run daily
```

Two steps: `fetch` (1h TTL, so a daily run always pulls live) and `snapshot`,
which appends today's consensus to `data/footballers_history_2026.parquet`,
idempotent per day.

**The snapshot is the part that cannot be recovered later.** Their page serves
today's projections only, nobody sells the history, and `footballers.movement()`
— which analyst has quietly walked a player down 40 points across three August
revisions — only works on days that were actually pulled. It reads empty until
two days exist. Started 2026-08-10.
