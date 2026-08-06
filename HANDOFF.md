# ff-edge — handoff

**Session date:** 2026-08-02
**State:** all 128 tests pass, app boots clean, everything below is committed to
the working tree but **not committed to git**. Nothing has been pushed.

Read this before doing anything else. It is written to be the whole context —
you should not need to re-derive any of it.

---

## The one-paragraph version

The project set out to find under- and overrated players by clustering on
advanced metrics. **The clustering did not work and has been removed.** What
replaced it is better: a stability gate that tells you which metrics are real, a
quality/opportunity split that survived testing, and a measured answer to
"does any of this beat ADP" (no, to ±0.01). The most useful finding of the
session was accidental — the criteria that predict a promoted player's
production are *completely different at running back* than at receiver, and the
project had been applying the receiver logic everywhere.

---

## What changed this session

**Window widened to 2018.** `FEATURE_SEASONS` 2018-2025, `LABEL_SEASONS`
2019-2024. Out-of-sample rows went 284 → 540 and test folds 2 → 4. This was worth
more than every modelling change combined — it roughly halved every interval in
the project. 2018 is a hard floor because FFC publishes no ADP before it.

**New: `src/context.py`** — play-context features from tables already on disk.
Red-zone and end-zone target share, goal-line carry share, touchdown-equity
share, neutral-script shares, NGS rushing (RYOE, box counts), FTN charting
(catchable/contested/screen). No new downloads were needed: the `ff_opportunity`
play-level tables already carry `score_differential` and `vegas_wp`, so
neutral-script filtering came free without the 1GB `pbp` pull.

**New: `src/stability.py`** — year-over-year percentile correlation per metric.
This is the module that reorganized everything. See below.

**New: `src/projection.py`** — the beat-ADP question with a continuous target
instead of a binary one. Measures the same gap ~5× more precisely.

**Removed: k-means clustering** from `src/archetypes.py`. `cluster()`,
`cluster_profiles()`, `choose_k()`, `K_CEILING` and
`features.cluster_feature_columns()` are gone. Replaced by `archetypes.scores()`,
which returns the same quality/opportunity scores without the clustering step.
`neighbors()` survives and is now the module's main output.

**Also:** stability-weighted quality scores, 6 noisy features pruned, 2 new MCP
tools (`metric_stability`, `touchdown_equity`), 2 new app sections, README and
docstrings updated with real numbers including the negative ones.

---

## The findings, in order of how much they should shape what you do next

### 1. Opportunity persists. Quality mostly doesn't.

Rank every qualified player within position and season, pair each player-season
with his own next one, correlate the percentiles. No model, no outcome.

| position | opportunity | quality |
|---|---:|---:|
| WR | 0.549 | 0.442 |
| RB | 0.526 | 0.278 |
| TE | 0.510 | 0.400 |
| QB | 0.472 | 0.394 |

Six columns failed the 0.20 noise floor and were removed from the quality sets:

| dropped | r |
|---|---:|
| `contested_catch_rate` | 0.061 WR, **−0.118** TE |
| `drop_rate` | 0.096 WR, 0.183 TE |
| `catch_rate` | 0.046 RB, 0.142 TE |
| `ypt` | 0.174 RB |

`ryoe_per_att` was added expecting it to be the running back's yards-per-route-
run. It persists at 0.202 — kept, but demoted. Selecting features this way uses
no outcome data, so it is not the "search until something scores well" trap.

**Run `src/stability.py` before adding any new metric.** It is the cheapest
filter in the project and it is now enforced by a test.

### 2. Nothing beats ADP, and this is now precisely measured

| | rank correlation |
|---|---:|
| model | 0.495 |
| ADP alone | 0.495 |
| **difference** | **−0.000, 95% CI [−0.010, +0.011]** |

Predicting next season's finish is easy — usage persists, so anything sane gets
to ~0.5. Beating the *price* is the question. Approaches tried and measured:

| approach | gap vs ADP |
|---|---:|
| play-context features in the model | −0.013 |
| quality/opportunity composite scores | −0.013 |
| GMM soft cluster membership | −0.022 |
| partial pooling across positions | −0.030 |
| PCA on the quality block | worse everywhere |

At 2.5-6.5 events per variable another feature costs more variance than it buys.
**The binding constraint is label seasons, not features or algorithms.** Do not
spend another session on model architecture; there is nothing there.

Quarterback is actively harmful: −0.099, CI [−0.173, −0.033], entirely below
zero. Rushing share and expected-points share are fully priced at QB.

### 3. The promotion test — the finding to build on

Backups (below-median role) whose role then grew by 10+ percentile points. What
predicts their next-season points:

| | RB (n=72) | WR (n=122) | TE (n=67) |
|---|---:|---:|---:|
| snap share | **0.43** | 0.35 | 0.28 |
| red-zone carry share | **0.41** | — | — |
| TD equity share | **0.37** | 0.22 | 0.16 |
| yards per route run | −0.00 | **0.26** | **0.38** |
| yards per carry | −0.02 | — | — |
| yards after contact | 0.02 | — | — |
| broken tackles per att | 0.04 | — | — |

**At running back, prior efficiency predicts nothing.** Five independent
efficiency metrics all land within 0.04 of zero. What predicts a promoted back is
whether the staff was already giving him the *valuable* touches — snaps,
red-zone carries, goal-line work. Mechanically sensible: a backup getting
red-zone carries is one the coaches already trust; a backup with a shiny YPC got
it in the fourth quarter against a light box.

At WR and TE the efficiency logic *does* hold (yprr 0.26 / 0.38).

**Quality is a filter, not a picker.** Promoted backups split by prior quality:

| prior quality | n | hit top quartile |
|---|---:|---:|
| top 30% | 66 | 19.7% |
| middle | 135 | 14.1% |
| bottom 30% | 104 | **4.8%** |

Monotonic, 4× spread. Much better at ruling players out than picking them.

**Nothing predicts the promotion itself.** Correlation with next-season role
growth: vacated targets **−0.04**, vacated carries **−0.03**, teammate's target
share 0.07, prior quality 0.02. The only real term is prior opportunity at −0.33,
which is pure mean reversion.

That last point is load-bearing for design. `vacated_target_share_next` is a
project feature and it is worthless for this. **The model cannot tell you who
gets the job.** You supply that from camp news and depth charts; the model grades
the player once you have.

---

## The proposed next build (BUILT 2026-08-03 — see `src/promotion.py`, the
## Screen tab, and `CLAIMS_SPEC.md` for the agreed Phase 2)

A **promotion screen** with that division of labor:

1. You mark a player as "role growing" — from beat reporting, a depth chart, a
   departure you know about. The model does not attempt this and should not.
2. It scores him on the criteria that work *at his position*: trust markers
   (snaps, red-zone/goal-line share, TD equity) for RBs; efficiency (yprr, tprr,
   separation) for WR/TE.
3. It reports the base rate — given a profile like his, how often did comparable
   promoted players hit — rather than a point projection.

Open questions to settle before building:

- **Position-specific weights, or one model with position interactions?** The RB
  criteria are so different that separate scorecards may be honest and simpler.
- **How does the "role growing" input get in?** A toggle per player in the app, a
  CSV you edit, or a column in a scratch file. Cheapest is probably a text box
  that takes a list of names.
- **What is the hit definition?** Top-quartile finish was used above. Might want
  "returned ADP value" instead, which needs the ADP join.
- **Sample sizes are 67-122.** The RB efficiency null is sturdy (five metrics all
  read zero) but the positive coefficients are individually thin. Decide how much
  weight the screen should put on them, and consider reporting the n beside every
  number the way the rest of the project does.

---

## Things to be careful about

**`config.ROOT` uses `parents[1]`, not `parent`.** `src/config.py:35`. Using
`.parent` silently repoints `DATA_DIR` at `src/data`, creates it, and orphans the
whole cache without raising. Highest-risk line in the project.

**`data/` is gitignored and must stay that way.** The repo is public.
`FF_EDGE_LEAGUE_ID` and `FF_EDGE_SLEEPER_USER` are read from the shell — a
Sleeper league id in a public repo exposes nine other people's data.

**Never push without asking.** `Projects/CLAUDE.md` requires explicit approval in
the current conversation for any remote git operation.

**Season-forward validation only.** Never a random split. `bk.season_forward_splits`.

**Traps already found and fixed — do not reintroduce:**
- Both `ff_opportunity` pbp tables carry weeks 19-22 with no `season_type` column
- 776 pass plays are two-point tries snapped from the two
- `play_id` is Float64 in ff_opportunity, Int32 in FTN
- The rush table has `rushing_td_exp` *and* `rush_touchdown_exp`; they disagree on
  4,558 rows and the short-named one is a sentinel
- `rank(descending=True)` gives rank 1 to the *largest* value — this inverted the
  valuation board once and put McCaffrey on the undervalued list
- `StandardScaler` applied after a deliberate column rescale undoes the rescale

---

## Where to pick up

```bash
uv run pytest                       # 128 tests, ~10s
uv run streamlit run app.py         # Landscape / Players / Strategy / Board / Glossary
uv run python -c "from src import stability as st; print(st.axis_summary(st.year_over_year()))"
```

The exploratory scripts behind the promotion finding are **not in the repo** —
they were run from a scratch directory. If you want to reproduce or extend that
analysis, it pairs each player-season to his own next one, computes the change in
`opportunity_pct`, filters to `opportunity_pct < 50 AND opp_change >= 10`, and
correlates each prior-season metric against `next_ppg_pct`. Everything it needs
is in `ft.build()` and `ar.scores()`.

Uncommitted work is the entire session. Review `git diff` before committing.
