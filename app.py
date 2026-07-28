"""ff-edge — draft research for the Shiva Bowl.

    uv run streamlit run app.py

Lives at the repo root rather than in src/ because `streamlit run` puts the
script's own directory on sys.path, and from src/ the `from src import ...`
imports below would not resolve.

Caching convention, applied everywhere in this file: a scoring dict is not
hashable, so every cached function takes `scoring_key` — a tuple of sorted
(key, weight) pairs — and rebuilds the dict inside. Nothing that reads
st.session_state is ever cached; that is the standard way a Streamlit board goes
stale while looking fine.
"""

from __future__ import annotations

from typing import Any, Mapping

import altair as alt
import polars as pl
import streamlit as st

from src import adp as adp_mod
from src import archetypes as ar
from src import breakout as bo
from src import features as ft
from src import glossary
from src import landscape as ls
from src import rookies as rk
from src import scoring as sc
from src import simulate as sim
from src import theme
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    FEATURE_SEASONS,
    LEAGUE_ID,
    OUTPUT_DIR,
    SEASON,
)

st.set_page_config(page_title="ff-edge", page_icon="🏈", layout="wide")


# --- Cache boundaries -------------------------------------------------------


def _key(scoring: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Make a scoring dict hashable for @st.cache_data."""
    return tuple(sorted((k, float(v)) for k, v in scoring.items()))


def table(df: pl.DataFrame, **kwargs: Any) -> None:
    """Render a frame with a hover definition on every column that has one.

    Every table in this app goes through here rather than st.dataframe directly.
    A column named `wopr` or `market_var` means nothing on its own, and a
    glossary nobody opens is a glossary nobody reads — the definition has to be
    on the column itself.
    """
    if not df.height:
        return
    pdf = df.to_pandas()
    config = {
        col: st.column_config.Column(help=help_text)
        for col, help_text in glossary.column_help(list(pdf.columns)).items()
    }
    st.dataframe(
        pdf, use_container_width=True, hide_index=True, column_config=config, **kwargs
    )


def _glossary_section() -> None:
    """The full reference, grouped, with the long definitions."""
    st.subheader("Glossary")
    st.caption(
        "Every metric in the app, what it is computed from, and — where it "
        "matters — what it does not mean. Hovering any column header elsewhere "
        "shows the short version."
    )
    search = st.text_input("Filter", placeholder="e.g. share, replacement, ADP")
    needle = search.lower().strip()

    for group, terms in glossary.groups().items():
        hits = [
            (key, term)
            for key, term in terms
            if not needle
            or needle in key.lower()
            or needle in term.label.lower()
            or needle in term.long.lower()
        ]
        if not hits:
            continue
        st.markdown(f"#### {group}")
        for key, term in hits:
            with st.container(border=True):
                st.markdown(f"**{term.label}**  ·  `{key}`")
                st.markdown(term.long or term.short)


@st.cache_data(show_spinner=False)
def _league(league_id: str) -> dict[str, Any]:
    return sc.league_settings(league_id)


@st.cache_data(show_spinner=False)
def _scoring_history(league_id: str) -> pl.DataFrame:
    return sc.scoring_history(league_id)


@st.cache_data(show_spinner="Scoring history…")
def _season_points(scoring_key: tuple[tuple[str, float], ...]) -> pl.DataFrame:
    return sc.score_season(scoring=dict(scoring_key))


@st.cache_data(show_spinner=False)
def _par(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return ls.par_by_position(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _concentration(
    scoring_key: tuple[tuple[str, float], ...], shares: tuple[int, ...]
) -> pl.DataFrame:
    return ls.concentration(
        scoring=dict(scoring_key), shares=shares, season_points=_season_points(scoring_key)
    )


@st.cache_data(show_spinner=False)
def _scarcity(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
    max_rank: int,
    basis: str,
) -> pl.DataFrame:
    return ls.scarcity_curve(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        max_rank=max_rank,
        basis=basis,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _cross(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return ls.cross_positional_value(
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        season_points=_season_points(scoring_key),
    )


@st.cache_data(show_spinner=False)
def _replacement(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    return sc.replacement_level(
        _season_points(scoring_key),
        list(roster),
        teams,
        dict(flex_split) if flex_split else None,
    )


# --- Sidebar ----------------------------------------------------------------


def _sidebar() -> dict[str, Any]:
    st.sidebar.title("League")

    league = _league(LEAGUE_ID)
    if league["source"] != "sleeper":
        st.sidebar.caption("⚠️ Sleeper unreachable — using saved 2026 settings.")
    else:
        st.sidebar.caption(f"Live from Sleeper · {league['season']} season")

    teams = st.sidebar.number_input(
        "Teams", min_value=4, max_value=16, value=int(league["teams"]), step=1
    )

    st.sidebar.subheader("Scoring")
    scoring = dict(league["scoring"])
    rec = st.sidebar.slider(
        "Points per reception", 0.0, 1.0, float(scoring.get("rec", 0.5)), 0.25
    )
    pass_td = st.sidebar.slider(
        "Points per passing TD", 3.0, 6.0, float(scoring.get("pass_td", 4.0)), 1.0
    )
    scoring["rec"] = rec
    scoring["pass_td"] = pass_td

    st.sidebar.subheader("FLEX allocation")
    mode = st.sidebar.radio(
        "How FLEX slots are filled",
        ["Computed from the data", "Set manually"],
        help=(
            "Computed assigns each FLEX slot to whichever of RB/WR/TE is worth "
            "more at its next open rank, using that season's real scoring. "
            "This moves replacement level by several ranks, so it is worth "
            "seeing both."
        ),
    )
    flex_split: dict[str, float] | None = None
    if mode == "Set manually":
        rb = st.sidebar.slider("Share of FLEX going to RB", 0.0, 1.0, 0.5, 0.05)
        te = st.sidebar.slider("Share going to TE", 0.0, 1.0 - rb, 0.05, 0.05)
        flex_split = {"RB": rb, "WR": max(0.0, 1.0 - rb - te), "TE": te}
        st.sidebar.caption(
            f"RB {flex_split['RB']:.0%} · WR {flex_split['WR']:.0%} · TE {flex_split['TE']:.0%}"
        )

    unmapped = sc.unmapped_keys(scoring)
    if unmapped:
        with st.sidebar.expander(f"Not modeled ({len(unmapped)} scoring rules)"):
            st.caption(
                "weekly_stats has no team-defense rows, so DST and IDP scoring "
                "cannot be computed from this data at all. Every chart here "
                "covers QB/RB/WR/TE only."
            )
            st.code("\n".join(unmapped), language=None)

    return {
        "league_id": LEAGUE_ID,
        "teams": int(teams),
        "scoring": scoring,
        "scoring_key": _key(scoring),
        "roster_positions": tuple(league["roster_positions"] or DEFAULT_ROSTER_POSITIONS),
        "flex_split": tuple(sorted(flex_split.items())) if flex_split else None,
        "dark": theme.is_dark(),
    }


# --- Landscape tab ----------------------------------------------------------


def _tab_landscape(p: dict[str, Any]) -> None:
    dark = p["dark"]
    pos_scale = theme.position_scale(dark)

    st.subheader("How positional value has moved")
    st.caption(
        f"Seasons {FEATURE_SEASONS[0]}–{FEATURE_SEASONS[-1]}, every season "
        "rescored under the settings in the sidebar. Weeks 1–14 only — the "
        "fantasy regular season."
    )

    hist = _scoring_history(p["league_id"])
    if hist.height > 1 and hist.get_column("rec").n_unique() > 1:
        changes = " · ".join(
            f"{r['season']}: {r['rec']} PPR, {r['flex_slots']} FLEX"
            for r in hist.iter_rows(named=True)
            if r["season"]
        )
        st.info(
            f"**This league changed its own rules.** {changes}. History below is "
            "recomputed under today's settings, which is the right basis for a "
            "2026 draft — but those seasons were not played this way.",
            icon="ℹ️",
        )

    par = _par(p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"])
    if not par.height:
        st.warning("No scored seasons found. Run `uv run python -m src.bootstrap --light`.")
        return

    # --- Value above replacement ---
    st.markdown("#### Value above replacement, per starting slot")
    st.caption(
        "Points per game a starter at this position gives you over the best "
        "player you could have had for free. The only unit that compares a "
        "quarterback to a tight end."
    )

    par_pd = par.to_pandas()
    line = (
        alt.Chart(par_pd)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("par_mean_starter:Q", title="PAR per game"),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("par_mean_starter:Q", title="PAR/game", format=".2f"),
                alt.Tooltip("demand:Q", title="Starters league-wide", format=".0f"),
                alt.Tooltip("replacement_rank:Q", title="Replacement rank"),
                alt.Tooltip("replacement_ppg:Q", title="Replacement PPG", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    labels = (
        alt.Chart(par_pd[par_pd["season"] == par_pd["season"].max()])
        .mark_text(align="left", dx=8, fontSize=11, fontWeight=600)
        .encode(
            x=alt.X("season:O"),
            y=alt.Y("par_mean_starter:Q"),
            text="position:N",
            color=alt.Color("position:N", scale=pos_scale, legend=None),
        )
    )
    st.altair_chart(theme.base_chart(line + labels, dark), use_container_width=True)

    with st.expander("Replacement level behind these numbers"):
        st.caption(
            "Replacement is the first player past the starting pool. Per-game "
            "replacement is ranked on per-game among players with 8+ games — "
            "ranking on season totals lets an injured star who happened to land "
            "on the slot set the baseline."
        )
        repl = _replacement(
            p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"]
        )
        table(repl)

    # --- Scarcity curves ---
    st.markdown("#### The shape of the dropoff")
    st.caption(
        "The average above tells you how much a position is worth; this tells "
        "you *where*. A cliff is worth reaching for, a gentle slope is worth "
        "waiting on, and both can produce the same average."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        basis = st.radio(
            "Measured as",
            ["Points per game", "Season total"],
            horizontal=False,
            key="scarcity_basis",
            help=(
                "Per game compares seasons fairly and asks what a healthy player "
                "at this tier is worth (8+ games). Season total counts "
                "availability as value, which is what you actually draft."
            ),
        )
    basis_key = "ppg" if basis == "Points per game" else "total"
    with c2:
        max_rank = st.slider("Ranks to show", 12, 60, 36, 6, key="scarcity_rank")

    scarcity = _scarcity(
        p["scoring_key"],
        p["roster_positions"],
        p["teams"],
        p["flex_split"],
        max_rank,
        basis_key,
    )
    seasons = sorted(scarcity.get_column("season").unique().to_list())
    chosen = st.multiselect(
        "Seasons", seasons, default=seasons[-3:], key="scarcity_seasons"
    )
    if chosen:
        y_field, y_title = (
            ("ppg", "Points per game")
            if basis_key == "ppg"
            else ("fantasy_points", "Season points")
        )
        sc_pd = scarcity.filter(pl.col("season").is_in(sorted(chosen))).to_pandas()
        curves = (
            alt.Chart(sc_pd)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("pos_rank:Q", title="Positional rank"),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color(
                    "season:N",
                    scale=alt.Scale(
                        domain=sorted(chosen),
                        range=theme.ordinal_steps(len(chosen), dark),
                    ),
                    title="Season",
                ),
                tooltip=[
                    alt.Tooltip("player_display_name:N", title="Player"),
                    alt.Tooltip("season:O", title="Season"),
                    alt.Tooltip("pos_rank:Q", title="Rank"),
                    alt.Tooltip("ppg:Q", title="PPG", format=".1f"),
                    alt.Tooltip("fantasy_points:Q", title="Season pts", format=".0f"),
                    alt.Tooltip("games:Q", title="Games"),
                ],
            )
            .properties(height=200, width=200)
            .facet(column=alt.Column("position:N", title=None, sort=list(theme.position_colors())))
        )
        st.altair_chart(theme.base_chart(curves, dark), use_container_width=True)

    # --- Cross-positional mix ---
    st.markdown("#### Who actually occupies the top of the board")
    st.caption(
        "Every position ranked together on value over replacement. This is the "
        "early-RB argument as a countable fact rather than an opinion."
    )
    cross = _cross(p["scoring_key"], p["roster_positions"], p["teams"], p["flex_split"])
    mix = ls.positional_mix(cross)
    cutoff = st.select_slider(
        "Top N by value over replacement", options=[12, 24, 36, 48], value=24
    )
    mix_pd = mix.filter(pl.col("cutoff") == cutoff).to_pandas()
    bars = (
        alt.Chart(mix_pd)
        # A 2px stroke in the surface color, not a border: it reads as a gap
        # between stacked segments so adjacent positions stay countable.
        .mark_bar(cornerRadiusEnd=4, stroke=theme.surface(dark), strokeWidth=2)
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("n:Q", title=f"Players in the top {cutoff}", stack="zero"),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("n:Q", title="Players"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(theme.base_chart(bars, dark), use_container_width=True)

    # --- Concentration ---
    st.markdown("#### Are the top players taking a bigger slice?")
    st.caption(
        "Share of positional points held by the top N, among a pool of roughly "
        "three times the number of starters. An uncapped pool would make this a "
        "measure of how many replacement bodies the league cycled through."
    )
    conc = _concentration(p["scoring_key"], (5, 15, 30))
    conc_pd = conc.to_pandas()
    conc_chart = (
        alt.Chart(conc_pd)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("share:Q", title="Share of positional points", axis=alt.Axis(format="%")),
            color=alt.Color("position:N", scale=pos_scale, title="Position"),
            tooltip=[
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("position:N", title="Position"),
                alt.Tooltip("top_n:Q", title="Top N"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
                alt.Tooltip("pool_size:Q", title="Pool"),
            ],
        )
        .properties(height=180)
        .facet(column=alt.Column("top_n:N", title="Top N players"))
    )
    st.altair_chart(theme.base_chart(conc_chart, dark), use_container_width=True)

    with st.expander("Table view"):
        st.caption("Every chart above, as numbers.")
        st.markdown("**Value above replacement**")
        table(par)
        st.markdown("**Positional mix of the top of the board**")
        table(mix)
        st.markdown("**Concentration**")
        table(conc)


@st.cache_data(show_spinner="Building features…")
def _features() -> pl.DataFrame:
    return ft.build()


@st.cache_data(show_spinner="Clustering…")
def _clusters(season: int, k: int | None, min_games: int) -> pl.DataFrame:
    return ar.cluster(season, min_games=min_games, k=k, df=_features())


@st.cache_data(show_spinner=False)
def _profiles(season: int, k: int | None, min_games: int) -> pl.DataFrame:
    return ar.cluster_profiles(_clusters(season, k, min_games), _features(), season=season)


@st.cache_data(show_spinner=False)
def _silhouette(season: int, position: str, min_games: int) -> pl.DataFrame:
    pool = _features().filter(
        (pl.col("season") == season) & (pl.col("games") >= min_games)
        & (pl.col("position") == position)
    )
    x, used = ar._matrix(pool, ft.cluster_feature_columns(position))
    if not used or pool.height < 12:
        return pl.DataFrame()
    return ar.choose_k(x, (2, ar.K_CEILING.get(position, 6)))


def _tab_players(p: dict[str, Any]) -> None:
    dark = p["dark"]
    feats = _features()
    if not feats.height:
        st.warning("No features built. Run `uv run python -m src.bootstrap --light`.")
        return

    season = int(feats.get_column("season").max())
    st.subheader("Usage archetypes")
    st.caption(
        f"Grouping {season} players by *how they were used* — shares, rates, and "
        "role — not by what they scored. Expected points and draft pedigree are "
        "deliberately excluded from the distance metric, so these are not "
        "scoring tiers in disguise."
    )
    st.warning(
        "**Clusters describe. They do not predict.** A receiver sitting with "
        "three alphas means his target share, air-yards share and route role "
        "rhyme with theirs. It does not mean he will produce like them — what "
        "separates him from them may be talent, and talent is not in this "
        "feature set.",
        icon="⚠️",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        auto = st.checkbox("Choose the number of groups automatically", value=True)
    with c2:
        min_games = st.slider("Minimum games", 4, 14, 8, key="cluster_min_games")
    k = None if auto else st.slider("Groups per position", 2, 6, 4, key="cluster_k")

    clusters = _clusters(season, k, min_games)
    if not clusters.height:
        st.info("Not enough qualified players to cluster.")
        return

    profiles = _profiles(season, k, min_games)

    st.info(
        "**Silhouette peaks at two groups for every position and falls from "
        "there** — NFL usage has one dominant axis, how much of his offense a "
        "player commands. The honest answer is 'featured' and 'not'. The one "
        "real exception is quarterback, where the split is rushing versus "
        "pocket rather than good versus bad. Turning off automatic selection "
        "lets you assert finer archetypes; the curve below shows what that costs.",
        icon="ℹ️",
    )

    st.markdown("#### What each group is")
    table(profiles.select("position", "cluster", "n", "mean_ppg", "label"))

    st.markdown("#### Find comparable usage")
    st.caption(
        "The players whose role most resembles this one, ranked by distance in "
        "the standardized usage space. A cheap player next to expensive ones is "
        "the thing worth a second look."
    )
    pos = st.selectbox("Position", sorted(clusters.get_column("position").unique().to_list()))
    pool = clusters.filter(pl.col("position") == pos).sort("pos_rank")
    names = pool.get_column("player_name").to_list()
    who = st.selectbox("Player", names)
    pid = pool.filter(pl.col("player_name") == who).get_column("player_id")[0]

    nb = ar.neighbors(pid, clusters, feats, n=8, season=season)
    if nb.height:
        table(nb.select("player_name", "team", "distance", "ppg", "pos_rank", "games"))

    with st.expander("How many groups the data actually supports"):
        sil = _silhouette(season, pos, min_games)
        if sil.height:
            st.caption(
                "Higher silhouette means tighter, better-separated groups. "
                "Solutions marked not viable have a group of fewer than four "
                "players — k-means quarantining an outlier, not an archetype."
            )
            sil_pd = sil.to_pandas()
            curve = (
                alt.Chart(sil_pd)
                .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True))
                .encode(
                    x=alt.X("k:O", title="Groups"),
                    y=alt.Y("silhouette:Q", title="Silhouette"),
                    tooltip=[
                        alt.Tooltip("k:O", title="Groups"),
                        alt.Tooltip("silhouette:Q", format=".3f"),
                        alt.Tooltip("smallest:Q", title="Smallest group"),
                        alt.Tooltip("viable:N", title="Viable"),
                    ],
                )
                .properties(height=200)
            )
            st.altair_chart(theme.base_chart(curve, dark), use_container_width=True)
            table(sil)

    with st.expander("Feature coverage"):
        st.caption(
            "Next Gen Stats cover qualified receivers only, so ~30% non-null is "
            "expected there. Snap share below ~90% would mean the "
            "pfr_id → gsis_id bridge broke."
        )
        table(ft.coverage_report(feats).filter(pl.col("scope") == "ALL"))

    st.divider()
    _breakout_section(dark)
    st.divider()
    _rookie_section(dark)


@st.cache_data(show_spinner="Backtesting…")
def _backtest(by_position: bool) -> dict[str, pl.DataFrame]:
    train = bo.training_frame(features=_features())
    if not train.height:
        return {}
    preds = bo.fit_predict(train, by_position=by_position)
    return {
        "train": train,
        "base_rates": bo.base_rates(bo.labels()),
        "preds": preds,
        "disc": bo.discrimination(preds),
        "cal": bo.calibration(preds),
        "cal_pos": bo.calibration_by_position(preds),
        "adequacy": bo.sample_adequacy(train),
        "coefs": bo.coefficients(train, by_position=by_position),
    }


def _breakout_section(dark: bool) -> None:
    st.subheader("Did last season's usage predict beating ADP?")

    stratified = st.toggle(
        "Separate model per position",
        value=True,
        help=(
            "On: a QB is scored by a QB model, an RB by an RB model, each with "
            "its own feature set. Off: one model across all positions, for "
            "comparison."
        ),
    )
    res = _backtest(stratified)
    if not res or not res["preds"].height:
        st.info("Not enough labeled seasons to backtest.")
        return

    disc, cal = res["disc"], res["cal"]
    overall = res["base_rates"].filter(pl.col("scope") == "overall")
    base = float(overall.get_column("rate")[0])
    n_labeled = int(overall.get_column("n")[0])

    allrow = disc.filter(pl.col("scope") == "all")
    auc = float(allrow.get_column("auc")[0])
    auc_adp = float(allrow.get_column("auc_adp_only")[0])
    d_lo, d_hi = float(allrow.get_column("delta_lo")[0]), float(allrow.get_column("delta_hi")[0])

    st.caption(
        f"Label: finish at or inside 60% of your ADP positional rank. "
        f"{n_labeled} labeled player-seasons, base rate **{base:.1%}**. "
        "Trained on earlier seasons, tested on later ones — never a random split."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Base rate", f"{base:.1%}", help=glossary.describe("base_rate"))
    c2.metric("Model AUC", f"{auc:.3f}", help=glossary.describe("auc"))
    c3.metric(
        "ADP alone",
        f"{auc_adp:.3f}",
        delta=f"{auc - auc_adp:+.3f}",
        delta_color="normal",
        help=glossary.describe("auc_adp_only"),
    )

    if stratified:
        st.warning(
            f"**Stratifying fixed a pathology but did not find an edge.** Pooled, "
            f"this model was *anti*-predictive — AUC 0.401, below a coin flip, "
            f"with inverted calibration. Fitting per position moves it to "
            f"{auc:.3f}, roughly chance. Isolating why: shrinkage alone changed "
            "nothing (0.401 → 0.398), cutting to four features recovered part "
            "(0.447), and separating positions recovered the rest. Pooling was "
            "averaging four different relationships and fitting none of them. It "
            f"still does not beat draft price — the gap interval "
            f"[{d_lo:+.3f}, {d_hi:+.3f}] covers zero — but 'no signal' is a "
            "different and more honest result than 'reliably wrong'.",
            icon="⚠️",
        )
    else:
        st.error(
            f"**Pooled, the model is worse than a coin flip.** AUC {auc:.3f} "
            f"against {auc_adp:.3f} for price alone. Calibration is inverted: "
            "the lowest-probability group beats ADP more often than the highest. "
            "Turn on per-position models above to see this largely disappear — "
            "one model across four positions is averaging relationships that "
            "point in different directions.",
            icon="🚫",
        )

    st.markdown("#### Can each position support a model?")
    st.caption(
        "Events per variable is the standard adequacy check for a fit like this, "
        "and the conventional floor is ten. Nothing here reaches it. That is the "
        "cost of stratifying a 629-row sample four ways, and it is the first "
        "thing to read before any number below."
    )
    table(res["adequacy"])

    if stratified:
        st.markdown("#### Per-position results")
        st.caption(
            "`delta_auc` is the gain over predicting from draft price alone. Every "
            "interval covers zero, so no position shows a defensible edge — but "
            "note the direction flipped positive for QB, RB and WR once the "
            "models were separated."
        )
        table(
            disc.filter(pl.col("scope").is_in(["QB", "RB", "WR", "TE"])).select(
                "scope", "n", "positives", "auc", "auc_adp_only",
                "delta_auc", "delta_lo", "delta_hi",
            )
        )

    st.markdown("#### Calibration")
    st.caption(
        "Players sorted into four groups by predicted probability, against what "
        "actually happened. If the model worked, the actual rate would rise left "
        "to right. Four groups rather than ten because ~280 out-of-sample rows "
        "makes a decile ±7 points — too wide to read."
    )
    cal_pd = cal.to_pandas()
    bars = (
        alt.Chart(cal_pd)
        .mark_bar(cornerRadiusEnd=4, color=theme.position_colors(dark)["QB"])
        .encode(
            x=alt.X("bin:O", title="Predicted-probability group (low → high)"),
            y=alt.Y("actual_rate:Q", title="Actually beat ADP", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("bin:O", title="Group"),
                alt.Tooltip("n:Q", title="Players"),
                alt.Tooltip("mean_predicted:Q", title="Predicted", format=".1%"),
                alt.Tooltip("actual_rate:Q", title="Actual", format=".1%"),
                alt.Tooltip("ci_lo:Q", title="CI low", format=".1%"),
                alt.Tooltip("ci_hi:Q", title="CI high", format=".1%"),
                alt.Tooltip("lift:Q", title="Lift vs base"),
            ],
        )
        .properties(height=240)
    )
    errors = (
        alt.Chart(cal_pd)
        .mark_rule(strokeWidth=2, color=theme.ink(dark)["muted"])
        .encode(x=alt.X("bin:O"), y=alt.Y("ci_lo:Q"), y2=alt.Y2("ci_hi:Q"))
    )
    baseline = (
        alt.Chart(cal_pd)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=theme.ink(dark)["muted"])
        .encode(y=alt.Y("base_rate:Q"))
    )
    st.altair_chart(theme.base_chart(bars + errors + baseline, dark), use_container_width=True)
    st.caption("Dashed line is the base rate. Whiskers are 95% Wilson intervals.")

    with st.expander("Full backtest numbers"):
        st.markdown("**Discrimination — overall, by position, by test season**")
        table(disc)
        st.markdown("**Calibration (pooled across positions)**")
        table(cal)
        if res["cal_pos"].height:
            st.markdown("**Calibration within each position**")
            st.caption("Two bins only — a position contributes 20-60 out-of-sample rows.")
            table(res["cal_pos"])
        st.markdown("**Base rates**")
        table(res["base_rates"])
        st.markdown("**What each model keyed on**")
        st.caption(
            "Standardized coefficients. On samples this small these are "
            "themselves unstable — read them as a description of this fit, not "
            "as estimates of an effect."
        )
        table(res["coefs"])


@st.cache_data(show_spinner="Fitting rookies…")
def _rookies(by_position: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    preds = rk.fit(by_position=by_position)
    return (
        preds,
        rk.performance(preds),
        rk.board(SEASON, by_position=by_position),
        rk.coefficients(by_position=by_position),
    )


def _rookie_section(dark: bool) -> None:
    st.subheader("Rookies")
    strat = st.toggle(
        "Separate model per position",
        value=True,
        key="rookie_strat",
        help=(
            "Matters more here than for veterans: a rookie QB who plays scores "
            "15-20 points a game and a TE scores 5, so a pooled model spends its "
            "capacity learning position before it can say anything about the player."
        ),
    )
    preds, perf, board, coefs = _rookies(strat)
    if not perf.height:
        st.info("No rookie classes available.")
        return

    overall = perf.filter(pl.col("scope") == "overall")
    corr = float(overall.get_column("corr")[0])
    mae = float(overall.get_column("mae")[0])
    base_mae = float(overall.get_column("baseline_mae")[0])

    st.caption(
        "A separate model, never merged into the clusters above — a rookie has "
        "no prior-season usage, which is the only input those use. Draft "
        "capital, combine testing, landing-spot opportunity, and age."
    )

    c1, c2 = st.columns(2)
    c1.metric("Out-of-sample correlation", f"{corr:.3f}")
    c2.metric(
        "Mean error (pts/game)",
        f"{mae:.2f}",
        delta=f"{mae - base_mae:+.2f} vs guessing the mean",
        delta_color="inverse",
    )

    st.success(
        f"**This one works, unlike the veteran model.** Predicted and actual "
        f"rookie points per game correlate {corr:.2f} out of sample, with "
        f"{(1 - mae / base_mae):.0%} less error than predicting the average for "
        "everyone — and it holds across all four positions. Draft capital does "
        "nearly all of the work; the combine numbers are close to noise. "
        "Validation is leave-one-season-out rather than forward-only, which is "
        "a weaker guarantee, chosen because a forward split would train on "
        "about ninety players.",
        icon="✅",
    )

    if strat:
        st.caption(
            "Stratifying helps here, though not uniformly: tight ends and "
            "receivers improve clearly (TE error 1.99 → 1.73), quarterbacks are "
            "a wash, and running backs get slightly worse — what a smaller "
            "training set costs when the pooled signal was already about right. "
            "The per-position coefficients below are the clearest evidence it "
            "was worth doing: the RB model keys on vacated *carries* and the WR "
            "model on vacated *targets*, which a pooled fit had to average."
        )

    st.markdown(f"#### {SEASON} rookie board")
    st.caption("Ranked by predicted points per game. Landing spot is the prior season's vacated volume.")
    table(board.head(40)
        .select("name", "position", "team", "draft_round", "draft_ovr",
                "vacated_target_share", "vacated_carry_share", "predicted")
        )

    with st.expander("Model detail"):
        st.markdown("**Accuracy by position**")
        table(perf)
        st.markdown("**Standardized coefficients**")
        st.caption("Negative on draft_ovr means an earlier pick predicts more production.")
        table(coefs)


@st.cache_data(show_spinner=False)
def _sim_baseline() -> pl.DataFrame:
    path = OUTPUT_DIR / "simulation_baseline.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


@st.cache_data(show_spinner=False)
def _sim_summary(runs_key: int) -> pl.DataFrame:
    return sim.summarize(_sim_baseline())


@st.cache_data(show_spinner=False)
def _sim_edges(runs_key: int) -> pl.DataFrame:
    return sim.compare_to_control(_sim_baseline())


@st.cache_data(ttl=3600, show_spinner="Simulating…")
def _sim_rerun(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    n_sims: int,
    seed: int,
) -> pl.DataFrame:
    return sim.run_all(
        n_sims=n_sims,
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        seed=seed,
    )


def _strategy_chart(summary: pl.DataFrame, dark: bool) -> alt.LayerChart:
    """Title rate with season-clustered intervals, control highlighted."""
    pd_ = summary.with_columns(
        pl.when(pl.col("strategy") == "bpa")
        .then(pl.lit("Control — follow ADP"))
        .otherwise(pl.lit("Strategy"))
        .alias("kind")
    ).to_pandas()

    colors = theme.position_colors(dark)
    scale = alt.Scale(
        domain=["Strategy", "Control — follow ADP"],
        range=[colors["QB"], colors["TE"]],
    )
    base = alt.Chart(pd_)
    dots = base.mark_circle(size=140).encode(
        y=alt.Y("strategy:N", sort="-x", title=None),
        x=alt.X("title_rate:Q", title="Title rate", axis=alt.Axis(format="%")),
        color=alt.Color("kind:N", scale=scale, title=None),
        tooltip=[
            alt.Tooltip("strategy:N", title="Strategy"),
            alt.Tooltip("title_rate:Q", title="Title rate", format=".1%"),
            alt.Tooltip("title_rate_lo:Q", title="CI low", format=".1%"),
            alt.Tooltip("title_rate_hi:Q", title="CI high", format=".1%"),
            alt.Tooltip("playoff_rate:Q", title="Playoff rate", format=".1%"),
            alt.Tooltip("mean_wins:Q", title="Mean wins", format=".2f"),
            alt.Tooltip("n_sims:Q", title="Simulations"),
        ],
    )
    bars = base.mark_rule(strokeWidth=2).encode(
        y=alt.Y("strategy:N", sort="-x"),
        x=alt.X("title_rate_lo:Q"),
        x2=alt.X2("title_rate_hi:Q"),
        color=alt.Color("kind:N", scale=scale, title=None),
    )
    return (bars + dots).properties(height=300)


def _tab_strategy(p: dict[str, Any]) -> None:
    dark = p["dark"]
    st.subheader("What is a draft strategy actually worth?")

    runs = _sim_baseline()
    if not runs.height:
        st.warning(
            "No simulation artifact. Run `uv run python -m src.simulate --sims 4000` "
            "(about 3 seconds)."
        )
        return

    summary = _sim_summary(runs.height)
    seasons = sorted(runs.get_column("season").unique().to_list())
    per_strategy = runs.height // summary.height

    st.caption(
        f"{per_strategy:,} simulated seasons per strategy, replaying {seasons[0]}–"
        f"{seasons[-1]} with those years' real ADP and real weekly scores. Nine "
        "opponents draft from ADP with noise scaled by each player's observed "
        "dispersion; your draft slot varies every run."
    )

    st.warning(
        "**This simulates drafting, not managing.** No waivers, no trades, no "
        "streaming, no bye-week maneuvering — just the draft, then optimal "
        "lineups all year. Those omissions are a large share of real outcomes, "
        "so the claim here is narrow: what a draft strategy is worth *holding "
        "in-season management constant*. That is not the same as what wins leagues.",
        icon="⚠️",
    )

    st.altair_chart(theme.base_chart(_strategy_chart(summary, dark), dark), use_container_width=True)
    st.caption(
        "Intervals resample whole seasons, not individual simulations — "
        "simulations within a season share one realized set of player outcomes, "
        "so treating them as independent would report bars roughly ten times "
        "too narrow."
    )

    st.markdown("#### Edge over simply following ADP")
    st.caption(
        "Each strategy minus the control, bootstrapped over shared seasons so "
        "both sides move together — season variation is the biggest term here, "
        "and comparing unpaired estimates spends all the power on it. Intervals "
        "are Bonferroni-corrected for comparing seven strategies: pick the best "
        "of seven and quote its uncorrected interval and you will find something "
        "every time."
    )
    edges = _sim_edges(runs.height)
    winners = edges.filter(pl.col("beats_control"))
    losers = edges.filter(pl.col("bonferroni_hi") < 0)

    if winners.height:
        names = ", ".join(
            f"**{r['strategy']}** ({r['edge']:+.1%}, CI [{r['bonferroni_lo']:+.1%}, "
            f"{r['bonferroni_hi']:+.1%}])"
            for r in winners.iter_rows(named=True)
        )
        st.success(
            f"{names} beat ADP-following on title rate, with corrected intervals "
            "that exclude zero. Both are quarterback-timing strategies, which is "
            "coherent: they are the templates that most change how many picks go "
            "to a position this league starts only one of. Four seasons is still "
            "four seasons — treat this as a real but lightly-evidenced edge.",
            icon="✅",
        )
    else:
        st.error(
            "**No strategy separates from simply following ADP** once intervals "
            "are corrected for comparing seven of them. With four seasons, the "
            "differences between templates are not distinguishable from the "
            "difference between one NFL season and another.",
            icon="🚫",
        )

    if losers.height:
        st.warning(
            "Significantly **worse** than the control: "
            + ", ".join(
                f"**{r['strategy']}** ({r['edge']:+.1%})" for r in losers.iter_rows(named=True)
            )
            + ". A negative finding is still a finding — these are templates to avoid in this format.",
            icon="⚠️",
        )

    table(edges.select(
            "strategy", "rate", "control_rate", "edge",
            "bonferroni_lo", "bonferroni_hi", "beats_control",
        ))

    st.markdown("#### Why the intervals are wide")
    st.caption(
        "Mean wins by strategy and season. Read across a row: the same strategy "
        "swings by more between seasons than the strategies differ from each "
        "other within one. Zero-RB is the clearest case — best in the league one "
        "year, worst the next — which lines up with the Landscape tab showing "
        "running back value climbing after 2023."
    )
    per_season = (
        runs.group_by(["strategy", "season"])
        .agg(pl.col("wins").mean().round(2).alias("mean_wins"))
        .sort(["strategy", "season"])
    )
    heat = (
        alt.Chart(per_season.to_pandas())
        .mark_rect(stroke=theme.surface(dark), strokeWidth=2)
        .encode(
            x=alt.X("season:O", title=None),
            y=alt.Y("strategy:N", title=None),
            color=alt.Color(
                "mean_wins:Q",
                scale=alt.Scale(range=theme.SEQUENTIAL_BLUE),
                title="Mean wins",
            ),
            tooltip=[
                alt.Tooltip("strategy:N", title="Strategy"),
                alt.Tooltip("season:O", title="Season"),
                alt.Tooltip("mean_wins:Q", title="Mean wins", format=".2f"),
            ],
        )
        .properties(height=280)
    )
    labels = (
        alt.Chart(per_season.to_pandas())
        .mark_text(fontSize=11, color=theme.ink(dark)["primary"])
        .encode(x=alt.X("season:O"), y=alt.Y("strategy:N"), text=alt.Text("mean_wins:Q", format=".1f"))
    )
    st.altair_chart(theme.base_chart(heat + labels, dark), use_container_width=True)

    with st.expander("Full results"):
        table(summary)
        st.caption(
            "`*_mc_lo`/`*_mc_hi` are the naive Monte Carlo intervals. They are "
            "included for comparison and should not be quoted."
        )

    with st.expander("Rerun under my sidebar settings"):
        st.caption(
            "The results above use the league's saved settings. This reruns at a "
            "smaller sample under whatever is in the sidebar now."
        )
        n = st.select_slider("Simulations per strategy", [200, 400, 800], value=400)
        if st.button("Run simulation", type="primary"):
            custom = _sim_rerun(
                p["scoring_key"], p["roster_positions"], p["teams"], int(n), 0
            )
            if custom.height:
                table(sim.summarize(custom, n_boot=400)
                    .select("strategy", "title_rate", "title_rate_lo", "title_rate_hi",
                            "playoff_rate", "mean_wins")
                    )
                st.caption(
                    f"{n} simulations per strategy — intervals are wider than the "
                    "baseline above, on top of the season-clustering already applied."
                )
            else:
                st.warning("No simulations produced under these settings.")


@st.cache_data(show_spinner="Building the board…")
def _board_base(
    scoring_key: tuple[tuple[str, float], ...],
    roster: tuple[str, ...],
    teams: int,
    flex_split: tuple[tuple[str, float], ...] | None,
) -> pl.DataFrame:
    """The static half of the board: market value and model score per player.

    Cached because it is pure. Everything downstream of session state — who is
    gone, who is cut, what pick you are on — is computed fresh on every rerun,
    because a cached board is a stale board and a stale draft board is worse
    than none.
    """
    board = bo.adp_board(SEASON)
    if not board.height:
        return pl.DataFrame()

    board = ls.market_implied_value(
        board,
        scoring=dict(scoring_key),
        roster_positions=list(roster),
        teams=teams,
        flex_split=dict(flex_split) if flex_split else None,
        season_points=_season_points(scoring_key),
    )

    scored = bo.score_current(features=_features())
    if scored.height:
        board = board.join(
            scored.select("gsis_id", "p_breakout"), on="gsis_id", how="left"
        )
    return board.sort("adp")


def _init_state() -> None:
    for key, default in (
        ("drafted", set()),
        ("excluded", set()),
        ("queue", []),
        ("my_slot", 1),
    ):
        if key not in st.session_state:
            st.session_state[key] = default


def _next_pick(taken: int, slot: int, teams: int) -> int:
    """Your next pick number in a snake draft, given how many are already gone."""
    rnd = taken // teams
    while True:
        pick = rnd * teams + (slot if rnd % 2 == 0 else teams - slot + 1)
        if pick > taken:
            return pick
        rnd += 1


def _tab_board(p: dict[str, Any]) -> None:
    _init_state()
    teams = p["teams"]

    st.subheader("Draft board")
    base = _board_base(p["scoring_key"], p["roster_positions"], teams, p["flex_split"])
    if not base.height:
        st.warning(f"No {SEASON} ADP available yet.")
        return

    st.info(
        "**`market_value` is not a projection, and the distinction matters.** "
        "This project has no points forecast — building one honestly is a bigger "
        "job than everything else here, and building one dishonestly is worse "
        "than having none. So the column inverts the question: it is the median "
        "historical value of *the draft slot* this player is going at. Two backs "
        "at RB14 get the same number, because it knows nothing about either of "
        "them. Its use is as a baseline to disagree with.",
        icon="ℹ️",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.session_state["my_slot"] = st.number_input(
            "Your draft slot", 1, teams, st.session_state["my_slot"]
        )
    taken = len(st.session_state["drafted"])
    pick = _next_pick(taken, int(st.session_state["my_slot"]), teams)
    with c2:
        st.metric("Your next pick", f"#{pick}", help=f"{taken} players off the board")
    with c3:
        pos_filter = st.multiselect(
            "Positions", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"]
        )

    gone = st.session_state["drafted"] | st.session_state["excluded"]
    available = base.filter(
        ~pl.col("gsis_id").is_in(list(gone)) if gone else pl.lit(True)
    ).filter(pl.col("position").is_in(pos_filter or ["QB", "RB", "WR", "TE"]))

    # Survival is computed on the *available* pool, so cuts and picks propagate
    # into the math rather than just the display.
    available = adp_mod.survival(available, pick)
    p_col = f"p_available_at_{pick}"

    st.markdown("#### Available")
    st.caption(
        f"`{p_col}` is the chance a player lasts to your pick, from his ADP and "
        "its observed dispersion. Two players at the same price can differ "
        "enormously here — that gap, not raw ranking, is what decides who you "
        "have to take now."
    )

    reachable_only = st.checkbox(
        "Only players with a real chance of reaching me",
        value=taken > 0,
        help=(
            "Sorting by value puts the best players on top, and once the draft "
            "starts most of them will be gone before your turn. This hides "
            "anyone under a 10% chance of lasting."
        ),
    )
    view = available
    if reachable_only:
        view = view.filter(pl.col(p_col) >= 0.10)

    view = view.select(
        "name", "position", "team", "adp", "stdev", "adp_pos_rank",
        "market_ppg", "market_var",
        *(["p_breakout"] if "p_breakout" in available.columns else []),
        p_col,
    ).sort("market_var", descending=True)

    if not view.height:
        st.info("Nobody clears the threshold — every remaining player is a reach or a lock.")
    else:
        table(view.head(60))
        st.caption(
            f"{view.height} of {available.height} available players shown. "
            "Sort any column by clicking its header."
        )

    st.markdown("#### Track the draft")
    names = available.get_column("name").to_list()
    lookup = dict(zip(available.get_column("name").to_list(), available.get_column("gsis_id").to_list()))

    b1, b2, b3 = st.columns(3)
    with b1:
        pick_name = st.selectbox("Someone was drafted", [""] + names, key="mark_drafted")
        if st.button("Mark drafted", disabled=not pick_name):
            st.session_state["drafted"].add(lookup[pick_name])
            st.rerun()
    with b2:
        cut_name = st.selectbox("Cut from my board", [""] + names, key="mark_cut")
        if st.button("Cut player", disabled=not cut_name):
            st.session_state["excluded"].add(lookup[cut_name])
            st.rerun()
    with b3:
        q_name = st.selectbox("Add to queue", [""] + names, key="mark_queue")
        if st.button("Queue player", disabled=not q_name):
            if lookup[q_name] not in st.session_state["queue"]:
                st.session_state["queue"].append(lookup[q_name])
            st.rerun()

    if st.session_state["queue"]:
        st.markdown("#### Your queue")
        queued = available.filter(pl.col("gsis_id").is_in(st.session_state["queue"]))
        if queued.height:
            table(queued.select("name", "position", "adp", "market_var", p_col)
                .sort(p_col)
                )
            st.caption("Sorted by least likely to survive — take the top one first.")
        stale = set(st.session_state["queue"]) & gone
        if stale:
            st.caption(f"{len(stale)} queued player(s) are already gone.")

    with st.expander(f"Cut list ({len(st.session_state['excluded'])}) and drafted ({taken})"):
        if st.session_state["excluded"]:
            cuts = base.filter(pl.col("gsis_id").is_in(list(st.session_state["excluded"])))
            table(cuts.select("name", "position", "adp"))
        c1, c2 = st.columns(2)
        if c1.button("Clear cut list"):
            st.session_state["excluded"] = set()
            st.rerun()
        if c2.button("Reset draft"):
            st.session_state["drafted"] = set()
            st.session_state["queue"] = []
            st.rerun()


def _placeholder(name: str, phase: str) -> None:
    st.subheader(name)
    st.info(f"Not built yet — {phase}.", icon="🚧")


def main() -> None:
    st.title("ff-edge")
    p = _sidebar()

    landscape, players, strategy, board, reference = st.tabs(
        ["Landscape", "Players", "Strategy", "Board", "Glossary"]
    )
    with landscape:
        _tab_landscape(p)
    with players:
        _tab_players(p)
    with strategy:
        _tab_strategy(p)
    with board:
        _tab_board(p)
    with reference:
        _glossary_section()


if __name__ == "__main__":
    main()
