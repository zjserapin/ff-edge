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

from src import archetypes as ar
from src import board as bd
from src import breakout as bo
from src import features as ft
from src import glossary
from src import claims as cm
from src import landscape as ls
from src import projection as pj
from src import promotion as pm
from src import rookies as rk
from src import scoring as sc
from src import simulate as sim
from src import stability as stab
from src import theme
from src import valuation as val
from src.config import (
    DEFAULT_ROSTER_POSITIONS,
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    CURRENT_SEASON,
    FANTASY_POSITIONS,
    FEATURE_SEASONS,
    OUTPUT_DIR,
    SEASON,
)

st.set_page_config(page_title="ff-edge", page_icon="🏈", layout="wide")


# --- Cache boundaries -------------------------------------------------------


def _key(scoring: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Make a scoring dict hashable for @st.cache_data."""
    return tuple(sorted((k, float(v)) for k, v in scoring.items()))


# Acronyms that a naive capitalize() would mangle into "Ppg" or "Adp".
_ACRONYMS = {"ppg", "adp", "par", "auc", "yprr", "tprr", "ypt", "ypc", "ypa", "hhi", "ci"}


def _header(column: str) -> str:
    """Readable header for a column: its glossary label, else a tidied name."""
    term = glossary.lookup(column)
    if term:
        return term.label
    words = [
        w.upper() if w.lower() in _ACRONYMS else w
        for w in column.replace("_", " ").split()
    ]
    if not words:
        return column
    head, *rest = words
    return " ".join([head if head.isupper() else head.capitalize(), *rest])


def table(df: pl.DataFrame, pretty: bool = True, **kwargs: Any) -> None:
    """Render a frame with readable headers and a hover definition on each.

    Every table in this app goes through here rather than st.dataframe directly.
    A column named `wopr` or `par_mean_starter` means nothing on its own, and a
    glossary nobody opens is a glossary nobody reads — so the name is rewritten
    to the glossary's label and the definition is attached to the header.

    `pretty=False` keeps the raw column names, which is what you want when the
    frame is going to be copied out and used somewhere else.
    """
    if not df.height:
        return

    pdf = df.to_pandas()
    if pretty:
        renamed = {c: _header(c) for c in pdf.columns}
        # Two raw columns can share a label (`vacated_target_share` in the
        # rookie frame and the veteran one). Keep the raw name on the second so
        # pandas does not silently collapse them into one column.
        seen: set[str] = set()
        for raw, label in list(renamed.items()):
            if label in seen:
                renamed[raw] = raw
            seen.add(label)
        pdf = pdf.rename(columns=renamed)
        helps = {
            renamed[raw]: text
            for raw, text in glossary.column_help(list(df.columns)).items()
            if raw in renamed
        }
    else:
        helps = glossary.column_help(list(pdf.columns))

    config = {col: st.column_config.Column(help=text) for col, text in helps.items()}
    st.dataframe(
        pdf, use_container_width=True, hide_index=True, column_config=config, **kwargs
    )


def chart_note(columns: list[str], extra: str = "") -> None:
    """A small definitions footer under a chart.

    Sized to be read without competing with the chart above it: the same muted
    caption tone as the rest of the dashboard, one line per metric, definitions
    only for the fields actually plotted. A reader should not have to open the
    Glossary tab to know what an axis means.
    """
    # Bold via <strong>, not markdown asterisks: Streamlit does not run the
    # markdown parser inside a raw HTML block, so `**x**` renders literally.
    parts = []
    for column in columns:
        term = glossary.lookup(column)
        if term:
            parts.append(
                f"<strong style='color:#c3c2b7'>{term.label}</strong> — {term.short}"
            )
    if extra:
        parts.append(extra)
    if not parts:
        return

    st.markdown(
        "<div style='font-size:0.82rem; line-height:1.6; color:#898781; "
        "margin-top:-0.5rem; padding:0 0 0.5rem 2px'>"
        + "<br>".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def data_expander(df: pl.DataFrame, label: str = "Show the numbers") -> None:
    """The exact frame a chart just drew, so the table always matches the chart.

    Tied to each chart rather than collected at the bottom of the tab. A single
    shared table view drifts out of sync the moment a slider moves — it was
    showing every cutoff while the chart showed one — and a table that disagrees
    with the chart above it is worse than no table.
    """
    if not df.height:
        return
    with st.expander(f"{label}  ({df.height} rows)"):
        table(df)


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
def _resolved_league_id() -> str:
    """Whatever league we are reading, from the env or from discovery."""
    return sc.resolve_league_id()


@st.cache_data(show_spinner=False)
def _league(league_id: str) -> dict[str, Any]:
    return sc.league_settings(league_id or None)


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

    league_id = _resolved_league_id()
    league = _league(league_id)
    if league["source"] != "sleeper":
        st.sidebar.caption(
            "Using saved 2026 settings — export `FF_EDGE_LEAGUE_ID` (and "
            "`FF_EDGE_SLEEPER_USER`) to read your own league live."
        )
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

    superflex = sum(
        1 for s in (league["roster_positions"] or DEFAULT_ROSTER_POSITIONS)
        if "QB" in sc.FLEX_SLOTS.get(s, ())
    )
    if superflex:
        st.sidebar.caption(
            f"{superflex} SUPER_FLEX slot{'s' if superflex > 1 else ''} on this "
            "roster are always computed, never set by the sliders above — the "
            "sliders name RB/WR/TE shares, which cannot express the only "
            "question a superflex asks. They go to quarterbacks, which is why "
            "replacement QB sits near QB21 rather than QB11."
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
        "league_id": league_id,
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
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=110, filled=True))
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
        .properties(height=380)
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
    chart_note(["par_mean_starter", "demand", "replacement_rank", "replacement_ppg"])
    data_expander(
        par.select(
            "season", "position", "demand", "replacement_rank", "replacement_ppg",
            "par_mean_starter", "par_total",
        )
    )

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
            .mark_line(
                strokeWidth=2.5,
                point=alt.OverlayMarkDef(size=70, filled=True),
            )
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
            .properties(height=340, width=260)
            .facet(column=alt.Column("position:N", title=None, sort=list(theme.position_colors())))
        )
        st.altair_chart(theme.base_chart(curves, dark), use_container_width=True)
        chart_note(
            ["pos_rank", "ppg", "fantasy_points", "games"],
            extra=(
                f"Showing ranks 1–{max_rank} for {', '.join(str(c) for c in sorted(chosen))}. "
                "Narrow the slider to spread the early ranks out — that is where "
                "the dropoff actually decides your draft."
            ),
        )
        data_expander(
            scarcity.filter(pl.col("season").is_in(sorted(chosen))).select(
                "season", "position", "pos_rank", "player_display_name",
                "games", "fantasy_points", "ppg", "par_ppg",
            )
        )

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
        .properties(height=320)
    )
    st.altair_chart(theme.base_chart(bars, dark), use_container_width=True)
    chart_note(
        ["overall_par_rank", "par_ppg"],
        extra=(
            f"Counting how many of each position land in the top {cutoff} by value "
            "over replacement, per season. Move the slider to change the cutoff — "
            "the table below follows it."
        ),
    )
    data_expander(mix.filter(pl.col("cutoff") == cutoff))

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
    chart_note(
        ["share", "pool_size"],
        extra=(
            "Each panel is a different top-N. The pool is capped at roughly three "
            "times the number of starters, so this measures how top-heavy the "
            "draftable players are rather than how many bodies the league used."
        ),
    )
    data_expander(conc)


@st.cache_data(show_spinner="Building features…")
def _features() -> pl.DataFrame:
    return ft.build()


@st.cache_data(show_spinner="Scoring quality and opportunity…")
def _scores(season: int, min_games: int) -> pl.DataFrame:
    return ar.scores(season, min_games=min_games, df=_features())


def _tab_players(p: dict[str, Any]) -> None:
    dark = p["dark"]
    feats = _features()
    if not feats.height:
        st.warning("No features built. Run `uv run python -m src.bootstrap --light`.")
        return

    season = int(feats.get_column("season").max())
    st.subheader("Comparable players")
    st.caption(
        f"The {season} players whose per-opportunity profile sits closest to this "
        "one, by distance in the standardized quality space. A cheap player next "
        "to expensive ones is the thing worth a second look."
    )
    st.info(
        "**The k-means archetypes that used to sit here have been removed.** "
        "Silhouette topped out between 0.19 and 0.29, which is a partition of a "
        "continuum rather than a set of groups, and cluster membership added "
        "nothing downstream — fed to the projection model as soft memberships it "
        "scored 0.022 *below* ADP alone. Nearest neighbours answer the same "
        "question directly, without throwing the distances away to get a label.",
        icon="🗑️",
    )

    min_games = st.slider("Minimum games", 4, 14, 8, key="scores_min_games")
    scored = _scores(season, min_games)
    if not scored.height:
        st.info("Not enough qualified players to score.")
        return

    pos = st.selectbox("Position", sorted(scored.get_column("position").unique().to_list()))
    pool = scored.filter(pl.col("position") == pos).sort("pos_rank")
    who = st.selectbox("Player", pool.get_column("player_name").to_list())
    pid = pool.filter(pl.col("player_name") == who).get_column("player_id")[0]

    nb = ar.neighbors(pid, scored, feats, n=8, season=season)
    if nb.height:
        table(nb.select("player_name", "team", "distance", "ppg", "pos_rank", "games"))

    with st.expander("Quality and opportunity scores for the whole position"):
        st.caption(
            "Both are weighted means of standardized metrics, weighted by how "
            "well each one repeats year to year. Relative to this position and "
            "season — +1 is one standard deviation, not an absolute."
        )
        table(
            pool.select(
                "player_name", "team", "games", "quality_score", "quality_pct",
                "opportunity_score", "opportunity_pct", "ppg", "pos_rank",
            ).sort("quality_pct", descending=True)
        )

    with st.expander("Feature coverage"):
        st.caption(
            "Next Gen Stats cover qualified receivers only, so ~30% non-null is "
            "expected there. Snap share below ~90% would mean the "
            "pfr_id → gsis_id bridge broke."
        )
        table(ft.coverage_report(feats).filter(pl.col("scope") == "ALL"))

    st.divider()
    # Stability comes before the backtest deliberately. It answers whether any of
    # these measurements repeat at all, which is the question that has to be
    # settled before an AUC means anything.
    _stability_section(dark)
    st.divider()
    _breakout_section(dark)
    st.divider()
    _projection_section(dark)
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


@st.cache_data(show_spinner="Measuring metric stability…")
def _stability() -> tuple[pl.DataFrame, pl.DataFrame]:
    table = stab.year_over_year()
    return table, stab.axis_summary(table)


def _stability_section(dark: bool) -> None:
    st.subheader("Does a metric mean the same thing next year?")
    st.caption(
        "Rank every qualified player within his position and season, pair each "
        "player-season with his own next one, and correlate the two percentiles. "
        "No model and no outcome — this asks only whether a measurement repeats. "
        "A metric that does not repeat is describing last season's variance, and "
        "a board built on it ranks noise."
    )

    table_df, summary = _stability()
    if not table_df.height:
        st.info("Not enough paired seasons to measure stability.")
        return

    st.success(
        "**Opportunity persists better than quality at every position** — the "
        "premise this project is built on, with a number attached. It is also why "
        "quality and volume are kept on separate axes instead of being mixed into "
        "one distance metric: they carry forward at very different rates.",
        icon="✅",
    )

    pivot = summary.pivot(on="axis", index="position", values="mean_r")
    order = [c for c in ("position", "opportunity", "outcome", "quality") if c in pivot.columns]
    table(pivot.select(order).sort("position"))
    st.caption(
        "Mean year-over-year percentile correlation across the metrics on each "
        "axis. 1.0 would mean a metric repeats perfectly; 0.0 means this season "
        "tells you nothing about next."
    )

    position = st.selectbox(
        "Position", ["WR", "RB", "TE", "QB"], key="stability_pos",
        help="Stability is measured within position — a tight end's separation is not a receiver's.",
    )
    sub = table_df.filter(pl.col("position") == position).sort("r_yoy", descending=True)

    chart_df = sub.to_pandas()
    colors = theme.position_colors(dark)
    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            y=alt.Y("metric:N", sort="-x", title=None),
            x=alt.X("r_yoy:Q", title="Year-over-year correlation"),
            color=alt.Color(
                "axis:N",
                title="Axis",
                scale=alt.Scale(
                    domain=["opportunity", "quality", "outcome"],
                    range=[colors["RB"], colors["WR"], colors["TE"]],
                ),
            ),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("axis:N", title="Axis"),
                alt.Tooltip("r_yoy:Q", title="Correlation", format=".3f"),
                alt.Tooltip("n_pairs:Q", title="Pairs"),
                alt.Tooltip("verdict:N", title="Verdict"),
            ],
        )
        .properties(height=max(220, 18 * len(chart_df)))
    )
    floor = (
        alt.Chart(chart_df)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=theme.ink(dark)["muted"])
        .encode(x=alt.datum(stab.NOISE_FLOOR))
    )
    st.altair_chart(theme.base_chart(bars + floor, dark), use_container_width=True)
    chart_note(
        ["r_yoy", "n_pairs"],
        extra=(
            f"Dashed line is the noise floor ({stab.NOISE_FLOOR:.2f}). Below it a "
            "metric is measuring the season rather than the player, and it is "
            "kept out of every feature set."
        ),
    )
    data_expander(sub)

    dropped = stab.noisy_features(table_df)
    if dropped.height:
        with st.expander("What this check removed"):
            st.markdown(
                "Six columns were dropped from the quality sets for failing here, "
                "and they are all things the fantasy literature treats as skill. "
                "`contested_catch_rate` is the cleanest case: hand-charted, "
                "plausible-sounding, and a coin flip on a small denominator."
            )
            table(dropped)


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
            f"this model sits below a coin flip with inverted calibration. "
            f"Fitting per position moves it to {auc:.3f}, roughly chance — "
            "pooling was averaging four different relationships and fitting none "
            f"of them. It still does not beat draft price: the gap interval "
            f"[{d_lo:+.3f}, {d_hi:+.3f}] covers zero. 'No signal' is a different "
            "and more honest result than 'reliably wrong', and it is the one "
            "supported here. The continuous-target version below measures the "
            "same gap about five times more precisely.",
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
        "cost of stratifying an 831-row sample four ways, and it is the first "
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
        "to right. Four groups rather than ten because 540 out-of-sample rows "
        "makes a decile ±6 points — too wide to read."
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
    chart_note(
        ["base_rate", "lift"],
        extra=(
            "Dashed line is the base rate; whiskers are 95% Wilson intervals. "
            "A working model would step upward left to right."
        ),
    )
    data_expander(cal)

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


@st.cache_data(show_spinner="Fitting the continuous-target model…")
def _projection() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = bo.training_frame()
    if not train.height:
        return pl.DataFrame(), pl.DataFrame()
    preds = pj.fit_predict(train)
    return preds, pj.skill(preds)


def _projection_section(dark: bool) -> None:
    st.subheader("The same question, asked more precisely")
    st.caption(
        "The backtest above collapses the outcome into beat/miss, which throws "
        "away everything except which side of the line a player landed on — RB4 "
        "and RB40 are the same 'no' if you paid for RB3. This version keeps the "
        "outcome continuous: predict where a player finishes among drafted "
        "players at his position, and score it by rank correlation."
    )

    preds, skill = _projection()
    if not skill.height:
        st.info("Not enough labeled seasons to fit.")
        return

    row = skill.filter(pl.col("scope") == "all")
    model = float(row.get_column("spearman")[0])
    price = float(row.get_column("spearman_adp_only")[0])
    lo, hi = float(row.get_column("delta_lo")[0]), float(row.get_column("delta_hi")[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", f"{model:.3f}", help=glossary.describe("spearman"))
    c2.metric("ADP alone", f"{price:.3f}", help=glossary.describe("spearman_adp_only"))
    c3.metric(
        "Difference",
        f"{model - price:+.3f}",
        delta=f"[{lo:+.3f}, {hi:+.3f}]",
        delta_color="off",
        help="95% interval from a season-level bootstrap.",
    )

    st.info(
        f"**Predicting the finish is easy. Beating the price is not.** Both the "
        f"model and ADP alone rank next season at about {model:.2f} — usage "
        "persists, so anything sane gets there. The number that matters is the "
        f"gap between them, and it is {model - price:+.3f} with an interval of "
        f"[{lo:+.3f}, {hi:+.3f}]. That interval is roughly five times tighter "
        "than the beat/miss version above, which is the entire point of the "
        "continuous target: 'no edge' measured to ±0.01 is a finding, while 'no "
        "edge' measured to ±0.06 is only an absence of evidence.",
        icon="📏",
    )

    st.markdown("#### Per position")
    st.caption(
        "Quarterback is the one position where the usage features make things "
        "actively worse — the interval sits entirely below zero. Rushing share "
        "and expected-points share are already fully priced there, and adding "
        "them to a small sample adds variance and nothing else."
    )
    table(
        skill.filter(pl.col("scope").is_in(["QB", "RB", "WR", "TE"])).select(
            "scope", "n", "spearman", "spearman_adp_only", "delta", "delta_lo", "delta_hi"
        )
    )

    with st.expander("Full projection numbers"):
        st.markdown("**Skill — overall, by position, by test season**")
        table(skill)
        st.markdown("**Out-of-sample projections**")
        table(
            preds.select(
                "label_season", "name", "position", "adp_pos_rank",
                "pred", "pred_adp_only", "finish_pos_rank", "finish_pct",
            ).sort(["label_season", "pred"], descending=[True, True])
        )


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

    if any("QB" in sc.FLEX_SLOTS.get(s, ()) for s in p["roster_positions"]):
        st.error(
            "**These numbers describe the 2024–25 format, not your superflex "
            "league.** A strategy comparison only means something when the "
            "draft market and the templates match the format, and neither does "
            "yet: the board is priced from 1QB ADP, and every template here was "
            "written for two FLEX slots — `elite_qb` drafts a single "
            "quarterback, which cannot fill a superflex. Run as-is under a "
            "superflex roster and the sim would report that quarterbacks are "
            "nearly free *and* start twice, which is an artifact of the "
            "mismatched market rather than a finding. The replacement-level "
            "and PAR numbers on every other tab **are** superflex-correct; only "
            "this tab is pinned. Unpinning it needs 2QB ADP boards (FFC has "
            "them back to 2020) and QB-heavier templates — not new machinery.",
            icon="🔒",
        )

    st.altair_chart(theme.base_chart(_strategy_chart(summary, dark), dark), use_container_width=True)
    chart_note(
        ["title_rate", "playoff_rate", "mean_wins"],
        extra=(
            "Intervals resample whole seasons, not individual simulations — "
            "simulations within a season share one realized set of player "
            "outcomes, so treating them as independent would report bars roughly "
            "ten times too narrow."
        ),
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
    chart_note(
        ["mean_wins"],
        extra=(
            "Read across a row, not down a column: the swing within one strategy "
            "across seasons is the point."
        ),
    )
    data_expander(per_season)

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


@st.cache_data(show_spinner="Valuing the board…")
def _valuation() -> pl.DataFrame:
    return val.board(df=_features())


def _tab_board(p: dict[str, Any]) -> None:
    dark = p["dark"]
    st.subheader("Who the market has wrong")

    board = _valuation()
    if not board.height:
        st.warning(f"No {SEASON} ADP available yet, or the feature table is cold.")
        return

    st.caption(
        f"Every drafted WR, RB and TE with at least 100 routes in "
        f"{CURRENT_SEASON}, scored on two axes the draft market treats as one."
    )
    st.info(
        "**The argument in one line: ADP prices volume well and quality badly.** "
        "Everyone can see a 28% target share; far fewer are pricing 2.4 yards "
        "per route run on 300 routes. So `quality_pct` is compared against "
        "`market_pct` — both percentiles within position — and the difference is "
        "`value_gap`. This is a disagreement score, not a projection: it says "
        "where this project's read differs from the market's, which is a reason "
        "to look closer, not a reason to be right.",
        icon="ℹ️",
    )

    counts = val.summary(board)
    c1, c2, c3 = st.columns(3)
    for col, verdict, label in (
        (c1, "undervalued", "Undervalued"),
        (c2, "fairly priced", "Fairly priced"),
        (c3, "overvalued", "Overvalued"),
    ):
        n = int(counts.filter(pl.col("verdict") == verdict).get_column("n").sum())
        col.metric(label, n)

    # --- the two-axis picture ---
    st.markdown("#### Quality against price")
    st.caption(
        "Up is better per opportunity; right is more expensive. The top-left "
        "quadrant is the one worth your time — good players the market has not "
        "paid for. Bottom-right is the reverse. Point size is how much room he "
        "has to grow into more volume."
    )
    pos_pick = st.multiselect(
        "Positions", list(val.SKILL_POSITIONS), default=list(val.SKILL_POSITIONS)
    )
    view = board.filter(pl.col("position").is_in(pos_pick or list(val.SKILL_POSITIONS)))
    if not view.height:
        return

    scatter_pd = view.to_pandas()
    points = (
        alt.Chart(scatter_pd)
        .mark_circle(opacity=0.85, stroke=theme.surface(dark), strokeWidth=1)
        .encode(
            x=alt.X("market_pct:Q", title="Draft price percentile (100 = most expensive)"),
            y=alt.Y("quality_pct:Q", title="Quality percentile (100 = best per opportunity)"),
            size=alt.Size("path_score:Q", title="Room to grow", scale=alt.Scale(range=[40, 400])),
            color=alt.Color("position:N", scale=theme.position_scale(dark), title="Position"),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("adp:Q", title="ADP", format=".1f"),
                alt.Tooltip("quality_pct:Q", title="Quality %ile", format=".0f"),
                alt.Tooltip("market_pct:Q", title="Price %ile", format=".0f"),
                alt.Tooltip("value_gap:Q", title="Value gap", format="+.0f"),
                alt.Tooltip("path_score:Q", title="Room to grow", format=".0f"),
                alt.Tooltip("yprr:Q", title="Yds/route", format=".2f"),
                alt.Tooltip("verdict:N", title="Verdict"),
            ],
        )
        .properties(height=420)
    )
    parity = (
        alt.Chart(pl.DataFrame({"a": [0, 100]}).to_pandas())
        .mark_line(strokeDash=[5, 5], strokeWidth=1, color=theme.ink(dark)["muted"])
        .encode(x=alt.X("a:Q"), y=alt.Y("a:Q"))
    )
    labels = (
        alt.Chart(scatter_pd[scatter_pd["value_gap"].abs() >= 42])
        .mark_text(align="left", dx=9, fontSize=10, color=theme.ink(dark)["primary"])
        .encode(x=alt.X("market_pct:Q"), y=alt.Y("quality_pct:Q"), text="name:N")
    )
    st.altair_chart(theme.base_chart(parity + points + labels, dark), use_container_width=True)
    chart_note(
        ["quality_pct", "market_pct", "value_gap", "path_score"],
        extra=(
            "Dashed line is where quality and price agree; distance from it is the "
            "value gap. Only players more than 42 points off the line are labeled — "
            "hover for the rest."
        ),
    )

    # --- the lists ---
    show = ["name", "position", "team", "adp", "quality_pct", "opportunity_pct",
            "market_pct", "value_gap", "path_score", "yprr", "teammate_top_share", "verdict"]

    st.markdown("#### Undervalued — quality above price, with somewhere to go")
    st.caption(
        "Filtered on room to grow, which is what separates this from a list of "
        "good players on crowded depth charts. A 90th-percentile receiver who is "
        "already his team's alpha has no volume left to gain — his price is high "
        "because he earned it."
    )
    min_path = st.slider(
        "Minimum room to grow", 0, 100, 50, 5,
        help="Blends unused opportunity, volume his team just lost, and how big the teammate in front of him is.",
    )
    under = val.undervalued(view, min_path=float(min_path), limit=25)
    if under.height:
        table(under.select([c for c in show if c in under.columns]))
    else:
        st.info("Nobody clears both thresholds at these settings.")

    st.markdown("#### Overvalued — price above quality")
    st.caption(
        "Read this list more sceptically than the one above. Expensive players "
        "are expensive partly because they command volume, and volume genuinely "
        "scores points — a mediocre efficiency profile on 30% of the targets is "
        "still a good fantasy season. This flags where the price is paying for "
        "last year's production rather than for the player."
    )
    over = val.overvalued(view, limit=25)
    if over.height:
        table(over.select([c for c in show if c in over.columns]))

    # --- comparables ---
    st.markdown("#### Who does this player look like?")
    st.caption(
        "Nearest quality profiles at the same position, shown next to what they "
        "cost. If the neighbours are all going four rounds earlier, that is the "
        "finding."
    )
    names = view.sort("adp").get_column("name").to_list()
    who = st.selectbox("Player", names, key="comp_player")
    target = view.filter(pl.col("name") == who)
    if target.height:
        row = target.row(0, named=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ADP", f"{row['adp']:.1f}")
        m2.metric("Quality %ile", f"{row['quality_pct']:.0f}")
        m3.metric("Price %ile", f"{row['market_pct']:.0f}")
        m4.metric("Value gap", f"{row['value_gap']:+.0f}", help=glossary.describe("value_gap"))

        comps = val.comparables(view, row["gsis_id"], n=8, df=_features())
        if comps.height:
            table(
                comps.select(
                    "name", "team", "distance", "quality_pct", "opportunity_pct",
                    "adp", "prior_ppg", "verdict",
                )
            )
            cheaper = comps.filter(pl.col("adp") < row["adp"]).height
            st.caption(
                f"{comps.height - cheaper} of {comps.height} comparable profiles "
                f"cost more than {who} does."
            )

    with st.expander("Full board"):
        table(view.select([c for c in show if c in view.columns]))


@st.cache_data(show_spinner="Building the promotion cohort…")
def _promotion_cohort() -> pl.DataFrame:
    return pm.cohort(_features())


@st.cache_data(show_spinner="Aggregating weekly usage…")
def _weekly_trust(season: int) -> pl.DataFrame:
    return pm.weekly_trust(season)


@st.cache_data(show_spinner="Scoring the claims ledger…")
def _claim_flags() -> pl.DataFrame:
    return cm.flags()


def _screen_sections(dark: bool) -> None:
    """The promotion screen and the claims ledger.

    Was its own tab. It is a tie-break layer rather than a destination — the
    board orders players, and this is what separates two of them inside a tier —
    so it now sits at the bottom of Draft Day, which is the moment it is read
    under time pressure.
    """
    feats = _features()
    if not feats.height:
        st.warning("No features built. Run `uv run python -m src.bootstrap --light`.")
        return

    st.subheader("Promotion screen")
    st.caption(
        "You say whose role is growing — from camp reports, a depth chart, a "
        "departure. The screen does the half the data can do: grade the player "
        "on what predicted production among historically promoted players at "
        "his position, and report the base rate for profiles like his."
    )
    st.info(
        "**The model cannot tell you who gets the job, and does not try.** "
        "Correlation with next-season role growth: vacated targets −0.04, "
        "vacated carries −0.03, prior quality 0.02. The one real term is prior "
        "opportunity at −0.33 — pure mean reversion. What *is* predictable is "
        "how a promoted player does once you know the role is coming: trust "
        "markers at RB (snaps, red-zone carries), efficiency at WR/TE.",
        icon="🧭",
    )

    # The claims ledger supplies candidates; the user stays the editor. A
    # button rather than an auto-fill, so a hand-typed list is never clobbered.
    ledger_flags = _claim_flags()
    if ledger_flags.height:
        flagged = ledger_flags.filter(pl.col("grade").is_in(["A", "B", "C"]))
        if flagged.height and st.button(
            f"Load {flagged.height} flagged names from the claims ledger",
            help="Players the ledger currently grades A/B/C for role growth. "
            "Every grade decomposes into quoted claims below.",
        ):
            st.session_state["screen_names"] = "\n".join(
                flagged.get_column("player_name").to_list()
            )

    raw_names = st.text_area(
        "Players whose role is growing (one per line)",
        placeholder="Bhayshul Tuten\nJaylen Wright",
        key="screen_names",
    )
    names = [n for n in (raw_names or "").splitlines() if n.strip()]
    if not names:
        st.caption("Type a name or two to get a grade.")
    else:
        coh = _promotion_cohort()
        grades, missing = pm.screen(names, df=feats, coh=coh)
        if missing:
            st.warning("Not found among qualified players: " + ", ".join(missing))
        if grades.height:
            lead = [
                "player_name", "position", "team", "games", "screen_pct",
                "quality_tier", "tier_hit_rate", "tier_ci_lo", "tier_ci_hi", "tier_n",
            ]
            table(grades.select([c for c in lead if c in grades.columns]))
            st.caption(
                "The hit rate is a base rate over comparable promoted players, "
                "not a projection for this one. A null tier means fewer than 8 "
                "games — a season fragment is not graded."
            )
            with st.expander("Criteria percentiles (RB efficiency shown as reference only)"):
                crit_cols = [c for c in grades.columns if c.endswith("_pct") and c != "screen_pct"]
                table(grades.select(["player_name", "position"] + crit_cols))

            # Weekly trend for one graded player: a December role change should
            # be visible, not averaged into a season number.
            season = int(feats.get_column("season").max())
            who = st.selectbox(
                "Weekly usage trend", grades.get_column("player_name").to_list(),
                key="screen_trend_player",
            )
            row = feats.filter(
                (pl.col("season") == season) & (pl.col("player_name") == who)
            )
            wk = _weekly_trust(season)
            if row.height and wk.height:
                pid = row.get_column("player_id")[0]
                position = row.get_column("position")[0]
                metric = st.selectbox(
                    "Metric",
                    ["rz_carry_share_wk", "carry_share_wk", "target_share_wk"]
                    if position == "RB"
                    else ["target_share_wk", "carry_share_wk", "rz_carry_share_wk"],
                    format_func=lambda c: glossary.TERMS[c].label,
                    key="screen_trend_metric",
                )
                mine = (
                    wk.filter(pl.col("player_id") == pid)
                    .select("week", metric)
                    .drop_nulls()
                    .sort("week")
                    .with_columns(
                        pl.col(metric).rolling_mean(4, min_samples=1).alias("trend")
                    )
                )
                if mine.height:
                    blue = theme.SEQUENTIAL_BLUE[3]
                    pts = (
                        alt.Chart(mine.to_pandas())
                        .mark_point(size=70, filled=True, color=theme.ink(dark)["muted"])
                        .encode(
                            x=alt.X("week:Q", title="Week", axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y(f"{metric}:Q", title=glossary.TERMS[metric].label,
                                    axis=alt.Axis(format=".0%")),
                            tooltip=[
                                alt.Tooltip("week:Q", title="Week"),
                                alt.Tooltip(f"{metric}:Q", title="Share", format=".1%"),
                            ],
                        )
                    )
                    line = (
                        alt.Chart(mine.to_pandas())
                        .mark_line(strokeWidth=2, color=blue)
                        .encode(x="week:Q", y=alt.Y("trend:Q", title=None))
                    )
                    st.altair_chart(
                        theme.base_chart(pts + line, dark), use_container_width=True
                    )
                    chart_note(
                        [metric],
                        extra="Points are single weeks; the line is a 4-week rolling mean. "
                        "A week with no team red-zone carries is a gap, not a zero.",
                    )
                    data_expander(mine)

    st.divider()
    coh = _promotion_cohort()
    if not coh.height:
        return

    with st.expander("The evidence: what predicts a promoted player, by position"):
        st.caption(
            "Rank correlation of each prior-season metric with next-season PPG "
            "percentile, among promoted players only. The RB efficiency rows "
            "near zero are the finding — five metrics, all noise. Criteria were "
            "fixed from the original exploratory session, not re-searched here."
        )
        table(pm.validate(coh))

    with st.expander("Base rates by prior quality tier"):
        st.caption(
            "Quality is a filter, not a picker — monotone, roughly 4× from "
            "bottom to top, and the intervals say how little the per-position "
            "cells can carry."
        )
        table(pm.quality_tiers(coh))

    with st.expander("The archetype split (decay-rate thread, closed)"):
        st.caption(
            "Promoted RBs split at the median prior pass-touch mix. The "
            "hypothesis said receiving-profile backs should hit less often "
            "under a bigger role; the point estimates lean the other way and "
            "the intervals overlap. Two cells of ~30 is all this cohort "
            "supports — measured, reported, and not a feature."
        )
        table(pm.archetype_split(coh))

    st.divider()
    st.subheader("Claims ledger")
    st.caption(
        "Automated role-change claims: depth-chart moves (no LLM), plus "
        "beat-report extraction when an API key is set. Every score "
        "decomposes into the quoted claims that produced it — the moment "
        "that stops being true, delete the layer. Absence of claims about a "
        "boring veteran starter is bullish stability, not missing data."
    )
    ledger = cm.load()
    if not ledger.height:
        st.info(
            "Ledger is empty. It fills as `uv run python -m src.bootstrap "
            "--light` (or a daily cron) runs `claims.pull()` — depth-chart "
            "diffs need two snapshots on different days, and news extraction "
            "needs `ANTHROPIC_API_KEY` in the shell.",
            icon="📓",
        )
    else:
        st.caption(
            f"{ledger.height} claims · newest "
            f"{ledger.get_column('claimed_on').max()}"
        )
        if ledger_flags.height:
            table(
                ledger_flags.select(
                    "player_name", "team", "role_score", "grade", "n_claims",
                    "n_novel", "best_tier", "latest_claim", "adp_at_latest",
                )
            )
        who_claims = st.selectbox(
            "Decompose a player's flag into its claims",
            ledger_flags.get_column("player_name").to_list()
            if ledger_flags.height
            else [],
            key="ledger_player",
        )
        if who_claims:
            table(
                cm.player_claims(who_claims, ledger).select(
                    "claimed_on", "claim_type", "direction", "specificity",
                    "source", "source_tier", "novel", "claim_score", "quote",
                )
            )
        with st.expander("Source grades (converge over seasons, not weeks)"):
            st.caption(
                "Per-source hit rate on resolved claims — did the claimed "
                "usage materialize within three weeks? This is what earns a "
                "source a better tier than the hand-set starting dict. "
                "Empty until the season provides resolutions; season one is "
                "labeled-data collection by design."
            )
            grades_df = cm.source_grades(cm.resolve(ledger))
            if grades_df.height:
                table(grades_df)
            else:
                st.caption("No resolved claims yet.")


# --- Draft Day tab ----------------------------------------------------------


@st.cache_data(show_spinner="Building the draft board…")
def _draft_board() -> dict[str, Any]:
    """The board, with team environment attached.

    `board.build` deliberately stops before the environment join so its own
    tests do not depend on Vegas lines being reachable. The app wants both, and
    `context_flags` cannot run without `env_swing`, so the two are composed here
    rather than widening the module's contract.
    """
    out = bd.build()
    if out["players"].height:
        out["players"] = bd.attach_quality(bd.attach_environment(out["players"]))
    return out


@st.cache_data(show_spinner="Reading your picks…")
def _my_picks() -> pl.DataFrame:
    return bd.picks()


def _tab_draft_day(p: dict[str, Any]) -> None:
    dark = p["dark"]
    data = _draft_board()

    for warning in data.get("warnings", []):
        st.warning(warning)

    players = data["players"]
    if not players.height:
        st.info(
            "No board yet. This tab needs an ADP board for the season and a "
            "league to price it against — run with `FF_EDGE_LEAGUE_ID` set.",
            icon="🗒️",
        )
        return

    profile = data["profile"]
    st.subheader("Draft day")
    st.caption(
        f"{profile.label} · priced off the {profile.adp_scoring} board · "
        f"{players.height} draftable players"
    )

    # --- your picks ---------------------------------------------------------
    picks = _my_picks()
    if picks.height:
        usable = picks.filter(pl.col("usable"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Picks owned", picks.height)
        c2.metric("Usable", usable.height)
        c3.metric(
            "First pick",
            int(usable.get_column("pick_no")[0]) if usable.height else "—",
        )
        st.caption(
            "A keeper is slotted onto a specific pick and spends it, so owned "
            "and usable are different numbers. Picks acquired in a trade sit at "
            "the *original* owner's draft slot, not yours."
        )
        table(
            picks.select("round", "pick_no", "from_owner", "keeper", "usable"),
            pretty=False,
        )
    else:
        st.caption("No pick list — needs a league with a draft order set.")

    st.divider()

    # --- the board ----------------------------------------------------------
    st.markdown("#### The board")
    st.warning(
        "**`par` is a function of price, not a rating of the player.** Expected "
        "points come from a curve mapping *positional ADP rank* to what players "
        "at that rank have historically scored, so within a position this board "
        "reproduces the market's order exactly. Two receivers eleven picks apart "
        "have different PAR for one reason: eleven picks. It answers *what is "
        "this draft slot worth*, which is a real question, and it is silent on "
        "*how good is this player*.\n\n"
        "**`value_gap` is the second opinion, and it is not derived from ADP.** "
        "Quality percentile within position, minus price percentile within "
        "position. Positive means this project rates him above his cost. "
        "Quarterbacks are blank because the metrics that carry it — yards per "
        "route run, separation, yards after contact — have no QB analogue, so "
        "a blank means *not measured*, never *bad*.",
        icon="⚠️",
    )
    st.info(
        "**`adj_adp` is where he actually goes once the keepers are off the "
        "board.** Public ADP is priced in leagues where nobody is kept, and "
        "every keeper here sits inside the top 150 — so each one is a player "
        "the market expects to be drafted who will not be, and everyone behind "
        "him moves up. Any mock that ignores keepers shows players roughly "
        "fifteen picks late.\n\n"
        "**Compare `exp_pick` against your own picks, not `adj_adp`.** A keeper "
        "does not only leave the pool, he also spends a pick: this draft has "
        "150 picks but fewer selections. `adj_adp` counts selections, "
        "`exp_pick` counts picks, and using the first against a pick number "
        "would count the keeper adjustment twice.",
        icon="🎯",
    )

    board_cols = [
        c for c in (
            "board_rank", "name", "position", "team", "adp", "adj_adp",
            "exp_pick", "par", "tier", "quality_pct", "value_gap", "env_swing",
        ) if c in players.columns
    ]
    positions = st.multiselect(
        "Positions",
        list(FANTASY_POSITIONS),
        default=list(FANTASY_POSITIONS),
        key="draft_board_positions",
    )
    view = players.filter(pl.col("position").is_in(positions or list(FANTASY_POSITIONS)))
    table(view.select(board_cols).head(60))
    chart_note(
        ["par", "tier", "adp"],
        "<strong style='color:#c3c2b7'>adj_adp</strong> — ADP with kept players "
        "removed from the pool.<br>"
        "<strong style='color:#c3c2b7'>exp_pick</strong> — the actual pick "
        "number that selection lands on.",
    )

    st.divider()

    # --- availability at a chosen pick --------------------------------------
    st.markdown("#### Who is likely to be there when you are on the clock")
    st.caption(
        "Best PAR among players who plausibly last, using the dispersion FFC "
        "publishes alongside ADP. Two players at the same price with different "
        "dispersion are completely different planning problems — the one with a "
        "tight distribution is the one you cannot wait on."
    )

    options = (
        _my_picks().filter(pl.col("usable")).get_column("pick_no").to_list()
        if picks.height
        else []
    )
    left, right = st.columns([1, 1])
    with left:
        pick_no = (
            st.selectbox("Your pick", options, key="draft_pick_no")
            if options
            else st.number_input("Pick number", 1, 200, 24, key="draft_pick_manual")
        )
    with right:
        floor = st.slider(
            "Minimum chance he lasts", 0.05, 0.95, 0.35, 0.05, key="draft_floor"
        )

    got = bd.targets(players, int(pick_no), min_available=float(floor), top=15)
    if got.height:
        table(got, pretty=False)
    else:
        st.caption(
            "Nobody clears that floor at this pick — either everyone worth "
            "taking is gone by then, or the dispersion column did not survive "
            "the board build."
        )

    st.divider()

    # --- context flags ------------------------------------------------------
    st.markdown("#### Where the offence outweighs the board")
    st.info(
        "Team skill-position points move **45.3 per point of implied team "
        "total**, measured over 128 team-seasons. These are the pairs where the "
        "board's own edge is smaller than the gap between the two offences — "
        "the picks where the ranking is not really the deciding input.\n\n"
        "**Read this as the size of the argument, not a correction to "
        "subtract.** The expected-points curve is rank-based and therefore "
        "team-blind, so this adds information it genuinely does not have. But "
        "some of the discount is already in the ADP level, and `env_swing` "
        "cannot know how much, so it is an upper bound.",
        icon="🏟️",
    )
    flags = bd.context_flags(players)
    if flags.height:
        table(flags, pretty=False)
    else:
        st.caption("No pairs where environment outweighs the board edge.")

    st.divider()

    # --- one position at a time --------------------------------------------
    st.markdown("#### Quality against price, one position at a time")
    st.caption(
        "The board above orders across positions, which is what PAR is for. "
        "This asks the other question — inside one position, who is rated above "
        "what he costs. Up is better per opportunity, right is more expensive, "
        "so the top-left is where this project disagrees with the market in "
        "your favour."
    )
    scored = players.filter(
        pl.col("quality_pct").is_not_null() & pl.col("market_pct").is_not_null()
    )
    if scored.height:
        which = st.radio(
            "Position",
            [p for p in ("WR", "RB", "TE")
             if scored.filter(pl.col("position") == p).height],
            horizontal=True,
            key="draft_quality_position",
        )
        sub = scored.filter(pl.col("position") == which)
        pdf = sub.to_pandas()
        base = alt.Chart(pdf)
        pts = base.mark_circle(size=150, opacity=0.85).encode(
            x=alt.X("market_pct:Q", title="Draft price percentile (within position)"),
            y=alt.Y("quality_pct:Q", title="Quality percentile (within position)"),
            color=alt.Color(
                "value_gap:Q",
                scale=alt.Scale(scheme="redyellowblue"),
                legend=alt.Legend(title="value gap"),
            ),
            tooltip=["name", "team", "adp", "par", "tier", "quality_pct",
                     "market_pct", "value_gap"],
        )
        # The diagonal is agreement: quality percentile equal to price
        # percentile. Distance from it is the whole message of the chart.
        fair = (
            alt.Chart(pl.DataFrame({"x": [0, 100], "y": [0, 100]}).to_pandas())
            .mark_line(strokeDash=[4, 4], color="#898781")
            .encode(x="x:Q", y="y:Q")
        )
        labels = base.mark_text(align="left", dx=8, fontSize=10).encode(
            x="market_pct:Q", y="quality_pct:Q", text="name:N",
            opacity=alt.condition(
                "abs(datum.value_gap) > 25", alt.value(0.9), alt.value(0.0)
            ),
        )
        st.altair_chart(
            (fair + pts + labels).properties(height=520).interactive(),
            use_container_width=True,
        )
        chart_note(
            ["quality_pct", "market_pct", "value_gap"],
            "The dashed line is agreement. Only players more than 25 percentile "
            "points from it are labelled.",
        )
        data_expander(
            sub.select("name", "team", "adp", "par", "tier", "quality_pct",
                       "market_pct", "value_gap", "path_score")
            .sort("value_gap", descending=True, nulls_last=True),
            "Show the numbers",
        )
    else:
        st.caption(
            "No quality scores available — needs the feature table built "
            "(`uv run python -m src.bootstrap --light`)."
        )

    if data["unmatched"].height:
        st.warning(
            "These keepers did not match a row on the ADP board. Most are "
            "benign — a keeper too deep to be in the top 200 never appears — "
            "but a match failure would leave a kept player on your board."
        )
        table(data["unmatched"], pretty=False)

    st.divider()

    # --- tie-breaks: the promotion screen and the claims ledger -------------
    st.markdown("#### Breaking a tie inside a tier")
    st.caption(
        "The board cannot separate players inside a tier — that is what a tier "
        "means. These two do, on information the market has not priced yet: a "
        "role that is visibly growing, and claims about it that resolve."
    )
    _screen_sections(dark)


def _placeholder(name: str, phase: str) -> None:
    st.subheader(name)
    st.info(f"Not built yet — {phase}.", icon="🚧")


def main() -> None:
    st.title("ff-edge")
    p = _sidebar()

    # Draft Day leads because it is the only tab with a deadline on it. The
    # Board tab is `valuation.py` — quality against price — and is deliberately
    # not the draft board, which lives under Draft Day. See DASHBOARD_SPEC.md.
    draft_day, landscape, players, strategy, board, reference = st.tabs(
        ["Draft Day", "Landscape", "Players", "Strategy", "Board", "Glossary"]
    )
    with draft_day:
        _tab_draft_day(p)
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
