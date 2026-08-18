"""Chart builders — authored in Python with Altair, rendered by vendored vega.

The browser is only a renderer: each function returns a Vega-Lite spec (as a
dict) that static/app.js hands to vega-embed. Colors come from src/theme.py,
the one validated palette in this repo — substituting a hue here without
re-running the palette validator is how a chart two leaguemates can't read
ships quietly.
"""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import polars as pl

from src import theme


def _spec(chart: alt.TopLevelMixin, dark: bool) -> dict[str, Any]:
    """Recessive chrome from the shared theme, transparent over the page CSS."""
    spec = json.loads(theme.base_chart(chart, dark).to_json())
    spec["background"] = "transparent"
    return spec


def dropoff(players: pl.DataFrame, dark: bool) -> dict[str, Any] | None:
    """The curves behind the `drop` column: `par_env` by positional rank.

    Small multiples rather than one shared plot, because each position's
    demand line sits at a different rank and a rule only means anything drawn
    on its own panel. One series per panel, so the panel title carries
    identity and no legend is needed.
    """
    needed = {"position", "pos_rank", "par_env"}
    if not players.height or not needed.issubset(players.columns):
        return None
    keep = [c for c in ("position", "pos_rank", "par_env", "name", "in_demand") if c in players.columns]
    view = players.select(keep).drop_nulls(subset=["pos_rank", "par_env"])
    if not view.height:
        return None

    colors = theme.position_colors(dark)
    panels: list[alt.LayerChart | alt.Chart] = []
    for position, color in colors.items():
        rows = view.filter(pl.col("position") == position).sort("pos_rank")
        if not rows.height:
            continue
        base = alt.Chart(alt.Data(values=rows.to_dicts()))
        line = base.mark_line(strokeWidth=2, color=color).encode(
            x=alt.X("pos_rank:Q", title="Positional rank"),
            y=alt.Y("par_env:Q", title="PAR + environment"),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("pos_rank:Q", title="Rank"),
                alt.Tooltip("par_env:Q", title="PAR+env", format=".1f"),
            ],
        )
        # Hover targets bigger than the mark: invisible points carry the tooltip.
        hover = base.mark_point(size=90, opacity=0, color=color).encode(
            x="pos_rank:Q",
            y="par_env:Q",
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("pos_rank:Q", title="Rank"),
                alt.Tooltip("par_env:Q", title="PAR+env", format=".1f"),
            ],
        )
        layers: list[alt.Chart] = [line, hover]
        if "in_demand" in rows.columns:
            inside = rows.filter(pl.col("in_demand"))
            if inside.height:
                # The roster-demand line, drawn where this position's demand
                # runs out — the same cut the board's blank `block` column marks.
                edge = int(inside.get_column("pos_rank").max())
                layers.append(
                    alt.Chart(alt.Data(values=[{"edge": edge}]))
                    .mark_rule(strokeDash=[4, 3], color=theme.ink(dark)["axis"])
                    .encode(x="edge:Q")
                )
        panels.append(
            alt.layer(*layers).properties(width=290, height=170, title=position)
        )

    if not panels:
        return None
    rows_of_two = [alt.hconcat(*panels[i : i + 2]) for i in range(0, len(panels), 2)]
    return _spec(alt.vconcat(*rows_of_two), dark)
