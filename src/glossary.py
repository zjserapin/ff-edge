"""What every column means, in one place.

This project computes a lot of things that look like they might be standard and
are not — `market_var` is not a projection, `air_yards_share` can be negative,
`p_breakout` is not comparable across positions. A number whose definition lives
only in the head of whoever wrote it is a number nobody else can act on, and in
six months that includes you.

Each entry has three fields for three different moments:

    label   what to call it in a header
    short   one line, shown as a tooltip on hover — the "what is this"
    long    the full definition, including how it is computed and what it does
            *not* mean, shown in the reference section

`describe()` and `column_help()` are the two entry points; the app uses the
latter to attach tooltips to every table it renders.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Term(NamedTuple):
    label: str
    short: str
    long: str
    group: str


def _t(label: str, short: str, long: str, group: str) -> Term:
    return Term(label, short, " ".join(long.split()), group)


TERMS: dict[str, Term] = {
    # --- Usage and role ----------------------------------------------------
    "snap_pct": _t(
        "Snap share",
        "Share of his team's offensive snaps he was on the field for.",
        """The closest thing to a measure of whether a player has a job. Averaged
        over his regular-season games. Snap share is stickier year to year than
        any efficiency stat, which is why it carries weight in the models.""",
        "Usage",
    ),
    "target_share": _t(
        "Target share",
        "Share of his team's targets, summed over the weeks he played.",
        """Player targets divided by team targets over the same weeks — a true
        season share, not an average of weekly ratios, so a 12-target game counts
        more than a 2-target game. The single most durable predictor of receiving
        production.""",
        "Usage",
    ),
    "air_yards_share": _t(
        "Air-yards share",
        "Share of his team's passing yardage thrown his way. Can be negative.",
        """Air yards are measured from the line of scrimmage, so a screen thrown
        two yards behind it contributes -2. A back targeted only on checkdowns
        therefore finishes with negative air yards and a negative share — 347 of
        2,610 player-seasons here are negative, and every one is a running back
        with a negative aDOT. That is a real distinction from a player who is
        never targeted at all, so it is not clipped at zero.""",
        "Usage",
    ),
    "rush_share": _t(
        "Rush share",
        "Share of his team's carries over the weeks he played.",
        """Carries divided by team carries. For quarterbacks this is the feature
        that separates a fantasy asset from a merely good passer.""",
        "Usage",
    ),
    "tgt_per_game": _t(
        "Targets per game",
        "Average targets in games he appeared in.",
        """A volume rate rather than a share — useful when a team's total passing
        volume is itself unusual.""",
        "Usage",
    ),
    "carry_per_game": _t(
        "Carries per game",
        "Average carries in games he appeared in.",
        "As with targets per game, volume without the team-context denominator.",
        "Usage",
    ),
    "adot": _t(
        "aDOT",
        "Average depth of target — air yards per target.",
        """How far downfield he is used. Negative for checkdown backs. A high aDOT
        means a boom-bust profile; a low one means a stable floor and a lower
        ceiling.""",
        "Usage",
    ),
    "wopr": _t(
        "WOPR",
        "Weighted opportunity: 1.5x target share + 0.7x air-yards share.",
        """The one composite the fantasy literature broadly agrees on. The weights
        are convention, not fitted here. It combines how often a player is
        targeted with how valuable those targets are, which is why the receiver
        model uses it rather than the two components separately — on samples this
        small, two correlated columns split into a large positive and a large
        negative coefficient and key on the noise between them.""",
        "Usage",
    ),
    # --- Efficiency --------------------------------------------------------
    "catch_rate": _t(
        "Catch rate",
        "Receptions divided by targets.",
        """Efficiency, which reverts far harder year to year than volume does.
        Present as a feature but not expected to carry much.""",
        "Efficiency",
    ),
    "ypt": _t("Yards per target", "Receiving yards divided by targets.",
              "Blends catch rate and depth. Efficiency, so treat it as noisy.", "Efficiency"),
    "ypc": _t("Yards per carry", "Rushing yards divided by carries.",
              """Famously unstable — a back's yards per carry one season barely
              predicts the next. Included for description, not signal.""", "Efficiency"),
    "yac_per_rec": _t("YAC per reception", "Yards gained after the catch, per reception.",
                      "Separates a player creating on his own from one being schemed open.",
                      "Efficiency"),
    # --- Expected points ---------------------------------------------------
    "exp_ppg": _t(
        "Expected points per game",
        "Points his opportunity was worth, under your league's scoring.",
        """Rebuilt from ff_opportunity's component expectations rather than taken
        from its `total_fantasy_points_exp` column, which is hardcoded to full
        PPR and is therefore wrong by half a point per expected reception in a
        half-PPR league — roughly 1.5 points a game for a target hog, which is
        exactly the population you would use expected points to evaluate.""",
        "Expected points",
    ),
    "pts_over_exp_per_game": _t(
        "Points over expected",
        "Actual minus expected points, per game. Positive means he outran his usage.",
        """Large negative values are usually the better buy: the volume was there
        and the finishing wasn't, and volume is much stickier than finishing luck.
        Large positive values are production you already paid for and probably
        won't get again at the same rate. It cannot tell you which players
        underperformed because they are simply not good.""",
        "Expected points",
    ),
    "exp_pts_share": _t(
        "Share of team's expected points",
        "His expected points divided by his whole offense's.",
        """The best single answer to "is this his offense?" available in this
        data. Can be very slightly negative for a backup quarterback, since an
        interception is worth -2 and a few attempts with no production nets
        below zero.""",
        "Expected points",
    ),
    # --- Next Gen Stats ----------------------------------------------------
    "avg_separation": _t("Separation", "Yards of separation from the nearest defender at catch point.",
                         """NGS tracking data, qualified receivers only — about 120 players a
                         season, so it is null for most of the pool by design, not by error.""",
                         "Next Gen Stats"),
    "avg_cushion": _t("Cushion", "Yards the defender lines up off him before the snap.",
                      "How much respect the defense shows pre-snap. NGS, qualified receivers only.",
                      "Next Gen Stats"),
    "avg_yac_above_expectation": _t(
        "YAC over expected", "Yards after catch versus what the tracking model expected.",
        "Isolates a player creating yards from a player being handed them by scheme.",
        "Next Gen Stats"),
    # --- Context -----------------------------------------------------------
    "age": _t("Age", "Age in years as of September 1 of that season.",
              "Computed from birth date, not from experience.", "Context"),
    "seasons_exp": _t("Seasons of experience", "Seasons since his rookie year, inclusive.",
                      """Excluded from every model: it correlates 0.964 with age, and
                      fitting both makes them split into a large positive and negative
                      pair keyed on the noise between two measures of the same thing.""",
                      "Context"),
    "draft_round": _t("Draft round", "NFL draft round. 8 means undrafted.",
                      """Undrafted is encoded as round 8 / pick 262 rather than left null,
                      because going undrafted is a strong signal and a null would be
                      imputed to a mid-round median.""", "Context"),
    "draft_pick": _t("Draft pick", "Overall pick number. 262 means undrafted.", "See draft round.",
                     "Context"),
    "undrafted": _t("Undrafted", "Whether he entered the league as an undrafted free agent.",
                    "Excluded from clustering — a standardized boolean spikes the distance metric.",
                    "Context"),
    # --- Value and replacement ---------------------------------------------
    "pos_rank": _t("Positional finish", "Where he finished among his position that season.",
                   "Ranked on total points over weeks 1-14, the fantasy regular season.",
                   "Value"),
    "par": _t("Points above replacement", "Season points minus the replacement player's.",
              "The only unit that compares a quarterback to a tight end.", "Value"),
    "par_ppg": _t("PAR per game", "Points per game above replacement.",
                  """Value over replacement on a per-game basis, so seasons of different
                  length compare fairly.""", "Value"),
    "par_mean_starter": _t(
        "Average starter PAR", "Average per-game edge a starter at this position gives you.",
        """Averaged over the players who actually start league-wide at that
        position. The headline number for "should I spend early picks here".""",
        "Value"),
    "par_total": _t("Total PAR", "Positive PAR summed across the starting pool.",
                    "Captures a steep top even when the average is unremarkable.", "Value"),
    "demand": _t("Starters league-wide", "How many of this position start in any given week.",
                 """Dedicated roster slots times teams, plus this position's computed
                 share of the FLEX slots. In a 10-team league with two FLEX, RB/WR/TE
                 sum to 70.""", "Value"),
    "replacement_rank": _t("Replacement rank", "The first player past the starting pool.",
                           "If 28 backs start, replacement is RB29 — who you could have had free.",
                           "Value"),
    "replacement_ppg": _t(
        "Replacement points per game", "What a freely available starter at this position scores.",
        """Ranked on per-game among players with 8+ games, not by reading the
        per-game figure off whoever placed at the replacement rank on season
        totals. That mistake let an injured Tucker Kraft — TE11 on total points in
        8 games at 12.65 ppg — set the 2025 tight end baseline 60% too high.""",
        "Value"),
    "overall_par_rank": _t("Overall value rank", "Rank among all positions on PAR per game.",
                           "The common axis that makes the early-RB question countable.",
                           "Value"),
    "share": _t("Concentration share", "Share of positional points held by the top N.",
                """Denominator is a capped pool of roughly three times starter demand.
                Uncapped, this measures how many replacement bodies the league cycled
                through rather than how top-heavy it is.""", "Value"),
    # --- Draft market ------------------------------------------------------
    "adp": _t("ADP", "Average draft position across public drafts.",
              "From Fantasy Football Calculator, at your league's format and size.",
              "Draft market"),
    "stdev": _t("ADP dispersion", "Standard deviation of his draft slot, in picks.",
                """The underused column. Two players at ADP 40 with dispersion 3 and 14
                are 0.4% and 28% likely to last to pick 48 — identical price,
                completely different decision. Floored at 0.5.""", "Draft market"),
    "adp_pos_rank": _t("ADP positional rank", "Where the market ranks him at his position.",
                       "The price the model has to beat.", "Draft market"),
    "adp_tier": _t("ADP tier", "His ADP positional rank in blocks of ten.",
                   "Display only — the breakout label no longer uses tiers. See beat_adp.",
                   "Draft market"),
    "market_ppg": _t(
        "Market-implied points per game", "What his draft slot has historically returned.",
        """NOT a projection. The median points per game of players who finished at
        his ADP positional rank, across the seasons in the window. It knows
        nothing about the player — two backs at RB14 get the same number. Its use
        is as a baseline to disagree with.""", "Draft market"),
    "market_var": _t(
        "Market value over replacement", "Market-implied points per game, minus replacement.",
        """The Board's value column. Same caveat as market_ppg: it describes the
        draft slot, not the player. This project has no points projection, because
        building one honestly is a larger job than everything else here and
        building one dishonestly is worse than having none.""", "Draft market"),
    # --- Backtest ----------------------------------------------------------
    "beat_adp": _t(
        "Beat ADP", "Finished at or inside 60% of his ADP positional rank.",
        """The ratio rule, chosen after measuring the tier rule and finding it
        degenerate: a tier-1 player would need to finish above rank zero, so its
        base rate is exactly 0.000 for a quarter of the sample and then climbs to
        0.451 by tier 6 — it measures how cheap you were, not how he played. The
        ratio rule is flat across tiers at 0.224.""", "Backtest"),
    "base_rate": _t("Base rate", "How often the label fires with no model at all.",
                    "The number every model result has to be read against. Here, 22.4%.",
                    "Backtest"),
    "p_breakout": _t(
        "Breakout probability", "Model probability he beats his ADP. Not comparable across positions.",
        """Each position has its own model, so the four probability scales are not
        on a common footing — a 0.30 at tight end and a 0.30 at receiver came from
        different fits. Quartiles are therefore computed within position. And read
        this next to the calibration table: out of sample the model does not beat
        draft price alone.""", "Backtest"),
    "p_adp_only": _t("Price-only probability", "The same model fit on draft price alone.",
                     """The null hypothesis. Without it an AUC is uninterpretable, because
                     ADP by itself already predicts beating ADP reasonably well.""",
                     "Backtest"),
    "auc": _t("AUC", "Chance the model ranks a random breakout above a random non-breakout.",
              "0.5 is a coin flip. Below 0.5 means it is ranking them backwards.",
              "Backtest"),
    "auc_adp_only": _t("Price-only AUC", "AUC of predicting from draft price alone.",
                       "The bar the model has to clear to be worth anything.", "Backtest"),
    "delta_auc": _t("AUC gain over price", "Model AUC minus price-only AUC.",
                    """The number that answers the actual question. If its interval covers
                    zero, prior-season usage adds nothing the market has not priced.""",
                    "Backtest"),
    "lift": _t("Lift", "A group's actual rate divided by the base rate.",
               "Above 1 means better than guessing. Below 1 means worse.", "Backtest"),
    "events_per_variable": _t(
        "Events per variable", "Positives in the smallest training fold, divided by feature count.",
        """The standard adequacy check for a logistic fit; the conventional floor
        is ten. No position here reaches it — WR is 5.5, TE is 1.5 — which is why
        every result is reported with its denominator rather than on its own.""",
        "Backtest"),
    "n_train": _t("Training rows", "How many player-seasons the model was fit on.",
                  "Read every coefficient and score against this.", "Backtest"),
    # --- Simulation --------------------------------------------------------
    "title_rate": _t("Title rate", "Share of simulated seasons this strategy won the league.",
                     "One in ten is the no-edge baseline in a 10-team league.", "Simulation"),
    "playoff_rate": _t("Playoff rate", "Share of simulated seasons it made the playoffs.",
                       "Six of ten make it, so 0.60 is the no-edge baseline.", "Simulation"),
    "mean_wins": _t("Mean wins", "Average regular-season wins out of 14.",
                    "Seven is the no-edge baseline.", "Simulation"),
    "edge": _t("Edge over ADP", "Title rate minus the ADP-following control's.",
               """Bootstrapped over shared seasons so both sides move together, which
               removes the season effect that otherwise dominates.""", "Simulation"),
    "bonferroni_lo": _t("Corrected CI low", "Lower bound, widened for multiple comparisons.",
                        """Seven strategies are compared, so intervals are Bonferroni-
                        corrected. Quote the uncorrected interval of the best of seven and
                        you will find a result whether or not one exists.""", "Simulation"),
    "bonferroni_hi": _t("Corrected CI high", "Upper bound, widened for multiple comparisons.",
                        "See corrected CI low.", "Simulation"),
    # --- Board -------------------------------------------------------------
    "p_available": _t(
        "Chance he lasts to your pick", "Probability he is still there when you next pick.",
        """From his ADP and its dispersion, treating his draft slot as normal.
        Computed on the pool remaining after your cuts and the players already
        gone, so it reflects the board you actually face. This reframes the draft
        from "who is best available" to "who do I lose by waiting".""", "Board"),
    # --- Clustering --------------------------------------------------------
    "cluster": _t("Usage group", "Which usage archetype he falls into, within his position.",
                  """From k-means on role features only — production and draft pedigree are
                  excluded so the groups are not scoring tiers in disguise. Clusters
                  describe; they do not predict.""", "Archetypes"),
    "silhouette": _t("Silhouette", "How well-separated the groups are. Higher is better.",
                     """Peaks at two groups for every position here and falls after, which
                     means NFL usage has one dominant axis rather than many archetypes.""",
                     "Archetypes"),
    "dist_to_center": _t("Distance to group center", "How typical he is of his group.",
                         "Large values are players who fit no archetype well.", "Archetypes"),
    "distance": _t("Usage distance", "How different two players' usage profiles are.",
                   """Euclidean distance in standardized role space. Zero would be
                   identical usage. This is the comparable-player number.""", "Archetypes"),
    # --- Rookies -----------------------------------------------------------
    "draft_ovr": _t("Overall draft pick", "Where the NFL drafted him.",
                    """The league's own aggregated scouting opinion, and by a wide margin
                    the strongest single predictor in the rookie model.""", "Rookies"),
    "age_at_draft": _t("Age at draft", "Age when drafted.",
                       "A 21-year-old declaring early is a different prospect than a 23-year-old senior.",
                       "Rookies"),
    "forty": _t("40-yard dash", "Combine 40 time in seconds.",
                "Partially observed — plenty of players skip drills, and not at random.",
                "Rookies"),
    "wt": _t("Weight", "Listed weight in pounds at the combine.", "", "Rookies"),
    "vacated_target_share": _t(
        "Vacated target share", "Share of last year's targets left by players no longer on the team.",
        """The landing-spot signal for receivers and tight ends — the closest thing
        to an opportunity projection available for someone who has never played.
        For past seasons this uses an end-of-season roster snapshot, so "who is
        still here" is approximate; for the current season it is exact.""",
        "Rookies"),
    "vacated_carry_share": _t(
        "Vacated carry share", "Share of last year's carries left by departed players.",
        "The landing-spot signal for running backs, and the feature the RB model keys on most after draft capital.",
        "Rookies"),
    "vacated_exp_points": _t("Vacated expected points", "Expected fantasy points left behind by departed players.",
                             "A position-agnostic measure of how much opportunity a team has to replace.",
                             "Rookies"),
    "predicted": _t("Predicted points per game", "Rookie model's first-year points-per-game estimate.",
                    """Out-of-sample correlation 0.59, about 24% less error than
                    predicting the average for everyone. Unlike the veteran breakout
                    probability, these are on one scale and comparable across positions.""",
                    "Rookies"),
}

# Columns whose meaning depends on a number in the name, e.g. p_available_at_20.
_PREFIXES: tuple[tuple[str, str], ...] = (("p_available_at_", "p_available"),)


def lookup(column: str) -> Term | None:
    """The Term for a column name, resolving numbered variants."""
    if column in TERMS:
        return TERMS[column]
    for prefix, key in _PREFIXES:
        if column.startswith(prefix):
            return TERMS.get(key)
    return None


def describe(column: str) -> str:
    """One-line description, or empty string if the column has no entry."""
    term = lookup(column)
    return term.short if term else ""


def column_help(columns: list[str]) -> dict[str, str]:
    """{column: tooltip} for the columns that have definitions."""
    out: dict[str, str] = {}
    for column in columns:
        term = lookup(column)
        if term:
            out[column] = term.short
    return out


def groups() -> dict[str, list[tuple[str, Term]]]:
    """Terms bucketed by group, for rendering a reference section."""
    out: dict[str, list[tuple[str, Term]]] = {}
    for key, term in TERMS.items():
        out.setdefault(term.group, []).append((key, term))
    for items in out.values():
        items.sort(key=lambda kv: kv[1].label)
    return out


def as_rows() -> list[dict[str, Any]]:
    """Flat rows, for rendering the whole glossary as a table."""
    return [
        {"group": t.group, "column": k, "term": t.label, "definition": t.short, "detail": t.long}
        for k, t in sorted(TERMS.items(), key=lambda kv: (kv[1].group, kv[1].label))
    ]
