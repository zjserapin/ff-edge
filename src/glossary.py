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
    "ppg": _t("Points per game", "Fantasy points per game played, in your league's scoring.",
              """Divides by games he actually appeared in, not by the length of the
              season — so it measures how good he was when available, not how
              available he was. `games` carries the second half of that.""", "Value"),
    "fantasy_points": _t("Fantasy points", "Season total in your league's scoring.",
                         "Weeks 1-14 only, the fantasy regular season.", "Value"),
    "games": _t("Games", "Games he appeared in.", "", "Value"),
    "season": _t("Season", "NFL season.", "", "Context"),
    "position": _t("Position", "QB, RB, WR or TE.", "", "Context"),
    "team": _t("Team", "NFL team.", "", "Context"),
    "player_display_name": _t("Player", "Player name.", "", "Context"),
    "player_name": _t("Player", "Player name.", "", "Context"),
    "name": _t("Player", "Player name.", "", "Context"),
    "n": _t("Players", "How many players are in this group.", "", "Context"),
    "pool_size": _t("Pool size", "How many players the share is measured against.",
                    """Capped at roughly three times starter demand. Uncapped, a
                    concentration metric measures how many replacement bodies the
                    league cycled through rather than how top-heavy it is.""", "Value"),
    "top_n": _t("Top N", "How many top players the share covers.", "", "Value"),
    "cutoff": _t("Cutoff", "How deep into the combined value ranking this row counts.",
                 "", "Value"),
    # --- Quality (per opportunity) -----------------------------------------
    "routes": _t("Routes run", "Dropbacks he was on the field for.",
                 """Counted from play participation on plays with pass rushers, which
                 correctly includes sacks and scrambles — routes are run on those too.
                 Yields ~21,000 league dropbacks in 2025, matching reality; the obvious
                 alternative fields are populated on every play and double it.""",
                 "Quality"),
    "route_share": _t("Route share", "Share of his team's dropbacks he ran a route on.",
                      "A better availability measure than snap share for pass catchers.",
                      "Quality"),
    "yprr": _t(
        "Yards per route run", "Receiving yards divided by routes run.",
        """The single most useful metric added to this project, and the cleanest
        public separator of "good but blocked by a teammate" from "not good".
        Target share cannot make that distinction: a receiver behind an alpha and
        a receiver who loses his matchups both post low shares. Elite is 2.0+,
        good 1.7-2.0, average around 1.3. Needs ~100 routes to mean anything —
        on 40 routes one long catch moves it half a yard.""", "Quality"),
    "tprr": _t("Targets per route run", "Targets divided by routes run.",
               """How often the offense looks for him when he is out there, stripped of
               how often he is out there. Closer to a measure of trust than of talent.""",
               "Quality"),
    "drop_rate": _t("Drop rate", "Share of catchable targets dropped. Lower is better.",
                    "Charted by Pro Football Reference.", "Quality"),
    "rec_broken_tackles": _t("Broken tackles (rec)", "Tackles broken after a catch.",
                             "PFR charting. Yards created rather than yards schemed.", "Quality"),
    "receiving_rat": _t("Passer rating when targeted", "Quarterback rating on throws to him.",
                        """Blends catch rate, yards, and touchdowns against interceptions
                        on his targets. A blunt but genuinely independent quality read.""",
                        "Quality"),
    "rush_broken_tackles_per_att": _t(
        "Broken tackles per carry", "Tackles broken divided by carries.",
        "Volume-independent, so a backup is comparable to a workhorse.", "Quality"),
    "yards_after_contact_per_att": _t(
        "Yards after contact per carry", "Rushing yards gained after first contact.",
        """The running back quality metric that is least contaminated by the
        offensive line. A back on a bad line has poor yards per carry and can
        still be excellent after contact — that gap is the whole point.""",
        "Quality"),
    "ypa": _t("Yards per attempt", "Passing yards per pass attempt.", "", "Quality"),
    "pts_over_exp_per_att": _t(
        "Points over expected per attempt", "Fantasy points above what his opportunities were worth.",
        """Actual minus expected, divided by pass attempts. Expected points are
        rebuilt under your league's scoring rather than taken from the source,
        which hardcodes full PPR. Positive means he converted his chances better
        than an average passer would have — and it repeats at only 0.22 year to
        year, so read a big number as a season he had, not a skill he has.""",
        "Quality"),
    "ryoe_per_att": _t(
        "Rush yards over expected per carry", "Yards gained above what the blocking and box predicted.",
        """Tracking data models what an average back gains given the blocking,
        the box count and where every defender was, then charges the runner only
        with the difference — which is what yards per carry fails to do. Added
        here expecting it to be the running back's yards per route run. It
        persists year to year at 0.20, barely above the noise floor and well
        under the receiving metrics at the same position, so read it as the best
        available rushing measure rather than a strong one.""", "Quality"),
    "rush_efficiency": _t(
        "Rushing directness", "How straight a line he runs. Higher is more direct.",
        """Next Gen Stats measures distance travelled per yard gained; the sign
        is flipped here so that, like every other quality column in this project,
        higher is better.""", "Quality"),
    "catch_rate_on_catchable": _t(
        "Catch rate on catchable balls", "Catches divided by throws that were actually catchable.",
        """Catch rate charges a receiver for every ball thrown nowhere near him,
        so it is partly a measurement of his quarterback. Splitting on whether
        the throw was catchable puts the passer's contribution in
        `catchable_rate` and leaves the receiver's hands here. Hand-charted by
        FTN, 2022 onward, and it only survives the persistence check at wide
        receiver.""", "Quality"),
    "quality_score": _t(
        "Quality score", "Weighted standardized per-opportunity quality, within position.",
        """Relative, not absolute: +1 means one standard deviation better per
        opportunity than other players at his position that season. Built only
        from rate metrics — nothing in it scales with how much he plays. Each
        metric is weighted by how well it repeats year to year rather than
        counted equally, because a flat average gives a column that is half noise
        the same vote as one that is not.""", "Quality"),
    "r_yoy": _t(
        "Year-over-year correlation", "How well a metric repeats for the same player next season.",
        """Percentile rank within position and season, correlated against that
        player's own next season. Below about 0.20 a metric is describing the
        season rather than the player. No outcome data enters, so selecting
        features on this is not the same as selecting them on what predicts
        fantasy points.""", "Backtest"),
    "n_pairs": _t("Player-season pairs", "How many consecutive-season pairs the correlation used.", "",
                  "Backtest"),
    "spearman": _t("Rank correlation", "How well predicted order matched actual order. 1.0 is perfect.",
                   """Uses every pair of players rather than only those spanning a
                   threshold, which is why it can measure on 540 rows what AUC
                   cannot.""", "Backtest"),
    "spearman_adp_only": _t("Rank correlation, price alone", "The same, for a model that sees only ADP.",
                            "The number the model has to beat to be worth anything.", "Backtest"),
    "finish_pct": _t("Finish percentile", "Where he finished among drafted players at his position. 1.0 is best.",
                     """Continuous rather than a beat/miss flag, which is what
                     makes the difference against ADP measurable to within
                     ±0.01. A season missed entirely ranks last, because missing
                     it is the outcome.""", "Backtest"),
    "pred": _t("Projected finish percentile", "The model's out-of-sample projection.",
               "From a model that never saw this player's season.", "Backtest"),
    "pred_adp_only": _t("Projected finish, price alone", "The same projection from ADP alone.", "", "Backtest"),
    # --- Opportunity and situation -----------------------------------------
    "opportunity_score": _t(
        "Opportunity score", "Weighted standardized volume, within position.",
        """The axis ADP prices well, and the one that carries forward: volume
        repeats year to year at 0.47-0.55 against 0.28-0.44 for quality. Kept on
        a separate axis from quality on purpose — mixing them is what made the
        old clustering rediscover the market.""", "Opportunity"),
    "teammate_top_share": _t(
        "Best teammate's target share", "The largest target share on his team other than his own.",
        """What makes a good player cheap, and what makes him a buy when that
        teammate leaves. A 15% share behind a 30% alpha is a different situation
        from 15% on a team where nobody clears 18%: the first is capped by
        someone else, the second is losing his own matchups. For a team's alpha
        this is the second-largest share, not null.""", "Opportunity"),
    "teammate_share": _t("Teammate target share", "Combined target share of everyone else on his offense.", "",
                         "Opportunity"),
    "team_target_hhi": _t("Target concentration", "How concentrated his team's targets are.",
                          "Sum of squared target shares. High means one player dominates.",
                          "Opportunity"),
    "is_team_alpha": _t("Team alpha", "Whether he led his team in target share.", "", "Opportunity"),
    "vacated_target_share_next": _t(
        "Vacated target share", "Share of his team's targets held by players who left.",
        """Opportunity about to be handed out. A player whose team just lost a
        quarter of its targets is in a materially different spot from an
        identical player whose depth chart is unchanged, and neither his usage
        nor his efficiency says so.""", "Opportunity"),
    "vacated_carry_share_next": _t("Vacated carry share", "Share of his team's carries held by players who left.",
                                   "The running back version of vacated targets.", "Opportunity"),
    # --- Where the work happened -------------------------------------------
    "exp_td_share": _t(
        "Touchdown equity share", "Share of his offense's expected touchdowns, running and passing.",
        """Every play carries a modelled probability of ending in a touchdown,
        given where it was snapped and how far the ball travelled. Sum his,
        divide by his team's, and you get the fraction of the offence's scoring
        chances that run through him. This is the largest thing target share
        cannot see: two receivers on 25% of targets are not the same asset when
        one of them is the goal-line look. Measured over the weeks he played, so
        it is not a reward for staying healthy.""", "Opportunity"),
    "exp_td_per_touch": _t(
        "Touchdown equity per touch", "Expected touchdowns per target or carry.",
        """Where his touches happen rather than how many he gets. High means he
        is used near the goal line; it describes the role, not the player.""",
        "Opportunity"),
    "rz_target_share": _t(
        "Red-zone target share", "His share of team targets from inside the 20.",
        "Measured over the weeks he played, against his own team's total.", "Opportunity"),
    "ez_target_share": _t(
        "End-zone target share", "His share of team targets thrown to or past the goal line.",
        """A throw counts when its air yards reach the end zone, which does not
        require the offence to already be inside the twenty — so this catches the
        forty-yard post that is also a scoring chance, not just the fade.""",
        "Opportunity"),
    "neutral_target_share": _t(
        "Neutral-script target share", "His target share with the game still live.",
        """Restricted to plays where Vegas win probability sat between 0.2 and
        0.8. Targets accumulated down three scores count the same in the raw
        share and are worth far less going forward, because next season's team
        may not trail by three scores. Win probability rather than score, because
        it already accounts for time: down seven in the first quarter is a normal
        game, down seven with two minutes left is not.""", "Opportunity"),
    "rz_carry_share": _t("Red-zone carry share", "His share of team carries from inside the 20.", "",
                         "Opportunity"),
    "gz_carry_share": _t(
        "Goal-line carry share", "His share of team carries from inside the 5.",
        """The single largest swing in a running back's scoring. A back with 18%
        of the carries and 60% of the goal-line work is a different asset from
        one with the same 18% and none of it, and `rush_share` reports them
        identically.""", "Opportunity"),
    "neutral_rush_share": _t("Neutral-script carry share", "His carry share with the game still live.",
                             "The rushing version of neutral target share.", "Opportunity"),
    "catchable_rate": _t(
        "Catchable target rate", "Share of throws at him that were catchable.",
        """A property of his quarterback, not of him. Catch rate charges a
        receiver for every ball thrown nowhere near him; this is the half of that
        which belongs to the passer. Hand-charted by FTN, 2022 onward.""",
        "Opportunity"),
    "screen_target_rate": _t(
        "Screen target rate", "Share of his targets that were screens.",
        """A discount, not a virtue. Screen yards are manufactured by the
        play-caller rather than earned against coverage, so efficiency resting on
        a heavy screen diet is a worse bet to repeat under a new coordinator.""",
        "Opportunity"),
    "contested_rate": _t("Contested target rate", "Share of his targets that were contested.",
                         "How often he is asked to win a ball in traffic. FTN charting, 2022 onward.",
                         "Opportunity"),
    "stacked_box_rate": _t("Stacked box rate", "Share of his carries against eight or more defenders.",
                           "Context for rushing efficiency: a back facing loaded boxes has a harder job.",
                           "Opportunity"),
    "time_to_los": _t("Time to line of scrimmage", "Average seconds from handoff to crossing the line.",
                      "Lower means a one-cut runner; higher means he dances behind the line.",
                      "Opportunity"),
    # --- Valuation ----------------------------------------------------------
    "quality_pct": _t("Quality percentile", "Where his per-opportunity quality ranks in his position. 100 is best.",
                      "Percentile rather than raw score, so it is comparable to a price percentile.",
                      "Valuation"),
    "opportunity_pct": _t("Opportunity percentile", "Where his volume ranks in his position. 100 is most.", "",
                          "Valuation"),
    "market_pct": _t("Price percentile", "Where his draft price ranks in his position. 100 is most expensive.",
                     "Derived from ADP positional rank, inverted so high always means expensive.",
                     "Valuation"),
    "value_gap": _t(
        "Value gap", "Quality percentile minus price percentile. Positive means cheap for the quality.",
        """The disagreement score this whole project builds toward. Positive means
        the market ranks him below where his per-opportunity quality does. It is
        not a projection and not a ranking — it says where this project's read
        differs from the market's, which is a reason to look closer rather than a
        reason to be right. Twenty points is roughly two tiers.""", "Valuation"),
    "path_score": _t(
        "Room to grow", "Whether he has anywhere to gain volume. Higher is more room.",
        """Quality alone is not a buy signal — a good player permanently stuck
        behind a better one stays stuck. This blends unused opportunity, volume
        his team just lost, and how big the teammate in front of him is. It is
        what separates a genuine buy from a good player who is already maxed out
        and priced accordingly.""", "Valuation"),
    "verdict": _t("Verdict", "Undervalued, fairly priced, or overvalued.",
                  "From the value gap, with a 20-point threshold in each direction.",
                  "Valuation"),
    "prior_ppg": _t("Last season's points per game", "What he actually scored last year.", "", "Valuation"),
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
                    "Excluded from the quality score — a standardized boolean spikes the mean.",
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
    # --- Comparables -------------------------------------------------------
    "distance": _t("Profile distance", "How different two players' per-opportunity profiles are.",
                   """Euclidean distance in the standardized quality space. Zero
                   would be identical. This replaced the k-means archetypes, which
                   topped out at a silhouette of 0.29 — a partition of a continuum
                   rather than real groups — and added nothing downstream.""",
                   "Comparables"),
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
    # --- Promotion screen ---------------------------------------------------
    "screen_pct": _t(
        "Screen score", "Mean percentile on his position's promotion criteria.",
        """Trust markers at RB (snap share, red-zone and goal-line carry share,
        TD equity), efficiency at WR/TE (yards per route run, plus snap share
        and TD equity). Percentiles are within position, this season, among
        players with 4+ games. It is a grade of the profile, not a prediction
        that the role actually grows — the user supplies that.""",
        "Promotion"),
    "quality_tier": _t(
        "Quality tier", "Bottom 30% / middle / top 30% of prior-season quality.",
        """Quality is a filter, not a picker: bottom-tercile promoted players
        hit a top-quartile finish ~4% of the time, top-tercile ~18%. Much
        better at ruling players out than ranking them in.""",
        "Promotion"),
    "tier_hit_rate": _t(
        "Tier hit rate", "How often promoted players in this quality tier hit top-quartile PPG.",
        """A base rate over the historical promotion cohort, not a projection
        for this player. Always read with `tier_n` and the interval beside it.""",
        "Promotion"),
    "tier_n": _t("Tier n", "Promoted players in the tier the base rate is computed from.", "", "Promotion"),
    "tier_ci_lo": _t("Tier CI low", "Lower bound of the Wilson interval on the tier hit rate.", "", "Promotion"),
    "tier_ci_hi": _t("Tier CI high", "Upper bound of the Wilson interval on the tier hit rate.", "", "Promotion"),
    "next_ppg_pct": _t(
        "Next-season PPG percentile", "Points-per-game percentile within position, the following season.",
        "The promotion cohort's outcome measure; a hit is the top quartile of it.",
        "Promotion"),
    "opp_change": _t(
        "Role growth", "Percentile points of opportunity gained season over season.",
        "The promotion cohort keeps players who grew by 10+ points from a below-median role.",
        "Promotion"),
    "hit": _t("Hit", "Finished in the top quartile of position PPG the next season.", "", "Promotion"),
    "carry_share_wk": _t(
        "Weekly carry share", "His share of the team's carries, that week only.",
        "Week-level shares are noisy by construction — read the trend, not a point.",
        "Promotion"),
    "rz_carry_share_wk": _t(
        "Weekly red-zone carry share", "His share of the team's red-zone carries, that week only.",
        """The trust marker that predicts promoted RBs, at week grain so a
        December role change is not averaged away. Null in weeks the team had
        no red-zone carries.""",
        "Promotion"),
    "target_share_wk": _t(
        "Weekly target share", "His share of the team's targets, that week only.", "",
        "Promotion"),
    # --- Claims ledger ------------------------------------------------------
    "role_score": _t(
        "Role score", "Cumulative signed claim score: tier × specificity × novelty × recency.",
        """Positive means the ledger's claims point at a growing role. The
        weights are hand-set priors, not fitted parameters — season one exists
        to collect the data that could justify better ones. Every score
        decomposes into its quoted claims in the table below it.""",
        "Ledger"),
    "grade": _t(
        "Ledger grade", "A/B/C/watch for role growth, from the cumulative claim score.",
        """A additionally requires at least one concrete tier-1/2 claim — hype
        volume alone cannot reach A no matter how loud. Not a projection: the
        promotion screen grades the player; this grades the evidence that his
        role is changing.""",
        "Ledger"),
    "claim_score": _t(
        "Claim score", "One claim's signed weight in the role score.",
        "The four factors are shown beside it so a score is never a mystery.",
        "Ledger"),
    "claim_type": _t(
        "Claim type", "What kind of role-change evidence this is.",
        """depth_chart, first_team_reps, coach_usage, injury_teammate,
        departure, or role_change_observed. Performance takes are excluded at
        extraction — they are the high-volume, low-value tier.""",
        "Ledger"),
    "specificity": _t(
        "Specificity", "Concrete (falsifiable) vs vibes (coachspeak).",
        "'We want to get him more involved' is vibes and weighs 0.4.",
        "Ledger"),
    "source_tier": _t(
        "Source tier", "1 beat/structured, 2 national, 3 aggregator.",
        """Hand-assigned to start, coarse on purpose: a season yields a handful
        of resolvable claims per source, so three tiers is all the sample can
        carry. Source grades earn changes over seasons.""",
        "Ledger"),
    "novel": _t(
        "Novel", "First claim of its kind inside the novelty window.",
        """Dedupe is by claim, not mention: forty aggregators quoting one beat
        report count once, and echoes weigh 0.15. This is the mechanical fix
        for stars generating high-volume, low-value context.""",
        "Ledger"),
    "n_novel": _t("Novel claims", "How many of the player's claims were first reports.", "", "Ledger"),
    "best_tier": _t("Best tier", "The most trusted source among the player's claims.", "", "Ledger"),
    "adp_at_latest": _t(
        "ADP at claim", "The player's ADP the day the latest claim landed.",
        """The already-priced check: a claim's value is only the part not yet
        in price, and ADP moves within days on big news. Null means the market
        has not priced him at all — a real claim there is pure edge.""",
        "Ledger"),
    "resolved_hit": _t(
        "Resolved", "Did the claimed usage change materialize within three weeks?",
        "Null is pending, not failure — off-season claims wait for games.",
        "Ledger"),
    "n_resolved": _t("Resolved claims", "Claims old enough to check against actual usage.", "", "Ledger"),
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
