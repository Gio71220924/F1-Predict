"""Streamlit UI for the 2026 tyre, qualifying and race-outcome models."""
from pathlib import Path

import pandas as pd
import streamlit as st

from f1_predict import export, live, model, predict, quali, race, replay, strategy

LAPS_CSV = "data/processed/laps.csv"

# Green-flag pit loss per circuit, measured by strategy.pit_loss over the
# cached 2026 race sessions. The second value is how many matched green-flag
# stops the median rests on; below strategy.MIN_STOPS_FOR_STABLE_MEDIAN the
# figure is thin, and the UI says so rather than presenting it like the
# well-sampled ones.
PIT_LOSS_S = {
    "Australian Grand Prix": (22.28, 10),
    "Austrian Grand Prix": (21.39, 33),
    "Barcelona Grand Prix": (24.17, 40),
    "Belgian Grand Prix": (22.48, 10),
    "British Grand Prix": (19.86, 22),
    "Canadian Grand Prix": (27.77, 15),
    "Chinese Grand Prix": (33.07, 4),
    "Dutch Grand Prix": (19.74, 37),
    "Hungarian Grand Prix": (22.51, 34),
    "Japanese Grand Prix": (23.86, 14),
    "Miami Grand Prix": (18.89, 20),
    "Monaco Grand Prix": (21.55, 19),
}

st.set_page_config(page_title="F1 2026 Predictor", layout="wide")


@st.cache_data
def load_laps():
    return pd.read_csv(LAPS_CSV)


@st.cache_data
def load_model():
    return model.load()


@st.cache_data
def load_temp_comparison():
    """MAE with and without the tyre-age x track-temperature interaction.

    Not persisted in models/degradation.json -- that file only carries the
    fitted (with_temp=True) result. Recomputed here, cached, so the Model
    card's improvement sentence is derived from two live numbers rather
    than a hardcoded percentage that can go stale independently of them.
    """
    return model.compare_temp()


@st.cache_data(show_spinner=False)
def outcome_intervals(path, fraction=None):
    """Per outcome: the Brier gap and its 95% interval, or None if no table.

    Resampled over races by race.bootstrap_gap. Cached because the answer
    cannot change until the saved table does, and resampling on every rerun
    would cost seconds for nothing.
    """
    frame = load_predictions(path)
    if frame is None:
        return None
    if fraction is not None:
        frame = frame[frame["fraction"] == fraction]
    out = {}
    for outcome in race.OUTCOMES:
        gap = race.brier(
            frame[f"base_{outcome}"], frame[f"actual_{outcome}"]
        ) - race.brier(frame[outcome], frame[f"actual_{outcome}"])
        low, high, share = race.bootstrap_gap(
            frame, outcome, f"base_{outcome}", f"actual_{outcome}",
            n_boot=1000, seed=1,
        )
        out[outcome] = (gap, low, high, share)
    return out


@st.cache_data
def load_predictions(path):
    """A prediction table written by `f1_predict.export`, or None.

    Returning None rather than raising lets each tab name the command that
    rebuilds its own table. Neither table is computable inside a page load:
    the race simulation alone loads every weekend's practice sessions and
    runs two thousand simulations per race.
    """
    return pd.read_csv(path) if Path(path).exists() else None


laps = load_laps()
fitted = load_model()
coef = fitted["coef"]
temp_mean = fitted["track_temp_mean"]

# Race distance, circuit names and the temperature range all come from the
# data rather than a hardcoded table, so they cannot drift away from it.
LAPS_BY_EVENT = laps.groupby("event_name")["lap_number"].max().astype(int).to_dict()
ROUND_BY_EVENT = laps.groupby("event_name")["round"].first().astype(int).to_dict()
EVENT_BY_ROUND = {r: e for e, r in ROUND_BY_EVENT.items()}
TEMP_LO = float(laps["track_temp"].min())
TEMP_HI = float(laps["track_temp"].max())


def temp_control(label, key):
    """A track-temperature slider defaulting to the model's fitted centre.

    The interaction was fitted against deviations from track_temp_mean, so
    what strategy receives is a pre-centred delta, never raw Celsius.
    """
    temp = st.slider(label, TEMP_LO, TEMP_HI, float(temp_mean), step=0.5, key=key)
    return temp, temp - temp_mean


def degradation_loss(compound, age, temp_delta):
    """Seconds per lap lost to tyre age, at a given track temperature."""
    return (
        coef.get(f"age_{compound}", 0.0) * age
        + coef.get(f"age_temp_{compound}", 0.0) * age * temp_delta
    )


degradation, next_tab, strategy_tab, quali_tab, odds_tab, midrace_tab, live_tab, replay_tab, card = st.tabs(
    ["Degradation", "Next race", "Strategy", "Qualifying", "Race odds",
     "Mid-race", "Live", "Replay", "Model card"]
)

with degradation:
    st.header("Time lost per lap as tyres age")
    left, right = st.columns(2)
    with left:
        max_age = st.slider("Tyre age (laps)", 5, 40, 25)
    with right:
        temp, temp_delta = temp_control("Track temperature (C)", "deg_temp")

    curve = pd.DataFrame(
        {
            c: [degradation_loss(c, age, temp_delta) for age in range(max_age + 1)]
            for c in model.COMPOUNDS
        },
        index=range(max_age + 1),
    )
    curve.index.name = "Tyre age (laps)"
    st.line_chart(curve)

    st.caption(
        f"Seconds lost versus a fresh tyre, fitted across "
        f"{len(fitted['rounds'])} 2026 races. The "
        f"temperature slider is centred on the model's training mean of "
        f"{temp_mean:.1f} C, where the interaction contributes nothing; the further "
        f"you move from it the more you are extrapolating. Per-circuit rates are not "
        f"shown, because each circuit was raced once and circuit cannot be separated "
        f"from race-day conditions."
    )
    if fitted.get("compound_ordering_note"):
        st.info(
            "The three compound lines sit close together and their order is not "
            "meaningful here. The Model card explains why."
        )

# Reading a live weekend hits the network, so it runs behind a button rather
# than on every rerun, and the result is cached per round.
@st.cache_data(show_spinner=False)
def run_forward_quali(round_no):
    return predict.quali_order(2026, int(round_no))


@st.cache_data(show_spinner=False)
def run_forward_race(round_no):
    return predict.race_odds(2026, int(round_no))


with next_tab:
    st.header("Predict a weekend that has not happened yet")
    st.caption(
        "Every other tab is retrospective: it scores how well a model would "
        "have predicted a race, against that race's own result. This one "
        "reads only sessions that have already run and forecasts the ones "
        "that have not. The round being predicted is excluded from the "
        "training history, so pointing it at a completed race gives an "
        "honest holdout rather than a lookup."
    )

    round_no = st.number_input(
        "Round", min_value=1, max_value=24, value=12, step=1,
        help="12 is the Dutch Grand Prix. A round whose sessions have not "
             "run yet will say so rather than guess.",
    )
    # Nested rather than guarded by st.stop(): st.stop() halts the whole
    # script, so a weekend with no data yet would blank every tab after this
    # one instead of just this one.
    if st.button("Run prediction", type="primary"):
        try:
            order, meta = run_forward_quali(round_no)
        except Exception as exc:  # noqa: BLE001 - a future session is not an error
            st.warning(f"No qualifying prediction available yet. {exc}")
        else:
            st.subheader(f"{meta['event']} -- predicted qualifying order")
            scored = load_predictions(export.QUALI_OUT)
            if scored is None:
                st.info(
                    "**Rank by raw practice pace is the recommendation**, "
                    "because it beat the model when both were measured. Run "
                    "`python -m f1_predict.export` to see by how much."
                )
            else:
                pace_score = quali.score(scored, scored["baseline_position"])
                model_score = quali.score(scored, scored["pred_position"])
                st.info(
                    f"**Rank by raw practice pace is the recommendation.** "
                    f"Measured leave-one-race-out over "
                    f"{scored['round'].nunique()} rounds it missed by "
                    f"{pace_score['position_mae']:.2f} places, against "
                    f"{model_score['position_mae']:.2f} for the Ridge model "
                    f"built on top of it. The model column is shown for "
                    f"comparison, not as the answer."
                )
            st.caption(
                f"Pace read from {meta['primary_session']}. Sessions used: "
                f"{', '.join(meta['sessions_used'])}. Unavailable: "
                f"{', '.join(meta['unavailable']) or 'none'}. Trained on "
                f"{meta['history_rounds']} rounds that ran BEFORE this one; "
                f"later rounds are excluded too, not just this one."
            )
            shown = order[
                ["driver", "baseline_position", "model_position", "gap_primary"]
            ].copy()
            shown.columns = ["Driver", "Predicted (pace)", "Model", "Gap to best"]
            st.dataframe(
                shown.set_index("Driver").style.format({"Gap to best": "{:.3%}"}),
                width="stretch",
            )
            if meta["n_imputed"]:
                st.caption(
                    f"{meta['n_imputed']} of {meta['n_drivers']} drivers set "
                    f"no lap in {meta['primary_session']} and were filled "
                    f"from elsewhere. F1 requires teams to run rookies in "
                    f"FP1, and those drivers never qualify, so this order can "
                    f"be longer than the real grid -- Hungary predicts 27 "
                    f"where 22 qualified."
                )

        try:
            odds, race_meta = run_forward_race(round_no)
        except Exception as exc:  # noqa: BLE001 - qualifying may not have run yet
            st.warning(f"No race odds yet. {exc}")
        else:
            st.subheader(f"{race_meta['event']} -- race outcome probabilities")
            measured = load_predictions(export.RACE_OUT)
            n_scored = measured["round"].nunique() if measured is not None else 0
            st.info(
                f"**The grid table is the better bet.** A lookup of what each "
                f"grid slot has historically converted to beat this simulation "
                f"on all three outcomes across {n_scored or 'the scored'} "
                f"races. Where the two disagree, trust the table."
            )
            if race_meta["dropped_no_pace"]:
                st.caption(
                    f"Dropped for want of practice pace: "
                    f"{', '.join(race_meta['dropped_no_pace'])}. They have a "
                    f"grid slot but set no lap in "
                    f"{race_meta['primary_session']}."
                )
            st.caption(
                f"{race_meta['total_laps']} scheduled laps, grid from "
                f"qualifying, pace from {race_meta['primary_session']}. Tyre "
                f"model fitted on rounds {race_meta['tyre_model_rounds']}, "
                f"excluding this one."
            )
            columns = ["driver", "grid"] + [
                c for outcome in race.OUTCOMES
                for c in (outcome, f"base_{outcome}")
            ]
            table = odds[columns].copy()
            table.columns = [
                "Driver", "Grid", "Sim P(win)", "Table P(win)",
                "Sim P(podium)", "Table P(podium)", "Sim P(points)",
                "Table P(points)",
            ]
            percent = [c for c in table.columns if "P(" in c]
            st.dataframe(
                table.set_index("Driver").style.format(
                    {c: "{:.1%}" for c in percent}
                ),
                width="stretch",
            )
            st.caption(
                f"Safety cars are not modelled, so the simulation is "
                f"overconfident by construction. Neither number is a betting "
                f"tip: {n_scored or 'a dozen'} races is a small sample and no "
                f"significance test stands behind any of it."
            )


with strategy_tab:
    st.header("Which strategy finishes first?")
    event = st.selectbox("Race", sorted(fitted["baseline_by_event"]))
    baseline = fitted["baseline_by_event"][event]

    if event not in PIT_LOSS_S:
        st.error(
            f"No measured pit loss for {event!r}. The table is keyed to event names "
            f"taken from the 2026 data, and this one is missing, so any figure shown "
            f"here would be invented."
        )
    else:
        measured_loss, n_stops = PIT_LOSS_S[event]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            total_laps = int(
                st.number_input(
                    "Race distance (laps)", 30, 80, LAPS_BY_EVENT.get(event, 52)
                )
            )
        with col_b:
            loss = st.number_input(
                "Pit loss (seconds)", 10.0, 40.0, float(measured_loss), step=0.5
            )
        with col_c:
            temp, temp_delta = temp_control("Track temperature (C)", "strat_temp")

        if n_stops < strategy.MIN_STOPS_FOR_STABLE_MEDIAN:
            st.warning(
                f"The {measured_loss:.2f} s default is a median over only {n_stops} "
                f"green-flag stops, below the "
                f"{strategy.MIN_STOPS_FOR_STABLE_MEDIAN}-stop stability threshold. "
                f"Treat it as rough."
            )
        else:
            st.caption(
                f"Default pit loss measured from {n_stops} green-flag stops in this race."
            )

        best_lap, best_time = strategy.best_pit_lap(
            coef, baseline, loss, total_laps, "MEDIUM", "HARD", temp_delta=temp_delta
        )
        low, high = strategy.pit_window(
            coef, baseline, loss, total_laps, "MEDIUM", "HARD", temp_delta=temp_delta
        )
        third = total_laps // 3
        two_stop = strategy.simulate(
            coef, baseline, loss, total_laps,
            [("MEDIUM", third), ("HARD", third), ("HARD", total_laps - 2 * third)],
            temp_delta=temp_delta,
        )

        left, right = st.columns(2)
        left.metric("Best one-stop", f"{best_time:.1f} s", f"pit lap {best_lap}")
        right.metric("Two-stop", f"{two_stop:.1f} s", f"{two_stop - best_time:+.1f} s")
        st.info(
            f"Pit window: laps {low} to {high}. The window is the recommendation and lap "
            f"{best_lap} is only its centre. Across those {high - low + 1} laps the "
            f"predicted difference stays under half a second, which is finer than this "
            f"model can resolve."
        )
        st.caption(
            "Clean air only: no traffic, safety cars, or rival undercuts are modelled. "
            "Moving the temperature re-solves the optimum, and which way it shifts "
            "depends on which compound carries the larger fitted temperature term, so "
            "there is no universal rule that a hotter track means stopping earlier."
        )

with quali_tab:
    st.header("Predicted qualifying order")
    predicted_quali = load_predictions(export.QUALI_OUT)
    if predicted_quali is None:
        st.error(
            f"No {export.QUALI_OUT} yet. Build it with "
            f"`python -m f1_predict.export`."
        )
    else:

        # Scored with the same function that produced the verdict, so this
        # headline cannot drift away from what the model actually did.
        model_scores = quali.score(predicted_quali, predicted_quali["pred_position"])
        base_scores = quali.score(predicted_quali, predicted_quali["baseline_position"])

        a, b, c = st.columns(3)
        a.metric(
            "Model error",
            f"{model_scores['position_mae']:.2f} places",
            f"{model_scores['position_mae'] - base_scores['position_mae']:+.2f} vs baseline",
            delta_color="inverse",
        )
        b.metric("Baseline: raw practice pace", f"{base_scores['position_mae']:.2f} places")
        c.metric(
            "Top-3 hit rate",
            f"{model_scores['top3_hit']:.0%}",
            f"baseline {base_scores['top3_hit']:.0%}",
        )
        st.warning(
            f"**The model loses.** Ranking drivers by their raw fastest practice lap "
            f"misses by {base_scores['position_mae']:.2f} places; the Ridge model built "
            f"on top of that pace misses by {model_scores['position_mae']:.2f}. "
            f"Everything below is shown so you can see where it goes wrong, not "
            f"because it works."
        )

        quali_rounds = sorted(predicted_quali["round"].unique())
        quali_event = st.selectbox(
            "Race",
            [EVENT_BY_ROUND.get(r, f"Round {r}") for r in quali_rounds],
            key="quali_event",
        )
        one = predicted_quali[
            predicted_quali["round"] == ROUND_BY_EVENT.get(quali_event, -1)
        ].copy()
        one["miss"] = one["pred_position"] - one["quali_position"]
        table = one.sort_values("quali_position")[
            ["driver", "quali_position", "pred_position", "baseline_position", "miss"]
        ]
        table.columns = ["Driver", "Actual", "Predicted", "Baseline", "Miss"]
        st.dataframe(table.set_index("Driver"), width="stretch")

        session_used = one["primary_session"].iloc[0] if len(one) else "?"
        st.caption(
            f"Pace for this weekend was read from {session_used}, chosen by "
            f"`sessions.pick_primary`: sprint qualifying if the weekend has one, "
            f"otherwise the latest dry practice session. Predictions are "
            f"leave-one-race-out, so this weekend's own qualifying never reached "
            f"the model that predicted it. `Miss` is places out, positive meaning "
            f"the model expected the driver further back than they qualified."
        )
        if one["imputed"].sum():
            st.info(
                f"{int(one['imputed'].sum())} driver(s) set no usable lap in "
                f"{session_used} and fell back to their FP1 gap, or to the field's "
                f"worst gap if they set nothing at all."
            )
        st.caption(
            f"{predicted_quali['round'].nunique()} of "
            f"{laps['round'].nunique()} completed rounds appear here. A "
            f"weekend whose practice ran wet has no dry pace comparable to a dry "
            f"qualifying, and is skipped rather than guessed at."
        )

with odds_tab:
    st.header("Race outcome probabilities")
    odds = load_predictions(export.RACE_OUT)
    if odds is None:
        st.error(
            f"No {export.RACE_OUT} yet. Build it with "
            f"`python -m f1_predict.export` -- it takes several minutes."
        )
    else:

        model_brier = {o: race.brier(odds[o], odds[f"actual_{o}"]) for o in race.OUTCOMES}
        base_brier = {
            o: race.brier(odds[f"base_{o}"], odds[f"actual_{o}"]) for o in race.OUTCOMES
        }

        labels = {"p_win": "Win", "p_podium": "Podium", "p_points": "Points"}
        for column, outcome in zip(st.columns(3), race.OUTCOMES):
            column.metric(
                f"Brier: {labels[outcome]}",
                f"{model_brier[outcome]:.4f}",
                f"{model_brier[outcome] - base_brier[outcome]:+.4f} vs grid table",
                delta_color="inverse",
            )
        st.warning(
            "**The simulation loses on all three.** A one-column lookup table -- for "
            "each grid slot, how often drivers starting there won, finished on the "
            "podium and scored -- beats it every time. Brier is a squared error on "
            "the stated probability: lower is better, and a confident wrong answer "
            "is punished hardest."
        )

        odds_event = st.selectbox(
            "Race", sorted(odds["event_name"].unique()), key="odds_event"
        )
        one = odds[odds["event_name"] == odds_event].sort_values("grid")

        chart = one.set_index("driver")[["p_win", "base_p_win"]]
        chart.columns = ["Simulation", "Grid table"]
        st.bar_chart(chart)
        st.caption("P(win) per driver, simulation against the grid-slot baseline.")

        shown = one[
            ["driver", "grid", "position", "p_win", "p_podium", "p_points",
             "base_p_win", "base_p_podium", "base_p_points"]
        ].copy()
        shown.columns = [
            "Driver", "Grid", "Finished", "P(win)", "P(podium)", "P(points)",
            "Grid P(win)", "Grid P(podium)", "Grid P(points)",
        ]
        percent = [c for c in shown.columns if "P(" in c]
        st.dataframe(
            shown.set_index("Driver").style.format({c: "{:.1%}" for c in percent}),
            width="stretch",
        )

        winner = one.loc[one["position"] == 1.0, "driver"]
        if len(winner):
            given = float(one.loc[one["driver"] == winner.iloc[0], "p_win"].iloc[0])
            st.info(f"{winner.iloc[0]} won this race. The simulation gave them {given:.1%}.")

        st.caption(
            f"Leave-one-race-out: this race's own laps, passing record and "
            f"reliability never reached the simulation that predicted it. Pace comes "
            f"from the weekend's practice, the tyre model is refitted without this "
            f"round, and the race runs its originally scheduled distance so a red "
            f"flag cannot leak backwards. "
            f"{int(one['n_excluded_total'].iloc[0])} of "
            f"{int(one['n_rows_total'].iloc[0])} season entries are excluded: drivers "
            f"with no grid slot or no practice pace, plus every entry from a skipped "
            f"round."
        )
        if str(one["skipped_rounds"].iloc[0]).strip():
            st.caption(f"Skipped rounds -- {one['skipped_rounds'].iloc[0]}")
        st.caption(
            "Safety cars are not modelled at all, and they are the single largest "
            "source of race randomness. The simulation is therefore overconfident by "
            "construction: its probabilities sit closer to 0 and 1 than reality "
            "warrants. Eleven races contain eleven wins, so P(win) is the headline "
            "number with the least evidence behind it."
        )

with midrace_tab:
    st.header("Prediction from part-way through the race")
    mid = load_predictions(export.MIDRACE_OUT)
    if mid is None:
        st.error(
            f"No {export.MIDRACE_OUT} yet. Build it with "
            f"`python -m f1_predict.export` -- it takes several minutes."
        )
    else:

        rows = []
        for fraction in sorted(mid["fraction"].unique()):
            at = mid[mid["fraction"] == fraction]
            entry = {"Race distance": f"{fraction:.0%}"}
            for outcome in race.OUTCOMES:
                entry[f"Model {outcome}"] = race.brier(
                    at[outcome], at[f"actual_{outcome}"]
                )
                entry[f"Baseline {outcome}"] = race.brier(
                    at[f"base_{outcome}"], at[f"actual_{outcome}"]
                )
            rows.append(entry)
        curve = pd.DataFrame(rows).set_index("Race distance")

        # The verdict is generated from the curve, never restated from a report.
        # It was written out by hand once and went stale the first time a new
        # race entered the season: the Dutch Grand Prix flipped P(points) at 90%
        # from the model to the baseline while the sentence still claimed the
        # model won it.
        LABEL = {"p_win": "P(win)", "p_podium": "P(podium)", "p_points": "P(points)"}
        # An advantage is claimed only where the 95% interval excludes zero. On
        # point estimate alone the model led all three outcomes at 25% and 50%,
        # but four of those six gaps sit inside the interval. Calling those wins
        # would be a false signal, which is the one thing this project cannot
        # afford to publish.
        proven, leaning, against = [], [], []
        for fraction in sorted(mid["fraction"].unique()):
            stats = outcome_intervals(export.MIDRACE_OUT, fraction)
            label = f"{fraction:.0%}"
            for outcome in race.OUTCOMES:
                gap, low, high, _share = stats[outcome]
                if low > 0:
                    proven.append(f"{LABEL[outcome]} at {label}")
                elif high < 0:
                    against.append(f"{LABEL[outcome]} at {label}")
                elif gap > 0:
                    leaning.append(f"{LABEL[outcome]} at {label}")

        if proven:
            verdict = (
                "**The model beats the baseline, with the interval to back it, on "
                + " and ".join(proven)
                + ".**"
            )
        else:
            verdict = (
                "**No advantage in either direction survives its confidence "
                "interval.** Every gap sits inside the noise."
            )
        if leaning:
            verdict += (
                " It leads on point estimate but sits inside the interval on "
                + ", ".join(leaning)
                + " -- suggestive, and not a result."
            )
        if against:
            verdict += (
                " The baseline is significantly better on " + ", ".join(against) + "."
            )

        n_races = int(mid["round"].nunique())
        st.line_chart(curve[["Model p_win", "Baseline p_win"]])
        st.info(
            f"{verdict} Early in a race, track position is a weak signal, "
            f"because most of the race and every pit stop is still ahead, so a "
            f"simulation that knows pace and tyre state adds real information. "
            f"Late in a race, position becomes strongly informative about who "
            f"is about to win or reach the podium, and the simulation has less "
            f"left to add on top of it."
        )
        st.dataframe(curve.style.format("{:.4f}"), width="stretch")

        latest = curve.index[-1]
        perfect = curve.loc[latest, "Baseline p_win"] == 0.0
        st.caption(
            f"Brier for each outcome against how far into the race the "
            f"prediction was made. Lower is better."
            + (
                f" At {latest} distance the baseline's P(win) Brier is exactly "
                f"0.0000: the leader at that point went on to win all "
                f"{n_races} races, a baseline nothing can beat."
                if perfect
                else ""
            )
        )
        st.warning(
            f"Intervals are 95%, bootstrapped over {n_races} races rather "
            f"than over rows: drivers in one race share its conditions and "
            f"are not independent observations, so resampling rows would "
            f"report a confidence the data does not support. {n_races} races "
            f"is still a small sample -- adding the Dutch Grand Prix narrowed "
            f"the model's lead at 25% and 50% rather than widening it, which "
            f"is what a margin this size can do."
        )

        mid_event = st.selectbox(
            "Race", sorted(mid["event_name"].unique()), key="mid_event"
        )
        mid_fraction = st.select_slider(
            "Predict from",
            options=sorted(mid["fraction"].unique()),
            format_func=lambda f: f"{f:.0%} distance",
        )
        one = mid[
            (mid["event_name"] == mid_event) & (mid["fraction"] == mid_fraction)
        ].sort_values("position_at_n")
        if one.empty:
            st.info("This race was not scored at that point.")
        else:
            shown = one[
                ["driver", "position_at_n", "position", "p_win", "p_podium",
                 "p_points", "base_p_win"]
            ].copy()
            shown.columns = [
                "Driver", f"Position at lap {int(one['lap'].iloc[0])}", "Finished",
                "P(win)", "P(podium)", "P(points)", "Baseline P(win)",
            ]
            percent = [c for c in shown.columns if c.startswith(("P(", "Baseline"))]
            st.dataframe(
                shown.set_index("Driver").style.format({c: "{:.1%}" for c in percent}),
                width="stretch",
            )
        st.caption(
            "Safety cars are still not modelled, and they matter more mid-race "
            "than before the start: a caution can hand a stopped driver the "
            "whole field's track position back in a single lap. That is "
            f"subsystem C2. The {len(curve)} points on this curve come from "
            f"the same {n_races} races, so it is a trend, not "
            f"{len(curve)} independent measurements."
        )

with live_tab:
    st.header("A race as it runs")
    st.caption(
        "Reads a live timing recording and predicts the rest of the race "
        "from it. Start the recorder before the session with "
        "`python scripts/record_session.py <path>` -- an unrecorded "
        "session cannot be recovered afterwards."
    )

    left, right = st.columns(2)
    with left:
        recording = st.text_input(
            "Recording file", value="recordings/monza-race.txt"
        )
        round_no = st.number_input("Round", 1, 24, 13)
    with right:
        every = st.selectbox("Refresh every (seconds)", [15, 30, 60], index=1)
        runs = st.selectbox("Simulations per refresh", [500, 2000], index=0)

    if not Path(recording).exists():
        st.info(
            f"No recording at `{recording}` yet. This tab has nothing to "
            f"show until a session has been recorded -- that is the cost "
            f"of a live feed, and there is no archive of the stream to "
            f"fall back on."
        )
    else:
        @st.fragment(run_every=every)
        def live_panel():
            try:
                out, meta, odds_meta, elapsed = live.run_odds(
                    recording, year=2026, round_no=int(round_no), n_runs=int(runs)
                )
            except ValueError as exc:
                st.warning(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - one bad read must not kill the tab
                st.error(f"{type(exc).__name__}: {exc}")
                return

            # `elapsed` covers only this refresh's read-and-simulate. The
            # caption below stays on screen for the whole refresh interval,
            # so the bound it states must add `every` -- otherwise it
            # understates the true lag by up to the length of that
            # interval, which is exactly the false confidence this tab
            # exists to avoid.
            st.caption(
                live.describe_age(
                    elapsed + every, odds_meta["lap"], odds_meta["total_laps"]
                )
            )
            if meta["errorcount"] > 20:
                st.warning(
                    f"{meta['errorcount']} unparseable lines in the "
                    f"recording. One or two is the half-written final "
                    f"line and is normal; this many means the recording "
                    f"is not what it is being read as."
                )
            st.dataframe(out.style.format("{:.3f}"), width="stretch")
            st.caption(
                "The safety car is not modelled -- subsystem C2 measured "
                "it across twelve races and it helped in none of them, so "
                "it ships switched off. These probabilities assume green "
                "flag racing to the end."
            )

        live_panel()

with replay_tab:
    st.header("Lap-by-lap replay")
    st.caption(
        "A finished race fed back one lap at a time. This is not live timing: a live "
        "feed needs a recorder running during the session, which cannot be "
        "reconstructed after the fact."
    )
    replay_event = st.selectbox("Race", sorted(ROUND_BY_EVENT), key="replay_event")
    round_no = ROUND_BY_EVENT[replay_event]
    race_laps = laps[laps["round"] == round_no]
    driver = st.selectbox("Driver", sorted(race_laps["driver"].unique()))

    frames = list(replay.stream(laps, round_no, driver))
    index = st.slider("Lap", 1, len(frames), 1) - 1
    frame = frames[index]

    # A lap can reach here with no tyre age: FastF1 leaves TyreLife unset on
    # some laps, filter_laps has no reason to drop them (they are perfectly
    # good laps, just missing one field), and the training path drops them
    # separately. Show the lap rather than crashing on int(NaN) or skipping
    # it, which would silently renumber the replay.
    known_age = pd.notna(frame["tyre_age"])

    a, b, c = st.columns(3)
    a.metric("Compound", str(frame["compound"]))
    b.metric("Tyre age", f"{int(frame['tyre_age'])} laps" if known_age else "not recorded")
    c.metric(
        "Lost vs fresh tyre",
        f"{degradation_loss(str(frame['compound']), int(frame['tyre_age']), 0.0):.2f} s/lap"
        if known_age
        else "n/a",
    )
    if not known_age:
        st.caption(
            "This lap has no recorded tyre age, so the wear estimate is "
            "unavailable for it. The lap time below is unaffected."
        )
    shown = pd.DataFrame(frames[: index + 1]).set_index("lap_number")
    st.line_chart(shown["lap_seconds"])
    st.caption(
        "Lap time in seconds, up to the selected lap. The per-lap loss above is "
        "quoted at the model's training temperature."
    )

with card:
    st.header("Model card")
    cv = fitted["cv"]
    temp_comparison = load_temp_comparison()
    a, b, c = st.columns(3)
    a.metric(
        "MAE (leave-races-out)", f"{cv['mae_mean']:.3f} s", f"+/-{cv['mae_std']:.3f}"
    )
    b.metric("Baseline: no degradation", f"{cv['mae_zero']:.3f} s")
    c.metric("Baseline: one shared slope", f"{cv['mae_global_slope']:.3f} s")

    st.subheader("Coefficients")
    st.caption(
        "age_* is seconds per lap of tyre age. age_temp_* is the extra seconds per "
        "lap of age for each degree above the training mean. fuel is seconds per lap "
        "of fuel still to burn."
    )
    st.json(coef)
    st.caption(
        f"Fitted on {fitted['n_rows']} clean laps from rounds {fitted['rounds']}. "
        f"{fitted['n_dropped_null']} laps were dropped for a missing tyre age."
    )

    st.subheader("Safety cars were modelled, measured, and switched back off")
    st.warning(
        "**The safety car does not improve these predictions, and the code "
        "for it ships disabled.** Of twelve cells -- three outcomes at four "
        "points of race distance -- no cell showed an improvement whose 95% "
        "interval excluded zero. One, P(points) at quarter distance, fell "
        "marginally outside on the worse side, by less than the "
        "simulation's own run-to-run noise. The other eleven sat inside "
        "the interval."
    )
    st.caption(
        "This is not a finding that safety cars do not matter. Their effect "
        "on track is large and was measured: across four periods in 2026 the "
        "spread between the leader and the last car on the lead lap fell to "
        "a median 0.25 of what it was. The finding is that eight deployments "
        "and four measurable periods are not enough to model one usefully -- "
        "how often and how long one happens can only be guessed from a "
        "sample that small, and the guess adds as much noise as it removes. "
        "With twelve cells scored at a 95% threshold, about one apparent "
        "result is expected from chance alone -- which is exactly what the "
        "single marginal cell above is, not a finding in either direction."
    )
    st.caption(
        "The prediction that it would fail was written into the design "
        "document before any of the code existed, so that a negative result "
        "could not be explained away afterwards and a positive one would "
        "have carried the weight of being predicted against. Nothing was "
        "tuned to rescue it: not the compression constant, not the observed "
        "durations, not the number of simulation runs. With twelve cells and "
        "a 5% threshold, a few attempts would very likely have manufactured "
        "one apparent win."
    )

    if fitted.get("compound_ordering_note"):
        st.subheader("Compound ordering is not identifiable")
        st.warning(fitted["compound_ordering_note"])

    if fitted["physics_violations"]:
        st.error("Physics checks failed: " + "; ".join(fitted["physics_violations"]))

    st.subheader("What this model does and does not establish")
    st.markdown(
        f"""
**Splitting degradation by compound is not shown to be worth anything.** Before the
temperature term existed, three per-compound slopes scored 0.616 s against a single
shared slope's 0.614 s: effectively tied. That comparison has not been re-run since,
and the shared-slope baseline carries no temperature term, so it is no longer a clean
test. The per-compound numbers above are reported, not validated.

**The tyre-age by track-temperature interaction was measured, and kept.**
Leave-races-out MAE improved from {temp_comparison['without']:.4f} s to
{temp_comparison['with']:.4f} s, a
{(temp_comparison['without'] - temp_comparison['with']) / temp_comparison['without']:.2%}
reduction, clearing a 1% threshold that was set in advance so noise could not be
promoted to a finding.

**Pit loss is a green-flag figure by definition.** Pooled across every stop it reaches
59 s at some circuits, because teams pit under safety car whenever they can, and the
in-lap plus out-lap excess then measures the caution period rather than the pit lane.

Other limits:

- {len(fitted['rounds'])} races, each circuit raced once. Per-circuit degradation is not identifiable and is not claimed.
- Bahrain and Saudi Arabia are absent from F1's own 2026 timing archive. They were not dropped by choice.
- SOFT, MEDIUM and HARD are relative to each circuit's Pirelli allocation, so "HARD" is not the same rubber everywhere. Pooling across circuits blurs the compounds together.
- 2026 energy deployment strongly affects lap time and is invisible in timing data. It lands in the residual.
- Wet running is excluded entirely.
- Cross-validation uses {len(cv['mae_folds'])} folds (`GroupKFold` capped at 5, over {len(fitted['rounds'])} race groups), so the fold spread of +/-{cv['mae_std']:.3f} s is wide by construction.
- The simulator assumes clean air.
"""
    )
