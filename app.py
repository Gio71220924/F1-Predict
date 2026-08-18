"""Streamlit UI for the 2026 tyre degradation model."""
import pandas as pd
import streamlit as st

from f1_predict import model, replay, strategy

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
    "Hungarian Grand Prix": (22.51, 34),
    "Japanese Grand Prix": (23.86, 14),
    "Miami Grand Prix": (18.89, 20),
    "Monaco Grand Prix": (21.55, 19),
}

st.set_page_config(page_title="F1 2026 Tyre Strategy", layout="wide")


@st.cache_data
def load_laps():
    return pd.read_csv(LAPS_CSV)


@st.cache_data
def load_model():
    return model.load()


laps = load_laps()
fitted = load_model()
coef = fitted["coef"]
temp_mean = fitted["track_temp_mean"]

# Race distance, circuit names and the temperature range all come from the
# data rather than a hardcoded table, so they cannot drift away from it.
LAPS_BY_EVENT = laps.groupby("event_name")["lap_number"].max().astype(int).to_dict()
ROUND_BY_EVENT = laps.groupby("event_name")["round"].first().astype(int).to_dict()
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


degradation, strategy_tab, replay_tab, card = st.tabs(
    ["Degradation", "Strategy", "Replay", "Model card"]
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
        f"Seconds lost versus a fresh tyre, fitted across all 11 2026 races. The "
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
        st.stop()
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
Leave-races-out MAE improved from 0.6161 s to {cv['mae_mean']:.4f} s, a 3.6% reduction,
clearing a 1% threshold that was set in advance so noise could not be promoted to a
finding.

**Pit loss is a green-flag figure by definition.** Pooled across every stop it reaches
59 s at some circuits, because teams pit under safety car whenever they can, and the
in-lap plus out-lap excess then measures the caution period rather than the pit lane.

Other limits:

- Eleven races, each circuit raced once. Per-circuit degradation is not identifiable and is not claimed.
- Bahrain and Saudi Arabia are absent from F1's own 2026 timing archive. They were not dropped by choice.
- SOFT, MEDIUM and HARD are relative to each circuit's Pirelli allocation, so "HARD" is not the same rubber everywhere. Pooling across circuits blurs the compounds together.
- 2026 energy deployment strongly affects lap time and is invisible in timing data. It lands in the residual.
- Wet running is excluded entirely.
- Cross-validation uses 11 groups, so the fold spread of +/-{cv['mae_std']:.3f} s is wide by construction.
- The simulator assumes clean air.
"""
    )
