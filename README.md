# F1 2026 — tyre, qualifying and race prediction

Five models over the 2026 Formula 1 season, and a live feed that predicts a race
while it is still running.

**Four of the five lose to a one-column baseline. This README says so, and so
does the app.** That is the point of the project, not an admission buried in it:
every model here is scored against the dumbest thing that could have been done
instead, on races it never saw, and reported at whatever it actually measured.

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## What was measured

Twelve completed rounds of 2026, 11,321 clean laps. Every model is validated
leave-one-race-out — drivers within a race share its weather, its safety cars
and its circuit, so scoring on held-out *rows* would leak the answer into the
question. Confidence intervals are bootstrapped over **races**, never rows, for
the same reason: resampling rows would report a confidence far narrower than the
data supports.

### Tyre degradation — wins

Seconds lost per lap as a tyre ages. Mean absolute error, leave-races-out:

| | MAE |
|---|---:|
| **Model** | **0.554 s** ±0.075 |
| Baseline: no degradation at all | 0.803 s |
| Baseline: one shared slope for every compound | 0.581 s |

It beats "tyres do not degrade" convincingly and "all tyres degrade the same"
narrowly. Splitting by compound is *reported*, not validated — before the
temperature term existed, three per-compound slopes scored 0.616 s against a
shared slope's 0.614 s. Effectively tied.

The tyre-age × track-temperature interaction cleared a 1% improvement threshold
set in advance, so noise could not be promoted to a finding afterwards.

### Qualifying position — loses

Mean absolute error in grid places, 260 rows across 12 races:

| | Places |
|---|---:|
| Model (practice pace + driver form) | 2.169 |
| **Baseline: order drivers by their fastest practice lap** | **1.554** |

Adding a model to the raw practice pace made the prediction worse by 0.6 of a
grid slot. Sorting one column beats it.

### Race outcome from the grid — loses, significantly

Brier score, 256 rows across 12 races. Lower is better. The baseline is a lookup
table: what fraction of drivers starting from each grid slot historically won,
podiumed or scored.

| Outcome | Model | Baseline | 95% interval on the gap |
|---|---:|---:|---|
| P(win) | 0.0529 | **0.0209** | [−0.050, −0.012] |
| P(podium) | 0.1065 | **0.0713** | [−0.068, −0.003] |
| P(points) | 0.2231 | **0.1921** | [−0.053, −0.008] |

All three intervals exclude zero on the baseline's side. This is not "too close
to call" — the Monte-Carlo race simulation is measurably worse than a lookup
table, and it ships with that stated on its own tab.

### Mid-race outcome — wins 2 cells of 12

Given the race as it stood at lap N, what happens from here. 903 rows, 12 races.
Model Brier / baseline Brier, where the baseline is a lookup on track position
at lap N:

| Distance | P(win) | P(podium) | P(points) |
|---|---|---|---|
| 25% | 0.0300 / 0.0420 | 0.0582 / 0.0683 | 0.1138 / 0.1333 |
| 50% | **0.0193 / 0.0362** | 0.0580 / 0.0655 | **0.1031 / 0.1190** |
| 75% | 0.0118 / 0.0099 | 0.0519 / 0.0538 | 0.0794 / 0.0817 |
| 90% | 0.0009 / **0.0000** | 0.0385 / 0.0273 | 0.0595 / 0.0564 |

The model leads on point estimate in nine of twelve cells. **Two survive their
confidence interval** — P(win) and P(points) at half distance, in bold. The rest
sit inside the noise, and at 90% distance the baseline's P(win) Brier is exactly
0.0000, because the leader with 10% to go won all twelve races. Nothing beats
that.

This is the only model here that beats its baseline on evidence, and it does so
in a narrow window: early enough that track position is still weak, late enough
that pace and tyre state have shown themselves.

### Safety car — no effect, ships disabled

Modelled as a per-lap hazard with a measured field-compression constant, then
evaluated across the same twelve cells. **None showed an improvement whose
interval excluded zero.** One fell marginally the other way, which is about what
twelve cells at a 5% threshold produce by chance.

The failure was predicted in the design document *before the code existed*, so a
negative result could not be explained away and a positive one would have
carried the weight of having been bet against. Nothing was tuned to rescue it.
The code ships behind a switch, off.

Their effect on track is real and was measured: across four periods, the spread
from the leader to the last car on the lead lap fell to a median 0.25 of what it
was. The finding is not that safety cars do not matter. It is that eight
deployments cannot teach a model when one will happen.

---

## The live feed

Win, podium and points probabilities that update **while a race is running**.

The predictor is unchanged — `midrace.state_from_laps` was already a pure
function taking a lap frame, so the live path only has to produce that frame from
a recording that is still being written.

```bash
# during the session — nothing runs this for you
python scripts/record_session.py recordings/monza-race.txt

# in another terminal, watch it update
streamlit run app.py        # the "Live" tab
```

**A session that is not recorded is gone.** There is no archive of the live
stream to replay. The machine must be awake and online for the whole session, and
a laptop that is on but asleep drops the connection.

The display trails the cars by the refresh interval — tens of seconds, under one
lap at Monza, and never zero. It says so on screen, in seconds, every refresh.

What was established offline: a half-written recording is readable. Two complete
lines plus a truncated fragment yields `errorcount == 1` without raising. What
could **not** be established offline, and is stated in the design document rather
than papered over: whether FastF1 builds usable *laps* from a partial recording.
That is what a Practice 1 rehearsal is for, two days before the race.

---

## Layout

```
f1_predict/
  data.py        Load and filter laps: accurate, green-flag, dry, non-outlier
  model.py       Tyre degradation fit, leave-races-out cross-validation
  strategy.py    Stint simulation and pit-loss measurement
  quali.py       Qualifying position model and its baseline
  race.py        Monte-Carlo race simulation, Brier scoring, race-level bootstrap
  predict.py     Forward prediction for a race that has not run
  midrace.py     Race state at lap N, and prediction from it
  safety.py      Safety-car hazard and compression (measured, disabled)
  live.py        Record, read and predict from a live timing stream
  replay.py      Lap-by-lap playback of a finished race
app.py           Nine-tab Streamlit interface
scripts/         Command-line recorder
tests/           132 tests, no network, warnings as errors
```

```bash
python -m pytest -W error        # the whole suite, about five seconds
```

Every test uses synthetic FastF1-shaped fixtures. Nothing in the suite touches
the network, and warnings are errors.

---

## Limits

- **Twelve races.** Each circuit ran once, so per-circuit degradation is not
  identifiable and is not claimed.
- **Bahrain and Saudi Arabia are missing** from F1's own 2026 timing archive.
  They were not dropped by choice.
- **Compounds are relative.** SOFT, MEDIUM and HARD are per-circuit Pirelli
  allocations, so "HARD" is not the same rubber everywhere. Pooling blurs them.
- **2026 energy deployment** strongly affects lap time and is invisible in timing
  data. It lands in the residual.
- **Wet running is excluded** from the tyre model entirely.
- **The simulator assumes clean air** and does not model dirty-air degradation.
- **Pit loss is a green-flag figure** by definition. Pooled across every stop it
  reaches 59 s at some circuits, because teams pit under caution whenever they
  can, and the in-lap plus out-lap excess then measures the safety car rather
  than the pit lane.

Data: [FastF1](https://github.com/theOehrly/Fast-F1). 2026 season only.
