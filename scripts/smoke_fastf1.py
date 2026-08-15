"""One-off check that FastF1 can reach 2026 timing data. Not part of the test suite."""
import logging
import warnings

import fastf1

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

fastf1.Cache.enable_cache("cache")
session = fastf1.get_session(2026, "Silverstone", "R")
session.load(telemetry=False, weather=True, messages=False)

laps = session.laps
print("laps:", len(laps))
print("accurate:", int(laps.IsAccurate.sum()))
print("compounds:", sorted(laps.Compound.dropna().unique()))
print("weather rows:", len(session.weather_data))
