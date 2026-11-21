"""Generates a SYNTHETIC hourly price series purely to test the model.
These are NOT real prices and must never be used for results in the project.
Replace with real NO2 day-ahead prices from the ENTSO-E Transparency Platform."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
h = idx.hour.values
doy = idx.dayofyear.values
daily = 15 * np.exp(-((h - 8) ** 2) / 6) + 22 * np.exp(-((h - 18) ** 2) / 5) - 10 * np.exp(-((h - 13) ** 2) / 8)
seasonal = 20 * np.cos(2 * np.pi * (doy - 15) / 365)
wind = np.convolve(rng.normal(0, 1, len(idx)), np.ones(48) / 48 ** 0.5, mode="same")  # multi-day wind regimes
price = 70 + seasonal + daily - 18 * wind + rng.normal(0, 4, len(idx))
pd.DataFrame({"time": idx.strftime("%Y-%m-%d %H:%M"), "price_eur_mwh": price.round(2)}).to_csv(
    "SYNTHETIC_TEST_prices_NOT_REAL.csv", index=False)
print("wrote SYNTHETIC_TEST_prices_NOT_REAL.csv (test data only)")
