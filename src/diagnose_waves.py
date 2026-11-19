"""Show where wave_height is missing, by year, by month for one site, and first/last valid hour."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

df = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])
df["missing"] = df["wave_height"].isna()

print("Missing share by year:")
print(df.groupby(df["time"].dt.year)["missing"].mean().round(3).to_string())

site = df["site"].iloc[0]
one = df[df["site"] == site].set_index("time")["missing"]
print(f"\nMonths with any missing wave data, {site}:")
monthly = one.groupby(one.index.to_period("M")).mean().round(2)
print(monthly[monthly > 0].to_string() or "none")

valid = df.dropna(subset=["wave_height"])
print("\nFirst and last hour with wave data:")
print(valid.groupby("site")["time"].agg(["min", "max"]).to_string())
