"""Check the collected weather CSV before any modelling.

Prints row counts, missing values, the wave grid cell each site really uses,
the pairwise distance between those cells, and the wave-height correlation
between sites. Flags problems that need a site to be moved.

Usage:
    python src/check_metocean.py
"""

import sys
from itertools import combinations
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two points."""
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = p2 - p1, radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def check(path=cfg.CSV_PATH):
    """Run all checks on the merged CSV and return a list of warning strings."""
    df = pd.read_csv(path, parse_dates=["time"])
    warnings = []

    print("Rows:", len(df))
    print("Time range:", df["time"].min(), "to", df["time"].max())
    print("\nRows per site:")
    print(df.groupby("site").size().to_string())

    print("\nShare of missing values per column:")
    print(df.isna().mean().round(4).to_string())

    wave_missing = df.groupby("site")["wave_height"].apply(lambda s: s.isna().mean())
    print("\nMissing wave_height share per site:")
    print(wave_missing.round(4).to_string())
    by_year = df.assign(miss=df["wave_height"].isna()).groupby(df["time"].dt.year)["miss"].mean()
    print("\nMissing wave_height share by year (all sites):")
    print(by_year.round(3).to_string())
    for site, share in wave_missing.items():
        if share > 0.02:
            warnings.append(
                f"{site}: {share:.0%} of wave_height is missing. If only this site is affected, "
                "move the point offshore. If every site is affected, check the year table above."
            )

    dup = df.duplicated(["site", "time"]).sum()
    print("\nDuplicate site/time rows:", dup)
    if dup:
        warnings.append(f"{dup} duplicate site/time rows")

    cells = df.groupby("site")[["grid_lat", "grid_lon"]].first()
    print("\nWave grid cell used by each site:")
    print(cells.to_string())
    for a, b in combinations(cells.index, 2):
        if tuple(cells.loc[a]) == tuple(cells.loc[b]):
            warnings.append(f"{a} and {b} use the same wave grid cell, move one")
        else:
            d = haversine_km(*cells.loc[a], *cells.loc[b])
            if d < 50:
                warnings.append(f"{a} and {b} grid cells are only {d:.0f} km apart")

    waves = df.pivot(index="time", columns="site", values="wave_height")
    corr = waves.corr()
    print("\nWave height correlation between sites:")
    print(corr.round(3).to_string())
    corr_columns = list(corr.columns)
    corr_values = corr.to_numpy(dtype=float)
    for i, a in enumerate(corr_columns):
        for j in range(i + 1, len(corr_columns)):
            b = corr_columns[j]
            correlation = corr_values[i, j]
            if correlation > 0.97:
                warnings.append(f"{a} and {b} wave correlation is {correlation:.3f}, nearly identical")

    print("\nGust speed (m/s) by site, 99th percentile and maximum:")
    print(df.groupby("site")["wind_gusts_10m"].agg(p99=lambda s: s.quantile(0.99), max="max").round(1).to_string())

    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("No problems found.")
    return warnings


if __name__ == "__main__":
    if not cfg.CSV_PATH.exists():
        raise SystemExit(f"{cfg.CSV_PATH} not found. Run collect_weather.py first.")
    check()
