"""Show class shares for each threshold set in config.THRESHOLD_SETS.

For each set it prints the class shares in the training years, by year, by site, and the
number of Critical hours in the 2026 test period. Use it to choose and defend the
thresholds before any model is trained. Whatever split results is reported as it is.

Usage:
    python src/check_labels.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import labels

NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


def class_table(frame, by):
    """Class shares in percent for each value of `by`."""
    t = pd.crosstab(frame[by], frame["y"], normalize="index").reindex(columns=[0, 1, 2], fill_value=0) * 100
    return t.rename(columns=NAMES).round(1)


def check(path=cfg.CSV_PATH):
    """Print the class balance for every threshold set."""
    df = pd.read_csv(path, parse_dates=["time"])
    sites = {s: labels.regular(g.assign(site=s)) for s, g in df.groupby("site")}
    for name, th in cfg.THRESHOLD_SETS.items():
        print("=" * 70)
        print(f"Threshold set: {name}")
        print(th)
        parts = []
        for s, g in sites.items():
            y, _ = labels.targets(g, th)
            parts.append(pd.DataFrame({"site": s, "time": g.index, "y": y}))
        f = pd.concat(parts)
        f = f[f["y"] >= 0]
        f["year"] = f["time"].dt.year
        train = f[f["time"] < cfg.VAL_START]
        print(f"\nTraining years (before {cfg.VAL_START}), share of hours in %:")
        print((train["y"].value_counts(normalize=True).sort_index() * 100).rename(NAMES).round(1).to_string())
        print("\nBy year:")
        print(class_table(f, "year").to_string())
        print("\nBy site, training years:")
        print(class_table(train, "site").to_string())
        test = f[f["time"] >= cfg.TEST_START]
        print(f"\nTest period (from {cfg.TEST_START}): {len(test)} hours, Critical hours: {(test['y'] == 2).sum()}")
        print(test[test["y"] == 2].groupby("site").size().to_string() or "none")


if __name__ == "__main__":
    if not cfg.CSV_PATH.exists():
        raise SystemExit(f"{cfg.CSV_PATH} not found. Run collect_weather.py first.")
    check()
