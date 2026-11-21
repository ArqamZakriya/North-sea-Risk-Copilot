"""Offline tests for labels, features and the split."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
import labels

TH = dict(wind_col="wind_speed_100m", wind_warn=12.0, wind_crit=18.0, wave_warn=1.5, wave_crit=3.0)


def synthetic(sites=("A", "B"), start="2024-12-01", end="2026-01-31", seed=0):
    t = pd.date_range(start, end, freq="h")
    frames = []
    for i, s in enumerate(sites):
        rng = np.random.default_rng(seed + i)
        n = len(t)
        w = rng.gamma(4, 2, n)
        frames.append(pd.DataFrame({
            "time": t, "site": s, "wind_speed_10m": w, "wind_speed_100m": w * 1.3,
            "wind_gusts_10m": w * 1.6, "wind_direction_100m": rng.uniform(0, 360, n),
            "temperature_2m": rng.normal(9, 4, n), "surface_pressure": rng.normal(1010, 8, n),
            "wave_height": rng.gamma(3, 0.5, n), "wave_period": rng.uniform(4, 12, n),
            "wave_direction": rng.uniform(0, 360, n), "grid_lat": 55.0, "grid_lon": 1.0}))
    return pd.concat(frames, ignore_index=True)


def test_label_from():
    lab = labels.label_from(np.array([5, 13, 20, 5, 5]), np.array([0.5, 0.5, 0.5, 1.6, 3.2]), TH)
    assert lab.tolist() == [0, 1, 2, 1, 2]


def test_target_looks_only_at_next_six_hours():
    t = pd.date_range("2025-03-01", periods=40, freq="h")
    g = pd.DataFrame({"wind_speed_100m": 5.0, "wave_height": 0.5}, index=t)
    g.loc[t[20], "wave_height"] = 3.5
    y, persist = labels.targets(g, TH)
    assert y[14:20].tolist() == [2] * 6
    assert y[13] == 0 and y[20] == 0
    assert persist[20] == 2 and persist[19] == 0
    assert (y[-6:] == -1).all()


def test_features_do_not_use_the_future():
    df = synthetic(sites=("A",), end="2025-06-30")
    full = labels.build(df, TH)
    cut = pd.Timestamp("2025-03-10 12:00")
    altered = df.copy()
    altered.loc[altered["time"] > cut, ["wind_speed_100m", "wave_height", "surface_pressure", "wind_gusts_10m"]] *= 3
    after = labels.build(altered, TH)
    cols = labels.feature_columns(full)
    a = full[full["time"] <= cut].set_index("time")[cols]
    b = after[after["time"] <= cut].set_index("time")[cols]
    pd.testing.assert_frame_equal(a, b)


def test_feature_count_and_split():
    df = synthetic()
    d = labels.build(df, TH)
    cols = labels.feature_columns(d)
    assert len(cols) == 28
    assert not set(cols) & set(cfg.NON_FEATURE_COLUMNS + ["time", "site"])
    tr, va, te = labels.time_split(d)
    b1, b2 = pd.Timestamp(cfg.VAL_START), pd.Timestamp(cfg.TEST_START)
    assert tr["time"].max() == b1 - pd.Timedelta(hours=cfg.HORIZON + 1)
    assert va["time"].min() == b1 and va["time"].max() == b2 - pd.Timedelta(hours=cfg.HORIZON + 1)
    assert te["time"].min() == b2
    assert len(tr) + len(va) + len(te) < len(d)
