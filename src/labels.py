"""Hazard labels, features and the time split.

Target: the worst class over the next cfg.HORIZON hours (t+1 to t+6). A class is set
by the highest wind speed and the highest significant wave height in that window.
    0 Normal, 1 Warning, 2 Critical
Thresholds come from cfg.THRESHOLD_SETS. They are assumptions taken from published
operating limits and must be cited as such in the report.

Features use the current hour and earlier hours only. y_persist labels the next
window from the current conditions alone and is the no-model baseline.

The split is by time: train before cfg.VAL_START, validation for that year, test from
cfg.TEST_START. The last cfg.HORIZON hours before each boundary are dropped because
their targets look into the next period.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROLL_COLUMNS = ["wind_speed_100m", "wind_gusts_10m", "wave_height"]


def label_from(wind, wave, th):
    """Class array from wind and wave arrays: 0 Normal, 1 Warning, 2 Critical."""
    lab = np.zeros(len(wind), dtype=int)
    lab[(wind >= th["wind_warn"]) | (wave >= th["wave_warn"])] = 1
    lab[(wind >= th["wind_crit"]) | (wave >= th["wave_crit"])] = 2
    return lab


def regular(g):
    """One site on a strict hourly index, filling gaps of up to three hours."""
    g = g.sort_values("time").drop_duplicates("time").set_index("time").asfreq("h")
    return g.ffill(limit=3)


def targets(g, th):
    """Return (y, y_persist) for a regular hourly site frame. y is -1 where the window is incomplete."""
    wcol = th["wind_col"]
    fut_w = g[wcol].rolling(cfg.HORIZON).max().shift(-cfg.HORIZON)
    fut_h = g["wave_height"].rolling(cfg.HORIZON).max().shift(-cfg.HORIZON)
    lab = label_from(fut_w.fillna(0).values, fut_h.fillna(0).values, th)
    y = np.where(fut_w.isna() | fut_h.isna(), -1, lab)
    persist = label_from(g[wcol].fillna(0).values, g["wave_height"].fillna(0).values, th)
    return y, persist


def add_features(g):
    """Add rolling, change, ratio and calendar features that use only the current and earlier hours."""
    g = g.copy()
    for c in ROLL_COLUMNS:
        for w in (6, 24):
            g[f"{c}_mean_{w}h"] = g[c].rolling(w).mean()
            g[f"{c}_std_{w}h"] = g[c].rolling(w).std()
    g["pressure_delta_1h"] = g["surface_pressure"].diff(1)
    g["pressure_delta_3h"] = g["surface_pressure"].diff(3)
    g["wave_delta_1h"] = g["wave_height"].diff(1)
    g["gust_factor"] = g["wind_gusts_10m"] / (g["wind_speed_10m"] + 0.1)
    g["marine_stress_index"] = g["wave_height"] * g["wind_speed_100m"]
    h, m = g.index.hour, g.index.month
    g["hour_sin"], g["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    g["month_sin"], g["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return g


def build(df, th=None):
    """Features and labels for every site. Rows without a full feature history or target window are dropped."""
    th = th or cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]
    parts = []
    for site, g in df.groupby("site"):
        g = add_features(regular(g.assign(site=site)))
        g["y"], g["y_persist"] = targets(g, th)
        g["site"] = site
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True).dropna()
    return out[out["y"] >= 0].reset_index(drop=True)


def feature_columns(d):
    """Numeric columns used as model inputs."""
    return [c for c in d.select_dtypes("number").columns if c not in cfg.NON_FEATURE_COLUMNS]


def time_split(d, purge=True):
    """Return (train, validation, test) split by time, dropping the last HORIZON hours before each boundary."""
    gap = pd.Timedelta(hours=cfg.HORIZON if purge else 0)
    b1, b2 = pd.Timestamp(cfg.VAL_START), pd.Timestamp(cfg.TEST_START)
    t = d["time"]
    return d[t < b1 - gap], d[(t >= b1) & (t < b2 - gap)], d[t >= b2]
