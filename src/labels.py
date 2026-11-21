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


def add_extended_features(g):
    """Add longer history, recent extremes, wave steepness and direction features for one site."""
    g = g.copy()
    for c in ("wind_speed_100m", "wave_height"):
        for k in (1, 3, 6, 12):
            g[f"{c}_lag_{k}h"] = g[c].shift(k)
        g[f"{c}_mean_3h"] = g[c].rolling(3).mean()
    for c in ROLL_COLUMNS:
        for w in (6, 12, 24):
            g[f"{c}_max_{w}h"] = g[c].rolling(w).max()
    g["wind_speed_100m_delta_3h"] = g["wind_speed_100m"].diff(3)
    g["wind_speed_100m_delta_6h"] = g["wind_speed_100m"].diff(6)
    g["wave_delta_3h"] = g["wave_height"].diff(3)
    g["wave_delta_6h"] = g["wave_height"].diff(6)
    g["pressure_delta_6h"] = g["surface_pressure"].diff(6)
    g["pressure_delta_12h"] = g["surface_pressure"].diff(12)
    g["wave_steepness"] = g["wave_height"] / (g["wave_period"] ** 2 + 0.1)
    wd, vd = np.deg2rad(g["wind_direction_100m"]), np.deg2rad(g["wave_direction"])
    g["wind_dir_sin"], g["wind_dir_cos"] = np.sin(wd), np.cos(wd)
    g["wave_dir_sin"], g["wave_dir_cos"] = np.sin(vd), np.cos(vd)
    g["wind_wave_align"] = np.cos(wd - vd)
    g["site_lat"], g["site_lon"] = g["grid_lat"], g["grid_lon"]
    return g


def cross_site_features(frames):
    """For each site, regional wind and wave summaries from the other sites at the same hour and their 3 hour change."""
    idx = sorted(set().union(*[f.index for f in frames.values()]))
    wind = pd.DataFrame({s: f["wind_speed_100m"].reindex(idx) for s, f in frames.items()})
    wave = pd.DataFrame({s: f["wave_height"].reindex(idx) for s, f in frames.items()})
    out = {}
    for s in wind.columns:
        others = [c for c in wind.columns if c != s]
        wm, vm = wind[others].mean(axis=1), wave[others].mean(axis=1)
        out[s] = pd.DataFrame({
            "others_wind_mean": wm, "others_wind_max": wind[others].max(axis=1),
            "others_wave_mean": vm, "others_wave_max": wave[others].max(axis=1),
            "others_wind_mean_delta_3h": wm.diff(3), "others_wave_mean_delta_3h": vm.diff(3)})
    return out


def build(df, th=None, feature_set="base"):
    """Features and labels for every site. Rows without a full feature history or target window are dropped.

    feature_set "base" uses each site's own recent weather. "extended" adds longer history, recent maxima,
    direction and steepness features, the site position, and summaries of the other sites at the same hour.
    Every feature uses only data from the current hour and earlier.
    """
    th = th or cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]
    extended = feature_set == "extended"
    frames = {site: regular(g.assign(site=site)) for site, g in df.groupby("site")}
    cross = cross_site_features(frames) if extended and len(frames) > 1 else {}
    parts = []
    for site, g in frames.items():
        g = add_features(g)
        if extended:
            g = add_extended_features(g)
            if site in cross:
                g = g.join(cross[site])
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
