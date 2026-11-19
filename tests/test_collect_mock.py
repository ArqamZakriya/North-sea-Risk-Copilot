"""Offline test of the collector and checker using fake API responses."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import check_metocean
import collect_weather as cw
import config as cfg


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def fake_get(url, params, timeout, headers):
    """Return hourly data shaped like the Open-Meteo responses."""
    times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
    rng = np.random.default_rng(abs(hash((params["latitude"], params["longitude"]))) % 2**32)
    n = len(times)
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    if "archive" in url:
        base = rng.gamma(4, 2, n)
        hourly.update(wind_speed_10m=base, wind_speed_100m=base * 1.3,
                      wind_gusts_10m=base * 1.5, wind_direction_100m=rng.uniform(0, 360, n),
                      temperature_2m=rng.normal(9, 4, n), surface_pressure=rng.normal(1010, 10, n))
        payload = {"latitude": params["latitude"], "longitude": params["longitude"]}
    else:
        hourly.update(wave_height=rng.gamma(3, 0.5, n), wave_period=rng.uniform(4, 12, n),
                      wave_direction=rng.uniform(0, 360, n))
        payload = {"latitude": round(params["latitude"] * 2) / 2,
                   "longitude": round(params["longitude"] * 2) / 2}
    payload["hourly"] = {k: (v if k == "time" else [float(x) for x in v]) for k, v in hourly.items()}
    return FakeResponse(payload)


def test_collect_and_check(tmp_path, monkeypatch):
    """Collect two sites for two years, rerun from cache, then run the checker."""
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path)
    monkeypatch.setattr(cfg, "JSON_DIR", tmp_path / "json")
    monkeypatch.setattr(cfg, "CSV_PATH", tmp_path / "metocean.csv")
    monkeypatch.setattr(cfg, "LOG_PATH", tmp_path / "log.csv")
    monkeypatch.setattr(cfg, "LAST_DATE", "2021-03-31")
    monkeypatch.setattr(cfg, "REQUEST_DELAY_S", 0)
    monkeypatch.setattr(cw.requests, "get", fake_get)

    sites = {"A": (58.0, -2.0), "B": (54.0, 2.0)}
    out = cw.collect(sites, first_year=2020)
    assert set(out["site"]) == {"A", "B"}
    assert len(out) == 2 * (366 + 90) * 24
    assert out.groupby("site")["time"].apply(lambda s: s.is_monotonic_increasing).all()

    calls = []
    monkeypatch.setattr(cw.requests, "get", lambda *a, **k: calls.append(1) or fake_get(*a, **k))
    cw.collect(sites, first_year=2020)
    assert calls == []

    log = pd.read_csv(cfg.LOG_PATH)
    assert {"downloaded", "cached"} <= set(log["status"])
    assert check_metocean.check(cfg.CSV_PATH) == []
