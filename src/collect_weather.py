"""Collect hourly weather and sea-state data for North Sea wind sites.

Data source: Open-Meteo historical weather API and marine API
(https://open-meteo.com, free for non-commercial use, attribution required).

The collector is resumable. Every raw JSON response is saved under
data/raw/json, and a file that already exists is not requested again. Each
request is appended to data/raw/collection_log.csv, which is the evidence of
collection for the report.

Usage:
    python src/collect_weather.py
    python src/collect_weather.py --sites EOWDC Seagreen --first-year 2025
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def log_request(site, year, kind, status, rows, note=""):
    """Append one line to the collection log, creating the file with a header if needed."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    new = not cfg.LOG_PATH.exists()
    with open(cfg.LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "site", "year", "kind", "status", "rows", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    site, year, kind, status, rows, note])


def fetch(url, lat, lon, start, end, hourly, atmos):
    """Request one block of hourly data, retrying with growing waits on errors.

    Returns the parsed JSON. Raises RuntimeError after cfg.MAX_RETRIES failures.
    """
    params = {"latitude": lat, "longitude": lon, "start_date": start,
              "end_date": end, "hourly": hourly, "timezone": "UTC"}
    if atmos:
        params["wind_speed_unit"] = "ms"
    else:
        params["models"] = cfg.MAR_MODEL
    last = "no response"
    for k in range(cfg.MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=cfg.TIMEOUT_S,
                             headers={"User-Agent": cfg.USER_AGENT})
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"{url} failed after {cfg.MAX_RETRIES} tries: {last}")


def get_block(site, lat, lon, year, kind):
    """Return the raw JSON for one site, year and data kind, from disk or the API."""
    cfg.JSON_DIR.mkdir(parents=True, exist_ok=True)
    tag = kind if kind == "atm" else f"{kind}_{cfg.MAR_MODEL}"
    path = cfg.JSON_DIR / f"{site}_{year}_{tag}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f), False
    start = f"{year}-01-01"
    end = min(f"{year}-12-31", cfg.LAST_DATE)
    if kind == "atm":
        obj = fetch(cfg.ATM_URL, lat, lon, start, end, cfg.ATM_VARS, True)
    else:
        obj = fetch(cfg.MAR_URL, lat, lon, start, end, cfg.MAR_VARS, False)
    with open(path, "w") as f:
        json.dump(obj, f)
    time.sleep(cfg.REQUEST_DELAY_S)
    return obj, True


def collect(sites=None, first_year=cfg.FIRST_YEAR):
    """Download all blocks, merge them and write the combined CSV.

    A block that still fails after retries is logged and skipped, so one bad
    request does not lose the others. Run the script again to fill the gaps.
    """
    sites = sites or cfg.SITES
    last_year = int(cfg.LAST_DATE[:4])
    frames = []
    for site, (lat, lon) in sites.items():
        for year in range(first_year, last_year + 1):
            try:
                atm, fresh_a = get_block(site, lat, lon, year, "atm")
                mar, fresh_m = get_block(site, lat, lon, year, "mar")
            except RuntimeError as exc:
                log_request(site, year, "both", "failed", 0, str(exc)[:200])
                print(f"FAILED {site} {year}: {exc}")
                continue
            a = pd.DataFrame(atm["hourly"])
            m = pd.DataFrame(mar["hourly"])
            df = a.merge(m, on="time", how="left")
            df["site"] = site
            df["grid_lat"] = mar.get("latitude")
            df["grid_lon"] = mar.get("longitude")
            frames.append(df)
            status = "downloaded" if (fresh_a or fresh_m) else "cached"
            log_request(site, year, "both", status, len(df))
            print(f"{site} {year}: {len(df)} rows ({status})")
    if not frames:
        raise SystemExit("No data collected.")
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values(["site", "time"]).reset_index(drop=True)
    out.to_csv(cfg.CSV_PATH, index=False)
    print(f"Saved {len(out)} rows to {cfg.CSV_PATH}")
    return out


def main():
    """Parse command line options and run the collection."""
    p = argparse.ArgumentParser(
        description="Collect hourly weather and sea-state data for North Sea wind sites."
    )
    p.add_argument("--sites", nargs="*", help="subset of site names from config.SITES")
    p.add_argument("--first-year", type=int, default=cfg.FIRST_YEAR)
    args = p.parse_args()
    sites = cfg.SITES
    if args.sites:
        unknown = [s for s in args.sites if s not in cfg.SITES]
        if unknown:
            raise SystemExit(f"Unknown sites: {unknown}. Choose from {list(cfg.SITES)}")
        sites = {s: cfg.SITES[s] for s in args.sites}
    collect(sites, args.first_year)


if __name__ == "__main__":
    main()
