"""Shared settings for data collection.

Site coordinates are approximate and must be verified with check_metocean.py
before the full run. Sites should sit at least 50 km apart because the
Open-Meteo marine data is a coarse grid and nearby points can share values.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
JSON_DIR = RAW_DIR / "json"
CSV_PATH = RAW_DIR / "north_sea_metocean.csv"
LOG_PATH = RAW_DIR / "collection_log.csv"

SITES = {
    "MorayEast": (58.18, -2.72),
    "EOWDC": (57.22, -1.97),
    "Seagreen": (56.60, -1.80),
    "Teesside": (54.70, -1.00),
    "DoggerBankA": (54.80, 2.00),
    "HornseaOne": (53.90, 1.80),
    "Dudgeon": (53.25, 1.40),
    "GreaterGabbard": (51.90, 2.00),
}

FIRST_YEAR = 2020
LAST_DATE = "2026-11-10"

ATM_URL = "https://archive-api.open-meteo.com/v1/archive"
MAR_URL = "https://marine-api.open-meteo.com/v1/marine"
ATM_VARS = (
    "wind_speed_10m,wind_speed_100m,wind_gusts_10m,wind_direction_100m,"
    "temperature_2m,surface_pressure"
)
MAR_VARS = "wave_height,wave_period,wave_direction"
MAR_MODEL = "era5_ocean"

USER_AGENT = "CS5079-student-project (University of Aberdeen coursework)"
REQUEST_DELAY_S = 1.5
MAX_RETRIES = 5
TIMEOUT_S = 90
