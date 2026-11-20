# North Sea Offshore Operations Risk & Safety Copilot

CS5079 Applied AI, CA1 Task 1 (University of Aberdeen). Forecasts hazardous North Sea working conditions for the next 6 hours and retrieves matching UK HSE and MAIB safety guidance with citations.

Weather and sea-state data: Open-Meteo (https://open-meteo.com), used under their free non-commercial terms with attribution.

## Setup

Python 3.11 recommended.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run order

1. `python src/collect_weather.py` downloads hourly data for every site in `src/config.py` and writes `data/raw/north_sea_metocean.csv`. Raw JSON goes to `data/raw/json/`. Every request is logged in `data/raw/collection_log.csv`. The script is resumable: run it again after a failure and it skips files already saved.
2. `python src/check_metocean.py` prints row counts, missing values, wave grid cells and site correlations. Fix every warning before moving on.
3. Offline self-test of the collector: `python -m pytest -q tests`.

4. `python src/diagnose_waves.py` shows where wave data is missing, if the checker reports gaps. Waves come from the Open-Meteo ERA5-Ocean model (0.5 degree grid), which covers the whole period.
5. `python src/scrape_corpus.py` builds the safety corpus from MAIB (GOV.UK Search API, report pages and PDFs) and HSE (offshore notices, safety bulletins, information sheets). It checks robots.txt for every host, waits 2 seconds between requests to the same host and writes `data/raw/corpus/documents.jsonl` and `data/raw/corpus/scrape_log.csv`. It is resumable. Use `--source hse` or `--source maib` to run one source, and `--limit 5` for a trial.
6. `python src/check_corpus.py` prints document counts, words, duplicates, log outcomes and an estimate of chunk count against the targets.

More steps are added as the project progresses.

## Notes

- Site coordinates in `src/config.py` are approximate. Sites must be offshore and at least 50 km apart.
- Do not run the collector in a loop. The 1.5 second delay between requests is deliberate.
- Seeds are fixed where randomness is used, so results can be reproduced.
