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

7. `python src/build_chunks.py` cleans the documents and writes `data/processed/chunks.jsonl`: 200-word chunks with a 40-word overlap, a `core` flag for offshore-operations documents, and the source URL on every chunk.

8. `python src/check_labels.py` prints class shares for each threshold set in `src/config.py` (training years, by year, by site, Critical hours in the test year).
9. `python src/train_models.py` builds labels and features, tunes Logistic Regression, Random Forest and XGBoost on the validation year, applies the selection rule, scores the 2026 test year once, and writes `models/best_model.joblib`, `scaler.joblib`, `features.joblib`, `feature_medians.joblib`, `results.json` and `data/processed/test_sample.csv`. Use `--thresholds guide_default --out-dir models/sensitivity_guide_default` for the sensitivity run, which leaves the main artefacts untouched, and `--quick` for a smoke test.

10. `python src/build_embeddings.py` embeds every chunk with MiniLM and saves `data/processed/embeddings.npy` (float16). It takes several minutes on a laptop CPU.
11. Retrieval benchmark. Each member writes 10 queries into `data/benchmark/queries.csv` (columns `qid,author,query`, see `queries_template.csv`). Use different wording and synonyms, because that is where BM25 and dense search differ. Then:
    - `python src/benchmark_retrieval.py pool` writes `data/benchmark/pool.csv`, the top 5 documents from each of BM25, dense and hybrid search for every query.
    - Open `pool.csv`, read each document through its `url` and fill the `relevant` column with 1 (the document answers the query) or 0. Agree the rule in the group first and mark independently when possible.
    - `python src/benchmark_retrieval.py evaluate` prints hit@3, recall@3, MRR and mean query time per method and writes `data/benchmark/retrieval_results.json`. Add `--core-only` to restrict search to core offshore documents.

More steps are added as the project progresses.

## Notes

- Site coordinates in `src/config.py` are approximate. Sites must be offshore and at least 50 km apart.
- Do not run the collector in a loop. The 1.5 second delay between requests is deliberate.
- Seeds are fixed where randomness is used, so results can be reproduced.
