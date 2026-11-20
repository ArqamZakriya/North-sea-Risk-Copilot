"""Summarise the scraped corpus and compare it with the project targets.

Prints documents and words per source, short or duplicated documents, how often
each URL outcome appears in the scrape log, and an estimate of chunk count.

Usage:
    python src/check_corpus.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

TARGET_DOCS = 150
TARGET_CHUNKS = 800
WORDS_PER_CHUNK = 200


def check():
    """Print the corpus summary and return the DataFrame of documents."""
    with open(cfg.DOCS_PATH, encoding="utf-8") as f:
        docs = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    docs["text_hash"] = docs["text"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())

    print("Documents:", len(docs))
    print("\nBy source and type:")
    print(docs.groupby(["source", "doc_type"]).agg(docs=("url", "size"), words=("n_words", "sum"),
                                                    median_words=("n_words", "median")).to_string())

    total_words = int(docs["n_words"].sum())
    est_chunks = total_words // WORDS_PER_CHUNK
    print(f"\nTotal words: {total_words}")
    print(f"Estimated chunks at {WORDS_PER_CHUNK} words: {est_chunks}")

    dup = docs.duplicated("text_hash").sum()
    print("Duplicate texts:", dup)

    dated = docs[docs["date"] != ""]
    if len(dated):
        print("Date range:", dated["date"].min(), "to", dated["date"].max())

    print("\nShortest five documents:")
    print(docs.nsmallest(5, "n_words")[["source", "n_words", "title"]].to_string(index=False))

    if cfg.SCRAPE_LOG_PATH.exists():
        log = pd.read_csv(cfg.SCRAPE_LOG_PATH)
        print("\nScrape log outcomes:")
        print(log.groupby(["source", "status"]).size().to_string())

    print()
    print(f"Documents: {len(docs)} of {TARGET_DOCS} target")
    print(f"Chunks (estimate): {est_chunks} of {TARGET_CHUNKS} target")
    if len(docs) < TARGET_DOCS or est_chunks < TARGET_CHUNKS:
        print("Below target. Report the real numbers and say why, or add another source.")
    return docs


if __name__ == "__main__":
    if not cfg.DOCS_PATH.exists():
        raise SystemExit(f"{cfg.DOCS_PATH} not found. Run scrape_corpus.py first.")
    check()
