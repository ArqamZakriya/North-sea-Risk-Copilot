"""Clean the scraped documents and split them into overlapping word chunks.

Cleaning removes page-number lines, table-of-contents dot leaders and short lines
that repeat many times in one document (running headers and footers). Each document
is then cut into windows of cfg.CHUNK_WORDS words that overlap by cfg.CHUNK_OVERLAP
words. Chunks shorter than cfg.MIN_CHUNK_WORDS are dropped unless they are the only
chunk of a document, and chunks with identical text are kept once.

Every chunk carries a `core` flag that marks documents about offshore operations,
so the retrieval step can compare the full corpus with the focused subset.

Output: data/processed/chunks.jsonl

Usage:
    python src/build_chunks.py
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

PAGE_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
LEADER_RE = re.compile(r"\.{4,}|(\. ){4,}")


def clean_text(text):
    """Remove page numbers, contents leaders and repeated header lines, then join with spaces."""
    lines = text.splitlines()
    counts = Counter(line for line in lines if len(line) < 100)
    kept = []
    for line in lines:
        if PAGE_RE.match(line) or LEADER_RE.search(line):
            continue
        if counts.get(line, 0) >= cfg.REPEAT_LINE_LIMIT:
            continue
        kept.append(line)
    return " ".join(kept)


def split_words(words, size, overlap):
    """Cut a word list into windows of `size` words that overlap by `overlap` words."""
    step = size - overlap
    pieces = []
    for start in range(0, max(len(words) - overlap, 1), step):
        piece = words[start:start + size]
        if piece:
            pieces.append(piece)
        if start + size >= len(words):
            break
    return pieces


def is_core(title, text, url=""):
    """True for documents about offshore operations.

    HSE pages under /offshore/ are core. Other documents are core when a strong offshore term
    appears in the title or in the first cfg.CORE_HEAD_WORDS words, which is where a report
    states what it is about. Mentions deep inside a long report do not count.
    """
    if "hse.gov.uk/offshore/" in url:
        return True
    head = f"{title} {' '.join(text.split()[:cfg.CORE_HEAD_WORDS])}"
    return re.search(cfg.CORE_PATTERN, head, re.I) is not None


def build(docs_path=None, out_path=None):
    """Chunk every document and write the chunks file. Returns the chunks as a DataFrame."""
    docs_path = docs_path or cfg.DOCS_PATH
    out_path = out_path or cfg.CHUNKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            text = clean_text(doc["text"])
            core = is_core(doc["title"], text, doc["url"])
            pieces = split_words(text.split(), cfg.CHUNK_WORDS, cfg.CHUNK_OVERLAP)
            for i, piece in enumerate(pieces):
                if len(piece) < cfg.MIN_CHUNK_WORDS and len(pieces) > 1:
                    continue
                chunk = " ".join(piece)
                digest = hashlib.sha1(chunk.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                rows.append({
                    "chunk_id": f"{doc['doc_id']}-{i:04d}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "doc_type": doc["doc_type"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "core": core,
                    "n_words": len(piece),
                    "text": chunk,
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return pd.DataFrame(rows)


def summarise(df):
    """Print chunk counts and sizes."""
    print("Chunks:", len(df))
    print("Documents with chunks:", df["doc_id"].nunique())
    print("\nChunks by source and core flag:")
    print(df.groupby(["source", "core"]).size().to_string())
    print("\nWords per chunk:")
    print(df["n_words"].describe().round(1).to_string())
    print(f"\nCore share of chunks: {df['core'].mean():.1%}")
    docs = df.drop_duplicates("doc_id")
    print(f"Core documents: {int(docs['core'].sum())} of {len(docs)}")
    print(docs.groupby("source")["core"].agg(core="sum", total="size").to_string())
    for flag, label in ((True, "core"), (False, "not core")):
        sample = docs[docs["core"] == flag]
        print(f"\nRandom {label} titles (20):")
        for t in sample.sample(min(20, len(sample)), random_state=cfg.SEED)["title"]:
            print(" -", t[:110])


if __name__ == "__main__":
    if not cfg.DOCS_PATH.exists():
        raise SystemExit(f"{cfg.DOCS_PATH} not found. Run scrape_corpus.py first.")
    summarise(build())
