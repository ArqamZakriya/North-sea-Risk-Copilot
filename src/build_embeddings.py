"""Embed every chunk with MiniLM and save the vectors next to the chunk file.

The vectors are stored as float16 to keep the file small and are in the same order as
data/processed/chunks.jsonl. MiniLM reads at most 256 word pieces, so the end of a long
chunk may be ignored. Mention this as a limit in the report.

Usage:
    python src/build_embeddings.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import retrieval


def main():
    """Encode all chunks in batches and write embeddings.npy."""
    chunks = retrieval.load_chunks()
    encode = retrieval.load_encoder()
    texts = [c["text"] for c in chunks]
    parts, t0 = [], time.time()
    for i in range(0, len(texts), 512):
        parts.append(np.asarray(encode(texts[i:i + 512]), dtype=np.float32))
        done = min(i + 512, len(texts))
        print(f"{done}/{len(texts)} chunks, {time.time() - t0:.0f}s", flush=True)
    emb = np.vstack(parts).astype(np.float16)
    cfg.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.EMBEDDINGS_PATH, emb)
    print("Saved", emb.shape, "to", cfg.EMBEDDINGS_PATH, f"({cfg.EMBEDDINGS_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
