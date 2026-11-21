"""Three retrievers over the chunk file behind one interface: BM25, dense and hybrid.

bm25    keyword ranking with BM25 Okapi over lower-cased word tokens
dense   cosine similarity between a query embedding and precomputed chunk embeddings
hybrid  reciprocal rank fusion of the BM25 and dense rankings

Each retriever ranks chunks. docs() turns a chunk ranking into a ranking of unique
documents, which is what the benchmark scores. top_chunks() returns the best chunk of
each of the top documents, which is what the app shows as cited passages.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

METHODS = ["bm25", "dense", "hybrid"]
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split())


def tokenize(text):
    """Lower-case word tokens without common stop words."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_chunks(path=None):
    """Read chunks.jsonl into a list of dicts."""
    with open(path or cfg.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_encoder():
    """Return a function that encodes a list of texts into normalised MiniLM vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.EMBED_MODEL)
    return lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


class Retrievers:
    """BM25, dense and hybrid search over one list of chunks."""

    def __init__(self, chunks, embeddings, encode, rrf_k=cfg.RRF_K, depth=cfg.RRF_DEPTH):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings. Rebuild the embeddings.")
        self.chunks = chunks
        self.emb = np.asarray(embeddings, dtype=np.float32)
        self.encode = encode
        self.rrf_k = rrf_k
        self.depth = depth
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.core = np.array([bool(c.get("core", False)) for c in chunks])

    def bm25_rank(self, query):
        """Chunk indices ordered by BM25 score."""
        return np.argsort(-self.bm25.get_scores(tokenize(query)), kind="stable")

    def dense_rank(self, query):
        """Chunk indices ordered by cosine similarity to the query embedding."""
        qv = np.asarray(self.encode([query])[0], dtype=np.float32)
        return np.argsort(-(self.emb @ qv), kind="stable")

    def hybrid_rank(self, query):
        """Reciprocal rank fusion of the top `depth` chunks from BM25 and dense search."""
        score = {}
        for ranking in (self.bm25_rank(query)[: self.depth], self.dense_rank(query)[: self.depth]):
            for pos, idx in enumerate(ranking):
                score[int(idx)] = score.get(int(idx), 0.0) + 1.0 / (self.rrf_k + pos + 1)
        return np.array(sorted(score, key=score.get, reverse=True))

    def rank(self, method, query, core_only=False):
        """Chunk indices for the chosen method, optionally limited to core offshore documents."""
        ranked = {"bm25": self.bm25_rank, "dense": self.dense_rank, "hybrid": self.hybrid_rank}[method](query)
        if core_only:
            ranked = ranked[self.core[ranked]]
        return ranked

    def docs(self, method, query, core_only=False):
        """Unique document ids in ranked order."""
        seen, out = set(), []
        for i in self.rank(method, query, core_only):
            d = self.chunks[int(i)]["doc_id"]
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def top_chunks(self, method, query, k=3, core_only=False):
        """The best chunk of each of the top k documents, in ranked order."""
        seen, out = set(), []
        for i in self.rank(method, query, core_only):
            c = self.chunks[int(i)]
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                out.append(c)
                if len(out) == k:
                    break
        return out


def load_retrievers():
    """Load chunks and embeddings from disk and build the retrievers."""
    if not cfg.EMBEDDINGS_PATH.exists():
        raise SystemExit(f"{cfg.EMBEDDINGS_PATH} not found. Run python src/build_embeddings.py first.")
    return Retrievers(load_chunks(), np.load(cfg.EMBEDDINGS_PATH), load_encoder())
