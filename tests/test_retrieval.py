"""Offline tests of the retrievers and the benchmark, using a fake encoder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import benchmark_retrieval as br
import retrieval as rt

TEXTS = {
    "d1": "crew transfer vessel wave height limit turbine access catamaran fender",
    "d2": "crane lifting operation high wind dropped object deck",
    "d3": "helicopter landing platform gusts helideck",
    "d4": "fishing vessel capsized trawler stability",
}
DIM = 128


def fake_encode(texts):
    """Hashed bag-of-words vectors, normalised. Stands in for MiniLM."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in rt.tokenize(t):
            out[i, sum(map(ord, tok)) % DIM] += 1.0
        out[i] /= np.linalg.norm(out[i]) + 1e-9
    return out


def make_retrievers():
    chunks = [dict(chunk_id=f"{d}-0", doc_id=d, title=f"Title {d}", url=f"https://x/{d}", core=d != "d4", text=t)
              for d, t in TEXTS.items()]
    return rt.Retrievers(chunks, fake_encode([c["text"] for c in chunks]), fake_encode)


def test_rankings():
    R = make_retrievers()
    assert R.docs("bm25", "wave height limit for crew transfer")[0] == "d1"
    assert R.docs("dense", "crane lifting in high wind")[0] == "d2"
    assert R.docs("hybrid", "helicopter gusts")[0] == "d3"
    assert len(R.docs("hybrid", "vessel")) == len(set(R.docs("hybrid", "vessel")))
    assert "d4" not in R.docs("bm25", "vessel", core_only=True)
    top = R.top_chunks("bm25", "crane", k=2)
    assert top[0]["doc_id"] == "d2" and len(top) == 2


def test_pool_and_evaluate(tmp_path):
    R = make_retrievers()
    q = tmp_path / "queries.csv"
    pd.DataFrame({"qid": ["Q1", "Q2"], "author": "t", "complexity": ["simple", "complex"],
                  "query": ["wave height limit crew transfer", "crane lifting high wind"]}).to_csv(q, index=False)
    pool = br.make_pool(R, q, tmp_path / "pool.csv", top=2)
    assert {"qid", "doc_id", "found_by", "relevant"} <= set(pool.columns)
    marked = pd.read_csv(tmp_path / "pool.csv")
    marked["relevant"] = [1 if (r.qid == "Q1" and r.doc_id == "d1") or (r.qid == "Q2" and r.doc_id == "d2") else 0
                          for r in marked.itertuples()]
    marked.to_csv(tmp_path / "pool.csv", index=False)
    res = br.evaluate(R, tmp_path / "pool.csv", queries_path=q)
    assert res["queries_scored"] == 2
    assert set(res["by_complexity"]) == {"simple", "complex"}
    assert res["by_complexity"]["complex"]["bm25"]["queries"] == 1
    for m in rt.METHODS:
        assert res["methods"][m]["hit_at_3"] == 1.0 and res["methods"][m]["mrr"] == 1.0
