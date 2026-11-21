"""Build and score the retrieval benchmark.

Step 1, pool:  python src/benchmark_retrieval.py pool [--core-only]
    Reads data/benchmark/queries.csv (columns qid, author, query, and optionally complexity). For each query it takes the
    top cfg.POOL_TOP documents from BM25, dense and hybrid search and writes the union to
    data/benchmark/pool.csv with an empty `relevant` column.

Step 2, by hand:
    Open pool.csv, read each document through its url, and enter 1 (answers the query) or 0.
    Pooling the three methods together avoids favouring one of them.

Step 3, evaluate:  python src/benchmark_retrieval.py evaluate [--core-only]
    Scores each method against the marked pool and writes data/benchmark/retrieval_results.json.

hit@3 is the share of queries with at least one relevant document in the top 3. recall@3 is the
average share of a query's relevant documents found in the top 3. mrr is the mean reciprocal rank
of the first relevant document. Documents outside the pool count as not relevant, so the
benchmark is small and self-made, which is a limit to state in the report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import retrieval


def make_pool(R, queries_path=None, pool_path=None, top=None, core_only=False):
    """Write the pooled candidate documents for every query. Returns the pool DataFrame."""
    queries = pd.read_csv(queries_path or cfg.QUERIES_PATH)
    top = top or cfg.POOL_TOP
    rows = []
    for q in queries.itertuples():
        found, best = {}, {}
        for m in retrieval.METHODS:
            for chunk in R.top_chunks(m, q.query, k=top, core_only=core_only):
                found.setdefault(chunk["doc_id"], []).append(m)
                best.setdefault(chunk["doc_id"], chunk)
        for doc_id, methods in found.items():
            c = best[doc_id]
            rows.append(dict(qid=q.qid, query=q.query, doc_id=doc_id, title=c["title"], url=c["url"],
                             found_by="+".join(methods), snippet=c["text"][:300], relevant=""))
    pool = pd.DataFrame(rows)
    out = pool_path or cfg.POOL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)
    return pool


def aggregate(rows):
    """Mean hit@3, recall@3, MRR and query time over per-query result rows."""
    return dict(hit_at_3=float(np.mean([r["hit"] for r in rows])), recall_at_3=float(np.mean([r["recall"] for r in rows])),
                mrr=float(np.mean([r["rr"] for r in rows])), mean_query_ms=float(np.mean([r["ms"] for r in rows])),
                queries=len(rows))


def evaluate(R, pool_path=None, k=3, core_only=False, queries_path=None):
    """Score every method against the marked pool. Returns a results dict.

    If the queries file has a `complexity` column, results are also reported for each value of it.
    """
    pool = pd.read_csv(pool_path or cfg.POOL_PATH)
    missing = pool["relevant"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} pool rows have no relevance mark. Enter 1 or 0 in the `relevant` column.")
    rel = {q: set(g.loc[g["relevant"] == 1, "doc_id"]) for q, g in pool.groupby("qid")}
    queries = pool.drop_duplicates("qid").set_index("qid")["query"]
    usable = {q: r for q, r in rel.items() if r}
    skipped = sorted(set(rel) - set(usable))
    groups = {}
    qfile = Path(queries_path or cfg.QUERIES_PATH)
    if qfile.exists():
        qdf = pd.read_csv(qfile)
        if "complexity" in qdf.columns:
            groups = dict(zip(qdf["qid"], qdf["complexity"]))
    results, by_group = {}, {}
    for m in retrieval.METHODS:
        R.docs(m, next(iter(queries)), core_only)
        rows = []
        for q, relevant in usable.items():
            t0 = time.perf_counter()
            docs = R.docs(m, queries[q], core_only)
            ms = (time.perf_counter() - t0) * 1000
            flags = [d in relevant for d in docs]
            rows.append(dict(qid=q, hit=float(any(flags[:k])), recall=len(relevant & set(docs[:k])) / len(relevant),
                             rr=next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0), ms=ms))
        results[m] = aggregate(rows)
        for g in sorted({groups[r["qid"]] for r in rows if r["qid"] in groups}):
            by_group.setdefault(g, {})[m] = aggregate([r for r in rows if groups.get(r["qid"]) == g])
    out = dict(queries_scored=len(usable), queries_without_relevant=skipped, pool_rows=len(pool),
               relevant_documents=int(sum(len(r) for r in usable.values())), core_only=core_only, methods=results)
    if by_group:
        out["by_complexity"] = by_group
    return out


def main():
    """Run the pool or evaluate step."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["pool", "evaluate"])
    p.add_argument("--core-only", action="store_true")
    a = p.parse_args()
    if not cfg.QUERIES_PATH.exists():
        raise SystemExit(f"{cfg.QUERIES_PATH} not found. Create it with columns qid, author, query.")
    R = retrieval.load_retrievers()
    if a.step == "pool":
        pool = make_pool(R, core_only=a.core_only)
        print(f"Wrote {len(pool)} candidate rows for {pool['qid'].nunique()} queries to {cfg.POOL_PATH}")
        print("Now fill the `relevant` column with 1 or 0, then run: python src/benchmark_retrieval.py evaluate")
        return
    res = evaluate(R, core_only=a.core_only)
    cfg.RETRIEVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.RETRIEVAL_RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(pd.DataFrame(res["methods"]).T.round(3).to_string())
    for group, table in res.get("by_complexity", {}).items():
        print(f"\n{group} queries:")
        print(pd.DataFrame(table).T.round(3).to_string())
    print(f"\nQueries scored: {res['queries_scored']}, relevant documents: {res['relevant_documents']}")
    if res["queries_without_relevant"]:
        print("Queries with no relevant document marked (excluded):", res["queries_without_relevant"])


if __name__ == "__main__":
    main()
