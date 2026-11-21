"""Print the top three documents from each retriever for one query.

Usage:
    python src/search_demo.py "crew transfer in heavy swell"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval


def main():
    """Run the query through BM25, dense and hybrid search and print the results."""
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/search_demo.py "your query"')
    query = " ".join(sys.argv[1:])
    R = retrieval.load_retrievers()
    for method in retrieval.METHODS:
        print(f"\n== {method} ==")
        for c in R.top_chunks(method, query, k=3):
            print(f"- {c['title']} [{c['source']}]\n  {c['url']}\n  {c['text'][:200]}...")


if __name__ == "__main__":
    main()
