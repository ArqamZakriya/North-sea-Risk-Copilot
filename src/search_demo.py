"""Print the top three documents from each retriever for one query.

Usage:
    python src/search_demo.py "crew transfer in heavy swell"
    python src/search_demo.py --core "crew transfer in heavy swell"

The --core option searches only chunks from documents flagged as offshore operations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval


def main():
    """Run the query through BM25, dense and hybrid search and print the results."""
    args = [a for a in sys.argv[1:] if a != "--core"]
    core_only = len(args) != len(sys.argv) - 1
    if not args:
        raise SystemExit('Usage: python src/search_demo.py [--core] "your query"')
    query = " ".join(args)
    R = retrieval.load_retrievers()
    for method in retrieval.METHODS:
        print(f"\n== {method} ==")
        for c in R.top_chunks(method, query, k=3, core_only=core_only):
            print(f"- {c['title']} [{c['source']}]\n  {c['url']}\n  {c['text'][:200]}...")


if __name__ == "__main__":
    main()
