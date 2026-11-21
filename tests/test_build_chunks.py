"""Offline test of the chunker."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_chunks as bc
import config as cfg


def make_doc(doc_id, title, text):
    return {"doc_id": doc_id, "source": "MAIB", "doc_type": "maib_report", "url": f"https://x/{doc_id}",
            "title": title, "date": "", "n_words": len(text.split()), "has_pdf_text": False, "scraped_at": "", "text": text}


def test_clean_text_removes_noise():
    lines = ["Real sentence one.", "12", "Page 3 of 40", "Contents ........ 5", "Running header"] * 1
    text = "\n".join(lines + ["Running header"] * 5)
    out = bc.clean_text(text)
    assert out == "Real sentence one."


def test_split_words_overlap():
    words = [str(i) for i in range(450)]
    pieces = bc.split_words(words, 200, 40)
    assert [len(p) for p in pieces] == [200, 200, 130]
    assert pieces[1][:40] == pieces[0][-40:]
    assert bc.split_words(["a"] * 30, 200, 40) == [["a"] * 30]
    assert bc.split_words([], 200, 40) == []


def test_build(tmp_path):
    body = " ".join(f"word{i}" for i in range(500))
    docs = [make_doc("aaa", "Wind farm crew transfer", body),
            make_doc("bbb", "Fishing vessel capsize", body.replace("word", "term")),
            make_doc("ccc", "Duplicate of first", body)]
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    df = bc.build(src, tmp_path / "chunks.jsonl")
    assert set(df["doc_id"]) == {"aaa", "bbb"}
    assert df[df["doc_id"] == "aaa"]["core"].all() and not df[df["doc_id"] == "bbb"]["core"].any()
    assert df["chunk_id"].is_unique and df["chunk_id"].iloc[0] == "aaa-0000"
    assert (df["n_words"] >= cfg.MIN_CHUNK_WORDS).all()
    assert len((tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()) == len(df)
