"""Offline test of the corpus scraper using fake HTTP responses."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import check_corpus
import config as cfg
import scrape_corpus as sc

BODY = "Offshore crane lifting operation during heavy weather caused a dropped object. " * 12

HSE_SEED = """<html><body><a href="/safetybulletins/offshore-cranes.htm">Cranes</a>
<a href="/safetybulletins/ups-systems.htm">UPS</a>
<a href="https://webarchive.nationalarchives.gov.uk/x/old.htm">Old</a>
<a href="/offshore/notices/on_01.htm">Notice</a></body></html>"""

HSE_PAGE = f"<html><body><nav>menu</nav><main><h1>Offshore cranes</h1><p>{BODY}</p><p>Issued 14 January 2022</p></main></body></html>"
HSE_IRRELEVANT = "<html><body><main><h1>UPS</h1><p>" + "Battery maintenance schedule for server rooms. " * 30 + "</p></main></body></html>"
MAIB_PAGE = f"""<html><body><main><h1>Wind farm transfer incident</h1><p>{BODY}</p>
<a href="https://assets.publishing.service.gov.uk/media/abc/report.pdf">PDF</a></main></body></html>"""


class FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status=200, text="", content=b"", payload=None):
        self.status_code = status
        self.text = text
        self.content = content
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Serves canned responses by URL and counts requests."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("robots.txt"):
            return FakeResp(404)
        if "api/search.json" in url:
            start = params["start"]
            results = [] if start else [{"title": "Wind farm transfer incident", "link": "/maib-reports/wind-farm-transfer",
                                         "public_timestamp": "2021-05-04T10:00:00Z"}]
            return FakeResp(payload={"results": results, "total": len(results)})
        if url == "https://www.hse.gov.uk/offshore/notices/index.htm":
            return FakeResp(text=HSE_SEED)
        if url.startswith("https://www.hse.gov.uk/offshore/") and "notices/on_01" in url:
            return FakeResp(404)
        if url.endswith("offshore-cranes.htm"):
            return FakeResp(text=HSE_PAGE)
        if url.endswith("ups-systems.htm"):
            return FakeResp(text=HSE_IRRELEVANT)
        if url.endswith("wind-farm-transfer"):
            return FakeResp(text=MAIB_PAGE)
        if url.endswith("report.pdf"):
            return FakeResp(content=b"%PDF")
        return FakeResp(404)


def test_scrape_and_check(tmp_path, monkeypatch):
    """Scrape HSE and MAIB from fakes, rerun for resumability, then run the checker."""
    monkeypatch.setattr(cfg, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "DOCS_PATH", tmp_path / "documents.jsonl")
    monkeypatch.setattr(cfg, "SCRAPE_LOG_PATH", tmp_path / "scrape_log.csv")
    monkeypatch.setattr(cfg, "HSE_SEEDS", ["https://www.hse.gov.uk/offshore/notices/index.htm"])
    monkeypatch.setattr(cfg, "MAIB_QUERIES", ["wind farm"])
    monkeypatch.setattr(sc, "pdf_to_text", lambda content: "Report conclusion on gangway transfer in swell. " * 20)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)

    session = FakeSession()
    fetcher = sc.Fetcher(delay=0, session=session)
    known = sc.load_known()
    assert sc.scrape_hse(fetcher, known, 0) == 1
    assert sc.scrape_maib(fetcher, known, 0, True) == 1

    docs = pd.read_json(cfg.DOCS_PATH, lines=True)
    assert set(docs["source"]) == {"HSE", "MAIB"}
    hse = docs[docs["source"] == "HSE"].iloc[0]
    assert hse["title"] == "Offshore cranes" and hse["date"].strftime("%Y-%m-%d") == "2022-01-14"
    assert "menu" not in hse["text"]
    maib = docs[docs["source"] == "MAIB"].iloc[0]
    assert maib["has_pdf_text"] and maib["date"].strftime("%Y-%m-%d") == "2021-05-04"

    log = pd.read_csv(cfg.SCRAPE_LOG_PATH)
    assert {"saved", "dropped_not_relevant", "skipped_external", "failed"} <= set(log["status"])

    before = len(session.calls)
    known = sc.load_known()
    assert sc.scrape_hse(sc.Fetcher(delay=0, session=session), known, 0) == 0
    assert len(known) == 2
    assert all("offshore-cranes" not in c for c in session.calls[before:])

    assert len(check_corpus.check()) == 2


def test_robots_block(tmp_path, monkeypatch):
    """A host whose robots.txt disallows everything is not fetched."""
    session = FakeSession()
    session.get = lambda url, params=None, timeout=None: FakeResp(200, text="User-agent: *\nDisallow: /")
    fetcher = sc.Fetcher(delay=0, session=session)
    resp, note = fetcher.fetch("https://www.gov.uk/api/search.json")
    assert resp is None and note == "blocked by robots.txt"
