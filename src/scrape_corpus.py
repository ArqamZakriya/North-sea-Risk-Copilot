"""Scrape the offshore safety corpus from UK public sources.

Sources
    MAIB  Report records from the GOV.UK Search API, then each report page and its PDF.
    HSE   Links found on the offshore notices index, the safety bulletin archive and
          the offshore information sheet, publication and topic pages.

Every URL is checked against the host's robots.txt before it is requested, and a
delay is kept between requests to the same host. A source that robots.txt blocks
is skipped and logged, not worked around.

Outputs
    data/raw/corpus/documents.jsonl   one JSON object per document
    data/raw/corpus/scrape_log.csv    one row per URL attempted, the evidence for the report

The script is resumable: URLs already in documents.jsonl are not requested again.

Usage
    python src/scrape_corpus.py
    python src/scrape_corpus.py --source hse
    python src/scrape_corpus.py --source maib --limit 20 --no-pdfs
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"\b(\d{{1,2}}) ({MONTHS}) ((?:19|20)\d{{2}})\b")


def now_iso():
    """Current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_row(source, url, status, chars=0, note=""):
    """Append one row to the scrape log, writing the header on first use."""
    cfg.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    new = not cfg.SCRAPE_LOG_PATH.exists()
    with open(cfg.SCRAPE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "source", "url", "status", "chars", "note"])
        w.writerow([now_iso(), source, url, status, chars, note])


class Fetcher:
    """HTTP client that obeys robots.txt, rate limits per host and retries failures."""

    def __init__(self, delay=cfg.SCRAPE_DELAY_S, session=None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = cfg.USER_AGENT
        self.delay = delay
        self.robots = {}
        self.crawl_delay = {}
        self.last = {}

    def load_robots(self, base):
        """Fetch and parse robots.txt for one host. A missing file allows everything."""
        rp = robotparser.RobotFileParser()
        try:
            r = self.session.get(base + "/robots.txt", timeout=cfg.TIMEOUT_S)
        except requests.RequestException:
            rp.parse(["User-agent: *", "Disallow: /"])
            return rp
        if r.status_code in (404, 410):
            rp.parse(["User-agent: *", "Allow: /"])
        elif r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            rp.parse(["User-agent: *", "Disallow: /"])
        delay = rp.crawl_delay(cfg.USER_AGENT)
        if delay:
            self.crawl_delay[urlparse(base).netloc] = float(delay)
        return rp

    def allowed(self, url):
        """True when robots.txt of the URL's host permits fetching it."""
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.robots:
            self.robots[base] = self.load_robots(base)
        return self.robots[base].can_fetch(cfg.USER_AGENT, url)

    def fetch(self, url, params=None):
        """Return (response, note). The response is None when blocked or failed."""
        if not self.allowed(url):
            return None, "blocked by robots.txt"
        host = urlparse(url).netloc
        wait = max(self.delay, self.crawl_delay.get(host, 0))
        gap = wait - (time.monotonic() - self.last.get(host, 0))
        if gap > 0:
            time.sleep(gap)
        note = "no response"
        for k in range(3):
            try:
                r = self.session.get(url, params=params, timeout=cfg.TIMEOUT_S)
                self.last[host] = time.monotonic()
                if r.status_code == 200:
                    return r, ""
                note = f"HTTP {r.status_code}"
                if r.status_code in (403, 404, 410):
                    break
            except requests.RequestException as exc:
                note = f"{type(exc).__name__}: {exc}"
            time.sleep(5 * (k + 1))
        return None, note


def clean(text):
    """Collapse whitespace inside one line."""
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html):
    """Return (title, main text) from an HTML page, dropping navigation and scripts."""
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1") or soup.find("title")
    title = clean(heading.get_text(" ")) if heading else ""
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    root = soup.select_one("main") or soup.select_one("#content") or soup.select_one("article") or soup.body or soup
    lines = [clean(line) for line in root.get_text("\n").splitlines()]
    return title, "\n".join(line for line in lines if line)


def pdf_to_text(content):
    """Extract text from the first cfg.PDF_MAX_PAGES pages of a PDF."""
    import pdfplumber

    parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages[: cfg.PDF_MAX_PAGES]:
            parts.append(page.extract_text() or "")
    lines = [clean(line) for line in "\n".join(parts).splitlines()]
    return "\n".join(line for line in lines if line)


def find_date(text):
    """First 'D Month YYYY' date in the text as an ISO date, or an empty string."""
    m = DATE_RE.search(text)
    if not m:
        return ""
    return datetime.strptime(" ".join(m.groups()), "%d %B %Y").date().isoformat()


def is_relevant(title, text):
    """True when the title or the start of the text mentions an offshore operations topic."""
    return re.search(cfg.RELEVANT_PATTERN, f"{title} {text[:6000]}", re.I) is not None


def load_known():
    """URLs already stored in documents.jsonl."""
    if not cfg.DOCS_PATH.exists():
        return set()
    with open(cfg.DOCS_PATH, encoding="utf-8") as f:
        return {json.loads(line)["url"] for line in f if line.strip()}


def save_doc(source, doc_type, url, title, date, text, has_pdf):
    """Append one document to documents.jsonl."""
    cfg.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "doc_id": hashlib.sha1(url.encode()).hexdigest()[:12],
        "source": source,
        "doc_type": doc_type,
        "url": url,
        "title": title,
        "date": date,
        "n_words": len(text.split()),
        "has_pdf_text": has_pdf,
        "scraped_at": now_iso(),
        "text": text,
    }
    with open(cfg.DOCS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def hse_doc_type(path):
    """Label an HSE URL by where it lives."""
    if path.startswith("/safetybulletins/"):
        return "hse_bulletin"
    if "notices" in path or "/sa_" in path:
        return "hse_notice"
    return "hse_offshore_page"


def hse_links(fetcher):
    """Collect candidate HSE document URLs from the seed pages."""
    links = {}
    seeds = set(cfg.HSE_SEEDS)
    for seed in cfg.HSE_SEEDS:
        resp, note = fetcher.fetch(seed)
        if resp is None:
            log_row("HSE", seed, "seed_failed", 0, note)
            continue
        log_row("HSE", seed, "seed_ok", len(resp.text))
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            url = urldefrag(urljoin(seed, a["href"]))[0]
            p = urlparse(url)
            if p.netloc.endswith("nationalarchives.gov.uk"):
                log_row("HSE", url, "skipped_external", 0, "archived copy on another host")
                continue
            if p.netloc != cfg.HSE_HOST or url in seeds:
                continue
            if not p.path.startswith(cfg.HSE_PATH_PREFIXES):
                continue
            if not p.path.lower().endswith((".htm", ".html", ".pdf")):
                continue
            if p.path.endswith("index.htm"):
                continue
            links.setdefault(url, a.get_text(" ").strip())
    return links


def scrape_hse(fetcher, known, limit):
    """Scrape HSE pages and PDFs linked from the seed pages. Returns the number saved."""
    saved = 0
    for url in list(hse_links(fetcher))[: cfg.HSE_MAX_DOCS]:
        if limit and saved >= limit:
            break
        if url in known:
            log_row("HSE", url, "cached")
            continue
        resp, note = fetcher.fetch(url)
        if resp is None:
            log_row("HSE", url, "failed", 0, note)
            continue
        path = urlparse(url).path
        try:
            if path.lower().endswith(".pdf"):
                text, title, has_pdf = pdf_to_text(resp.content), "", True
                title = title or Path(path).stem.replace("-", " ").replace("_", " ")
            else:
                title, text = html_to_text(resp.text)
                has_pdf = False
        except Exception as exc:
            log_row("HSE", url, "parse_failed", 0, f"{type(exc).__name__}: {exc}")
            continue
        if len(text) < cfg.MIN_CHARS:
            log_row("HSE", url, "dropped_short", len(text))
            continue
        if not is_relevant(title, text):
            log_row("HSE", url, "dropped_not_relevant", len(text))
            continue
        save_doc("HSE", hse_doc_type(path), url, title, find_date(text), text, has_pdf)
        log_row("HSE", url, "saved", len(text))
        known.add(url)
        saved += 1
    return saved


def maib_records(fetcher):
    """Query the GOV.UK Search API for MAIB reports and return {path: record}."""
    records = {}
    for q in cfg.MAIB_QUERIES:
        start = 0
        while True:
            params = {
                "filter_format": "maib_report",
                "q": q,
                "count": 100,
                "start": start,
                "fields": ["title", "link", "public_timestamp"],
            }
            resp, note = fetcher.fetch(cfg.GOVUK_SEARCH_URL, params)
            if resp is None:
                log_row("MAIB", f"{cfg.GOVUK_SEARCH_URL}?q={q}", "search_failed", 0, note)
                break
            data = resp.json()
            results = data.get("results", [])
            for r in results:
                records.setdefault(r["link"], r)
            start += len(results)
            log_row("MAIB", f"{cfg.GOVUK_SEARCH_URL}?q={q}&start={start}", "search_ok", len(results))
            if not results or start >= data.get("total", 0) or start >= cfg.MAIB_MAX_PER_QUERY:
                break
    return records


def first_pdf_link(html, page_url):
    """URL of the first PDF linked from a report page, or None."""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if a["href"].lower().split("?")[0].endswith(".pdf"):
            return urljoin(page_url, a["href"])
    return None


def scrape_maib(fetcher, known, limit, use_pdfs):
    """Scrape MAIB report pages, adding the PDF text when available. Returns the number saved."""
    saved = 0
    for link, rec in maib_records(fetcher).items():
        if limit and saved >= limit:
            break
        url = urljoin(cfg.GOVUK_BASE, link)
        if url in known:
            log_row("MAIB", url, "cached")
            continue
        resp, note = fetcher.fetch(url)
        if resp is None:
            log_row("MAIB", url, "failed", 0, note)
            continue
        title, text = html_to_text(resp.text)
        title = title or rec.get("title", "")
        has_pdf = False
        pdf_url = first_pdf_link(resp.text, url) if use_pdfs else None
        if pdf_url:
            pdf_resp, pdf_note = fetcher.fetch(pdf_url)
            if pdf_resp is None:
                log_row("MAIB", pdf_url, "pdf_failed", 0, pdf_note)
            else:
                try:
                    text += "\n" + pdf_to_text(pdf_resp.content)
                    has_pdf = True
                except Exception as exc:
                    log_row("MAIB", pdf_url, "pdf_parse_failed", 0, f"{type(exc).__name__}: {exc}")
        if len(text) < cfg.MIN_CHARS:
            log_row("MAIB", url, "dropped_short", len(text))
            continue
        if not is_relevant(title, text):
            log_row("MAIB", url, "dropped_not_relevant", len(text))
            continue
        date = (rec.get("public_timestamp") or "")[:10] or find_date(text)
        save_doc("MAIB", "maib_report", url, title, date, text, has_pdf)
        log_row("MAIB", url, "saved", len(text))
        known.add(url)
        saved += 1
    return saved


def main():
    """Parse options and run the selected scrapers."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["all", "hse", "maib"], default="all")
    p.add_argument("--limit", type=int, default=0, help="stop each source after this many saved documents")
    p.add_argument("--no-pdfs", action="store_true", help="skip MAIB PDF downloads")
    args = p.parse_args()

    fetcher = Fetcher()
    known = load_known()
    if args.source in ("all", "hse"):
        print("HSE saved:", scrape_hse(fetcher, known, args.limit))
    if args.source in ("all", "maib"):
        print("MAIB saved:", scrape_maib(fetcher, known, args.limit, not args.no_pdfs))
    print("Now run: python src/check_corpus.py")


if __name__ == "__main__":
    main()
