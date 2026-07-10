"""Scrape Wikipedia article text from multiple language editions and save as markdown."""

from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS = [
    "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82",
    "https://en.wikipedia.org/wiki/India",
    "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4",
    "https://kn.wikipedia.org/wiki/%E0%B2%AD%E0%B2%BE%E0%B2%B0%E0%B2%A4",
]

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "scraped_india.md"

HEADERS = {
    "User-Agent": (
        "ERA-assignments-data-collector/1.0 "
        "(educational; contact: local-dev)"
    ),
}

# Elements inside the article body that are not written article prose.
_STRIP_SELECTORS = (
    "table",
    "style",
    "script",
    "noscript",
    "sup.reference",
    "span.mw-editsection",
    "div.navbox",
    "div.sistersitebox",
    "div.metadata",
    "div.hatnote",
    "div.ambox",
    "div.thumb",
    "figure",
    "ul.gallery",
)


def scrape_page_text(url: str, session: requests.Session | None = None) -> str:
    """Fetch a Wikipedia page and return its main article text."""
    client = session or requests.Session()
    response = client.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        content = soup.select_one("#mw-content-text")
    if content is None:
        raise ValueError(f"Could not find article content for {url}")

    for selector in _STRIP_SELECTORS:
        for node in content.select(selector):
            node.decompose()

    # Prefer block-level prose (headings, paragraphs, list items) so inline
    # links/spans do not force every word onto its own line.
    blocks: list[str] = []
    for el in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dd"]):
        chunk = " ".join(el.get_text(" ", strip=True).split())
        if chunk:
            blocks.append(chunk)

    if blocks:
        return "\n\n".join(blocks)

    fallback = " ".join(content.get_text(" ", strip=True).split())
    return fallback


def collect_and_save(
    urls: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Scrape written text from each URL, concatenate it, and save as markdown.

    The file contains only scraped page content — no headings, comments, or fences.
    Returns the path to the written markdown file.
    """
    urls = urls or URLS
    out = Path(output_path) if output_path else DEFAULT_OUTPUT

    session = requests.Session()
    bodies: list[str] = []
    for url in urls:
        print(f"Scraping: {url}")
        bodies.append(scrape_page_text(url, session=session))

    combined = "\n\n".join(bodies).rstrip() + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(combined, encoding="utf-8")
    print(f"Saved combined text to {out} ({len(combined):,} characters)")
    return out


if __name__ == "__main__":
    collect_and_save()
