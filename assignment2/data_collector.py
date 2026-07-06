import re

import requests
from bs4 import BeautifulSoup

WIKIPEDIA_LANGUAGES = ("en", "hi", "te")
LINKS_PER_LANGUAGE = 5


class DataCollector:
    _headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def FetchContent(self, url: str) -> str:
        """Fetch and return all visible text content from the given website URL."""
        response = requests.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def FetchLinks(self) -> list[str]:
        """Fetch 5 random Wikipedia article links each for English, Hindi, and Telugu."""
        links: list[str] = []
        for language in WIKIPEDIA_LANGUAGES:
            links.extend(self._fetch_random_wikipedia_links(language, LINKS_PER_LANGUAGE))
        return links

    def _fetch_random_wikipedia_links(self, language: str, count: int) -> list[str]:
        api_url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": count,
            "prop": "info",
            "inprop": "url",
            "format": "json",
        }

        response = requests.get(api_url, params=params, headers=self._headers, timeout=30)
        response.raise_for_status()

        pages = response.json().get("query", {}).get("pages", {})
        return [page["fullurl"] for page in pages.values()]

    def parseResponse(self, content: str) -> list[str]:
        """Split fetched content into two-word phrases."""
        words = content.split()
        return [
            f"{words[i]} {words[i + 1]}"
            for i in range(0, len(words) - len(words) % 2, 2)
        ]


if __name__ == "__main__":
    collector = DataCollector()
    # links = collector.FetchLinks()
    # print(f"Found {len(links)} links")
    # for link in links:
    #     print(link)
    content = collector.FetchContent("https://te.wikipedia.org/wiki/%E0%B0%98%E0%B0%9C%E0%B0%BF%E0%B0%AF%E0%B0%BE%E0%B0%AC%E0%B0%BE%E0%B0%A6%E0%B1%8D_%E0%B0%B2%E0%B1%8B%E0%B0%95%E0%B1%8D%E2%80%8C%E0%B0%B8%E0%B0%AD_%E0%B0%A8%E0%B0%BF%E0%B0%AF%E0%B1%8B%E0%B0%9C%E0%B0%95%E0%B0%B5%E0%B0%B0%E0%B1%8D%E0%B0%97%E0%B0%82")
    print(collector.parseResponse(content))