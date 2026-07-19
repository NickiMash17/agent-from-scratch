"""Raw search vs. agentic search — why agents use tools like Tavily.

Part 1 does what an agent would have to do with generic web access: search
DuckDuckGo, fetch the top result's HTML, and scrape the visible text. The
output is huge and messy — nav menus, cookie banners, footers — and the
model would have to burn tokens wading through it every iteration.

Part 2 asks Tavily the same question. It returns a direct answer plus a few
ranked, pre-cleaned snippets: small, structured, and ready to drop into a
tool_result.

Run:  python search_comparison.py "your query here"
      (Part 2 needs TAVILY_API_KEY in .env — free key at https://tavily.com)
"""

import os
import re
import sys

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS  # formerly the `duckduckgo-search` package
from dotenv import load_dotenv

load_dotenv()

DEFAULT_QUERY = "What is the ReAct pattern for LLM agents?"

# If DuckDuckGo rate-limits us (it aggressively blocks scripted queries),
# fall back to scraping a fixed page so Part 1 still demonstrates the mess.
FALLBACK_URL = "https://en.wikipedia.org/wiki/Intelligent_agent"

PREVIEW_CHARS = 1200


def separator(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Part 1: raw search — search engine + HTTP fetch + HTML scraping
# ---------------------------------------------------------------------------

def raw_search(query: str) -> str:
    separator("PART 1: raw search (DuckDuckGo + requests + BeautifulSoup)")

    try:
        hits = DDGS().text(query, max_results=3)
        if not hits:
            raise RuntimeError("search returned no results (rate-limited?)")
        url = hits[0]["href"]
        print(f"top result: {hits[0]['title']}\nurl:        {url}")
    except Exception as exc:
        print(f"search failed ({exc}); falling back to a fixed URL")
        url = FALLBACK_URL

    html = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    ).text
    print(f"fetched HTML: {len(html):,} chars")

    # Strip tags that never contain readable content, then extract text.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ")

    # Regex cleanup: collapse runs of whitespace. This is about as good as
    # generic scraping gets without per-site extraction rules.
    text = re.sub(r"\s+", " ", text).strip()

    print(f"scraped text: {len(text):,} chars (after cleanup!)\n")
    print(f"--- first {PREVIEW_CHARS} chars " + "-" * 30)
    print(text[:PREVIEW_CHARS])
    print("--- ... and thousands more chars of this ---")
    return text


# ---------------------------------------------------------------------------
# Part 2: agentic search — Tavily returns LLM-ready results
# ---------------------------------------------------------------------------

def agentic_search(query: str) -> None:
    separator("PART 2: agentic search (Tavily, include_answer=True)")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print(
            "TAVILY_API_KEY not set — skipping the live call.\n"
            "Get a free key at https://tavily.com and add a line to .env:\n"
            "    TAVILY_API_KEY=tvly-..."
        )
        return

    from tavily import TavilyClient  # imported here so Part 1 runs keyless

    response = TavilyClient(api_key=api_key).search(
        query, include_answer=True, max_results=3
    )

    print(f"\ndirect answer:\n  {response['answer']}")
    print("\nranked results:")
    total = len(response["answer"])
    for hit in response["results"]:
        total += len(hit["content"])
        print(f"\n  [{hit['score']:.2f}] {hit['title']}\n  {hit['url']}")
        print(f"  {hit['content'][:300]}")
    print(f"\ntotal content: {total:,} chars — answer + snippets, no boilerplate")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    print(f"query: {query!r}")
    scraped = raw_search(query)
    agentic_search(query)

    separator("THE POINT")
    print(
        "Same question. Raw search hands the model a "
        f"{len(scraped):,}-char wall of scraped text to dig through;\n"
        "Tavily hands it a direct answer plus a few clean snippets.\n"
        "Fewer tokens per iteration, no brittle per-site scraping, and the\n"
        "agent usually finishes in one reason-act-observe cycle instead of\n"
        "several. That's what 'agentic search' means: search shaped for a\n"
        "model to consume, not a human to click through."
    )
