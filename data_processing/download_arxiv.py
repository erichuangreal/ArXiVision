from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"

HEADERS = {
    "User-Agent": "ArxivResearchDataset/1.0 (contact: huangheeh@gmail.com)"
}

# arXiv's public API rate-limits fairly aggressively and asks callers not to
# hammer it; a 429 here is expected occasionally, not a bug, so it's retried
# with backoff rather than failing the expedition on the first hit.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5

# Every expedition runs in its own thread. Concurrent expeditions queue instead of colliding.
_ARXIV_LOCK = threading.Lock()
_last_request_at = 0.0
MIN_REQUEST_INTERVAL_SECONDS = 3


def _throttled_get(url, **kwargs):
    global _last_request_at
    with _ARXIV_LOCK:
        wait = MIN_REQUEST_INTERVAL_SECONDS - (time.time() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            return requests.get(url, **kwargs)
        finally:
            _last_request_at = time.time()


class ArxivUnavailable(requests.RequestException):
    """arXiv kept failing after retries (rate limit or outage).

    Subclasses RequestException so it's still caught wherever a single
    paper's download failure is handled as a per-paper (not whole-job)
    failure, while giving a message a user can actually act on.
    """


def _retry_request(attempt_fn, description):
    # attempt_fn does one full try (request + whatever validation it needs)
    # and either returns a result or raises a requests exception.
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return attempt_fn()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in RETRYABLE_STATUS_CODES:
                raise
            last_error = error
            retry_after = error.response.headers.get("Retry-After") if error.response is not None else None
        except requests.RequestException as error:
            last_error = error
            retry_after = None

        if attempt < MAX_RETRIES:
            wait = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(
                f"{description} failed ({last_error}); "
                f"retrying in {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait)

    raise ArxivUnavailable(
        f"{description} did not succeed after {MAX_RETRIES + 1} attempts "
        f"(last error: {last_error}). arXiv may be rate-limiting requests or "
        "temporarily unavailable - try again in a few minutes."
    ) from last_error


def clean_filename(text: str, max_length: int = 120) -> str:
    """
    Convert a paper title into a safe filename.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = text.replace(" ", "_")
    return text[:max_length]


def _run_search_query(search_query: str, max_results: int, description: str) -> list[dict]:
    # One raw arXiv query -> parsed paper list. No exact-phrase-vs-broad
    # decision here; search_arxiv() owns that.
    parameters = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urlencode(parameters)}"

    def attempt():
        response = _throttled_get(url, headers=HEADERS, timeout=90)
        response.raise_for_status()
        return response

    response = _retry_request(attempt, description)

    feed = feedparser.parse(response.content)

    if feed.bozo:
        raise RuntimeError(f"Could not parse arXiv response: {feed.bozo_exception}")

    papers = []

    for entry in feed.entries:
        arxiv_id = entry.id.rsplit("/", 1)[-1]

        authors = [
            author.name
            for author in entry.get("authors", [])
        ]

        categories = [
            tag["term"]
            for tag in entry.get("tags", [])
        ]

        pdf_url = None

        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": re.sub(r"\s+", " ", entry.title).strip(),
                "authors": authors,
                "abstract": re.sub(
                    r"\s+",
                    " ",
                    entry.summary,
                ).strip(),
                "published": entry.get("published"),
                "updated": entry.get("updated"),
                "categories": categories,
                "abstract_url": entry.id,
                "pdf_url": pdf_url,
            }
        )

    return papers


def search_arxiv(topic: str, max_results: int = 10) -> list[dict]:
    """
    Search arXiv and return paper metadata.

    Tries an exact-phrase match first (precise, but arXiv requires the
    literal wording to appear - a multi-word topic phrased differently than
    any paper's own text, e.g. "airplane wing designs", can legitimately
    match zero papers even though the subject is well covered). Falls back
    to requiring every word to appear somewhere in the paper (not
    necessarily adjacent or in order) - looser than an exact phrase, but
    still real boolean AND, not arXiv silently ignoring an unparseable
    query and returning its newest submissions regardless of relevance.
    """
    exact_query = f'all:"{topic}"'
    papers = _run_search_query(exact_query, max_results, f"arXiv exact-phrase search for '{topic}'")

    if papers:
        return papers

    words = topic.split()
    # A literal "+" here would be double-encoded by urlencode() into "%2B"
    # (data), not arXiv's AND operator - a real space is what turns into the
    # "+AND+" arXiv's query parser actually expects.
    broad_query = " AND ".join(f"all:{word}" for word in words) if words else f"all:{topic}"
    print(f"No exact-phrase match for '{topic}'; falling back to a broader search.")
    return _run_search_query(broad_query, max_results, f"arXiv broad search for '{topic}'")


def download_pdf(
    paper: dict,
    output_directory: Path,
    paper_number: int,
) -> Path:
    """
    Download one paper PDF.
    """
    safe_title = clean_filename(paper["title"])
    filename = f"{paper_number:03d}_{safe_title}.pdf"
    output_path = output_directory / filename

    if output_path.exists():
        print(f"Already exists: {filename}")
        return output_path

    print(f"Downloading: {paper['title']}")

    def attempt():
        with _throttled_get(paper["pdf_url"], headers=HEADERS, timeout=60, stream=True) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type:
                # Not a transient failure - retrying won't change the content type.
                raise ValueError(f"Expected a PDF but received: {content_type}")

            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        file.write(chunk)
        return output_path

    return _retry_request(attempt, f"Download of '{paper['title']}'")


def save_metadata(
    papers: list[dict],
    output_directory: Path,
) -> None:
    """
    Save metadata for all downloaded papers.
    """
    metadata_path = output_directory / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            papers,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Metadata saved to: {metadata_path}")


def main() -> None:
    topic = "AI safety" # CHOOSE TOPIC HERE
    number_of_papers = 10

    output_directory = Path("papers") / clean_filename(topic)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Searching arXiv for: {topic}")

    papers = search_arxiv(
        topic=topic,
        max_results=number_of_papers,
    )

    print(f"Found {len(papers)} papers.")

    downloaded_papers = []

    for index, paper in enumerate(papers, start=1):
        try:
            pdf_path = download_pdf(
                paper=paper,
                output_directory=output_directory,
                paper_number=index,
            )

            paper["local_pdf_path"] = str(pdf_path)
            paper["download_status"] = "success"

        except (requests.RequestException, ValueError) as error:
            print(f"Could not download {paper['title']}: {error}")
            paper["local_pdf_path"] = None
            paper["download_status"] = "failed"
            paper["download_error"] = str(error)

        downloaded_papers.append(paper)


        time.sleep(3)

    save_metadata(
        papers=downloaded_papers,
        output_directory=output_directory,
    )

    print("Finished.")


if __name__ == "__main__":
    main()