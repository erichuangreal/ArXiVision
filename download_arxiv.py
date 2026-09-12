from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"

HEADERS = {
    "User-Agent": "ArxivResearchDataset/1.0 (contact: huangheeh@gmail.com)"
}


def clean_filename(text: str, max_length: int = 120) -> str:
    """
    Convert a paper title into a safe filename.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = text.replace(" ", "_")
    return text[:max_length]


def search_arxiv(topic: str, max_results: int = 10) -> list[dict]:
    """
    Search arXiv and return paper metadata.
    """
    parameters = {
        "search_query": f'all:"{topic}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urlencode(parameters)}"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

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

    with requests.get(
        paper["pdf_url"],
        headers=HEADERS,
        timeout=60,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type:
            raise ValueError(
                f"Expected a PDF but received: {content_type}"
            )

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    file.write(chunk)

    return output_path


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