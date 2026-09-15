"""
Usage (one-time setup, requires a Kaggle account + API token in
~/.kaggle/kaggle.json - see https://www.kaggle.com/docs/api):

    python -m data_processing.kaggle_search download   # fetches the dataset
    python -m data_processing.kaggle_search build       # builds the FTS5 index

After that, search_local() answers topic searches with zero network calls.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RAW_JSONL_PATH = CACHE_DIR / "arxiv-metadata-oai-snapshot.json"
INDEX_DB_PATH = CACHE_DIR / "arxiv_metadata_index.sqlite3"

KAGGLE_DATASET = "Cornell-University/arxiv"


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def download_dataset(dest_dir: Path = CACHE_DIR) -> Path:
    import kaggle  # imported lazily: only needed for this one-time step

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} from Kaggle to {dest_dir} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=False)

    zip_path = dest_dir / "arxiv.zip"
    if not zip_path.exists():
        candidates = list(dest_dir.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError(f"Expected a downloaded .zip in {dest_dir}, found none.")
        zip_path = candidates[0]

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)

    if not RAW_JSONL_PATH.exists():
        candidates = list(dest_dir.glob("*.json"))
        if candidates:
            candidates[0].rename(RAW_JSONL_PATH)
    print(f"Dataset ready at {RAW_JSONL_PATH}")
    return RAW_JSONL_PATH


def build_index(
    jsonl_path: Path = RAW_JSONL_PATH,
    db_path: Path = INDEX_DB_PATH,
    progress_callback=None,
) -> None:
    # Streams the JSONL line-by-line rather than json.load()-ing the whole
    # ~4GB file - keeps memory flat regardless of dataset size.
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found - run download_dataset() first.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE VIRTUAL TABLE papers USING fts5(
            arxiv_id UNINDEXED,
            title,
            abstract,
            authors UNINDEXED,
            categories UNINDEXED,
            published UNINDEXED,
            updated UNINDEXED
        )"""
    )

    batch = []
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            authors_parsed = record.get("authors_parsed") or []
            authors = [
                _normalise_whitespace(f"{parts[1]} {parts[0]}")
                for parts in authors_parsed
                if len(parts) >= 2
            ]

            versions = record.get("versions") or []
            published = versions[0]["created"] if versions else record.get("update_date", "")
            updated = versions[-1]["created"] if versions else record.get("update_date", "")

            batch.append((
                record.get("id", ""),
                _normalise_whitespace(record.get("title", "")),
                _normalise_whitespace(record.get("abstract", "")),
                json.dumps(authors),
                record.get("categories", ""),
                published,
                updated,
            ))
            count += 1

            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO papers (arxiv_id, title, abstract, authors, categories, published, updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch = []
                if progress_callback:
                    progress_callback(count)
                else:
                    print(f"  indexed {count:,} papers...", end="\r")

    if batch:
        conn.executemany(
            "INSERT INTO papers (arxiv_id, title, abstract, authors, categories, published, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    conn.close()
    print(f"\nBuilt index of {count:,} papers at {db_path}")


def _row_to_paper(row) -> dict:
    arxiv_id, title, abstract, authors_json, categories, published, updated = row
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": json.loads(authors_json),
        "abstract": abstract,
        "published": published,
        "updated": updated,
        "categories": categories.split() if categories else [],
        "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def _fts5_quote(term: str) -> str:
    # Escapes a bare word/phrase for safe embedding in an FTS5 MATCH string.
    return '"' + term.replace('"', '""') + '"'


def search_local(topic: str, max_results: int = 10, db_path: Path = INDEX_DB_PATH) -> list[dict]:
    """
    Same two-stage behavior as download_arxiv.search_arxiv(): try an exact
    phrase match first, fall back to requiring every word to appear
    (a real AND, same reasoning as the live version) - just against the
    local FTS5 index instead of a live arXiv query, so no network call and
    nothing arXiv can rate-limit.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"No local index at {db_path} - run `python -m data_processing.kaggle_search build` first."
        )

    conn = sqlite3.connect(db_path)
    try:
        exact_query = f"title:{_fts5_quote(topic)} OR abstract:{_fts5_quote(topic)}"
        exact_rows = conn.execute(
            "SELECT arxiv_id, title, abstract, authors, categories, published, updated "
            "FROM papers WHERE papers MATCH ? ORDER BY rank LIMIT ?",
            (exact_query, max_results),
        ).fetchall()

        if len(exact_rows) >= max_results:
            return [_row_to_paper(r) for r in exact_rows]

        words = topic.split()
        if not words:
            return [_row_to_paper(r) for r in exact_rows]

        broad_query = " AND ".join(_fts5_quote(w) for w in words)
        remaining = max_results - len(exact_rows)
        seen_ids = {r[0] for r in exact_rows}

        broad_rows = conn.execute(
            "SELECT arxiv_id, title, abstract, authors, categories, published, updated "
            "FROM papers WHERE papers MATCH ? ORDER BY rank LIMIT ?",
            (broad_query, remaining + len(exact_rows)),
        ).fetchall()

        if not exact_rows:
            print(f"No exact-phrase match for '{topic}' in local index; falling back to a broader search.")

        combined = list(exact_rows)
        for row in broad_rows:
            if len(combined) >= max_results:
                break
            if row[0] not in seen_ids:
                combined.append(row)
                seen_ids.add(row[0])

        return [_row_to_paper(r) for r in combined]
    finally:
        conn.close()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"download", "build"}:
        print("Usage: python -m data_processing.kaggle_search [download|build]")
        raise SystemExit(1)

    if sys.argv[1] == "download":
        download_dataset()
    elif sys.argv[1] == "build":
        build_index()


if __name__ == "__main__":
    main()
