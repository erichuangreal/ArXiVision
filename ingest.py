"""Background pipeline: arXiv search -> download -> extract -> rebuild the RAG index.

Runs in a plain thread (not asyncio), since search/download/extract/embedding
are all blocking calls. JOBS is polled by the API for a "loading screen."
"""

import json
import threading
import time
import uuid
from pathlib import Path

import requests

from download_arxiv import clean_filename, download_pdf, save_metadata, search_arxiv
from extract_pdf import extract_all_pdfs
from preprocessing import copy_metadata_files

PAPERS_DIR = Path("papers")
PROCESSED_DIR = Path("processed_text")
JOBS_PATH = Path("data") / "ingest_jobs.json"


def _load_jobs():
    if not JOBS_PATH.exists():
        return {}
    jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    # A job that was "running" when the process died has no thread left to
    # finish it. Say so rather than leaving a client poll a job that will
    # never update again.
    for job in jobs.values():
        if job["status"] == "running":
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = "Interrupted by a server restart."
            job["message"] = job["error"]
    return jobs


def _save_jobs():
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(JOBS, indent=2), encoding="utf-8")


# Job status, keyed by job_id. Persisted to a flat file so a poll after a
# restart sees the last known state instead of a 404 or a stale in-memory dict.
JOBS = _load_jobs()


def _set(job_id, **fields):
    JOBS[job_id].update(fields)
    JOBS[job_id]["updated_at"] = time.time()
    _save_jobs()


def start_ingest(topic, num_papers, rebuild_rag):
    # rebuild_rag: no-arg callback that reloads and swaps in the RAG index
    # once new papers are on disk. Passed in so this module doesn't import api.py.
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "topic": topic,
        "num_papers": num_papers,
        "status": "running",
        "stage": "queued",
        "message": "Waiting to start...",
        "current": 0,
        "total": num_papers,
        "error": None,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _save_jobs()

    thread = threading.Thread(
        target=_run_ingest, args=(job_id, topic, num_papers, rebuild_rag), daemon=True
    )
    thread.start()
    return job_id


def _run_ingest(job_id, topic, num_papers, rebuild_rag):
    try:
        _run_pipeline(job_id, topic, num_papers, rebuild_rag)
        _set(job_id, status="ready", stage="ready")
    except Exception as error:
        _set(job_id, status="failed", stage="failed", message=str(error), error=str(error))


def _run_pipeline(job_id, topic, num_papers, rebuild_rag):
    topic_slug = clean_filename(topic)
    papers_dir = PAPERS_DIR / topic_slug
    processed_dir = PROCESSED_DIR / topic_slug
    papers_dir.mkdir(parents=True, exist_ok=True)

    _set(job_id, stage="searching_arxiv", message=f"Searching arXiv for '{topic}'...")
    papers = search_arxiv(topic=topic, max_results=num_papers)

    if not papers:
        raise RuntimeError(f"No arXiv results found for '{topic}'.")

    _set(job_id, total=len(papers), message=f"Found {len(papers)} papers on arXiv.")

    downloaded = []
    for index, paper in enumerate(papers, start=1):
        _set(
            job_id,
            stage="downloading",
            current=index,
            message=f"Downloading {index}/{len(papers)}: {paper['title']}",
        )
        try:
            pdf_path = download_pdf(paper, papers_dir, index)
            paper["local_pdf_path"] = str(pdf_path)
            paper["download_status"] = "success"
        except (requests.RequestException, ValueError) as error:
            paper["local_pdf_path"] = None
            paper["download_status"] = "failed"
            paper["download_error"] = str(error)
        downloaded.append(paper)
        time.sleep(3)  # stay polite to arXiv, matches download_arxiv.py's CLI pace

    save_metadata(downloaded, papers_dir)

    succeeded = [p for p in downloaded if p["download_status"] == "success"]
    if not succeeded:
        raise RuntimeError("All downloads failed for this topic; nothing to extract.")

    _set(
        job_id,
        stage="extracting",
        current=0,
        total=len(succeeded),
        message=f"Extracting text from {len(succeeded)} PDFs...",
    )

    def on_extract_progress(index, total, pdf_path, status):
        _set(
            job_id,
            current=index,
            total=total,
            message=f"Extracting {index}/{total}: {pdf_path.name} ({status})",
        )

    extract_all_pdfs(papers_dir, processed_dir, progress_callback=on_extract_progress)
    copy_metadata_files(papers_dir, processed_dir)

    _set(
        job_id,
        stage="indexing",
        current=0,
        total=1,
        message="Chunking and embedding the new papers...",
    )
    rebuild_rag()

    _set(
        job_id,
        current=1,
        message=f"Added {len(succeeded)} papers on '{topic}'. Corpus ready.",
    )
