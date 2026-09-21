import threading
import time
import traceback
import uuid
from pathlib import Path

import requests

import db
import dynamic_eval
import rag.rag_registry as rag_registry
from data_processing.download_arxiv import clean_filename, download_pdf, save_metadata, search_arxiv
from data_processing.extract_pdf import extract_all_pdfs
from data_processing.kaggle_search import search_local
from data_processing.preprocessing import copy_metadata_files


def _search(topic, num_papers):
    # Tries arXiv's live search first (one attempt, no retries - a 429/5xx
    # falls back immediately instead of running the full backoff sequence
    # search_arxiv() normally uses for a standalone call). Falling back
    # fast keeps every rate-limited request bounded, while still preferring
    # live results (arXiv's own index is always current; the local index is
    # only as fresh as the last `kaggle_search build` run) whenever arXiv
    # actually cooperates.
    try:
        return search_arxiv(topic=topic, max_results=num_papers, max_retries=0)
    except requests.RequestException as error:
        print(f"Live arXiv search failed ({error}); falling back to the local index.")

    try:
        return search_local(topic=topic, max_results=num_papers)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Live arXiv search failed and no local index is built yet. Run "
            "`python -m data_processing.kaggle_search download` then `build`."
        ) from error


def _describe_job_error(error):
    # Mirrors the frontend's api/client.js describeError(): translate an
    # exception into a message a user can actually read. Our own explicitly
    # raised RuntimeErrors above already carry a clean, specific message;
    # anything else (a raw HTTP/SDK exception) gets a calm, generic fallback
    # instead of a technical str(error) dump - request URLs, status lines,
    # retry counts - straight into the Field Log.
    if isinstance(error, RuntimeError):
        return str(error)
    return (
        "Something went wrong partway through this expedition, likely a temporary "
        "network or service issue. Try again - if it keeps happening, it's worth "
        "reporting."
    )


def start_ingest(user_id, topic, num_papers):
    job_id = uuid.uuid4().hex[:8]
    db.create_job(user_id, job_id, topic, num_papers)

    thread = threading.Thread(
        target=_run_ingest, args=(user_id, job_id, topic, num_papers), daemon=True
    )
    thread.start()
    return db.get_job(user_id, job_id)


def _run_ingest(user_id, job_id, topic, num_papers):
    try:
        _run_pipeline(user_id, job_id, topic, num_papers)
        db.update_job(user_id, job_id, status="ready", stage="ready")
    except Exception as error:
        traceback.print_exc()
        message = _describe_job_error(error)
        db.update_job(
            user_id, job_id, status="failed",
            message=message, error=message,
        )


def _run_pipeline(user_id, job_id, topic, num_papers):
    papers_dir, processed_dir, _ = rag_registry.user_paths(user_id)
    topic_slug = clean_filename(topic)
    papers_dir = papers_dir / topic_slug
    processed_dir = processed_dir / topic_slug
    papers_dir.mkdir(parents=True, exist_ok=True)

    db.update_job(user_id, job_id, stage="searching_arxiv", message=f"Searching arXiv for '{topic}'...")
    papers = _search(topic, num_papers)

    if not papers:
        raise RuntimeError(f"No arXiv results found for '{topic}'.")

    db.update_job(user_id, job_id, total=len(papers), message=f"Found {len(papers)} papers on arXiv.")

    downloaded = []
    for index, paper in enumerate(papers, start=1):
        paper["topic"] = topic
        db.update_job(
            user_id, job_id, stage="downloading", current=index,
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

    db.update_job(
        user_id, job_id, stage="extracting", current=0, total=len(succeeded),
        message=f"Extracting text from {len(succeeded)} PDFs...",
    )

    def on_extract_progress(index, total, pdf_path, status):
        db.update_job(
            user_id, job_id, current=index, total=total,
            message=f"Extracting {index}/{total}: {pdf_path.name} ({status})",
        )

    extract_all_pdfs(papers_dir, processed_dir, progress_callback=on_extract_progress)
    copy_metadata_files(papers_dir, processed_dir)

    db.update_job(
        user_id, job_id, stage="indexing", current=0, total=1,
        message="Chunking the new papers...",
    )

    def on_index_progress(message):
        db.update_job(user_id, job_id, message=message)

    rag = rag_registry.rebuild_rag(user_id, progress_callback=on_index_progress)

    db.update_job(
        user_id, job_id, stage="evaluating", current=0, total=1,
        message="Drafting verification questions...",
    )

    def on_eval_progress(message):
        db.update_job(user_id, job_id, message=message)

    new_paper_ids = [Path(p["local_pdf_path"]).stem for p in succeeded]

    eval_note = ""
    try:
        dynamic_eval.run_for_user(user_id, rag, new_paper_ids, progress_callback=on_eval_progress)
    except Exception as error:
        # Verification is a bonus signal on top of a successful ingest, not
        # the point of the expedition - a failure here shouldn't undo it.
        eval_note = f" Verification failed: {_describe_job_error(error)}"

    db.update_job(
        user_id, job_id, stage="ready", current=1,
        message=f"Added {len(succeeded)} papers on '{topic}'. Corpus ready.{eval_note}",
    )


db.mark_interrupted_jobs()
