"""Per-user RAG instances: each user gets their own papers/processed_text/
chroma_store directory and their own in-memory RAGClass, built lazily.
"""

from pathlib import Path

from rag.rag_implementation import RAGClass  # also validates OPENAI_API_KEY on import
import db

PAPERS_ROOT = Path("papers")
PROCESSED_ROOT = Path("processed_text")
CHROMA_ROOT = Path("chroma_store")

# user_id -> RAGClass, or user_id -> None for "checked, has no corpus yet".
_CACHE = {}


def user_paths(user_id):
    return (
        PAPERS_ROOT / user_id,
        PROCESSED_ROOT / user_id,
        CHROMA_ROOT / user_id,
    )


def has_corpus(user_id):
    _, processed_dir, _ = user_paths(user_id)
    return processed_dir.exists() and any(processed_dir.rglob("metadata.json"))


def _build(user_id, progress_callback=None):
    _, processed_dir, persist_dir = user_paths(user_id)
    settings = db.get_settings(user_id) or {}
    rag = RAGClass(processed_dir, persist_directory=persist_dir)
    rag.load_documents()
    rag.split_documents()
    rag.create_vectorstore(progress_callback=progress_callback)
    rag.setup_retriever()
    rag.setup_qa_chain(
        model=settings.get("model", db.DEFAULT_MODEL),
        temperature=settings.get("temperature", db.DEFAULT_TEMPERATURE),
        language_style=settings.get("language_style", db.DEFAULT_LANGUAGE_STYLE),
    )
    return rag


def get_rag(user_id):
    # Returns None if the user hasn't ingested anything yet, without raising.
    if user_id in _CACHE:
        return _CACHE[user_id]
    if not has_corpus(user_id):
        _CACHE[user_id] = None
        return None
    _CACHE[user_id] = _build(user_id)
    return _CACHE[user_id]


def rebuild_rag(user_id, progress_callback=None):
    # Called after an ingest job adds papers; always rebuilds from disk.
    _CACHE[user_id] = _build(user_id, progress_callback=progress_callback)
    return _CACHE[user_id]


def invalidate(user_id):
    # Called after a settings change - the cached qa_chain has the old
    # model/temperature/style baked in, so drop it and let the next request
    # rebuild lazily (cheap: create_vectorstore() hits its fast path since
    # the corpus itself hasn't changed).
    _CACHE.pop(user_id, None)
