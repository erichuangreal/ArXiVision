"""File-backed storage for research question collections. No DB for the MVP."""

import json
import uuid
from pathlib import Path

STORE_PATH = Path("data") / "collections.json"


def _load():
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def _save(store):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def create_collection(question, papers):
    # papers: [{paper_id, title, authors, arxiv_id, why_included}], caller
    # supplies why_included from the /search results they picked these from.
    store = _load()
    collection_id = uuid.uuid4().hex[:8]
    store[collection_id] = {
        "collection_id": collection_id,
        "question": question,
        "papers": papers,
        "comparison": None,
        "followups": None,
    }
    _save(store)
    return store[collection_id]


def get_collection(collection_id):
    return _load().get(collection_id)


def list_collections():
    return list(_load().values())


def update_collection(collection_id, **fields):
    store = _load()
    if collection_id not in store:
        return None
    store[collection_id].update(fields)
    _save(store)
    return store[collection_id]
