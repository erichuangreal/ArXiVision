from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.rag_class import split_abstention
from evaluate.grounding import (
    check_citations,
    check_citation_coverage,
    check_claim_support,
    check_names,
    check_numbers,
    collect_authors,
    format_context,
)
import db
import ingest
import rag.rag_registry as rag_registry
import synthesis


app = FastAPI(
    title="arXiv Research RAG API",
    version="1.0"
)

# Permissive during development; no frontend deploy origin is decided yet.
# Lock this down to the real origin before any public deployment - auth here
# is an API key, not a cookie, so this is lower-risk than it would be with
# cookie-based sessions, but still worth tightening later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_METRICS = {
    "hit@1": {
        "label": "Expected evidence appeared first",
        "higher_is_better": True,
    },
    "hit@4": {
        "label": "Expected evidence appeared within the first four results",
        "higher_is_better": True,
    },
    "paper_hit@4": {
        "label": "The expected paper appeared within the first four results",
        "higher_is_better": True,
    },
    "mrr@4": {
        "label": "How near the top the expected evidence generally ranked (1.0 = always first)",
        "higher_is_better": True,
    },
    "answers_citing_anything": {
        "label": "Share of answers that cited a source at all",
        "higher_is_better": True,
    },
    "citation_validity": {
        "label": "Share of citations pointing at a source actually retrieved",
        "higher_is_better": True,
    },
    "citation_correctness": {
        "label": "Share of citations pointing at the expected page (evaluation set only)",
        "higher_is_better": True,
    },
    "citation_coverage": {
        "label": "Share of claims sitting under a citation",
        "higher_is_better": True,
    },
    "claim_support": {
        "label": "Of claims with a checkable fact, share confirmed present in the cited source",
        "higher_is_better": True,
    },
    "claims_not_applicable": {
        "label": "Count of claims with no checkable fact (opinions, labels) - excluded from claim_support",
        "higher_is_better": None,  # a count, not a rate - no "better direction"
    },
    "answer_correctness": {
        "label": "Share of answers judged to state the same fact as the reference answer, regardless of which page they cite",
        "higher_is_better": True,
    },
    "numeric_grounding": {
        "label": "Share of numbers in the answer also present in the retrieved text (checks presence, not arithmetic correctness)",
        "higher_is_better": True,
    },
    "name_grounding": {
        "label": "Share of named people in the answer also present in the retrieved text",
        "higher_is_better": True,
    },
    "abstention_rate": {
        "label": "Share of unanswerable test questions the system correctly declined",
        "higher_is_better": True,
    },
    "false_refusal_rate": {
        "label": "Share of answerable test questions the system wrongly declined",
        "higher_is_better": False,
    },
}


def get_current_user(x_api_key: str = Header(..., alias="X-API-Key")):
    user_id = db.get_user_by_api_key(x_api_key)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return user_id


# Per-day caps.
DAILY_LIMITS = {
    "ingest": 20,
    "ask": 100,
    "search": 100,
    "compare": 50,
    "followups": 50,
}

# Papers per single expedition.
MAX_PAPERS_PER_INGEST = 100


def enforce_daily_limit(user_id, action):
    limit = DAILY_LIMITS[action]
    if not db.try_consume_usage(user_id, action, limit):
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached for '{action}' ({limit}/day per account). Try again tomorrow.",
        )


def get_user_rag(user_id):
    # Every route that queries a corpus needs this; centralized so the "no
    # papers yet" message is worded the same everywhere.
    rag = rag_registry.get_rag(user_id)
    if rag is None:
        raise HTTPException(
            status_code=400,
            detail="No papers ingested yet. POST /ingest with {topic, num_papers} first.",
        )
    return rag


class RegisterRequest(BaseModel):
    email: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    topic: Optional[str] = None


class AskRequest(BaseModel):
    query: str
    topic: Optional[str] = None


class PaperSelection(BaseModel):
    paper_id: str
    why_included: str


class CollectionRequest(BaseModel):
    question: str
    papers: List[PaperSelection]


class IngestRequest(BaseModel):
    topic: str
    num_papers: int = 10


ALLOWED_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-5"}
ALLOWED_LANGUAGE_STYLES = {"plain", "standard", "technical"}


class SettingsUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    language_style: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "arXiv RAG API is running"
    }


@app.get("/me")
def whoami(user_id: str = Depends(get_current_user)):
    # Lets a client resolve its own account id from the API key alone -
    # nothing else is needed to sign back in.
    return {"user_id": user_id}


@app.get("/settings")
def get_settings(user_id: str = Depends(get_current_user)):
    return db.get_settings(user_id)


@app.put("/settings")
def update_settings(request: SettingsUpdate, user_id: str = Depends(get_current_user)):
    fields = {}
    if request.model is not None:
        if request.model not in ALLOWED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"model must be one of {sorted(ALLOWED_MODELS)}.",
            )
        fields["model"] = request.model
    if request.temperature is not None:
        if not 0.0 <= request.temperature <= 1.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 1.0.")
        fields["temperature"] = request.temperature
    if request.language_style is not None:
        if request.language_style not in ALLOWED_LANGUAGE_STYLES:
            raise HTTPException(
                status_code=400,
                detail=f"language_style must be one of {sorted(ALLOWED_LANGUAGE_STYLES)}.",
            )
        fields["language_style"] = request.language_style

    if not fields:
        raise HTTPException(status_code=400, detail="Provide at least one of model, temperature, language_style.")

    updated = db.update_settings(user_id, **fields)
    # The cached qa_chain has the old model/temperature/style baked in.
    rag_registry.invalidate(user_id)
    return updated


@app.post("/users")
def register_user(request: RegisterRequest):
    user_id, api_key = db.create_user(email=request.email)
    return {
        "user_id": user_id,
        "api_key": api_key,
        "note": (
            "Store this API key now - it will not be shown again. "
            "Pass it as the X-API-Key header on every other request."
        ),
    }


@app.post("/search")
def search(request: SearchRequest, user_id: str = Depends(get_current_user)):
    enforce_daily_limit(user_id, "search")
    rag = get_user_rag(user_id)

    docs = rag.retriever.invoke({
        "input": request.query,
        "topic": request.topic
    })

    results = []

    for doc in docs:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    return {
        "query": request.query,
        "results": results
    }


@app.post("/ask")
def ask(request: AskRequest, user_id: str = Depends(get_current_user)):
    enforce_daily_limit(user_id, "ask")
    rag = get_user_rag(user_id)
    known_authors = collect_authors(rag.paper_metadata)

    response = rag.qa_chain.invoke({
        "input": request.query,
        "topic": request.topic
    })

    docs = response["context"]
    sources = []

    for doc in docs:
        sources.append({
            "title": doc.metadata.get("title"),
            "authors": doc.metadata.get("authors"),
            "arxiv_id": doc.metadata.get("arxiv_id"),
            "page_number": doc.metadata.get("page_number")
        })

    abstained, answer = split_abstention(response["answer"])

    # Evidence checks run live (no ground truth here, unlike evaluate/), so
    # "correct" always comes back None/not-applicable outside the eval set.
    citations = check_citations(answer, docs, {})
    context_text = format_context(docs)
    _numbers_found, numbers_missing = check_numbers(answer, context_text)
    _names_found, names_missing = check_names(answer, context_text, known_authors)
    claims, cited, _uncited = check_citation_coverage(answer)
    assessed, supported, not_applicable, weak = check_claim_support(answer, docs)

    evidence = {
        "citations": {
            "cited_anything": citations["cited_anything"],
            "valid": citations["valid"],
            "correct": "not applicable outside evaluation",
            "invalid_ids": citations["invalid_ids"],
            "invalid_pages": citations["invalid_pages"],
        },
        "claims_with_citations": f"{cited}/{claims}",
        "claims_assessed_for_support": f"{assessed} (of which {supported} supported)",
        "claims_not_applicable": not_applicable,
        "weak_claims": weak,
        "ungrounded_numbers": numbers_missing,
        "ungrounded_names": names_missing,
        "note": "Number/name checks confirm presence in retrieved text, not correctness.",
    }

    return {
        "query": request.query,
        "answer": answer,
        "abstained": abstained,
        "coverage": rag.coverage(docs),
        "sources": sources,
        "evidence": evidence
    }


@app.post("/ingest")
def start_ingest(request: IngestRequest, user_id: str = Depends(get_current_user)):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty.")
    if not 1 <= request.num_papers <= MAX_PAPERS_PER_INGEST:
        raise HTTPException(
            status_code=400,
            detail=f"num_papers must be between 1 and {MAX_PAPERS_PER_INGEST}.",
        )
    enforce_daily_limit(user_id, "ingest")

    return ingest.start_ingest(user_id, request.topic, request.num_papers)


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str, user_id: str = Depends(get_current_user)):
    job = db.get_job(user_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No ingest job with that id.")
    return job


@app.get("/ingest")
def list_ingest_jobs(user_id: str = Depends(get_current_user)):
    return {"jobs": db.list_jobs(user_id)}


@app.get("/papers")
def list_papers(user_id: str = Depends(get_current_user)):
    rag = get_user_rag(user_id)
    papers = []
    for paper_id, metadata in rag.paper_metadata.items():
        papers.append({
            "paper_id": paper_id,
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "arxiv_id": metadata.get("arxiv_id"),
            "topic": metadata.get("topic") or "uncategorized",
        })
    return {"papers": papers}


@app.post("/collections")
def create_collection(request: CollectionRequest, user_id: str = Depends(get_current_user)):
    rag = get_user_rag(user_id)
    papers = []
    for selection in request.papers:
        metadata = rag.paper_metadata.get(selection.paper_id, {})
        papers.append({
            "paper_id": selection.paper_id,
            "title": metadata.get("title", "Unknown"),
            "authors": ", ".join(metadata.get("authors", [])),
            "arxiv_id": metadata.get("arxiv_id", "Unknown"),
            "topic": metadata.get("topic") or "uncategorized",
            "why_included": selection.why_included,
        })

    topics = {p["topic"] for p in papers}
    if len(topics) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"All specimens in a comparison must come from the same expedition topic; got {sorted(topics)}.",
        )

    return db.create_collection(user_id, request.question, papers)


@app.get("/collections")
def list_collections(user_id: str = Depends(get_current_user)):
    return {"collections": db.list_collections(user_id)}


@app.get("/collections/{collection_id}")
def get_collection(collection_id: str, user_id: str = Depends(get_current_user)):
    collection = db.get_collection(user_id, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return collection


@app.post("/collections/{collection_id}/papers")
def add_paper(collection_id: str, request: PaperSelection, user_id: str = Depends(get_current_user)):
    rag = get_user_rag(user_id)
    metadata = rag.paper_metadata.get(request.paper_id, {})
    if not metadata:
        raise HTTPException(status_code=404, detail="No such paper_id in your corpus.")

    existing = db.get_collection(user_id, collection_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Collection not found.")

    new_topic = metadata.get("topic") or "uncategorized"
    existing_topics = {p.get("topic", "uncategorized") for p in existing["papers"]}
    if existing_topics and new_topic not in existing_topics:
        raise HTTPException(
            status_code=400,
            detail=f"This comparison is scoped to '{next(iter(existing_topics))}'; "
                   f"'{request.paper_id}' is from '{new_topic}'.",
        )

    paper = {
        "paper_id": request.paper_id,
        "title": metadata.get("title", "Unknown"),
        "authors": ", ".join(metadata.get("authors", [])),
        "arxiv_id": metadata.get("arxiv_id", "Unknown"),
        "topic": new_topic,
        "why_included": request.why_included,
    }

    collection = db.add_paper_to_collection(user_id, collection_id, paper)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return collection


@app.delete("/collections/{collection_id}/papers/{paper_id}")
def remove_paper(collection_id: str, paper_id: str, user_id: str = Depends(get_current_user)):
    result = db.remove_paper_from_collection(user_id, collection_id, paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    if result == "not_found":
        raise HTTPException(status_code=404, detail="That paper is not in this collection.")
    return result


@app.post("/collections/{collection_id}/compare")
def compare_collection(collection_id: str, user_id: str = Depends(get_current_user)):
    rag = get_user_rag(user_id)
    collection = db.get_collection(user_id, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    enforce_daily_limit(user_id, "compare")

    settings = db.get_settings(user_id)
    paper_ids = [p["paper_id"] for p in collection["papers"]]
    rows = synthesis.compare_papers(
        rag, collection["question"], paper_ids,
        model=settings["model"], temperature=settings["temperature"], language_style=settings["language_style"],
    )
    contradictions = synthesis.find_contradictions(
        collection["question"], rows,
        model=settings["model"], temperature=settings["temperature"], language_style=settings["language_style"],
    )

    return db.update_collection(user_id, collection_id, comparison=rows, contradictions=contradictions)


@app.post("/collections/{collection_id}/followups")
def followups_for_collection(collection_id: str, user_id: str = Depends(get_current_user)):
    collection = db.get_collection(user_id, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    if not collection.get("comparison"):
        raise HTTPException(
            status_code=400,
            detail="Run /collections/{id}/compare before requesting follow-ups.",
        )
    enforce_daily_limit(user_id, "followups")

    settings = db.get_settings(user_id)
    suggestions = synthesis.suggest_followups(
        collection["question"], collection["comparison"],
        model=settings["model"], temperature=settings["temperature"], language_style=settings["language_style"],
    )

    return db.update_collection(user_id, collection_id, followups=suggestions)


@app.get("/evaluation")
def evaluation_report(user_id: str = Depends(get_current_user)):
    # Per-user and dynamic: reflects YOUR corpus, from questions generated on
    # your own ingested papers (see dynamic_eval.py), not a fixed benchmark
    # against an unrelated demo corpus. Requires auth again as a result -
    # there's no longer one shared report to serve publicly.
    results = db.get_evaluation_results(user_id)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="No verification ledger yet. Run an expedition first - it generates one automatically.",
        )

    summary = results.get("summary", {})
    annotated = {}
    for section, metrics in summary.items():
        annotated[section] = {
            name: {
                "value": value,
                "label": EVAL_METRICS.get(name, {}).get("label", ""),
                "higher_is_better": EVAL_METRICS.get(name, {}).get("higher_is_better"),
            }
            for name, value in metrics.items()
        }

    return {
        "summary": annotated,
        "questions": results.get("questions", []),
        "unanswerable_questions": results.get("unanswerable_questions", []),
        "updated_at": results.get("updated_at"),
    }
