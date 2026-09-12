import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_implementation import RAGClass
from rag_class import split_abstention
from evaluate.grounding import (
    check_citations,
    check_citation_coverage,
    check_claim_support,
    check_names,
    check_numbers,
    collect_authors,
    format_context,
)
import research_store
import synthesis


app = FastAPI(
    title="arXiv Research RAG API",
    version="1.0"
)


# Initialize RAG once when the API starts
rag = RAGClass("processed_text")

rag.load_documents()
rag.split_documents()
rag.create_vectorstore()
rag.setup_retriever()
rag.setup_qa_chain()

known_authors = collect_authors(rag.paper_metadata)

EVAL_RESULTS_PATH = Path(__file__).resolve().parent / "evaluate" / "evaluation_results.json"

EVAL_LABELS = {
    "hit@1": "Expected evidence appeared first",
    "hit@4": "Expected evidence appeared within the first four results",
    "paper_hit@4": "The expected paper appeared within the first four results",
    "mrr@4": "How near the top the expected evidence generally ranked (1.0 = always first)",
    "answers_citing_anything": "Share of answers that cited a source at all",
    "citation_validity": "Share of citations pointing at a source actually retrieved",
    "citation_correctness": "Share of citations pointing at the expected page (evaluation set only)",
    "citation_coverage": "Share of claims sitting under a citation",
    "claim_support": "Of claims with a checkable fact, share confirmed present in the cited source",
    "claims_not_applicable": "Count of claims with no checkable fact (opinions, labels) - excluded from claim_support",
    "numeric_grounding": "Share of numbers in the answer also present in the retrieved text (checks presence, not arithmetic correctness)",
    "name_grounding": "Share of named people in the answer also present in the retrieved text",
    "abstention_rate": "Share of unanswerable test questions the system correctly declined",
    "false_refusal_rate": "Share of answerable test questions the system wrongly declined",
}


class SearchRequest(BaseModel):
    query: str


class AskRequest(BaseModel):
    query: str


class PaperSelection(BaseModel):
    paper_id: str
    why_included: str


class CollectionRequest(BaseModel):
    question: str
    papers: List[PaperSelection]


@app.get("/")
def root():
    return {
        "message": "arXiv RAG API is running"
    }


@app.post("/search")
def search(request: SearchRequest):

    docs = rag.retriever.invoke({
        "input": request.query
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
def ask(request: AskRequest):

    response = rag.qa_chain.invoke({
        "input": request.query
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
    numbers_found, numbers_missing = check_numbers(answer, context_text)
    names_found, names_missing = check_names(answer, context_text, known_authors)
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


@app.get("/papers")
def list_papers():
    papers = []
    for paper_id, metadata in rag.paper_metadata.items():
        papers.append({
            "paper_id": paper_id,
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "arxiv_id": metadata.get("arxiv_id"),
        })
    return {"papers": papers}


@app.post("/collections")
def create_collection(request: CollectionRequest):
    papers = []
    for selection in request.papers:
        metadata = rag.paper_metadata.get(selection.paper_id, {})
        papers.append({
            "paper_id": selection.paper_id,
            "title": metadata.get("title", "Unknown"),
            "authors": ", ".join(metadata.get("authors", [])),
            "arxiv_id": metadata.get("arxiv_id", "Unknown"),
            "why_included": selection.why_included,
        })

    collection = research_store.create_collection(request.question, papers)
    return collection


@app.get("/collections/{collection_id}")
def get_collection(collection_id: str):
    collection = research_store.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return collection


@app.post("/collections/{collection_id}/compare")
def compare_collection(collection_id: str):
    collection = research_store.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")

    paper_ids = [p["paper_id"] for p in collection["papers"]]
    rows = synthesis.compare_papers(rag, collection["question"], paper_ids)

    return research_store.update_collection(collection_id, comparison=rows)


@app.post("/collections/{collection_id}/followups")
def followups_for_collection(collection_id: str):
    collection = research_store.get_collection(collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    if not collection.get("comparison"):
        raise HTTPException(
            status_code=400,
            detail="Run /collections/{id}/compare before requesting follow-ups.",
        )

    suggestions = synthesis.suggest_followups(
        collection["question"], collection["comparison"]
    )

    return research_store.update_collection(collection_id, followups=suggestions)


@app.get("/evaluation")
def evaluation_report():
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results yet. Run evaluate/evaluate_results.py first.",
        )

    results = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    summary = results.get("summary", {})

    annotated = {}
    for section, metrics in summary.items():
        annotated[section] = {
            name: {"value": value, "label": EVAL_LABELS.get(name, "")}
            for name, value in metrics.items()
        }

    return {
        "summary": annotated,
        "questions": results.get("questions", []),
        "unanswerable_questions": results.get("unanswerable_questions", []),
    }
