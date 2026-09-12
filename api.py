from fastapi import FastAPI
from pydantic import BaseModel

from rag_implementation import RAGClass
from rag_class import split_abstention


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


class SearchRequest(BaseModel):
    query: str


class AskRequest(BaseModel):
    query: str


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

    sources = []

    for doc in response["context"]:
        sources.append({
            "title": doc.metadata.get("title"),
            "authors": doc.metadata.get("authors"),
            "arxiv_id": doc.metadata.get("arxiv_id"),
            "page_number": doc.metadata.get("page_number")
        })

    abstained, answer = split_abstention(response["answer"])

    return {
        "query": request.query,
        "answer": answer,
        "abstained": abstained,
        "coverage": rag.coverage(response["context"]),
        "sources": sources
    }