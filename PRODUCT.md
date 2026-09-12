# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite SPA, calling the existing FastAPI backend (`api.py`) as a JSON API over HTTP, authenticated with an `X-API-Key` header. No deploy target decided yet.

## Users

Two audiences for the same product, by explicit choice:

- **The builder, doing real work:** picks a research topic, ingests an arXiv corpus, compares papers against a specific research question, and gets grounded follow-up study ideas.
- **Recruiters/visitors evaluating this as a portfolio piece:** need to see competence quickly — a working retrieval-augmented system, real benchmark numbers, and a coherent product idea, not a generic chatbot-over-PDFs demo.

The frontend must hold up under actual research use and read as competent to a fast-skimming visitor — not be designed for one audience at the expense of the other.

## Product Purpose

A RAG research assistant that helps someone explore a research question, not just search papers. Given a topic, it ingests a small arXiv corpus, lets the user build a paper collection around a specific question, compares what the selected papers actually establish (approach, evaluation setting, main finding, author-stated limitation), and suggests follow-up studies grounded in specific evidence. Success is the user leaving having compared real papers and identified a concrete, evidence-backed next research direction — not just having gotten a summarized answer.

## Positioning

Distinct from a plain "chat with your PDFs" RAG demo in three ways:

1. Comparisons are structured (approach / evaluation setting / main finding / author limitation) with explicit `unclear` / `not applicable` states, rather than forcing an answer where none exists.
2. Follow-up suggestions are provenance-tagged — author-proposed, unresolved-in-selection, or assistant-proposed — and assistant-proposed ones never claim novelty, since a retrieval failure isn't evidence something hasn't been studied elsewhere.
3. The system exposes its own evaluation transparently: a public benchmark page (retrieval accuracy, citation/claim grounding, abstention behavior) plus a live evidence panel under every answer, including limitations the automated checks don't catch.

## Operating Context

- Backend: FastAPI (`api.py`), already built and working locally. This project is the frontend build against it.
- Auth: one API key per user (`POST /users` issues it once; sent as `X-API-Key` on every other call). No password, no session model, no key recovery — losing the key loses that corpus permanently.
- Each user has an isolated corpus: their own downloaded papers, extracted text, and Chroma vector store, populated by ingesting arXiv papers for a topic of their choosing. A new user's corpus is empty until they run an ingest.
- Ingest is a background job (arXiv search -> download -> extract -> embed) taking anywhere from seconds to a few minutes. The frontend must poll job status and show real progress (stage, message, current/total counts) — this was an explicit requirement, not a nice-to-have "loading spinner."
- Core workflow: ingest a topic -> browse/search the resulting papers -> create a collection around a research question (pick papers, state why each was included) -> run comparison -> run follow-up suggestions. Editing a collection's papers clears its existing comparison/follow-ups server-side (they no longer apply to the new paper set).
- Supporting workflow: ask an ad-hoc question and get a cited answer with a live evidence panel.
- The evaluation page reports a fixed benchmark corpus, not the visitor's own data, and is public (no auth) by design.

## Capabilities and Constraints

Existing backend API surface the frontend is built against:

- Auth: `POST /users`
- Corpus: `POST /ingest`, `GET /ingest`, `GET /ingest/{id}`, `GET /papers`, `POST /search`, `POST /ask`
- Collections: `POST /collections`, `GET /collections`, `GET /collections/{id}`, `POST /collections/{id}/papers`, `DELETE /collections/{id}/papers/{paper_id}`, `POST /collections/{id}/compare`, `POST /collections/{id}/followups`
- Evaluation: `GET /evaluation` (public, no auth)

Constraints:

- No frontend exists yet; this is a from-scratch build.
- Abuse/cost protection for public access (rate limiting, spend caps, or requiring visitors to bring their own OpenAI key) is an explicitly open, deferred decision. Do not assume it is solved, and do not design as though unlimited free usage is safe to imply.
- Every corpus-dependent screen needs a real empty state (no papers ingested yet), not just a loading state.

## Evidence on Hand

A real, working benchmark report (`GET /evaluation`): retrieval hit@1/hit@4/paper_hit@4/MRR@4, grounding metrics (citation validity/correctness/coverage, claim support, numeric/name grounding), and abstention rate/false-refusal rate — each with a plain-language label, plus the actual test questions and answers behind the numbers.

No name, logo, or brand identity established yet.

## Product Principles

1. Never let the interface imply more certainty than the system has — `unclear`/`not applicable` are first-class states, not failures, and follow-up ideas are always labeled by provenance.
2. Progress must be honest and visible — any operation that takes real time shows what's actually happening, not a generic spinner.
3. The evaluation/trust page is a product surface, not an afterthought — it's the credibility mechanism for both real research use and recruiter evaluation.
4. Design for two audiences without splitting the product: the same real workflow must hold up under actual use and read as competent work to a fast-skimming visitor.

## Accessibility & Inclusion

No product-specific requirement established yet.
