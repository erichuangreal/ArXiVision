"""Per-corpus verification ledger.

Instead of scoring every user against one fixed benchmark tied to an
unrelated demo corpus, each user's own ingested papers get their own test
questions - authored by a stronger model than the one that has to answer
them - then scored with the same retrieval/grounding/abstention checks used
everywhere else in this app (evaluate/retrieval.py, evaluate/grounding.py,
evaluate/abstention.py). Questions accumulate across expeditions: a paper
that already has one is never re-billed on a later expedition.
"""

import json
import re

from langchain_openai import ChatOpenAI

import db
from evaluate.abstention import evaluate_abstention
from evaluate.correctness import evaluate_correctness
from evaluate.grounding import claim_anchors, collect_authors, evaluate_grounding, SUPPORT_THRESHOLD
from evaluate.retrieval import evaluate_retrieval

# Deliberately independent of the user's own chat-model setting: a stronger
# model authors the exam than the one taking it.
QUESTION_MODEL = "gpt-5"

# Generic, corpus-independent probes that should never be answerable from an
# arXiv paper corpus. Kept static rather than generated per topic: "clearly
# outside any research paper corpus" doesn't need to be topic-tailored to be
# a valid abstention test, and generating a good adversarial near-miss
# reliably is a harder problem than this MVP takes on.
STATIC_UNANSWERABLE_QUESTIONS = [
    {"query": "What accuracy does ResNet-50 achieve on the ImageNet benchmark according to these papers?", "answerable": False},
    {"query": "What do these papers say about using CRISPR for gene editing?", "answerable": False},
    {"query": "How do these papers recommend configuring Kubernetes autoscaling for model serving?", "answerable": False},
]

QUESTION_PROMPT = """
You are authoring one test question for a retrieval-augmented QA system, to
verify it can correctly answer questions about a specific research paper.

You are given one excerpt from the paper "{title}" (arXiv {arxiv_id}), page {page}.

Write ONE factual question whose answer is fully contained in this excerpt,
plus the ground-truth answer.

Return a JSON object with exactly these keys:
- "query": the question, answerable using only this excerpt
- "ground_truth": a concise correct answer, grounded in the excerpt

The question must be specific enough that a system without access to this
excerpt could not guess it, and must not require outside knowledge.

Return ONLY the JSON object, no other text.

Excerpt:
{excerpt}
"""


def _extract_json(raw):
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return json.loads(text.strip())


def _pick_representative_chunk(rag, paper_id):
    # Prefer the chunk with the most body text, not a short title/author page.
    # Page 1 (abstract/intro) is avoided when a later page is available: an
    # abstract states a result tersely, while the paper's own results/methods
    # sections usually restate the same finding at greater length - which
    # out-competes the terser abstract passage in retrieval, making the
    # abstract a structurally bad source for a question meant to test
    # whether retrieval finds ITS specific passage.
    candidates = [c for c in rag.text_chunks if c.metadata.get("paper_id") == paper_id]
    if not candidates:
        return None
    body_candidates = [c for c in candidates if c.metadata.get("page_number", 1) > 1]
    pool = body_candidates or candidates
    return max(pool, key=lambda c: len(c.page_content))


def _find_additional_supporting_pages(rag, paper_id, query, ground_truth, source_page, k=4):
    # A fact stated on one page is often restated elsewhere in the same paper
    # (an abstract's terse version vs. a results section's elaboration) - if
    # the system lands on a different page that also genuinely contains the
    # ground truth, that is a correct answer, not a miss. Reuses the exact
    # anchor-presence check evaluate/grounding.py already uses for claim
    # support, applied here to the paper's other candidate pages instead of
    # a live answer. No new LLM calls.
    # A single generic anchor (e.g. just "python" for a paper centrally about
    # a Python course) recurs all over a paper regardless of context, and
    # can't actually distinguish "this page restates the fact" from "this
    # page happens to mention the same keyword." Require at least two
    # independent anchors before trusting a match at all.
    anchors = claim_anchors(ground_truth)
    if len(anchors) < 2:
        return []

    candidates = rag.vectorstore.similarity_search(query, k=k, filter={"paper_id": paper_id})

    extra_pages = []
    for doc in candidates:
        page = doc.metadata.get("page_number")
        if page is None or page == source_page or page in extra_pages:
            continue
        lowered = doc.page_content.lower()
        missing = [a for a in anchors if a not in lowered]
        score = (len(anchors) - len(missing)) / len(anchors)
        if score >= SUPPORT_THRESHOLD:
            extra_pages.append(page)

    return extra_pages


def generate_questions_for_papers(rag, paper_ids):
    llm = ChatOpenAI(model=QUESTION_MODEL)
    questions = []

    for paper_id in paper_ids:
        chunk = _pick_representative_chunk(rag, paper_id)
        if chunk is None:
            continue

        metadata = rag.paper_metadata.get(paper_id, {})
        title = metadata.get("title", "Unknown")
        arxiv_id = metadata.get("arxiv_id", "Unknown")
        page = chunk.metadata.get("page_number")

        prompt = QUESTION_PROMPT.format(
            title=title, arxiv_id=arxiv_id, page=page, excerpt=chunk.page_content
        )
        raw = llm.invoke(prompt).content

        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            continue

        if not parsed.get("query") or not parsed.get("ground_truth"):
            continue

        expected_pages = [page] if page is not None else []
        if page is not None:
            expected_pages += _find_additional_supporting_pages(
                rag, paper_id, parsed["query"], parsed["ground_truth"], page
            )

        questions.append({
            "paper_id": paper_id,
            "query": parsed["query"],
            "expected_paper": arxiv_id,
            "expected_pages": expected_pages,
            "ground_truth": parsed["ground_truth"],
            "topic": metadata.get("topic") or "uncategorized",
        })

    return questions


def run_for_user(user_id, rag, new_paper_ids, progress_callback=None):
    # Only pay generation cost for papers that don't already have a question.
    existing_paper_ids = db.verification_question_paper_ids(user_id)
    to_generate = [p for p in new_paper_ids if p not in existing_paper_ids]

    if to_generate:
        if progress_callback:
            progress_callback(f"Drafting verification questions for {len(to_generate)} new paper(s)...")
        new_questions = generate_questions_for_papers(rag, to_generate)
        if new_questions:
            db.add_verification_questions(user_id, new_questions)

    answerable = db.list_verification_questions(user_id)
    if not answerable:
        if progress_callback:
            progress_callback("No verification questions yet - skipping the verification run.")
        return None

    if progress_callback:
        progress_callback(f"Verifying retrieval over {len(answerable)} question(s)...")
    retrieval = evaluate_retrieval(rag, answerable)

    if progress_callback:
        progress_callback(f"Verifying grounding over {len(answerable)} question(s)...")
    known_authors = collect_authors(rag.paper_metadata)
    grounding, per_question = evaluate_grounding(rag, answerable, known_authors)

    if progress_callback:
        progress_callback(f"Judging answer correctness over {len(answerable)} question(s)...")
    correctness, correctness_by_query = evaluate_correctness(answerable, per_question)
    for q in per_question:
        q.update(correctness_by_query.get(q["query"], {}))

    if progress_callback:
        progress_callback(f"Verifying abstention over {len(STATIC_UNANSWERABLE_QUESTIONS)} out-of-corpus question(s)...")
    abstention, unanswerable_results = evaluate_abstention(rag, per_question, STATIC_UNANSWERABLE_QUESTIONS)

    results = {
        "summary": {
            "retrieval": retrieval,
            "grounding": grounding,
            "correctness": correctness,
            "abstention": abstention,
        },
        "questions": per_question,
        "unanswerable_questions": unanswerable_results,
    }
    db.save_evaluation_results(user_id, results)
    return results
