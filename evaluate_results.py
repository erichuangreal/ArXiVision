# 1. Retrieval
#   Hit@1
#   Hit@4
#   Paper Hit@4
#   MRR@4

# 2. Grounding
#   Answers citing anything
#   Citation validity
#   Citation correctness
#   Citation coverage
#   Numeric grounding
#   Name grounding

# 3. Abstention
#   Abstention rate
#   False refusal rate

import json
import re
from pathlib import Path

# Importing from rag_implementation also loads the .env / OPENAI_API_KEY check,
# the same way api.py does it.
from rag_implementation import RAGClass

DATA_PATH = "processed_text/"
QUESTIONS_PATH = "evaluation_questions.json"
RESULTS_PATH = "evaluation_results.json"


def build_rag():
    # Same startup sequence as rag_implementation.py and api.py.
    rag = RAGClass(DATA_PATH)
    rag.load_documents()
    rag.split_documents()
    rag.create_vectorstore()
    rag.setup_retriever()
    rag.setup_qa_chain()
    return rag


def load_tests():
    path = Path(QUESTIONS_PATH)
    tests = json.loads(path.read_text(encoding="utf-8"))
    if not tests:
        raise ValueError(f"No test questions found in {path.resolve()}")

    # Questions marked "answerable": false have no expected paper or ground
    # truth. They exist to check that the system refuses instead of inventing.
    answerable = [t for t in tests if t.get("answerable", True)]
    unanswerable = [t for t in tests if not t.get("answerable", True)]

    print(
        f"Loaded {len(tests)} evaluation questions "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)."
    )
    return answerable, unanswerable


def is_expected(doc, test):
    # A chunk counts as a hit if it comes from the expected paper AND from one
    # of the pages that actually contains the answer.
    return (
        doc.metadata.get("arxiv_id") == test["expected_paper"]
        and
        doc.metadata.get("page_number") in test["expected_pages"]
    )


# 1. Retrieval
# No model call here, so retrieval failures stay separate from generation ones.
# If Hit@4 is low, no prompt change will fix the answers.

def evaluate_retrieval(rag, tests):
    hits_at_1 = 0
    hits_at_4 = 0
    paper_hits = 0
    reciprocal_ranks = 0.0

    for test in tests:
        docs = rag.retriever.invoke({
            "input": test["query"]
        })

        rank = None
        for i, doc in enumerate(docs):
            if is_expected(doc, test):
                rank = i + 1
                break

        if any(d.metadata.get("arxiv_id") == test["expected_paper"] for d in docs):
            paper_hits += 1

        if rank is not None:
            hits_at_4 += 1
            reciprocal_ranks += 1 / rank
            if rank == 1:
                hits_at_1 += 1

        print(
            "\nQuery:", test["query"],
            "\nExpected:", test["expected_paper"], "pages", test["expected_pages"],
            "\nRetrieved:", [
                (d.metadata.get("arxiv_id"), d.metadata.get("page_number"))
                for d in docs
            ],
            "\nRank of first correct chunk:", rank
        )

    total = len(tests)
    results = {
        "hit@1": hits_at_1 / total,
        "hit@4": hits_at_4 / total,
        "paper_hit@4": paper_hits / total,
        "mrr@4": reciprocal_ranks / total
    }

    print("\n--- Retrieval ---")
    for name, value in results.items():
        print(f"{name}: {value * 100:.2f}%" if name != "mrr@4" else f"{name}: {value:.3f}")

    return results


# 2. Grounding
# These look at the answer text itself and compare it against the chunks that
# were actually retrieved. No model is involved, so nothing here can hallucinate.

ARXIV_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")

# "page 3", "pages 3 and 24", "pp. 7, 10", "p.16"
PAGE_PATTERN = re.compile(
    r"\b(?:pages?|pp?\.)\s*(\d+(?:\s*(?:,|and|-|–|&)\s*\d+)*)",
    re.IGNORECASE
)

# Numbers, with optional decimal part, thousands separators and percent sign.
NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")

# Spaces only, not \s: across a newline this glued a bullet's last word to the
# next block's first word and reported the pair as an ungrounded name.
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[ ]+[A-Z][a-z]+)+\b")

# Title-case phrases that are common in this domain and are not people.
NAME_STOPWORDS = {
    "Large Language", "Language Model", "Language Models", "Artificial Intelligence",
    "General Purpose", "General-Purpose", "Public Service", "Chain Of", "Chain Of Thought",
    "The Paper", "The Papers", "The Retrieved", "The Model", "The Dataset",
    "Table", "Section", "Figure", "Appendix", "Page"
}

# A refusal is a source word near a negated reporting verb. Matching stems this
# way catches "do not discuss", which a fixed phrase list missed.
SOURCE_WORDS = r"(?:papers?|context|sources?|documents?|excerpts?|text)"
NEGATIONS = (
    r"(?:do(?:es)?\s+not|do\s?n'?t|does\s?n'?t|did\s+not|cannot|can'?t"
    r"|lack\w*|contain\s+no|provide\s+no|include\s+no)"
)
REPORTING_VERBS = (
    r"(?:contain|discuss|mention|address|provide|cover|include|report"
    r"|specify|state|say|give|offer|answer)"
)

ABSTENTION_PATTERN = re.compile(
    rf"\b{SOURCE_WORDS}\b[^.!?]{{0,60}}?\b{NEGATIONS}\s+(?:\w+\s+){{0,2}}{REPORTING_VERBS}\b"
    rf"|\b{NEGATIONS}\s+(?:\w+\s+){{0,3}}\b{SOURCE_WORDS}\b"
    rf"|\b(?:insufficient|not\s+enough|no)\s+(?:relevant\s+)?information\b"
    rf"|\b(?:cannot|can'?t|unable\s+to)\s+(?:be\s+)?(?:answer|determine|establish)",
    re.IGNORECASE
)


def looks_like_abstention(answer):
    # Only the opening block: a refusal leads, while a real answer may note what
    # a paper "does not discuss" partway through and is not a refusal.
    head = re.split(r"\n\s*\n", answer.strip())[0]
    return bool(ABSTENTION_PATTERN.search(head))


def normalise_number(token):
    return token.replace(",", "").rstrip("%")


def extract_citations(answer):
    # Returns the arXiv IDs and page numbers the answer claims to cite, plus the
    # character spans they occupy so they can be masked out of other checks.
    ids = set(ARXIV_PATTERN.findall(answer))
    pages = set()
    spans = []

    for match in ARXIV_PATTERN.finditer(answer):
        spans.append(match.span())

    for match in PAGE_PATTERN.finditer(answer):
        for page in re.findall(r"\d+", match.group(1)):
            pages.add(int(page))
        spans.append(match.span())

    return ids, pages, spans


def mask_spans(text, spans):
    # Blank out citation spans so page numbers and arXiv IDs are not re-checked
    # as if they were factual figures in the prose.
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def check_citations(answer, docs, test):
    # Validity: every cited source was actually retrieved.
    # Correctness: at least one cited page is a page we expected.
    retrieved_ids = {d.metadata.get("arxiv_id") for d in docs}
    retrieved_pages = {d.metadata.get("page_number") for d in docs}

    cited_ids, cited_pages, _ = extract_citations(answer)

    invalid_ids = cited_ids - retrieved_ids
    invalid_pages = cited_pages - retrieved_pages

    # Checked as separate sets, not pairs: the model cites in free prose and
    # pairing IDs to pages reliably is not worth the parsing.
    valid = not invalid_ids and not invalid_pages
    cited_anything = bool(cited_ids or cited_pages)

    expected_pages = set(test.get("expected_pages", []))
    correct = bool(cited_pages & expected_pages) if expected_pages else None

    return {
        "cited_anything": cited_anything,
        "valid": valid,
        "correct": correct,
        "invalid_ids": sorted(invalid_ids),
        "invalid_pages": sorted(invalid_pages)
    }


def check_numbers(answer, context_text):
    # Every figure quoted in the answer should appear somewhere in the retrieved
    # text. Citation spans are masked first so page numbers are not counted here.
    _, _, spans = extract_citations(answer)
    prose = mask_spans(answer, spans)

    context_numbers = {
        normalise_number(t) for t in NUMBER_PATTERN.findall(context_text)
    }

    found = []
    missing = []
    for token in NUMBER_PATTERN.findall(prose):
        value = normalise_number(token)
        # Accept an exact match on the normalised value, or the raw digits
        # appearing in the context (catches "10%" against "10.0%").
        if value in context_numbers or value in context_text:
            found.append(token)
        else:
            missing.append(token)

    return found, missing


def check_names(answer, context_text, known_authors):
    # Corpus authors and title-case phrases must both appear in the retrieved
    # text. The second probe is a heuristic, so treat "missing" as a prompt to look.
    candidates = set()

    for author in known_authors:
        if author in answer:
            candidates.add(author)

    for phrase in NAME_PATTERN.findall(answer):
        if phrase not in NAME_STOPWORDS:
            candidates.add(phrase)

    found = []
    missing = []
    for name in sorted(candidates):
        surname = name.split()[-1]
        if name in context_text or surname in context_text:
            found.append(name)
        else:
            missing.append(name)

    return found, missing


# A blank line, bullet, or numbered item starts a block. The model cites once per
# block, so the block is the unit a citation covers.
BLOCK_PATTERN = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)]\s))")


def split_offsets(text, pattern):
    # Returns (start, end) ranges rather than substrings, so citation positions
    # found in the full answer can be matched back against each range.
    cuts = (
        [0]
        + [match.end() for match in pattern.finditer(text)]
        + [len(text)]
    )
    return list(zip(cuts, cuts[1:]))


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def check_citation_coverage(answer):
    # The system prompt asks for a citation on every factual claim. This counts
    # how many asserting sentences sit under one.
    _, _, spans = extract_citations(answer)

    # Both splits run on a masked copy so the period in "p. 11" is not read as a
    # sentence end, which would sever every citation from the claim it supports.
    masked = mask_spans(answer, spans)

    claims = 0
    cited = 0
    uncited = []

    for block_start, block_end in split_offsets(masked, BLOCK_PATTERN):
        # A trailing citation covers every claim in its bullet. Charging each
        # sentence separately caps a well-cited answer at roughly a third.
        block_cited = any(
            s < block_end and e > block_start for s, e in spans
        )

        block = masked[block_start:block_end]
        for offset_start, offset_end in split_offsets(block, SENTENCE_PATTERN):
            sentence = answer[
                block_start + offset_start:block_start + offset_end
            ].strip()

            # Skip very short fragments, bare citations, and refusals, which
            # assert nothing.
            if len(sentence) < 25 or looks_like_abstention(sentence):
                continue

            claims += 1
            if block_cited:
                cited += 1
            else:
                uncited.append(sentence)

    return claims, cited, uncited


def collect_authors(paper_metadata):
    authors = set()
    for metadata in paper_metadata.values():
        for author in metadata.get("authors", []):
            authors.add(author)
    return authors


def format_context(docs):
    # Must mirror the document_prompt in RAGClass.setup_qa_chain: the model sees
    # title and authors too, so grounding has to be checked against the same text.
    parts = []
    for doc in docs:
        parts.append(
            f"[{doc.metadata.get('arxiv_id')} page {doc.metadata.get('page_number')}]\n"
            f"Paper: {doc.metadata.get('title')}\n"
            f"Authors: {doc.metadata.get('authors')}\n\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def evaluate_grounding(rag, tests, known_authors):
    # Runs the full chain on every answerable question and checks the answer
    # text against the chunks it was actually given.
    answers_with_citations = 0
    answers_with_valid_citations = 0
    answers_with_correct_citations = 0
    total_numbers = 0
    grounded_numbers = 0
    total_names = 0
    grounded_names = 0
    total_claims = 0
    cited_claims = 0

    per_question = []

    for test in tests:
        response = rag.qa_chain.invoke({"input": test["query"]})
        answer = response["answer"]
        docs = response["context"]
        context_text = format_context(docs)

        citations = check_citations(answer, docs, test)
        numbers_found, numbers_missing = check_numbers(answer, context_text)
        names_found, names_missing = check_names(answer, context_text, known_authors)
        claims, cited, uncited = check_citation_coverage(answer)

        if citations["cited_anything"]:
            answers_with_citations += 1
            if citations["valid"]:
                answers_with_valid_citations += 1
            if citations["correct"]:
                answers_with_correct_citations += 1

        total_numbers += len(numbers_found) + len(numbers_missing)
        grounded_numbers += len(numbers_found)
        total_names += len(names_found) + len(names_missing)
        grounded_names += len(names_found)
        total_claims += claims
        cited_claims += cited

        per_question.append({
            "query": test["query"],
            "answer": answer,
            "abstained": looks_like_abstention(answer),
            "citations_valid": citations["valid"],
            "citations_correct": citations["correct"],
            "ungrounded_numbers": numbers_missing,
            "ungrounded_names": names_missing,
            "claims_cited": f"{cited}/{claims}"
        })

        print(
            "\nQuery:", test["query"],
            "\nModel Answer:", answer,
            "\nCitations valid:", citations["valid"],
            "| pointing at an expected page:", citations["correct"],
            "\nInvalid citations:", citations["invalid_ids"], citations["invalid_pages"],
            "\nUngrounded numbers:", numbers_missing,
            "\nUngrounded names:", names_missing,
            "\nClaims cited:", f"{cited}/{claims}",
            "\nUncited claims:", uncited
        )

    total = len(tests)
    grounding = {
        "answers_citing_anything": answers_with_citations / total,
        # Of the answers that cited anything, how many cited only real sources.
        "citation_validity": (
            answers_with_valid_citations / answers_with_citations
            if answers_with_citations else 0.0
        ),
        "citation_correctness": (
            answers_with_correct_citations / answers_with_citations
            if answers_with_citations else 0.0
        ),
        "citation_coverage": cited_claims / total_claims if total_claims else 0.0,
        "numeric_grounding": grounded_numbers / total_numbers if total_numbers else 1.0,
        "name_grounding": grounded_names / total_names if total_names else 1.0
    }

    print("\n--- Grounding ---")
    for name, value in grounding.items():
        print(f"{name}: {value * 100:.2f}%")

    return grounding, per_question


# 3. Abstention
# Unanswerable questions should be refused; answerable ones should not be.

def evaluate_abstention(rag, answerable_results, unanswerable):
    refused_when_should = 0

    for test in unanswerable:
        response = rag.qa_chain.invoke({"input": test["query"]})
        answer = response["answer"]
        abstained = looks_like_abstention(answer)

        if abstained:
            refused_when_should += 1

        print(
            "\nUnanswerable query:", test["query"],
            "\nModel Answer:", answer,
            "\nAbstained:", abstained
        )

    # The flip side: refusing a question the corpus can answer. Without this a
    # system that refuses everything would score perfectly above.
    false_refusals = sum(1 for r in answerable_results if r["abstained"])

    results = {
        "abstention_rate": (
            refused_when_should / len(unanswerable) if unanswerable else 0.0
        ),
        "false_refusal_rate": (
            false_refusals / len(answerable_results) if answerable_results else 0.0
        )
    }

    print("\n--- Abstention ---")
    print(f"abstention_rate (unanswerable, higher is better): {results['abstention_rate'] * 100:.2f}%")
    print(f"false_refusal_rate (answerable, lower is better): {results['false_refusal_rate'] * 100:.2f}%")

    return results


if __name__ == "__main__":
    rag = build_rag()
    answerable, unanswerable = load_tests()
    known_authors = collect_authors(rag.paper_metadata)

    # Retrieval only touches metadata, so it costs nothing beyond the embeddings.
    retrieval = evaluate_retrieval(rag, answerable)

    # Everything below calls the answering model.
    grounding, per_question = evaluate_grounding(rag, answerable, known_authors)
    abstention = evaluate_abstention(rag, per_question, unanswerable)

    summary = {
        "retrieval": retrieval,
        "grounding": grounding,
        "abstention": abstention
    }

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))

    # Saved so runs can be compared after a chunk size, k, or prompt change.
    Path(RESULTS_PATH).write_text(
        json.dumps({"summary": summary, "questions": per_question}, indent=2),
        encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")
