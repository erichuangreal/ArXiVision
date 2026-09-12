"""Grounding: compare answer text against the chunks actually retrieved."""

import re

from evaluate.abstention import looks_like_abstention


ARXIV_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")

PAGE_PATTERN = re.compile(
    r"\b(?:pages?|pp?\.)\s*(\d+(?:\s*(?:,|and|-|–|&)\s*\d+)*)",
    re.IGNORECASE
)

NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")

NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[ ]+[A-Z][a-z]+)+\b")

NAME_STOPWORDS = {
    "Large Language", "Language Model", "Language Models", "Artificial Intelligence",
    "General Purpose", "General-Purpose", "Public Service", "Chain Of", "Chain Of Thought",
    "The Paper", "The Papers", "The Retrieved", "The Model", "The Dataset",
    "Table", "Section", "Figure", "Appendix", "Page"
}


def normalise_number(token):
    return token.replace(",", "").rstrip("%")


def extract_citations(answer):
    # Returns cited arXiv IDs, page numbers, and their character spans.
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
    # Blanks citation spans so they are not re-checked as prose.
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def check_citations(answer, docs, test):
    # Were the cited sources retrieved, and do they point at an expected page?
    retrieved_ids = {d.metadata.get("arxiv_id") for d in docs}
    retrieved_pages = {d.metadata.get("page_number") for d in docs}

    cited_ids, cited_pages, _ = extract_citations(answer)

    invalid_ids = cited_ids - retrieved_ids
    invalid_pages = cited_pages - retrieved_pages

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
    # Every figure in the answer should appear in the retrieved text.
    _, _, spans = extract_citations(answer)
    prose = mask_spans(answer, spans)

    context_numbers = {
        normalise_number(t) for t in NUMBER_PATTERN.findall(context_text)
    }

    found = []
    missing = []
    for token in NUMBER_PATTERN.findall(prose):
        value = normalise_number(token)
        if value in context_numbers or value in context_text:
            found.append(token)
        else:
            missing.append(token)

    return found, missing


def check_names(answer, context_text, known_authors):
    # Every person named in the answer should appear in the retrieved text.
    candidates = set()

    for author in known_authors:
        if author in answer:
            candidates.add(author)

    for phrase in NAME_PATTERN.findall(answer):
        if phrase not in NAME_STOPWORDS:
            candidates.add(phrase)

    lowered_context = context_text.lower()

    found = []
    missing = []
    for name in sorted(candidates):
        surname = name.split()[-1].lower()
        if name.lower() in lowered_context or surname in lowered_context:
            found.append(name)
        else:
            missing.append(name)

    return found, missing


BLOCK_PATTERN = re.compile(r"\n\s*\n|\n(?=(?:[-*•]|\d+[.)])\s)")

CITATION_LINE_PATTERN = re.compile(
    r"^[\(\[]?\s*(?:sources?|papers?|citations?|references?)\b\s*:?",
    re.IGNORECASE
)


def is_claim(sentence):
    # True when a sentence asserts something, not just names a source.
    stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", sentence).strip()

    if not re.search(r"[A-Za-z]", stripped):
        return False
    if CITATION_LINE_PATTERN.match(stripped):
        return False
    if stripped.startswith("(") and stripped.endswith(")"):
        return False

    return not looks_like_abstention(stripped)


def split_offsets(text, pattern):
    # Returns (start, end) ranges so spans can be matched back to the answer.
    cuts = (
        [0]
        + [match.end() for match in pattern.finditer(text)]
        + [len(text)]
    )
    return list(zip(cuts, cuts[1:]))


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def check_citation_coverage(answer):
    # Share of claims sitting under a citation. One citation covers the answer.
    _, _, spans = extract_citations(answer)

    answer_cited = bool(spans)

    # Split on a masked copy so the period in "p. 11" is not a sentence end.
    masked = mask_spans(answer, spans)

    claims = 0
    cited = 0
    uncited = []

    for block_start, block_end in split_offsets(masked, BLOCK_PATTERN):
        block = masked[block_start:block_end]
        for offset_start, offset_end in split_offsets(block, SENTENCE_PATTERN):
            sentence = answer[
                block_start + offset_start:block_start + offset_end
            ].strip()

            if not is_claim(sentence):
                continue

            claims += 1
            if answer_cited:
                cited += 1
            else:
                uncited.append(sentence)

    return claims, cited, uncited


ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-\d+)?\b")
TECHNICAL_PATTERN = re.compile(r"\b[A-Za-z]+[-–]?\d+\w*\b|\b[a-z]+[A-Z]\w*\b")
PROPER_PATTERN = re.compile(r"(?<![.!?\n]\s)(?<!^)\b[A-Z][a-z]{2,}\b")

SUPPORT_THRESHOLD = 0.75


def claim_anchors(text):
    found = set()
    for pattern in (NUMBER_PATTERN, ACRONYM_PATTERN, TECHNICAL_PATTERN):
        found |= {m.group().lower().rstrip(".,") for m in pattern.finditer(text)}
    for match in PROPER_PATTERN.finditer(text):
        found.add(match.group().lower())
    return {a for a in found if len(a) > 1}


def check_claim_support(answer, docs):
    # Checks each claim against the page it cites, catching misattribution.
    _, _, spans = extract_citations(answer)
    masked = mask_spans(answer, spans)

    total = 0
    supported = 0
    weak = []

    for block_start, block_end in split_offsets(masked, BLOCK_PATTERN):
        block = answer[block_start:block_end].strip()
        if not is_claim(block):
            continue

        _, cited_pages, _ = extract_citations(block)
        if cited_pages:
            scoped = [d for d in docs if d.metadata.get("page_number") in cited_pages]
            # A page never retrieved supports nothing; no fallback.
            if not scoped:
                total += 1
                weak.append({
                    "claim": block[:120],
                    "missing": ["<cites unretrieved page>"]
                })
                continue
            source = format_context(scoped)
        else:
            source = format_context(docs)

        anchors = claim_anchors(block)
        if not anchors:
            continue

        lowered = source.lower()
        missing = [a for a in sorted(anchors) if a not in lowered]
        score = (len(anchors) - len(missing)) / len(anchors)

        total += 1
        if score >= SUPPORT_THRESHOLD:
            supported += 1
        else:
            weak.append({"claim": block[:120], "missing": missing[:6]})

    return total, supported, weak


def collect_authors(paper_metadata):
    authors = set()
    for metadata in paper_metadata.values():
        for author in metadata.get("authors", []):
            authors.add(author)
    return authors


def format_context(docs):
    # Mirrors the document_prompt so checks see what the model saw.
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
    # Runs the chain on every answerable question and scores each answer.
    answers_with_citations = 0
    answers_with_valid_citations = 0
    answers_with_correct_citations = 0
    total_numbers = 0
    grounded_numbers = 0
    total_names = 0
    grounded_names = 0
    total_claims = 0
    cited_claims = 0
    total_supported = 0
    supported_claims = 0

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
        n_claims, n_supported, weak = check_claim_support(answer, docs)

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
        total_supported += n_claims
        supported_claims += n_supported

        per_question.append({
            "query": test["query"],
            "answer": answer,
            "abstained": looks_like_abstention(answer),
            "citations_valid": citations["valid"],
            "citations_correct": citations["correct"],
            "ungrounded_numbers": numbers_missing,
            "ungrounded_names": names_missing,
            "claims_cited": f"{cited}/{claims}",
            "claims_supported": f"{n_supported}/{n_claims}",
            "weak_claims": weak
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
            "\nClaims supported by their cited page:", f"{n_supported}/{n_claims}",
            "\nWeakly supported:", weak,
            "\nUncited claims:", uncited
        )

    total = len(tests)
    grounding = {
        "answers_citing_anything": answers_with_citations / total,
        "citation_validity": (
            answers_with_valid_citations / answers_with_citations
            if answers_with_citations else 0.0
        ),
        "citation_correctness": (
            answers_with_correct_citations / answers_with_citations
            if answers_with_citations else 0.0
        ),
        "citation_coverage": cited_claims / total_claims if total_claims else 0.0,
        "claim_support": supported_claims / total_supported if total_supported else 1.0,
        "numeric_grounding": grounded_numbers / total_numbers if total_numbers else 1.0,
        "name_grounding": grounded_names / total_names if total_names else 1.0
    }

    print("\n--- Grounding ---")
    for name, value in grounding.items():
        print(f"{name}: {value * 100:.2f}%")

    return grounding, per_question
