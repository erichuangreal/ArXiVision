"""Abstention: unanswerable questions should be refused, answerable ones not."""

import re

from rag.rag_class import split_abstention


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
    # The sentinel is authoritative; the pattern catches prose refusals.
    abstained, _ = split_abstention(answer)
    if abstained:
        return True
    
    head = re.split(r"\n\s*\n", answer.strip())[0]
    return bool(ABSTENTION_PATTERN.search(head))


def evaluate_abstention(rag, answerable_results, unanswerable):
    # Unanswerable questions should be refused, answerable ones should not.
    refused_when_should = 0
    per_question = []

    for test in unanswerable:
        response = rag.qa_chain.invoke({"input": test["query"]})
        answer = response["answer"]
        abstained = looks_like_abstention(answer)
        by_sentinel, _ = split_abstention(answer)

        if abstained:
            refused_when_should += 1

        per_question.append({
            "query": test["query"],
            "answer": answer,
            "abstained": abstained,
            "used_sentinel": by_sentinel
        })

        print(
            "\nUnanswerable query:", test["query"],
            "\nModel Answer:", answer,
            "\nAbstained:", abstained, "| via sentinel:", by_sentinel
        )

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

    return results, per_question
