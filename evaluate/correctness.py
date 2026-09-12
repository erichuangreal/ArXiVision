"""Correctness: does the final answer state the same fact as the ground
truth, regardless of which chunk it came from or which page it cites?

Retrieval and grounding both use the expected page as a proxy for "the
answer is right" - neither actually compares the answer text to a known-good
answer. A correct fact restated from a different (also-valid) page reads as
a retrieval miss under that proxy even though the user got the right answer.
This closes that gap with a direct LLM-judge comparison.
"""

import json
import re

from langchain_openai import ChatOpenAI

# A judge call is one short comparison, not authorship - a cheaper model than
# QUESTION_MODEL is fine here.
JUDGE_MODEL = "gpt-5-mini"

JUDGE_PROMPT = """
You are grading whether a model's answer states the same fact as a reference answer.

Question: {query}
Reference answer: {ground_truth}
Model's answer: {answer}

Judge only whether the MODEL'S ANSWER conveys the same key fact(s) as the
reference, regardless of phrasing, page number, or which source it cites.
Extra correct detail beyond the reference is fine. A refusal to answer, a
hedge, or a different fact is incorrect.

Return ONLY a JSON object: {{"correct": true or false, "reason": "one short sentence"}}
"""


def _extract_json(raw):
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return json.loads(text.strip())


def judge_answer(llm, query, ground_truth, answer):
    prompt = JUDGE_PROMPT.format(query=query, ground_truth=ground_truth, answer=answer)
    raw = llm.invoke(prompt).content
    try:
        parsed = _extract_json(raw)
        return bool(parsed.get("correct")), parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Malformed judge output is ungraded, not a silent False - it must not
        # move the denominator either way.
        return None, "judge response could not be parsed"


def evaluate_correctness(tests, per_question):
    # Reuses the answers evaluate_grounding already generated - one judge
    # call per question, no extra QA-chain calls.
    answers_by_query = {q["query"]: q["answer"] for q in per_question}
    llm = ChatOpenAI(model=JUDGE_MODEL)

    correct_count = 0
    graded = 0
    by_query = {}

    for test in tests:
        answer = answers_by_query.get(test["query"])
        if answer is None:
            continue

        correct, reason = judge_answer(llm, test["query"], test["ground_truth"], answer)
        if correct is None:
            continue

        graded += 1
        if correct:
            correct_count += 1
        by_query[test["query"]] = {"answer_correct": correct, "correctness_reason": reason}

        print(
            "\nQuery:", test["query"],
            "\nGround truth:", test["ground_truth"],
            "\nModel answer:", answer,
            "\nJudged correct:", correct, "-", reason
        )

    results = {
        "answer_correctness": correct_count / graded if graded else 0.0
    }

    print("\n--- Correctness ---")
    print(f"answer_correctness: {results['answer_correctness'] * 100:.2f}%")

    return results, by_query
