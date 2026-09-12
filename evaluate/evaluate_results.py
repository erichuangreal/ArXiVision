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
#   Claim support
#   Numeric grounding
#   Name grounding

# 3. Abstention
#   Abstention rate
#   False refusal rate

import json
import sys
from pathlib import Path

# Running this file by path puts evaluate/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing rag_implementation also runs the .env / OPENAI_API_KEY check.
from rag_implementation import RAGClass  # noqa: E402

from evaluate.abstention import evaluate_abstention  # noqa: E402
from evaluate.grounding import collect_authors, evaluate_grounding  # noqa: E402
from evaluate.retrieval import evaluate_retrieval  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATA_PATH = EVAL_DIR.parent / "processed_text"
QUESTIONS_PATH = EVAL_DIR / "evaluation_questions.json"
RESULTS_PATH = EVAL_DIR / "evaluation_results.json"


def build_rag():
    # Same startup sequence as api.py.
    rag = RAGClass(DATA_PATH)
    rag.load_documents()
    rag.split_documents()
    rag.create_vectorstore()
    rag.setup_retriever()
    rag.setup_qa_chain()
    return rag


def load_tests():
    tests = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not tests:
        raise ValueError(f"No test questions found in {QUESTIONS_PATH}")

    # "answerable": false questions check that the system refuses.
    answerable = [t for t in tests if t.get("answerable", True)]
    unanswerable = [t for t in tests if not t.get("answerable", True)]

    print(
        f"Loaded {len(tests)} evaluation questions "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)."
    )
    return answerable, unanswerable


if __name__ == "__main__":
    rag = build_rag()
    answerable, unanswerable = load_tests()
    known_authors = collect_authors(rag.paper_metadata)

    retrieval = evaluate_retrieval(rag, answerable)

    grounding, per_question = evaluate_grounding(rag, answerable, known_authors)
    abstention, unanswerable_questions = evaluate_abstention(
        rag, per_question, unanswerable
    )

    summary = {
        "retrieval": retrieval,
        "grounding": grounding,
        "abstention": abstention
    }

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))

    # Saved so runs can be compared after a chunk size, k, or prompt change.
    RESULTS_PATH.write_text(
        json.dumps({
            "summary": summary,
            "questions": per_question,
            "unanswerable_questions": unanswerable_questions
        }, indent=2),
        encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")
