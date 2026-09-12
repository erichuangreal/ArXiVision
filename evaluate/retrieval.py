"""Retrieval: did the retriever put the right chunk in front of the model?"""


def is_expected(doc, test):
    # A hit needs the expected paper and a page that holds the answer.
    return (
        doc.metadata.get("arxiv_id") == test["expected_paper"]
        and
        doc.metadata.get("page_number") in test["expected_pages"]
    )


def evaluate_retrieval(rag, tests):
    # No model call, so retrieval failures stay separate from generation ones.
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
