"""Per-collection synthesis: compare selected papers, then suggest follow-ups.

Unlike rag.qa_chain (retrieval across the whole corpus for one question),
compare_papers scopes retrieval to one paper at a time, because a comparison
needs what THAT paper reports, not whatever chunk is most similar to the
question corpus-wide.
"""

import json
import re

from langchain_openai import ChatOpenAI

MODEL = "gpt-5-nano"

COMPARE_FIELDS = ("approach", "evaluation_setting", "main_finding", "author_limitation")

COMPARE_SYSTEM_PROMPT = """
You are filling in one row of a paper-comparison table for the research question:
"{question}"

You are given retrieved excerpts from a single paper. Using ONLY these excerpts:

Return a JSON object with exactly these keys: approach, evaluation_setting,
main_finding, author_limitation, supporting_pages.

For approach, evaluation_setting, main_finding, and author_limitation:
- Give a short (1-3 sentence) answer grounded in the excerpts.
- If the excerpts don't cover it, use exactly "unclear" (the paper may say this,
  the excerpts just didn't surface it).
- If the field doesn't apply to this paper (e.g. a position paper with no
  evaluation_setting), use exactly "not applicable". Do not invent an
  evaluation to fill the field.
- Do not use prior knowledge about this paper or topic.

supporting_pages: a list of the page numbers (integers) from the excerpts that
back the fields above.

Return ONLY the JSON object, no other text.

Retrieved excerpts:
{context}
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are proposing follow-up research directions for the question:
"{question}"

You are given a comparison table of papers the user selected. Propose up to 5
follow-ups. Each must be one of exactly three types, and you must not blur them:

- "author_proposed": the paper(s) explicitly state this as future work. Only
  use this type if the excerpts actually contain such a statement.
- "unresolved_in_selection": the selected papers leave this open (e.g. a
  stated limitation, or two papers that don't cover the same setting), but
  the papers themselves don't propose it as future work.
- "assistant_proposed": your own suggestion, grounded in a specific limitation
  or gap in the selection. Never claim this is novel or "has not been done" —
  retrieval failure here is not evidence no one has studied it. Always include
  "Still to check: whether other research has already evaluated this" or
  similar in still_to_check.

For each follow-up return an object with keys: type, possible_followup,
why_investigate, starting_experiment, still_to_check, evidence (a list of
objects with paper_id and page_number pointing at the specific passage this
is grounded in).

Do not propose anything that isn't traceable to a specific row in the table.
Return ONLY a JSON array, no other text.

Comparison table:
{table}
"""


def _extract_json(raw):
    # Model may wrap JSON in a code fence despite instructions; strip it.
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return json.loads(text.strip())


def _format_paper_context(docs):
    parts = []
    for doc in docs:
        parts.append(
            f"[page {doc.metadata.get('page_number')}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def compare_papers(rag, question, paper_ids, k=6):
    if rag.vectorstore is None:
        raise ValueError("Vectorstore not initialized.")

    llm = ChatOpenAI(model=MODEL)
    rows = []

    for paper_id in paper_ids:
        metadata = rag.paper_metadata.get(paper_id, {})
        docs = rag.vectorstore.similarity_search(
            question, k=k, filter={"paper_id": paper_id}
        )

        row = {
            "paper_id": paper_id,
            "title": metadata.get("title", "Unknown"),
            "arxiv_id": metadata.get("arxiv_id", "Unknown"),
        }

        if not docs:
            row.update({f: "unclear" for f in COMPARE_FIELDS})
            row["supporting_pages"] = []
            row["note"] = "No chunks found for this paper_id in the vectorstore."
            rows.append(row)
            continue

        prompt = COMPARE_SYSTEM_PROMPT.format(
            question=question, context=_format_paper_context(docs)
        )
        raw = llm.invoke(prompt).content

        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            parsed = {f: "unclear" for f in COMPARE_FIELDS}
            parsed["supporting_pages"] = []
            parsed["note"] = "Model output was not valid JSON; treated as unclear."

        for field in COMPARE_FIELDS:
            row[field] = parsed.get(field, "unclear")
        row["supporting_pages"] = parsed.get("supporting_pages", [])
        if "note" in parsed:
            row["note"] = parsed["note"]

        rows.append(row)

    return rows


def suggest_followups(question, comparison_rows, max_suggestions=5):
    llm = ChatOpenAI(model=MODEL)

    table_lines = []
    for row in comparison_rows:
        table_lines.append(
            f"- paper_id: {row['paper_id']} | title: {row['title']}\n"
            f"  approach: {row.get('approach')}\n"
            f"  evaluation_setting: {row.get('evaluation_setting')}\n"
            f"  main_finding: {row.get('main_finding')}\n"
            f"  author_limitation: {row.get('author_limitation')}\n"
            f"  supporting_pages: {row.get('supporting_pages')}"
        )
    table = "\n".join(table_lines)

    prompt = FOLLOWUP_SYSTEM_PROMPT.format(question=question, table=table)
    raw = llm.invoke(prompt).content

    try:
        suggestions = _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return []

    return suggestions[:max_suggestions]
