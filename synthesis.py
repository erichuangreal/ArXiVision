"""Per-collection synthesis: compare selected papers, then suggest follow-ups.

Unlike rag.qa_chain (retrieval across the whole corpus for one question),
compare_papers scopes retrieval to one paper at a time, because a comparison
needs what THAT paper reports, not whatever chunk is most similar to the
question corpus-wide.
"""

import json
import re

from langchain_openai import ChatOpenAI

from rag_class import DEFAULT_MODEL, DEFAULT_TEMPERATURE, LANGUAGE_STYLE_INSTRUCTIONS

COMPARE_FIELDS = ("approach", "evaluation_setting", "main_finding", "author_limitation")

COMPARE_SYSTEM_PROMPT = """
You are filling in one row of a paper-comparison table for the research question:
"{question}"

{style_instruction}

You are given retrieved excerpts from a single paper. Using ONLY these excerpts,
answer each field AS IT RELATES TO THE RESEARCH QUESTION ABOVE - not a generic
summary of the paper.

Return a JSON object with exactly these keys: approach, evaluation_setting,
main_finding, author_limitation, supporting_pages.

- approach: how this paper approaches the research question specifically.
- evaluation_setting: how this paper evaluates whatever it reports about the
  research question.
- main_finding: what this paper found about the research question.
- author_limitation: a limitation the authors state that bears on how much
  their answer to the research question can be trusted.

For all four fields:
- Give a short (1-3 sentence) answer grounded in the excerpts.
- If the excerpts don't cover it, use exactly "unclear" (the paper may say this,
  the excerpts just didn't surface it).
- If the field doesn't apply to this paper's relationship to the research
  question (e.g. the paper doesn't address the question at all, or is a
  position paper with no evaluation_setting), use exactly "not applicable".
  Do not invent an evaluation, or answer about the paper in general, to fill
  the field.
- Do not use prior knowledge about this paper or topic.

supporting_pages: a list of the page numbers (integers) from the excerpts that
back the fields above.

Return ONLY the JSON object, no other text.

Retrieved excerpts:
{context}
"""

CONTRADICTIONS_SYSTEM_PROMPT = """
You are auditing a paper-comparison table built to answer the research question:
"{question}"

{style_instruction}

You are given the comparison table below, one row per selected paper. Find:

- "contradictions": places where two or more papers disagree about the
  research question - one reports X, another reports not-X (or a materially
  different answer) for what is otherwise the same question. Only report a
  contradiction you can point to specific rows for; do not infer one from
  silence or from an "unclear"/"not applicable" field.
- "gaps": aspects of the research question that NONE of the selected papers
  address (every row is "unclear" or "not applicable" for that aspect), so
  the user knows what this selection cannot tell them.

For each contradiction, return an object with keys: description (one
sentence stating what the papers disagree about), paper_ids (list of the
paper_ids involved), evidence (list of objects with paper_id and
page_number, drawn from that row's supporting_pages).

For each gap, return an object with keys: description (one sentence naming
what the research question asks that isn't covered), paper_ids (the
paper_ids whose rows are silent on it).

Return ONLY a JSON object with exactly these keys: contradictions, gaps
(each a list, possibly empty). No other text.

Comparison table:
{table}
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are proposing follow-up research directions for the question:
"{question}"

{style_instruction}

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


def compare_papers(
    rag, question, paper_ids, k=6,
    model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, language_style="standard"
):
    if rag.vectorstore is None:
        raise ValueError("Vectorstore not initialized.")

    llm = ChatOpenAI(model=model, temperature=temperature)
    style_instruction = LANGUAGE_STYLE_INSTRUCTIONS.get(
        language_style, LANGUAGE_STYLE_INSTRUCTIONS["standard"]
    )
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
            question=question, style_instruction=style_instruction, context=_format_paper_context(docs)
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


def _format_comparison_table(comparison_rows):
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
    return "\n".join(table_lines)


def find_contradictions(
    question, comparison_rows,
    model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, language_style="standard"
):
    # A per-paper row is filled in isolation (compare_papers scopes retrieval
    # to one paper at a time), so nothing sees the other rows until here -
    # this is the only pass that can actually compare across papers.
    llm = ChatOpenAI(model=model, temperature=temperature)
    style_instruction = LANGUAGE_STYLE_INSTRUCTIONS.get(
        language_style, LANGUAGE_STYLE_INSTRUCTIONS["standard"]
    )

    prompt = CONTRADICTIONS_SYSTEM_PROMPT.format(
        question=question, style_instruction=style_instruction,
        table=_format_comparison_table(comparison_rows),
    )
    raw = llm.invoke(prompt).content

    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return {"contradictions": [], "gaps": []}

    return {
        "contradictions": parsed.get("contradictions", []),
        "gaps": parsed.get("gaps", []),
    }


def suggest_followups(
    question, comparison_rows, max_suggestions=5,
    model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, language_style="standard"
):
    llm = ChatOpenAI(model=model, temperature=temperature)
    style_instruction = LANGUAGE_STYLE_INSTRUCTIONS.get(
        language_style, LANGUAGE_STYLE_INSTRUCTIONS["standard"]
    )

    table = _format_comparison_table(comparison_rows)

    prompt = FOLLOWUP_SYSTEM_PROMPT.format(question=question, style_instruction=style_instruction, table=table)
    raw = llm.invoke(prompt).content

    try:
        suggestions = _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return []

    return suggestions[:max_suggestions]
