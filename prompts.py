from typing import List, Dict


def hypothetical_answer_prompt(query: str) -> str:
    return (
        "You are a QMS expert. Draft a short hypothetical answer that might appear in a quality "
        "management document. Keep it factual, concise, and neutral. Do not invent policies or "
        "dates. This is only to aid retrieval.\n\n"
        f"Question: {query}\nHypothetical Answer:"
    )


def final_answer_prompt(query: str, context_blocks: List[Dict[str, str]]) -> str:
    context_text = "\n\n".join(
        [
            f"[C{idx + 1}] {block['citation']}\n{block['text']}"
            for idx, block in enumerate(context_blocks)
        ]
    )
    return (
        "You are a strict QMS assistant.\n\n"
        "Instructions:\n"
        "1. Identify the specific context passages that directly relate to the question.\n"
        "2. Extract the key facts, figures, and definitions from those passages (quote or paraphrase briefly).\n"
        "3. Explain, in a few logical steps, how these facts lead to the answer.\n"
        "4. Provide a clear, concise final answer in one or two sentences.\n\n"
        "Constraints:\n"
        "- Use only the information in the Context. Do not add outside knowledge or assumptions.\n"
        "- If the context is ambiguous or incomplete, explicitly state what’s missing and what would be needed to answer.\n"
        "- Avoid repeating the same points. Be precise and direct.\n"
        "- If multiple interpretations exist, present the most likely answer and briefly note alternatives.\n\n"
        "Output format (respond ONLY with valid JSON):\n"
        "- Return a single JSON object with the following fields:\n"
        '  - "relevant_context": array of strings (passages or lines used, with brief rationale if needed)\n'
        '  - "key_information": array of strings (concise facts or figures)\n'
        '  - "reasoning": array of 2–4 short sentences connecting facts to the conclusion\n'
        '  - "final_answer": string (one concise sentence)\n\n'
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
