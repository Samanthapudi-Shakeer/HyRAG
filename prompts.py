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
        "You are a strict QMS assistant. Use only the provided context to answer. "
        "If the answer is not supported by the context, respond exactly with: "
        "Not available in the document.\n\n"
        "Provide concise answers and include citations like [C1].\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
