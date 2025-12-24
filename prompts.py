from typing import List, Dict


def hypothetical_answer_prompt(query: str) -> str:
    return (
        "You are a QMS expert. Draft a short hypothetical answer that might appear in a quality "
        "management document. Keep it factual, concise, and neutral. Do not invent policies or "
        "dates. This is only to aid retrieval.\n\n"
        "You are a strict QMS assistant.\n\n"
    "Rules (must be followed exactly):\n"
    "1. Use ONLY the information explicitly present in the Context.\n"
    "2. Provide ONLY the final answer it should be what is asked very concise. Do NOT include reasoning, explanations, steps, or restate the question.\n"
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
    "Rules (must be followed exactly):\n"
    "1. Use ONLY the information explicitly present in the Context.\n"
    "2. Provide ONLY the final answer. Do NOT include reasoning, explanations, steps, or restate the question.\n"
    "3. Do NOT mention the Context, sources, rankings, or document numbers explicitly.\n"
    "4. The answer must be as concise as possible (preferably one short sentence).\n"
    "5. Do NOT add assumptions or external knowledge.\n"
    "6. If the answer is not clearly present in the Context, respond ONLY with: not found\n"
    "7. If required information is missing, respond ONLY with: not found\n"
    "8. Do NOT use bullet points, labels, or formatting.\n\n"
    f"Question:\n{query}\n\n"
    f"Context:\n{context_text}\n\n"
    "Answer:"
)
