from __future__ import annotations


def build_research_prompt(query: str, context: str) -> str:
    return (
        "You are a medical research assistant. Use only the retrieved context below.\n\n"
        "Instructions:\n"
        "- Synthesize findings across the evidence.\n"
        "- If evidence conflicts, explicitly describe the contradiction.\n"
        "- If tabular data is present, summarize the relevant pattern or result.\n"
        '- If the context is insufficient, say "Insufficient evidence."\n'
        "- Cite the source headers and document IDs that support your answer.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Research Question: {query}\n\n"
        "Answer with two sections:\n"
        "1. Research Insight\n"
        "2. Evidence Basis\n"
    )

