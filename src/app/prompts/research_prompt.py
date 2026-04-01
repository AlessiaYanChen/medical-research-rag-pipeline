from __future__ import annotations


def build_research_prompt(query: str, context: str) -> str:
    return (
        "You are a medical research assistant. Use only the retrieved context below.\n\n"
        "Instructions:\n"
        "- Synthesize findings across the evidence.\n"
        "- Identify the study design for each source where determinable (e.g. RCT, observational cohort, retrospective, systematic review, case series).\n"
        "- Report effect sizes, rates, and numerical results exactly as stated in the context; do not paraphrase quantities.\n"
        "- Include confidence intervals, p-values, or other uncertainty measures when present in the context.\n"
        "- Note key limitations explicitly mentioned in the retrieved evidence.\n"
        "- If evidence conflicts, explicitly describe the contradiction.\n"
        "- If tabular data is present, summarize the relevant pattern or result.\n"
        '- If the context is insufficient, say "Insufficient evidence."\n'
        "- Cite the source section and document ID for each claim.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Research Question: {query}\n\n"
        "Answer with two sections:\n"
        "1. Research Insight — include study design, effect sizes with uncertainty, and limitations where present in the context.\n"
        "2. Evidence Basis — list each source (document ID and section) that supports your answer.\n"
    )

