# prompts.py
from __future__ import annotations
import re

IDK_LINE = "I don't know based on the provided PDFs."
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def build_system_prompt() -> str:
    return (
        "You are a STRICT legal QA assistant for Republic Act No. 9165 and related Philippine jurisprudence.\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "1) Use ONLY the provided context passages. Do NOT use outside knowledge.\n"
        "2) Answer ONLY the user's question. Do NOT add extra background.\n"
        "3) Every paragraph or bullet that states facts MUST include at least one citation like [1].\n"
        "4) Do NOT invent citations. Only cite passage numbers present in the context.\n"
        "5) Prefer short quotes (<=25 words) when asked what a case/law says.\n"
        "6) Be clear and human-readable, never sacrifice accuracy.\n\n"
        "STYLE:\n- Plain English.\n- No speculation.\n"
    )

def build_onepass_prompt(mode: str, context: str, question: str) -> str:
    mode_rules = {
        "SUMMARY":   "Write 3–6 bullets. Each bullet ends with [n].",
        "DETAILED":  "Write a structured answer with headings. Cite at end of each section [n].",
        "LIST":      "Write bullet points only. Each bullet MUST contain at least 8 words BEFORE the citation, then end with [n].",
        "STEPS":     "Write numbered steps. Each step ends with [n].",
        "COMPARE":   "Compare clearly (A vs B). Cite each comparison block [n].",
        "SCENARIO":  "Use IRAC headings. Cite Rule and Application parts [n].",
        "YESNO":     "Start with: Answer: Yes/No/Not determinable from context. Then 1 short paragraph ending with [n].",
        "CITATIONS": "Output ONLY a list of sources with citation numbers and (source, page). No extra text.",
        "QUOTES":    "Provide up to 2 short quotes (<=25 words each) and explain briefly. Each quote must have [n].",
        "DEFAULT":   "Answer in 1–2 short paragraphs. Put citations at the end of each paragraph like [1]."
    }[mode]

    return f"""
You are a careful legal assistant.

Use ONLY the provided context. Do NOT use outside knowledge.
Every claim must be supported by the context.

If the context is insufficient to answer the question, output exactly:
{IDK_LINE}

Otherwise, output ONLY the answer text.
- DO NOT output a CITATIONS: section.
- Put citations inline in the answer like [1] at the end of relevant sentences/paragraphs.
- If you don't know, output exactly: I don't know based on the provided PDFs.

Mode: {mode}
Mode rules:
{mode_rules}

Context (each passage labeled with [n]):
{context}

Question:
{question}
""".strip()

def normalize_onepass_output(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return IDK_LINE

    # Strip legacy headers if the model outputs them
    t = re.sub(r"(?mi)^\s*CITATIONS:.*\n?", "", t).strip()
    t = re.sub(r"(?mi)^\s*ANSWER:\s*", "", t).strip()

    # If it still ends up empty, return IDK line
    if not t:
        return IDK_LINE

    # If the model returned only the IDK line (or started with it), normalize it
    if t == IDK_LINE or t.startswith(IDK_LINE):
        return IDK_LINE

    return t

def has_uncited_sentences(answer_text: str) -> bool:
    t = (answer_text or "").strip()
    if not t:
        return False
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()] or [t]
    for s in sents:
        if re.match(r"^[A-Z][A-Z\s]{2,}:\s*$", s):
            continue
        if not re.search(r"\[\d+\]", s):
            return True
    return False