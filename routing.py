# routing.py
from __future__ import annotations

def route_corpus(query: str) -> str:
    q = (query or "").lower()

    juris_keywords = [
        "people v", "vs.", "g.r.", "gr no", "ruling", "case", "cases",
        "jurisprudence", "acquit", "acquitted", "convict", "convicted",
        "conviction", "court held", "doctrine",
        "lim", "consada", "sarabia", "padollo", "asaytuno"
    ]
    if any(x in q for x in juris_keywords):
        return "jurisprudence"

    statutes_keywords = [
        "schedule i", "schedule ii", "section", "article", "irr",
        "penalt", "penalty", "ra 9165", "ra 10640", "ddb", "pdea"
    ]
    if any(x in q for x in statutes_keywords):
        return "statutes_and_guidelines"

    return "statutes_and_guidelines"


def corpora_to_search(cleaned_query: str) -> list[str]:
    """
    Your multi-corpus detection used in retrieve_final_hits() (juris + statutes).
    """
    q = (cleaned_query or "").lower()
    needs_juris = any(x in q for x in ["people v", "ruling", "case", "lim", "consada", "sarabia", "padollo", "asaytuno"])
    needs_statutes = any(x in q for x in ["ra 9165", "ra 10640", "section 21", "irr", "ddb", "pdea", "section", "article"])

    out = []
    if needs_juris:
        # you used structured summaries for juris when question spans
        out.append("jurisprudence")
    if needs_statutes:
        out.append("statutes_and_guidelines")
    if not out:
        out = [route_corpus(cleaned_query)]
    return out