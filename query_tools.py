# query_tools.py  (NO fastcoref version — copy/paste this whole file)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re

CONTROL_TAGS = {
    "summary":  "SUMMARY",
    "tldr":     "SUMMARY",
    "lengthen": "DETAILED",
    "detailed": "DETAILED",
    "list":     "LIST",
    "bullets":  "LIST",
    "steps":    "STEPS",
    "procedure":"STEPS",
    "compare":  "COMPARE",
    "vs":       "COMPARE",
    "scenario": "SCENARIO",
    "yesno":    "YESNO",
    "cite":     "CITATIONS",
    "quotes":   "QUOTES",
}

YESNO_START = re.compile(r"^(is|are|was|were|do|does|did|can|could|will|would|should|may|might|must)\b", re.I)
PRON = re.compile(r"\b(it|this|that|they|them|those|these|the act|the law)\b", re.I)

def parse_control_block(user_query: str) -> Tuple[Optional[str], str]:
    q = user_query.strip()
    if not q.startswith("/"):
        return None, q
    parts = q.split()
    mode = None
    consumed = 0
    for p in parts:
        if not p.startswith("/"):
            break
        tag = p[1:].lower().strip()
        if tag in CONTROL_TAGS:
            mode = CONTROL_TAGS[tag]
            consumed += 1
        else:
            break
    cleaned = " ".join(parts[consumed:]).strip() or user_query.strip()
    return mode, cleaned

def detect_mode(query: str) -> str:
    q = query.lower().strip()

    juris_triggers = ["people v", "g.r.", "gr no", "supreme court", "ruling", "held", "doctrine", "case"]
    if any(t in q for t in juris_triggers):
        if any(x in q for x in ["quote", "exact wording", "verbatim"]):
            return "QUOTES"
        return "DETAILED"

    if "yes or no" in q or "yes/no" in q:
        return "YESNO"
    if YESNO_START.match(q):
        return "YESNO"

    if any(x in q for x in ["quote", "exact wording", "verbatim", "pinpoint", "where does it say"]):
        return "QUOTES"
    if any(x in q for x in ["show sources", "sources only", "citations only", "cite only", "provide citations"]):
        return "CITATIONS"

    if any(x in q for x in ["summarize", "summary", "tldr", "in short"]):
        return "SUMMARY"
    if any(x in q for x in ["explain in detail", "elaborate", "expand", "lengthen", "deep dive"]):
        return "DETAILED"
    if any(x in q for x in ["list", "enumerate", "what are", "give me all", "provide a list"]):
        return "LIST"
    if any(x in q for x in ["steps", "step-by-step", "procedure", "process", "how do i", "how to"]):
        return "STEPS"
    if any(x in q for x in ["compare", "difference", "versus", " vs "]):
        return "COMPARE"
    if any(x in q for x in ["what if", "suppose", "in this scenario", "situation", "if someone"]):
        return "SCENARIO"

    return "DEFAULT"

def dynamic_k(mode: str):
    if mode in ["SUMMARY"]:
        return {"cand": 35, "rerank": 20, "final": 6}
    if mode in ["YESNO", "QUOTES", "CITATIONS"]:
        return {"cand": 45, "rerank": 25, "final": 7}
    if mode in ["LIST", "STEPS"]:
        return {"cand": 55, "rerank": 30, "final": 9}
    if mode in ["COMPARE", "SCENARIO", "DETAILED"]:
        return {"cand": 70, "rerank": 40, "final": 10}
    return {"cand": 45, "rerank": 25, "final": 7}

def normalize_for_retrieval_v2(q: str, chat_history: List[Dict[str, str]], max_turns: int) -> str:
    q2 = q.strip()
    mentions_ra = re.search(r"\b(ra\s*9165|dangerous\s+drugs\s+act)\b", q2, re.I) is not None
    hist_text = " ".join([m.get("content","") for m in chat_history[-max_turns*2:]]).lower()
    hist_about_ra = ("ra 9165" in hist_text) or ("dangerous drugs act" in hist_text)

    # lightweight pronoun injection (acts as a cheap "coref")
    if (mentions_ra or hist_about_ra) and re.search(r"\b(it|this|that|the act|the law)\b", q2, re.I):
        q2 = re.sub(
            r"\b(it|this|that|the act|the law)\b",
            "RA 9165 (Dangerous Drugs Act of 2002)",
            q2,
            flags=re.I
        )

    q2 = re.sub(r"\bconsequence(s)?\b", "penalties and sanctions", q2, flags=re.I)
    q2 = re.sub(r"\bdisobey(ing)?\b", "violation", q2, flags=re.I)
    return q2

@dataclass
class QueryProcessor:
    # ✅ coref_model removed / optional
    llm: any                    # expects .generate_chat(messages)->str
    max_history_turns: int = 4
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    def build_history_window(self) -> str:
        recent = self.chat_history[-self.max_history_turns * 2:]
        parts = []
        for m in recent:
            role = (m.get("role","") or "").upper()
            content = (m.get("content","") or "").strip()
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def rewrite_query_for_retrieval(self, user_query: str) -> str:
        recent = self.chat_history[-self.max_history_turns * 2:]

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite follow-up questions into standalone search queries for RA 9165 (Dangerous Drugs Act of 2002). "
                    "Do NOT answer. Output ONLY ONE LINE. "
                    "Replace pronouns like it/this/that/the act/the law with 'RA 9165 (Dangerous Drugs Act of 2002)'. "
                    "If the user asks about consequences, rewrite using legal terms like 'penalties', 'sanctions', and 'criminal liability'. "
                    "Do not add facts not present in the conversation."
                )
            },
            {
                "role": "user",
                "content": f"""
Conversation (most recent last):
{recent}

Task:
Rewrite the user's question into a standalone query that includes the missing context.
Rules:
- Output exactly ONE line.
- Do NOT answer the question.
- Do NOT add facts not present in the conversation.
- If already standalone, return it unchanged.

User question: {user_query}
Standalone retrieval query:
""".strip()
            }
        ]

        out = self.llm.generate_chat(messages, max_tokens=80)
        rewritten = " ".join((out or "").split())
        return rewritten or " ".join(user_query.strip().split())

    def finalize_retrieval_query(self, cleaned_query: str) -> str:
        normalized = normalize_for_retrieval_v2(cleaned_query, self.chat_history, self.max_history_turns)

        # If still short/vague, use LLM rewrite (keeps your "follow-up" capability)
        if len(normalized.split()) < 6 or re.search(r"\b(it|this|that|the act|the law|those|them)\b", normalized, re.I):
            return self.rewrite_query_for_retrieval(normalized)

        return normalized

    def update_history(self, user: str, assistant: str):
        self.chat_history.append({"role": "user", "content": user})
        self.chat_history.append({"role": "assistant", "content": assistant})
        self.chat_history[:] = self.chat_history[-self.max_history_turns * 2:]