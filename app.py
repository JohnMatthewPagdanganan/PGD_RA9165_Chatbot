import os
import streamlit as st
from pathlib import Path
from main import build_bot

st.set_page_config(page_title="Lex9165", layout="wide")

api_key = st.secrets.get("TOGETHER_API_KEY", None) or os.getenv("TOGETHER_API_KEY")
if not api_key:
    st.error("Missing TOGETHER_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

os.environ["TOGETHER_API_KEY"] = api_key  # ensures your llm_together sees it

ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_bot():
    return build_bot(ROOT)

bot = load_bot()

st.title("Lex9165: RA 9165 Legal Assistant")

q = st.text_input("Ask a question:", placeholder="e.g., What are the requirements under Section 21 of RA 9165?")

if st.button("Ask") and q.strip():
    with st.spinner("Retrieving and generating..."):
        reply, hits = bot.answer(q.strip())

    st.subheader("Answer")
    st.text(reply)

    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        st.markdown(f"**[{i}]** `{h['source']}` (page {h['page']})")
        st.caption((h.get("text", "")[:600] + ("..." if len(h.get("text","")) > 600 else "")))