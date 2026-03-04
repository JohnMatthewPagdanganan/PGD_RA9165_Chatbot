import os
import streamlit as st
from pathlib import Path
from main import build_bot

st.set_page_config(page_title="Lex9165", layout="wide")
ROOT = Path(__file__).resolve().parent

# ---- Secrets / env ----
api_key = st.secrets.get("TOGETHER_API_KEY", None) or os.getenv("TOGETHER_API_KEY")
if not api_key:
    st.error("Missing TOGETHER_API_KEY. Add it in Streamlit Secrets.")
    st.stop()
os.environ["TOGETHER_API_KEY"] = api_key

hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Avoid HuggingFace Xet path (helps 429 issues)
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ---- Lazy-load bot only when needed ----
@st.cache_resource
def get_bot():
    return build_bot(ROOT)

st.title("Lex9165: RA 9165 Legal Assistant")

# UI: load button
if "bot_loaded" not in st.session_state:
    st.session_state.bot_loaded = False

col1, col2 = st.columns([1, 3])
with col1:
    if not st.session_state.bot_loaded:
        if st.button("Load models"):
            with st.spinner("Loading models (first time can take a while)..."):
                _ = get_bot()
            st.session_state.bot_loaded = True
            st.success("Models loaded.")

with col2:
    st.caption("Tip: First load downloads Hugging Face models. After that it’s cached in this Streamlit instance.")

q = st.text_input("Ask a question:", placeholder="e.g., What are the requirements under Section 21 of RA 9165?")

if st.button("Ask") and q.strip():
    if not st.session_state.bot_loaded:
        with st.spinner("Loading models (first time only)..."):
            _ = get_bot()
        st.session_state.bot_loaded = True

    bot = get_bot()
    with st.spinner("Retrieving and generating..."):
        reply, hits = bot.answer(q.strip())

    st.subheader("Answer")
    st.text(reply)

    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        st.markdown(f"**[{i}]** `{h['source']}` (page {h['page']})")
        st.caption((h.get("text", "")[:600] + ("..." if len(h.get("text","")) > 600 else "")))