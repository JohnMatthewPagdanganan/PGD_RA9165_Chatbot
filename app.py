import os
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Lex9165", layout="wide")

# ----------------------------
# 1) Secrets → Environment (BEFORE importing main/build_bot)
# ----------------------------
together_key = st.secrets.get("TOGETHER_API_KEY", None) or os.getenv("TOGETHER_API_KEY", "")
hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")

if not together_key:
    st.error("Missing TOGETHER_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

os.environ["TOGETHER_API_KEY"] = together_key

# HF token is strongly recommended to avoid 429 rate limits
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Helps avoid HF XET-related issues on some hosts
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Optional: keep downloads in your repo folder (Streamlit Cloud has write access here)
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".hf_cache"))

# ----------------------------
# 2) Now import build_bot (SAFE)
# ----------------------------
from main import build_bot


# ----------------------------
# 3) Load on start (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_bot():
    return build_bot(ROOT)

st.title("Lex9165: RA 9165 Legal Assistant")

with st.spinner("Loading models and indexes (first run can take a while)..."):
    try:
        bot = load_bot()
    except Exception as e:
        st.error("Failed to initialize the app. Check logs in 'Manage app' for the full traceback.")
        st.exception(e)  # shows full error to you; remove later if you want
        st.stop()

# ----------------------------
# 4) UI
# ----------------------------
q = st.text_input("Ask a question:", placeholder="e.g., What are the requirements under Section 21 of RA 9165?")

if st.button("Ask") and q.strip():
    with st.spinner("Retrieving and generating..."):
        reply, hits = bot.answer(q.strip())

    st.subheader("Answer")
    st.text(reply)

    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        st.markdown(f"**[{i}]** `{h['source']}` (page {h['page']})")
        txt = h.get("text", "") or ""
        st.caption((txt[:600] + ("..." if len(txt) > 600 else "")))