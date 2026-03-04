import os
from pathlib import Path
import gradio as gr

# Your builder
from main import build_bot

ROOT = Path(__file__).resolve().parent

# Global singleton (lazy-loaded)
BOT = None

def ensure_env():
    """
    Gradio doesn't have st.secrets. Use environment variables.
    Set these in your host (HF Spaces / Render / etc).
    """
    # Together
    if not os.getenv("TOGETHER_API_KEY", "").strip():
        raise RuntimeError("Missing TOGETHER_API_KEY environment variable.")

    # HuggingFace token strongly recommended to avoid 429 rate limit on model downloads
    # Not strictly required if models are already cached, but usually needed on hosts.
    # If you don't have it, leave it empty; but expect occasional 429.
    # os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    # Also supported:
    # os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")

def load_bot_once():
    global BOT
    if BOT is None:
        ensure_env()
        BOT = build_bot(ROOT)
    return BOT

def load_models():
    """
    Button action: load models on-demand.
    This avoids startup crash/timeouts.
    """
    try:
        load_bot_once()
        return "✅ Models loaded. You can now ask questions."
    except Exception as e:
        return f"❌ Failed to load models:\n{type(e).__name__}: {e}"

def answer_question(q: str):
    """
    Chat action: loads bot if not loaded, then answers.
    """
    q = (q or "").strip()
    if not q:
        return "Please enter a question.", ""

    try:
        bot = load_bot_once()
        reply, hits = bot.answer(q)

        # Format sources nicely
        sources_lines = []
        for i, h in enumerate(hits or [], 1):
            src = h.get("source", "unknown")
            page = h.get("page", "?")
            snippet = (h.get("text", "") or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            sources_lines.append(f"[{i}] {src} (page {page})\n{snippet}")

        sources = "\n\n".join(sources_lines) if sources_lines else "(no sources)"
        return reply, sources

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", ""

with gr.Blocks(title="Lex9165") as demo:
    gr.Markdown("# Lex9165: RA 9165 Legal Assistant")

    with gr.Row():
        load_btn = gr.Button("Load models (recommended first)", variant="primary")
        load_status = gr.Textbox(label="Status", value="Models not loaded yet.", interactive=False)

    load_btn.click(fn=load_models, inputs=None, outputs=load_status)

    q = gr.Textbox(
        label="Ask a question",
        placeholder="e.g., What are the requirements under Section 21 of RA 9165?",
        lines=2
    )
    ask_btn = gr.Button("Ask")

    answer = gr.Textbox(label="Answer", lines=12)
    sources = gr.Textbox(label="Sources", lines=12)

    ask_btn.click(fn=answer_question, inputs=q, outputs=[answer, sources])
    q.submit(fn=answer_question, inputs=q, outputs=[answer, sources])

if __name__ == "__main__":
    # For local run:
    # TOGETHER_API_KEY must be set in your shell env
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))