import time
import traceback
from pathlib import Path
import gradio as gr

from main import build_bot

ROOT = Path(__file__).resolve().parent

print("Loading Lex9165...")
bot = build_bot(ROOT)
print("Lex9165 ready")

# Optional warmup
try:
    bot.answer("warmup")
    print("Warmup done")
except Exception:
    pass


def _format_sources_details(hits, max_items=6, max_snip=260) -> str:
    if not hits:
        return "\n\n<details><summary><b>Sources</b></summary><br><br>(no sources)<br><br></details>"

    items = []
    for i, h in enumerate(hits[:max_items], 1):
        src = h.get("source", "unknown")
        page = h.get("page", "?")
        snip = (h.get("text") or "").strip().replace("\n", " ")
        if len(snip) > max_snip:
            snip = snip[:max_snip] + "…"
        items.append(f"<b>[{i}]</b> {src} (p. {page})<br>{snip}")

    joined = "<br><br>".join(items)
    return f"\n\n<details><summary><b>Sources</b></summary><br><br>{joined}<br><br></details>"


def _stream_words(text: str, delay=0.008):
    acc = ""
    for w in (text or "").split():
        acc += w + " "
        yield acc
        time.sleep(delay)


def respond(message, messages):
    """
    messages format for Gradio 6.x Chatbot:
    [{"role": "user"/"assistant", "content": "..."}]
    """
    msg = (message or "").strip()
    messages = list(messages or [])

    if not msg:
        # keep input as-is? return empty to clear anyway
        yield messages, ""
        return

    # Add user + placeholder assistant
    messages.append({"role": "user", "content": msg})
    messages.append({"role": "assistant", "content": ""})

    # Clear input immediately
    yield messages, ""

    try:
        reply, hits = bot.answer(msg)
        sources_block = _format_sources_details(hits)

        # Stream reply
        for partial in _stream_words(reply):
            messages[-1]["content"] = partial
            yield messages, ""

        # Append dropdown sources inside the SAME assistant message
        messages[-1]["content"] = (reply or "").strip() + sources_block
        yield messages, ""

    except Exception as e:
        tb = traceback.format_exc()
        messages[-1]["content"] = f"Error: {type(e).__name__}: {e}\n\n{tb}"
        yield messages, ""


def clear_all():
    return [], ""


CSS = """
/* Make the whole app fill the browser height */
.gradio-container {
  height: 100vh !important;
}

/* Make chat area expand; adjust offsets if you change header/input sizes */
#chatbot {
  height: calc(100vh - 210px) !important;
}

/* Reduce extra padding a bit for a cleaner ChatGPT-like look */
#header {
  margin-bottom: 0.25rem;
}
"""

with gr.Blocks(title="Lex9165", fill_height=True, css=CSS) as demo:
    gr.Markdown("# Lex9165", elem_id="header")

    chatbot = gr.Chatbot(elem_id="chatbot")  # Gradio 6.x uses messages format by default

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question and press Enter…",
            label="",
            lines=2,
            scale=8
        )
        send = gr.Button("Send", variant="primary", scale=1)
        new_chat = gr.Button("New chat", scale=1)

    # Wire events
    send.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
    new_chat.click(clear_all, inputs=None, outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)