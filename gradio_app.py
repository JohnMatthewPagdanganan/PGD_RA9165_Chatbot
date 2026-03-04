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
    return "\n\n<details><summary><b>Sources</b></summary><br><br>" + "<br><br>".join(items) + "<br><br></details>"


def _stream_words(text: str, delay=0.008):
    acc = ""
    for w in (text or "").split():
        acc += w + " "
        yield acc
        time.sleep(delay)


def respond(message, messages):
    msg = (message or "").strip()
    messages = list(messages or [])

    if not msg:
        yield messages, "", gr.update(interactive=True)
        return

    # add user message
    messages.append({"role": "user", "content": msg})

    # add assistant "thinking" placeholder (animated)
    thinking_html = '<span class="thinking">Thinking<span class="dots"></span></span>'
    messages.append({"role": "assistant", "content": thinking_html})

    # clear input + disable while busy
    yield messages, "", gr.update(interactive=False)

    try:
        # compute answer (this may take time)
        reply, hits = bot.answer(msg)
        sources_block = _format_sources_details(hits)

        # stream answer gradually
        acc = ""
        for w in (reply or "").split():
            acc += w + " "
            messages[-1]["content"] = acc
            yield messages, "", gr.update(interactive=False)
            time.sleep(0.008)

        # final: append sources dropdown
        messages[-1]["content"] = (reply or "").strip() + sources_block
        yield messages, "", gr.update(interactive=True)

    except Exception as e:
        tb = traceback.format_exc()
        messages[-1]["content"] = f"Error: {type(e).__name__}: {e}\n\n{tb}"
        yield messages, "", gr.update(interactive=True)


def clear_all():
    return [], ""


CSS = """
/* Keep CSS simple; don't touch Gradio wrappers */
html, body { height: 100%; margin: 0; }
.gradio-container { height: 100vh !important; }

/* 2-column layout fills available height */
#layout { height: 100vh; }

/* Left column */
#sidebar {
  height: 100%;
  padding: 12px;
  border-right: 1px solid rgba(255,255,255,0.10);
  box-sizing: border-box;
}
#title, #title * {
  font-size: 26px !important;
  font-weight: 800 !important;
}
#newchat {
  width: fit-content !important;
}

#newchat button {
  width: fit-content !important;
  padding: 5px 12px !important;
}

/* Right column: flex column so chat grows and input stays at bottom */
#main {
  height: 100%;
  padding: 12px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}
#chatbot { flex: 1 1 auto; min-height: 0; }
#inputrow { margin-top: 10px; padding-bottom: 10px; }

/* Same height input + send */
#query textarea { height: 44px !important; min-height: 44px !important; }
#send button { height: 44px !important; min-height: 44px !important; padding: 0 14px !important; }

/* Thinking animation */
.thinking {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 14px;
  opacity: 0.9;
}
.thinking .dots::after{
  content: "";
  display: inline-block;
  width: 2em;
  text-align: left;
  animation: dots 1s steps(4, end) infinite;
}
@keyframes dots {
  0%   { content: ""; }
  25%  { content: "."; }
  50%  { content: ".."; }
  75%  { content: "..."; }
  100% { content: ""; }
}

"""

with gr.Blocks(css=CSS, fill_height=True) as demo:
    with gr.Row(elem_id="layout"):
        # LEFT ~10%
        with gr.Column(scale=1, min_width=150, elem_id="sidebar"):
            gr.Markdown("⚖️ **Lex9165**", elem_id="title")
            new_chat = gr.Button("New Chat", elem_id="newchat")

        # RIGHT ~90%
        with gr.Column(scale=19, elem_id="main"):
            chatbot = gr.Chatbot(elem_id="chatbot")

            with gr.Row(elem_id="inputrow"):
                msg = gr.Textbox(
                    placeholder="Type your question…",
                    show_label=False,
                    lines=1,
                    elem_id="query",
                    scale=10,
                    interactive=True
                )
                send = gr.Button("Send", variant="primary", elem_id="send", scale=1)

    send.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg, msg])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg, msg])
    new_chat.click(clear_all, outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)