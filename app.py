"""
Gradio frontend for the Riyadh Metro RAG system.

Usage:
    python app.py
"""
import html
import re

import gradio as gr

from metro_rag.config import TOP_K
from metro_rag.pipeline import MetroRAG

# Official Riyadh Metro line colours, used to colour-code retrieved chunks by line
LINE_COLORS = {
    "Line1": ("#3B82F6", "Blue"),
    "Line2": ("#EF4444", "Red"),
    "Line3": ("#F97316", "Orange"),
    "Line4": ("#EAB308", "Yellow"),
    "Line5": ("#22C55E", "Green"),
    "Line6": ("#A855F7", "Purple"),
}
NEUTRAL_COLOR = "#64748B"

EXAMPLES = [
    ("Fact lookup", "What type of station is Dr Sulaiman Al Habib, and which line is it on?"),
    ("Neighbour", "What is the station immediately following KAFD on the Blue Line?"),
    ("Aggregation", "How many elevated stations are there in total on the Blue Line?"),
    ("District", "Which metro stations are in Al Olaya district?"),
    ("بالعربية", "في أي حي تقع محطة قصر الحكم؟"),
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0A0F1A;
    --surface:   #111A2B;
    --surface-2: #16223A;
    --border:    #22304A;
    --text:      #E8EDF5;
    --muted:     #8B9BB4;
    --accent:    #38BDF8;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
}

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 28px 20px 48px !important; }

/* Strip Gradio's default panel chrome; surfaces are re-added deliberately below. */
.block, .form, .panel { background: transparent !important; border: none !important; box-shadow: none !important; }

/* Arabic and English answers sit side by side, so let each paragraph pick its own
   direction instead of forcing LTR on Arabic text. */
#answer-body, #answer-body * , .src-text { unicode-bidi: plaintext; text-align: start; }

/* ---------- Header ---------- */
#masthead { border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 26px !important; }
#masthead h1 {
    font-size: 26px !important; font-weight: 700 !important; color: var(--text) !important;
    margin: 0 0 6px !important; letter-spacing: -0.4px !important;
}
#masthead h1 span { color: var(--accent) !important; }
#masthead p { color: var(--muted) !important; font-size: 14px !important; margin: 0 !important; }
#line-key { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 14px; }
#line-key i { display: flex; align-items: center; gap: 6px; font-style: normal; font-size: 11.5px; color: var(--muted); }
#line-key b { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

/* ---------- Input ---------- */
#input-box { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
#input-box:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important; }
#input-box textarea {
    background: transparent !important; border: none !important; color: var(--text) !important;
    font-size: 15px !important; line-height: 1.6 !important; padding: 14px 16px !important;
    font-family: 'Inter', sans-serif !important;
    unicode-bidi: plaintext; text-align: start;
}
#input-box textarea::placeholder { color: #5A6A85 !important; }
/* Gradio nests the textarea inside its <label>, so hide only the label text. */
#input-box label > span { display: none !important; }

/* ---------- Example chips ---------- */
#chips { gap: 8px !important; margin: 14px 0 6px !important; flex-wrap: wrap !important; }
#chips button {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--muted) !important; border-radius: 999px !important; padding: 7px 15px !important;
    font-size: 12.5px !important; font-weight: 500 !important; box-shadow: none !important;
    min-width: 0 !important; transition: all .15s ease !important;
}
#chips button:hover { border-color: var(--accent) !important; color: var(--accent) !important; background: var(--surface-2) !important; }

/* ---------- Action row ---------- */
#actions { gap: 10px !important; margin: 16px 0 28px !important; }
#ask-btn {
    background: var(--accent) !important; color: #06121F !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important; font-size: 14.5px !important;
    padding: 11px 26px !important; flex: 0 0 auto !important; min-width: 150px !important;
}
#ask-btn:hover { background: #7DD3FC !important; }
#clear-btn {
    background: transparent !important; color: var(--muted) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; font-weight: 500 !important; font-size: 14px !important;
    padding: 11px 20px !important; flex: 0 0 auto !important; min-width: 90px !important;
}
#clear-btn:hover { color: var(--text) !important; border-color: var(--muted) !important; }

/* ---------- Shared card ---------- */
.card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; overflow: hidden !important; }
.card-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 18px; border-bottom: 1px solid var(--border); background: var(--surface-2);
}
.card-head h2 {
    font-size: 11px !important; font-weight: 700 !important; letter-spacing: 1.1px !important;
    text-transform: uppercase; color: var(--muted) !important; margin: 0 !important;
}
.card-head em { font-style: normal; font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

/* ---------- Answer ---------- */
#answer-body { padding: 22px 24px 26px !important; min-height: 240px; }
#answer-body h1, #answer-body h2, #answer-body h3 {
    font-size: 16px !important; font-weight: 650 !important; color: var(--text) !important;
    margin: 0 0 14px !important; padding: 0 !important; border: none !important;
}
#answer-body p, #answer-body li { color: #CFD9E8 !important; font-size: 14.5px !important; line-height: 1.75 !important; }
#answer-body strong { color: #FFFFFF !important; font-weight: 650 !important; }
#answer-body ul, #answer-body ol { padding-inline-start: 22px !important; margin: 10px 0 !important; }
#answer-body li { margin-bottom: 6px !important; }
#answer-body hr { border-color: var(--border) !important; margin: 18px 0 !important; }

.placeholder { color: #55637A !important; font-size: 14px !important; text-align: center !important; padding: 62px 20px !important; line-height: 1.7 !important; }
.placeholder svg { opacity: .5; margin-bottom: 12px; }

/* ---------- Sources ---------- */
#source-wrap { max-height: 560px; overflow-y: auto; padding: 14px; }
#source-wrap::-webkit-scrollbar { width: 8px; }
#source-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
.src {
    background: var(--surface-2); border: 1px solid var(--border); border-inline-start: 3px solid var(--line, #64748B);
    border-radius: 9px; padding: 12px 14px; margin-bottom: 10px;
}
.src:last-child { margin-bottom: 0; }
.src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.src-rank {
    background: var(--line, #64748B); color: #06121F; font-weight: 700; font-size: 10.5px;
    width: 19px; height: 19px; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto;
}
.src-id { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #A9BAD2; flex: 1 1 auto; word-break: break-all; }
.src-score { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--muted); background: rgba(255,255,255,.05); padding: 2px 7px; border-radius: 5px; flex: 0 0 auto; }
.src-text { font-size: 12.5px; line-height: 1.65; color: #9FB0C7; margin: 0; }

/* Gradio's queue counter duplicates our own pending state - hide it. */
.progress-text, .eta-bar { display: none !important; }

footer, .built-with, #footer { display: none !important; }
"""


def _chunk_color(chunk_id):
    """Colour a retrieved chunk by the metro line it belongs to, when it has one."""
    match = re.search(r"Line\d", chunk_id or "")
    if match and match.group() in LINE_COLORS:
        return LINE_COLORS[match.group()][0]
    return NEUTRAL_COLOR


def format_sources(results):
    """Render the retrieved chunks as line-colour-coded cards."""
    cards = []
    for res in results:
        cards.append(
            f'<div class="src" style="--line:{_chunk_color(res["chunk_id"])}">'
            f'<div class="src-head">'
            f'<span class="src-rank">{res["rank"]}</span>'
            f'<span class="src-id">{html.escape(str(res["chunk_id"]))}</span>'
            f'<span class="src-score">{res["score"]:.3f}</span>'
            f"</div>"
            f'<p class="src-text">{html.escape(res["text"])}</p>'
            f"</div>"
        )
    return f'<div id="source-wrap">{"".join(cards)}</div>'


EMPTY_SOURCES = (
    '<div id="source-wrap"><p class="placeholder">'
    "Retrieved chunks will appear here,<br>colour-coded by metro line."
    "</p></div>"
)
EMPTY_ANSWER = (
    '<p class="placeholder">Ask a question about the Riyadh Metro '
    "in English or Arabic.<br>Answers are grounded strictly in the station data.</p>"
)
THINKING = '<p class="placeholder">Searching the knowledge base…</p>'


def make_answer_fn(rag):
    def answer_question_with_ui(user_question):
        """
        The bridge between Gradio and the RAG engine.
        Returns TWO things: the LLM's answer, and the retrieved context as HTML cards.
        """
        if not user_question or not user_question.strip():
            return "⚠️ Please enter a question about the Riyadh Metro.", EMPTY_SOURCES

        # Step 1: Retrieve the data
        raw_results = rag.retrieve(user_question, k=TOP_K)

        # Step 2: Generate the grounded answer
        try:
            llm_response = rag.ask(user_question, retrieved=raw_results)
        except Exception as e:
            # Log the details, but don't expose internal AWS errors to users
            print(f"Bedrock error: {e}")
            llm_response = "⚠️ The answer service is temporarily unavailable. The retrieved source data is shown alongside."

        # Return both items so Gradio can display them in two separate panels
        return llm_response, format_sources(raw_results)

    return answer_question_with_ui


def build_demo(rag):
    line_key = "".join(
        f'<i><b style="background:{color}"></b>{name}</i>' for color, name in LINE_COLORS.values()
    )

    with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="Riyadh Metro RAG") as demo:
        gr.HTML(
            "<div id='masthead'>"
            "<h1>🚇 Riyadh Metro <span>RAG Assistant</span></h1>"
            "<p>Bilingual question answering grounded in the 2024 station dataset.</p>"
            f"<div id='line-key'>{line_key}</div>"
            "</div>"
        )

        user_input = gr.Textbox(
            label="Question",
            placeholder="Which line is KAFD station on?   ·   ما هي محطات المترو في حي العليا؟",
            lines=2,
            elem_id="input-box",
        )

        with gr.Row(elem_id="chips"):
            chips = [(gr.Button(label, size="sm"), query) for label, query in EXAMPLES]

        with gr.Row(elem_id="actions"):
            ask_btn = gr.Button("Ask", variant="primary", elem_id="ask-btn")
            clear_btn = gr.Button("Clear", elem_id="clear-btn")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="card"):
                gr.HTML("<div class='card-head'><h2>Answer</h2></div>")
                answer_box = gr.Markdown(EMPTY_ANSWER, elem_id="answer-body")

            with gr.Column(scale=2, elem_classes="card"):
                gr.HTML(f"<div class='card-head'><h2>Retrieved context</h2><em>top {TOP_K}</em></div>")
                source_box = gr.HTML(EMPTY_SOURCES)

        answer_fn = make_answer_fn(rag)

        # Show a pending state first, then run retrieval + generation.
        def run(question):
            return answer_fn(question)

        for trigger in (user_input.submit, ask_btn.click):
            trigger(lambda: (THINKING, EMPTY_SOURCES), outputs=[answer_box, source_box]).then(
                run, inputs=user_input, outputs=[answer_box, source_box]
            )

        # Example chips fill the box and run straight away
        for button, query in chips:
            button.click(lambda q=query: q, outputs=user_input).then(
                lambda: (THINKING, EMPTY_SOURCES), outputs=[answer_box, source_box]
            ).then(run, inputs=user_input, outputs=[answer_box, source_box])

        clear_btn.click(
            lambda: ("", EMPTY_ANSWER, EMPTY_SOURCES),
            outputs=[user_input, answer_box, source_box],
        )

    return demo


if __name__ == "__main__":
    build_demo(MetroRAG()).launch()
