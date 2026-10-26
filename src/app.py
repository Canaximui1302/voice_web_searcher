import gradio as gr
from transformers import pipeline

import web_search as ws
import voice as vc


def gradio_fct(data):
    
    request = ws.transcribe(data)
    fetch_text = ws.fetch_content(request)
    summarized = ws.summarize_news(fetch_text)
    # print(summarized)
    return summarized, vc.text_to_speech(summarized)


with gr.Blocks() as demo:
    gr.Markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 3rem;">Voice News Assistant</h1>
        <h3>Record a request such as <em>"latest news about AI"</em>.</h3>
        <ol style="display: inline-block; text-align: left; font-size: 0.95rem; opacity: 0.75;">
            <li>Press <b>Record</b> and speak your request.</li>
            <li>Press <b>Stop</b> when you're done.</li>
            <li>Press <b>Summarize</b> and wait for the response.</li>
        </ol>
    </div>
    """
    )

    with gr.Row():
        gr.Column(scale=2)                      # empty left margin
        with gr.Column(scale=3, min_width=400):
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Record")
            btn = gr.Button("Summarize", variant="primary")
            summary_box = gr.Textbox(label="Summary", lines=5, elem_id="summary_box")
            audio_out = gr.Audio(label="Spoken summary", autoplay=True)
        gr.Column(scale=2)                      # empty right margin

    btn.click(gradio_fct, inputs=mic, outputs=[summary_box, audio_out])

theme = gr.Theme.from_hub("harsh8001/cartoon-style")

theme = theme.set(
    block_title_text_color=theme.block_label_text_color,
    block_title_text_color_dark=theme.block_label_text_color_dark,
    block_title_text_size=theme.block_label_text_size,
    block_title_text_weight=theme.block_label_text_weight,
)

css = """
#summary_box {
    padding-top: 8px !important;
}
"""

demo.launch(theme=theme, css=css)

# summarize_news(fetch_content("what are some latest AI news"))