import gradio as gr
from transformers import pipeline

import web_search as ws


def gradio_fct(data):
    
    request = ws.transcribe(data)
    fetch_text = ws.fetch_content(request)
    summarized = ws.summarize_news(fetch_text, False)
    # print(summarized)
    return summarized


demo = gr.Interface(
    fn=gradio_fct,
    inputs=gr.Audio(
        sources=["microphone"],
        type='numpy',
        waveform_options={"show_recording_waveform": False}
    ),
    outputs=gr.Textbox(),
    api_name="prediction"
)

demo.launch()

# summarize_news(fetch_content("what are some latest AI news"))