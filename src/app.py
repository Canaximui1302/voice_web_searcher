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


demo = gr.Interface(
    fn=gradio_fct,
    inputs=gr.Audio(
        sources=["microphone"],
        type='numpy',
        waveform_options={"show_recording_waveform": False}
    ),
    outputs=[
        gr.Textbox(label="Summary", lines=10),
        gr.Audio(label="Spoken summary", autoplay=True),
    ],
    api_name="prediction"
)

demo.launch()

# summarize_news(fetch_content("what are some latest AI news"))