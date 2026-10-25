import os
import torch
import gradio as gr
from transformers import pipeline
import numpy as np  
from ollama import Client, ResponseError

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

summarizer = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device=0, dtype=torch.bfloat16)
client = Client(headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"})

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en"
)

def transcribe(data):
    if data is None:
        return "No audio received. Please record again."

    sample_rate, audio = data

    # For debug

    # print("Sample rate:", sample_rate)
    # print("Audio shape:", audio.shape)
    # print("Audio dtype:", audio.dtype)

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Normalize integer-like amplitude ranges (if necessary)
    if np.max(np.abs(audio)) > 1.0:
        audio /= 32768.0

    result = transcriber({
        "sampling_rate": sample_rate,
        "raw": audio
    })
    return result["text"]


def fetch_content(text):
    # search on internet based on the prompt
    response = client.web_search(text)
    link = response['results'][0]['url']
    print("Retrieved URL: ", link)

    # fetch content of the first link
    try:
        summ = client.web_fetch(link)
        content = summ['content']
        return content 
    except ResponseError as e:
        print("status:", e.status_code)
        print("message:", e.error)
        return ""   # or raise a friendlier error for the UI

# for debugging with print, use 'stream_option=True'
def summarize_news(content, stream_option = True):
    torch.cuda.empty_cache()
    prompt = [{"role": "user", "content": "Summarize this into a news report.\n" + content}]
    to_pass = summarizer.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    final = summarizer(to_pass,  return_full_text=False)
    to_speak = final[0]['generated_text']

    # print("*** SUMMARIZED NEWS ***")
    # for i in final:
    #     to_speak += i['message']['content']
    #     if (stream_option):
    #         print(i['message']['content'], end='')
    # print("\n*** DONE NEWS ***\n")

    print(to_speak)
    return to_speak


def gradio_fct(data):
    request = transcribe(data)
    fetch_text = fetch_content(request)
    summarized = summarize_news(fetch_text, False)
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