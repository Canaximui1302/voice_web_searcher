import os
import torch
import gc
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


def fetch_content(text, max_tries=5, min_chars=300):
    response = client.web_search(text)
    results = response["results"][:max_tries]

    for i, result in enumerate(results, start=1):
        link = result["url"]
        print(f"[{i}/{len(results)}] Fetching URL: {link}")
        try:
            page = client.web_fetch(link)
            content = page["content"]
        except ResponseError as e:
            print("  status:", e.status_code, "| message:", e.error)
            continue                       # go to the next link
        except Exception as e:
            print("  unexpected error:", repr(e))
            continue

        if content and len(content.strip()) >= min_chars:
            print("  OK")
            return content           # stop at the first success

        print("  too little text, skipping")

    return None, None # nothing works

# for debugging with print, use 'stream_option=True'
def summarize_news(content):
    tok = summarizer.tokenizer

    # cap the input by tokens
    ids = tok(content, truncation=True, max_length=3000)["input_ids"]
    content = tok.decode(ids, skip_special_tokens=True)

    prompt = [{"role": "user", "content": "Summarize this into a news report (about 200 words). Give only readable text and no special symbols.\n" + content}]
    to_pass = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    final = summarizer(to_pass,  return_full_text=False)
    to_speak = final[0]['generated_text']

    print(to_speak)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return to_speak