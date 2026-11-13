import os
import torch
import gc
import gradio as gr
from transformers import pipeline
import numpy as np
from threading import Thread
from transformers import TextIteratorStreamer
from tinyfish import TinyFish

client = TinyFish()

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

summarizer = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device=0, dtype=torch.bfloat16)

SYSTEM = (
    "You summarize news for a text-to-speech reader. "
    "Write sentences of plain, flowing prose. "
    "Do not use bullet points, lists, headings, markdown, or emojis. "
    "Ignore menus, ads, event listings, dates and unrelated headlines. "
    "Use only information from the provided text."
)


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
    # Web search by query 'text'
    response = client.search.query(query=text)
    top_urls = [r.url for r in response.results[:max_tries]]

    # Try to fetch in list of URLs retrieved
    for i, url in enumerate(top_urls, start=1):
        link = url
        print(f"[{i}/{len(top_urls)}] Fetching URL: {link}")

        # fetche content from i-th url in the list
        fetched = client.fetch.get_contents(urls=[url], format="markdown").results[0]
        content = fetched.text
        print(f"Title: {fetched.title}")
        print(fetched.text)

        if content and len(content.strip()) >= min_chars:
            print("  OK")
            return text, content           # stop at the first success

        print("  too little text, skipping")

    return None, None # nothing works

# for debugging with print, use 'stream_option=True'
def summarize(query, content):
    tok = summarizer.tokenizer

    # cap the input by tokens
    ids = tok(content, truncation=True, max_length=3000)["input_ids"]
    content = tok.decode(ids, skip_special_tokens=True)

    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"<request>\n{query}\n</request>\n\n"
            f"<article>\n{content}\n</article>\n\n"
            "Give a short news summary according to the request."
        )},
    ]
    to_pass = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=120)
    thread = Thread(
        target=summarizer,
        args=(to_pass,),
        kwargs=dict(streamer=streamer, max_new_tokens=400, do_sample=False, return_full_text=False),
    )
    thread.start()

    text = ""
    for piece in streamer:
        text += piece
        yield text
    thread.join()