import wave
import numpy as np
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
SAMPLE_RATE = 24000                   

def text_to_speech(text):
    generator = pipeline(text, voice='af_heart')

    parts = []
    for _, _, audio in generator:
        if audio is None:
            continue
        if hasattr(audio, "detach"):          # torch tensor -> numpy
            audio = audio.detach().cpu().numpy()
        parts.append(np.asarray(audio, dtype=np.float32))

    if not parts:
        return None

    audio = np.concatenate(parts)
    audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)   # same format as the Piper version

    return SAMPLE_RATE, audio