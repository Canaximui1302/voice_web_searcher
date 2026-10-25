import wave
import numpy as np
from piper import PiperVoice

voice = PiperVoice.load("voice_model/en_GB-jenny_dioco-medium.onnx")
sample_rate = voice.config.sample_rate

def text_to_speech(text):
    pcm = []
    for chunk in voice.synthesize(text):
        pcm.append(chunk.audio_int16_bytes)
    audio = np.frombuffer(b"".join(pcm), dtype=np.int16)
    return sample_rate, audio