# audio_input.py

import queue
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, BLOCK_SIZE, CHANNELS, AUDIO_DEVICE

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

def start_audio_stream():
    stream = sd.InputStream(
        device=AUDIO_DEVICE,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    stream.start()
    return stream

def get_latest_audio_frame():
    latest = None
    while not audio_queue.empty():
        latest = audio_queue.get()

    if latest is None:
        return np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    return latest