# audio_input.py

import threading
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, BLOCK_SIZE, CHANNELS, AUDIO_DEVICE


_audio_lock = threading.Lock()
_latest_audio_frame = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)
_overflow_count = 0


def audio_callback(indata, frames, time, status):
    global _latest_audio_frame, _overflow_count

    if status and getattr(status, "input_overflow", False):
        _overflow_count += 1

    with _audio_lock:
        _latest_audio_frame = indata.copy()


def start_audio_stream():
    stream = sd.InputStream(
        device=AUDIO_DEVICE,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
    return stream


def get_latest_audio_frame():
    with _audio_lock:
        return _latest_audio_frame.copy()


def consume_overflow_count():
    global _overflow_count
    count = _overflow_count
    _overflow_count = 0
    return count
