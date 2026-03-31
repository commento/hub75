# audio_input.py

import queue
import numpy as np
import sounddevice as sd

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

def get_latest_audio_frame():
    latest = None
    while not audio_queue.empty():
        latest = audio_queue.get()

    if latest is None:
        return np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    return latest

SAMPLE_RATE = 48000
BLOCK_SIZE = 512
CHANNELS = 2   # change to 2 only if your device really supports stereo


def find_input_device():
    devices = sd.query_devices()

    print("=== AVAILABLE AUDIO DEVICES ===")
    for i, d in enumerate(devices):
        print(i, d["name"], "IN:", d["max_input_channels"], "OUT:", d["max_output_channels"])

    # choose first valid input device
    for i, d in enumerate(devices):
        if d["max_input_channels"] >= CHANNELS:
            print(f"[AUDIO] Using input device {i}: {d['name']}")
            return i

    raise RuntimeError("No suitable audio input device found")


def start_audio_stream():
    device_index = find_input_device()

    print("[AUDIO] Starting stream...")
    stream = sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback
    )
    stream.start()
    return stream

