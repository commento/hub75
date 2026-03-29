# main_matrix.py

import time
import cv2
import random
import numpy as np
from PIL import Image

from rgbmatrix import RGBMatrix, RGBMatrixOptions

from config import WIDTH, HEIGHT
from audio_input import start_audio_stream, get_latest_audio_frame
from audio_features import StereoFeatureExtractor
from visual_engine import VisualEngineClean


def setup_matrix():
    options = RGBMatrixOptions()

    # ===== pannello =====
    options.rows = 64
    options.cols = 64
    options.chain_length = 1
    options.parallel = 1

    # ===== hardware =====
    options.hardware_mapping = "adafruit-hat"   # oppure "regular" a seconda del tuo adattatore
    options.gpio_slowdown = 4                   # spesso utile su Pi 4/5
    options.brightness = 70
    options.pwm_bits = 11
    options.pwm_lsb_nanoseconds = 130
    options.disable_hardware_pulsing = False

    # ===== qualità =====
    options.limit_refresh_rate_hz = 120

    matrix = RGBMatrix(options=options)
    return matrix


def get_random_frame(cap, total_frames, width, height):
    frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def get_next_video_frame(cap, width, height):
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def main():
    print("Starting HUB75 visual engine...")

    # ===== Matrix =====
    matrix = setup_matrix()
    offscreen_canvas = matrix.CreateFrameCanvas()

    # ===== Audio =====
    stream = start_audio_stream()
    extractor = StereoFeatureExtractor()

    # ===== Visual engine =====
    visual = VisualEngineClean("base.jpg", WIDTH, HEIGHT)

    # ===== Video =====
    cap = cv2.VideoCapture("video.mov")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ===== Kick detection =====
    KICK_THRESHOLD = 0.50
    kick_triggered = False

    # ===== FPS =====
    target_fps = 30
    frame_duration = 1.0 / target_fps

    try:
        while True:
            loop_start = time.time()

            # =========================
            # AUDIO
            # =========================
            audio_frame = get_latest_audio_frame()
            features = extractor.extract(audio_frame)

            # =========================
            # VIDEO / KICK JUMP
            # =========================
            if features["rms"] > KICK_THRESHOLD and not kick_triggered:
                frame_rgb = get_random_frame(cap, total_frames, WIDTH, HEIGHT)
                kick_triggered = True

            else:
                if features["rms"] <= KICK_THRESHOLD:
                    kick_triggered = False

                # playback normale
                if features["rms"] != 0.0:
                    frame_rgb = get_next_video_frame(cap, WIDTH, HEIGHT)
                else:
                    # se silenzio, tieni il frame corrente
                    frame_rgb = visual.base_img.copy()

            # =========================
            # UPDATE VISUAL SOURCE
            # =========================
            visual.base_img = frame_rgb
            visual.luma = visual.compute_luma(frame_rgb)
            visual.edge_map = visual.compute_edge_map(visual.luma)
            visual.motion_map = visual.compute_motion_map(visual.luma)

            # =========================
            # APPLY VISUAL ENGINE
            # =========================
            out_frame = visual.update(features)

            # =========================
            # SEND TO MATRIX
            # =========================
            pil_img = Image.fromarray(out_frame)
            offscreen_canvas.SetImage(pil_img, 0, 0)
            offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

            # =========================
            # DEBUG CONSOLE
            # =========================
            print(
                f"RMS:{features['rms']:.2f} "
                f"LOW:{features['low']:.2f} "
                f"MID:{features['mid']:.2f} "
                f"HIGH:{features['high']:.2f}",
                end="\r"
            )

            # =========================
            # FPS LIMIT
            # =========================
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop()
        stream.close()
        cap.release()
        matrix.Clear()


if __name__ == "__main__":
    main()