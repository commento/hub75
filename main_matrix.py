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
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "base.jpg"
VIDEO_PATH = BASE_DIR / "video.mov"


# =========================================================
# HELPERS
# =========================================================
def is_black_frame(frame_rgb, threshold=18, dark_ratio=0.92):
    luma = (
        0.299 * frame_rgb[:, :, 0] +
        0.587 * frame_rgb[:, :, 1] +
        0.114 * frame_rgb[:, :, 2]
    )
    dark_pixels = np.mean(luma < threshold)
    return dark_pixels > dark_ratio


def setup_matrix():
    options = RGBMatrixOptions()

    # ===== pannello =====
    options.rows = 64
    options.cols = 64
    options.chain_length = 1
    options.parallel = 1

    # ===== hardware =====
    options.hardware_mapping = "regular"
    options.gpio_slowdown = 4
    options.brightness = 70
    options.pwm_bits = 11
    options.pwm_lsb_nanoseconds = 130
    options.disable_hardware_pulsing = True

    # ===== qualità =====
    options.limit_refresh_rate_hz = 120

    matrix = RGBMatrix(options=options)
    return matrix


# =========================================================
# VIDEO BUFFER PRELOAD
# =========================================================
def preload_random_frames(video_path, width, height, num_frames=80):
    """
    Carica in RAM un set di frame random per i jump kick.
    Così eviti seek random live sul file video durante il loop.
    """
    print(f"[PRELOAD] Loading {num_frames} random frames...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for preload: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Video has no readable frames.")

    buffer_frames = []
    attempts = 0
    max_attempts = num_frames * 4

    while len(buffer_frames) < num_frames and attempts < max_attempts:
        attempts += 1
        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # evita di precaricare troppi frame quasi neri
        if is_black_frame(frame_rgb):
            continue

        buffer_frames.append(frame_rgb.copy())

    cap.release()

    if not buffer_frames:
        raise RuntimeError("Preload buffer is empty.")

    print(f"[PRELOAD] Loaded {len(buffer_frames)} frames.")
    return buffer_frames


def get_random_preloaded_frame(random_buffer):
    return random.choice(random_buffer).copy()


def get_next_video_frame(cap, width, height, random_buffer=None):
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if not ret:
        # fallback estremo
        if random_buffer:
            return get_random_preloaded_frame(random_buffer)
        return None

    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # se il frame è troppo nero, usa un frame random già in RAM
    if is_black_frame(frame_rgb) and random_buffer:
        frame_rgb = get_random_preloaded_frame(random_buffer)

    return frame_rgb


# =========================================================
# MAIN
# =========================================================
def main():
    print("Starting HUB75 visual engine...")

    # ===== Matrix =====
    matrix = setup_matrix()
    offscreen_canvas = matrix.CreateFrameCanvas()

    # ===== Audio =====
    stream = start_audio_stream()
    extractor = StereoFeatureExtractor()

    # ===== Visual engine =====
    visual = VisualEngineClean(IMAGE_PATH, WIDTH, HEIGHT)

    # ===== Video =====
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise RuntimeError("Cannot read total video frames.")

    # ===== PRELOAD RANDOM JUMP BUFFER =====
    random_buffer = preload_random_frames(VIDEO_PATH, WIDTH, HEIGHT, num_frames=100)

    # ===== Stato iniziale =====
    initial_frame = get_next_video_frame(cap, WIDTH, HEIGHT, random_buffer=random_buffer)
    if initial_frame is None:
        initial_frame = get_random_preloaded_frame(random_buffer)

    visual.base_img = initial_frame.copy()
    visual.luma = visual.compute_luma(initial_frame)
    visual.edge_map = visual.compute_edge_map(visual.luma)
    visual.motion_map = visual.compute_motion_map(visual.luma)

    frozen_output = initial_frame.copy()

    # ===== Kick detection =====
    KICK_THRESHOLD = 0.50
    kick_triggered = False
    last_kick_time = 0.0
    KICK_COOLDOWN = 0.45  # secondi

    # ===== Silence freeze =====
    SILENCE_THRESHOLD = 0.015
    SILENCE_HOLD = 0.35
    last_audio_time = time.time()

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

            now = time.time()

            if features["rms"] >= SILENCE_THRESHOLD:
                last_audio_time = now

            no_audio = (now - last_audio_time) > SILENCE_HOLD

            # =========================
            # FREEZE SE SILENZIO
            # =========================
            if no_audio:
                pil_img = Image.fromarray(frozen_output)
                offscreen_canvas.SetImage(pil_img, 0, 0)
                offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

                print(
                    f"RMS:{features['rms']:.2f} "
                    f"LOW:{features['low']:.2f} "
                    f"MID:{features['mid']:.2f} "
                    f"HIGH:{features['high']:.2f} "
                    f"[FREEZE]",
                    end="\r"
                )

                elapsed = time.time() - loop_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # =========================
            # KICK JUMP (RAM BUFFER)
            # =========================
            # =========================
            # MUSICAL JUMP DETECTION
            # =========================
            kick_energy = (
                features["low"] * 0.75 +
                features.get("transient", 0.0) * 1.15 +
                features["rms"] * 0.20
            )

            jump_condition = (
                features["low"] > 0.18 and
                features.get("transient", 0.0) > 0.12 and
                features["rms"] > 0.03 and
                kick_energy > 0.32 and
                (now - last_kick_time) > KICK_COOLDOWN and
                not kick_triggered
            )

            if jump_condition:
                frame_rgb = get_random_preloaded_frame(random_buffer)
                kick_triggered = True
                last_kick_time = now
            else:
                # reset trigger solo quando il colpo è davvero finito
                if features["low"] < 0.10 and features.get("transient", 0.0) < 0.06:
                    kick_triggered = False

                # playback normale
                frame_rgb = get_next_video_frame(
                    cap,
                    WIDTH,
                    HEIGHT,
                    random_buffer=random_buffer
                )

                if frame_rgb is None:
                    frame_rgb = visual.base_img.copy()

                # playback normale
                frame_rgb = get_next_video_frame(
                    cap,
                    WIDTH,
                    HEIGHT,
                    random_buffer=random_buffer
                )

                if frame_rgb is None:
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

            # salva per freeze
            frozen_output = out_frame.copy()

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
                f"HIGH:{features['high']:.2f} "
                f"BUF:{len(random_buffer)}",
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