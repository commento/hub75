# main_matrix_preloaded.py

import time
import random
import numpy as np
from PIL import Image

from rgbmatrix import RGBMatrix, RGBMatrixOptions

from config import WIDTH, HEIGHT
from audio_input import start_audio_stream, get_latest_audio_frame
from audio_features import StereoFeatureExtractor
from visual_engine import VisualEngineClean
from video_loader import preload_video_frames


# =========================================
# MATRIX SETUP
# =========================================

def setup_matrix():
    options = RGBMatrixOptions()

    # ---- pannello base ----
    options.rows = 64
    options.cols = 64
    options.chain_length = 1
    options.parallel = 1

    # ---- mapping HUB75 ----
    # prova "adafruit-hat" oppure "regular"
    options.hardware_mapping = "regular"

    # ---- tuning ----
    options.gpio_slowdown = 4
    options.brightness = 70
    options.pwm_bits = 11
    options.pwm_lsb_nanoseconds = 130
    options.disable_hardware_pulsing = True
    options.limit_refresh_rate_hz = 120

    matrix = RGBMatrix(options=options)
    return matrix


# =========================================
# FRAME HELPERS
# =========================================

def get_random_frame(video_frames):
    idx = random.randint(0, len(video_frames) - 1)
    return video_frames[idx].copy(), idx

def get_next_frame(video_frames, current_idx):
    current_idx += 1
    if current_idx >= len(video_frames):
        current_idx = 0
    return video_frames[current_idx].copy(), current_idx


# =========================================
# MAIN
# =========================================

def main():
    print("=== STARTING AUDIO VIDEO MATRIX ENGINE ===")

    # -----------------------------------------
    # Matrix
    # -----------------------------------------
    matrix = setup_matrix()
    offscreen_canvas = matrix.CreateFrameCanvas()

    # -----------------------------------------
    # Audio
    # -----------------------------------------
    print("[AUDIO] Starting stream...")
    stream = start_audio_stream()
    extractor = StereoFeatureExtractor()

    # -----------------------------------------
    # Visual Engine
    # -----------------------------------------
    print("[VISUAL] Initializing visual engine...")
    visual = VisualEngineClean("assets/base.jpg", WIDTH, HEIGHT)

    # -----------------------------------------
    # Video preload
    # -----------------------------------------
    print("[VIDEO] Preloading video into RAM...")
    video_frames = preload_video_frames(
        video_path="video.mov",
        width=WIDTH,
        height=HEIGHT,
        max_frames=None,   # oppure es. 2000 se vuoi limitare
        step=3             # usa 2 o 3 se vuoi alleggerire
    )

    current_video_idx = 0
    current_frame_rgb = video_frames[current_video_idx].copy()

    # -----------------------------------------
    # Kick detection
    # -----------------------------------------
    KICK_THRESHOLD = 0.50
    kick_triggered = False

    # -----------------------------------------
    # Timing
    # -----------------------------------------
    target_fps = 30
    frame_duration = 1.0 / target_fps

    print("[SYSTEM] Running... Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()

            # =====================================
            # 1) AUDIO
            # =====================================
            audio_frame = get_latest_audio_frame()
            features = extractor.extract(audio_frame)

            # =====================================
            # 2) VIDEO CONTROL
            # =====================================
            # kick -> random jump
            if features["rms"] > KICK_THRESHOLD and not kick_triggered:
                current_frame_rgb, current_video_idx = get_random_frame(video_frames)
                kick_triggered = True

            else:
                if features["rms"] <= KICK_THRESHOLD:
                    kick_triggered = False

                # playback lineare se c'è segnale
                if features["rms"] > 0.01:
                    current_frame_rgb, current_video_idx = get_next_frame(video_frames, current_video_idx)
                # se silenzio, mantiene ultimo frame

            # =====================================
            # 3) UPDATE VISUAL SOURCE
            # =====================================
            visual.base_img = current_frame_rgb
            visual.luma = visual.compute_luma(current_frame_rgb)
            visual.edge_map = visual.compute_edge_map(visual.luma)
            visual.motion_map = visual.compute_motion_map(visual.luma)

            # =====================================
            # 4) APPLY VISUAL ENGINE
            # =====================================
            out_frame = visual.update(features)

            # =====================================
            # 5) SEND TO HUB75
            # =====================================
            pil_img = Image.fromarray(out_frame)
            offscreen_canvas.SetImage(pil_img, 0, 0)
            offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

            # =====================================
            # 6) DEBUG CONSOLE
            # =====================================
            print(
                f"RMS:{features['rms']:.2f}  "
                f"LOW:{features['low']:.2f}  "
                f"MID:{features['mid']:.2f}  "
                f"HIGH:{features['high']:.2f}  "
                f"FRAME:{current_video_idx}",
                end="\r"
            )

            # =====================================
            # 7) FPS LIMIT
            # =====================================
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping...")

    finally:
        stream.stop()
        stream.close()
        matrix.Clear()


if __name__ == "__main__":
    main()
