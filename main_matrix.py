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

PANEL_ORDER = ("p3", "p1", "p2", "p4")

PANEL_TRANSFORMS = {
    "p1": {"flip_x": False, "flip_y": False, "rotate": 270},
    "p2": {"flip_x": False, "flip_y": False, "rotate": 90},
    "p3": {"flip_x": False, "flip_y": False, "rotate": 270},
    "p4": {"flip_x": False, "flip_y": False, "rotate": 90},
}


# =========================================================
# PANEL MAPPING
# =========================================================
def transform_panel(panel, flip_x=False, flip_y=False, rotate=0):
    out = panel.copy()

    if flip_x:
        out = np.fliplr(out)
    if flip_y:
        out = np.flipud(out)

    if rotate == 90:
        out = np.rot90(out, k=1)
    elif rotate == 180:
        out = np.rot90(out, k=2)
    elif rotate == 270:
        out = np.rot90(out, k=3)

    return out


def map_128x128_to_4x64x64_chain(
    frame_128,
    order=("p3", "p1", "p2", "p4"),
    transforms=None
):
    if frame_128.shape[0] != 128 or frame_128.shape[1] != 128:
        raise ValueError(f"Expected frame shape (128,128,3), got {frame_128.shape}")

    panels = {
        "p1": frame_128[0:64,   0:64].copy(),
        "p2": frame_128[0:64,  64:128].copy(),
        "p3": frame_128[64:128, 0:64].copy(),
        "p4": frame_128[64:128, 64:128].copy(),
    }

    if transforms is None:
        transforms = {}

    mapped_panels = []
    for key in order:
        panel = panels[key]
        t = transforms.get(key, {})
        panel = transform_panel(
            panel,
            flip_x=t.get("flip_x", False),
            flip_y=t.get("flip_y", False),
            rotate=t.get("rotate", 0),
        )
        mapped_panels.append(panel)

    out = np.concatenate(mapped_panels, axis=1)
    return out


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

    options.rows = 64
    options.cols = 64
    options.chain_length = 4
    options.parallel = 1

    options.hardware_mapping = "regular"
    options.gpio_slowdown = 4
    options.brightness = 70
    options.pwm_bits = 11
    options.pwm_lsb_nanoseconds = 130
    options.disable_hardware_pulsing = True
    options.limit_refresh_rate_hz = 120

    return RGBMatrix(options=options)


# =========================================================
# VIDEO BUFFER PRELOAD
# =========================================================
def preload_random_frames(video_path, width, height, num_frames=80):
    print(f"[PRELOAD] Loading {num_frames} random frames...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for preload: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Video has no readable frames.")

    buffer_frames = []
    attempts = 0
    max_attempts = num_frames * 6

    while len(buffer_frames) < num_frames and attempts < max_attempts:
        attempts += 1
        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        if random_buffer:
            return get_random_preloaded_frame(random_buffer)
        return None

    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if is_black_frame(frame_rgb) and random_buffer:
        frame_rgb = get_random_preloaded_frame(random_buffer)

    return frame_rgb


# =========================================================
# SMALL UTILS
# =========================================================
def smooth_value(prev, new, alpha=0.18):
    return prev * (1.0 - alpha) + new * alpha


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


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

    # ===== Visual =====
    visual = VisualEngineClean(IMAGE_PATH, WIDTH, HEIGHT)

    # ===== Video =====
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Cannot read total video frames.")

    # ===== Random jump preload =====
    random_buffer = preload_random_frames(VIDEO_PATH, WIDTH, HEIGHT, num_frames=100)

    # ===== Initial frame =====
    initial_frame = get_next_video_frame(cap, WIDTH, HEIGHT, random_buffer=random_buffer)
    if initial_frame is None:
        initial_frame = get_random_preloaded_frame(random_buffer)

    visual.base_img = initial_frame.copy()
    visual.luma = visual.compute_luma(initial_frame)
    visual.edge_map = visual.compute_edge_map(visual.luma)
    visual.motion_map = visual.compute_motion_map(visual.luma)

    frozen_output = initial_frame.copy()

    # =====================================================
    # AUDIO BEHAVIOUR TUNING
    # =====================================================
    AUDIO_GAIN = 2.2
    LOW_GAIN = 1.35
    MID_GAIN = 1.15
    HIGH_GAIN = 1.10
    TRANSIENT_GAIN = 1.45
    KICK_GAIN = 1.65

    smoothed = {
        "rms": 0.0,
        "low": 0.0,
        "mid": 0.0,
        "high": 0.0,
        "transient": 0.0,
        "kick": 0.0,
    }

    prev_low = 0.0
    prev_rms = 0.0

    # =====================================================
    # JUMP CONTROL
    # =====================================================
    last_jump_time = 0.0
    jump_hold_until = 0.0

    JUMP_COOLDOWN = 0.72
    JUMP_HOLD = 0.18

    jump_accumulator = 0.0
    energy_memory = 0.0

    # =====================================================
    # SILENCE / STILLNESS
    # =====================================================
    SILENCE_THRESHOLD = 0.030
    SILENCE_HOLD = 0.80
    last_audio_time = time.time()

    # =====================================================
    # FPS
    # =====================================================
    target_fps = 30
    frame_duration = 1.0 / target_fps

    try:
        while True:
            loop_start = time.time()
            now = time.time()

            # =========================
            # AUDIO INPUT
            # =========================
            audio_frame = get_latest_audio_frame()
            raw = extractor.extract(audio_frame)

            # =========================
            # SOFTWARE GAIN
            # =========================
            boosted = {
                "rms": clamp(raw["rms"] * AUDIO_GAIN),
                "low": clamp(raw["low"] * LOW_GAIN),
                "mid": clamp(raw["mid"] * MID_GAIN),
                "high": clamp(raw["high"] * HIGH_GAIN),
                "transient": clamp(raw.get("transient", 0.0) * TRANSIENT_GAIN),
                "kick": clamp(raw.get("kick", 0.0) * KICK_GAIN),
            }

            # =========================
            # SMOOTHING
            # =========================
            smoothed["rms"] = smooth_value(smoothed["rms"], boosted["rms"], alpha=0.18)
            smoothed["low"] = smooth_value(smoothed["low"], boosted["low"], alpha=0.20)
            smoothed["mid"] = smooth_value(smoothed["mid"], boosted["mid"], alpha=0.16)
            smoothed["high"] = smooth_value(smoothed["high"], boosted["high"], alpha=0.14)
            smoothed["transient"] = smooth_value(smoothed["transient"], boosted["transient"], alpha=0.26)
            smoothed["kick"] = smooth_value(smoothed["kick"], boosted["kick"], alpha=0.24)

            # =========================
            # ONSET / ACTIVITY
            # =========================
            low_rise = max(0.0, smoothed["low"] - prev_low)
            rms_rise = max(0.0, smoothed["rms"] - prev_rms)

            onset_score = (
                smoothed["kick"] * 1.15 +
                smoothed["transient"] * 0.55 +
                low_rise * 0.45 +
                rms_rise * 0.20
            )

            prev_low = smoothed["low"]
            prev_rms = smoothed["rms"]

            # activity memory (quanto "vive" il brano)
            activity = (
                smoothed["rms"] * 0.35 +
                smoothed["low"] * 0.20 +
                smoothed["mid"] * 0.15 +
                smoothed["high"] * 0.10 +
                smoothed["kick"] * 0.20
            )
            energy_memory = smooth_value(energy_memory, activity, alpha=0.04)

            # =========================
            # SILENCE DETECTION
            # =========================
            if smoothed["rms"] >= SILENCE_THRESHOLD:
                last_audio_time = now

            no_audio = (now - last_audio_time) > SILENCE_HOLD

            # =========================
            # NO HARD FREEZE
            # mantieni ultimo frame ma continua loop
            # =========================
            if no_audio:
                frame_rgb = visual.base_img.copy()
            else:
                frame_rgb = None

            # =========================
            # JUMP ACCUMULATION
            # =========================
            jump_drive = (
                onset_score * 0.65 +
                smoothed["kick"] * 0.55
            )

            if jump_drive > 0.18:
                jump_accumulator += jump_drive * 0.055
            else:
                jump_accumulator *= 0.93

            jump_accumulator = clamp(jump_accumulator, 0.0, 1.0)

            # =========================
            # MUSICAL CHAOS
            # più il brano è vivo, più si apre
            # =========================
            chaos = clamp((energy_memory - 0.16) * 2.0, 0.0, 1.0)

            # =========================
            # JUMP DECISION
            # =========================
            jump_condition = (
                smoothed["kick"] > 0.22 and
                smoothed["transient"] > 0.05 and
                onset_score > 0.28 and
                jump_accumulator > 0.28 and
                now > jump_hold_until and
                (now - last_jump_time) > JUMP_COOLDOWN
            )

            jump_probability = clamp(
                (smoothed["kick"] - 0.18) * 1.9 +
                (onset_score - 0.24) * 1.1 +
                jump_accumulator * 0.55 +
                chaos * 0.25,
                0.0,
                0.90
            )

            do_jump = (not no_audio) and jump_condition and (random.random() < jump_probability)

            # =========================
            # VIDEO SOURCE SELECTION
            # =========================
            if do_jump:
                frame_rgb = get_random_preloaded_frame(random_buffer)
                last_jump_time = now
                jump_hold_until = now + JUMP_HOLD
                jump_accumulator *= 0.38

            elif frame_rgb is None:
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
            # FEATURES FOR VISUAL ENGINE
            # =========================
            visual_features = {
                "rms": smoothed["rms"],
                "low": smoothed["low"],
                "mid": smoothed["mid"],
                "high": smoothed["high"],
                "transient": smoothed["transient"],
                "kick": smoothed["kick"],
                "chaos": chaos,
                "activity": energy_memory,
            }

            out_frame = visual.update(visual_features)
            frozen_output = out_frame.copy()

            # =========================
            # SEND TO MATRIX
            # =========================
            mapped_frame = map_128x128_to_4x64x64_chain(
                out_frame,
                order=PANEL_ORDER,
                transforms=PANEL_TRANSFORMS,
            )

            pil_img = Image.fromarray(mapped_frame)
            offscreen_canvas.SetImage(pil_img, 0, 0)
            offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

            # =========================
            # DEBUG
            # =========================
            print(
                f"RMS:{smoothed['rms']:.2f} "
                f"LOW:{smoothed['low']:.2f} "
                f"MID:{smoothed['mid']:.2f} "
                f"HIGH:{smoothed['high']:.2f} "
                f"TR:{smoothed['transient']:.2f} "
                f"K:{smoothed['kick']:.2f} "
                f"ON:{onset_score:.2f} "
                f"ACC:{jump_accumulator:.2f} "
                f"CH:{chaos:.2f} "
                f"JP:{jump_probability:.2f} "
                f"{'[JUMP]' if do_jump else '[RUN ]'}    ",
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