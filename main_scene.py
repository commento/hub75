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


def map_128x128_to_4x64x64_chain(frame_128, order=PANEL_ORDER, transforms=None):
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

    return np.concatenate(mapped_panels, axis=1)


# =========================================================
# HELPERS
# =========================================================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth_value(prev, new, alpha=0.18):
    return prev * (1.0 - alpha) + new * alpha


def is_black_frame(frame_rgb, threshold=18, dark_ratio=0.92):
    luma = (
        0.299 * frame_rgb[:, :, 0] +
        0.587 * frame_rgb[:, :, 1] +
        0.114 * frame_rgb[:, :, 2]
    )
    dark_pixels = np.mean(luma < threshold)
    return dark_pixels > dark_ratio


def blend_frames(a, b, alpha):
    alpha = clamp(alpha)
    out = (a.astype(np.float32) * (1.0 - alpha) + b.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gain(frame, gain=1.0):
    out = frame.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_contrast(frame, contrast=1.0):
    f = frame.astype(np.float32)
    out = (f - 127.5) * contrast + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def shift_frame(frame, dx=0, dy=0):
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def zoom_frame(frame, zoom=1.0):
    if zoom <= 1.001:
        return frame.copy()

    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    crop = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def glitch_slice(frame, intensity=0.2):
    out = frame.copy()
    h, w = out.shape[:2]
    num_slices = int(2 + intensity * 10)

    for _ in range(num_slices):
        y = random.randint(0, h - 4)
        slice_h = random.randint(2, max(3, int(4 + intensity * 14)))
        offset = random.randint(-int(3 + intensity * 16), int(3 + intensity * 16))
        out[y:y+slice_h] = np.roll(out[y:y+slice_h], offset, axis=1)

    return out


def color_split(frame, amount=1):
    if amount <= 0:
        return frame.copy()

    out = frame.copy()
    r = np.roll(out[:, :, 0], amount, axis=1)
    g = out[:, :, 1]
    b = np.roll(out[:, :, 2], -amount, axis=0)
    return np.stack([r, g, b], axis=2)


# =========================================================
# MATRIX
# =========================================================
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
# VIDEO BUFFER
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
    max_attempts = num_frames * 5

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
# MAIN
# =========================================================
def main():
    print("Starting HUB75 visual engine...")

    matrix = setup_matrix()
    offscreen_canvas = matrix.CreateFrameCanvas()

    stream = start_audio_stream()
    extractor = StereoFeatureExtractor()

    visual = VisualEngineClean(IMAGE_PATH, WIDTH, HEIGHT)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Cannot read total video frames.")

    random_buffer = preload_random_frames(VIDEO_PATH, WIDTH, HEIGHT, num_frames=100)

    initial_frame = get_next_video_frame(cap, WIDTH, HEIGHT, random_buffer=random_buffer)
    if initial_frame is None:
        initial_frame = get_random_preloaded_frame(random_buffer)

    visual.base_img = initial_frame.copy()
    visual.luma = visual.compute_luma(initial_frame)
    visual.edge_map = visual.compute_edge_map(visual.luma)
    visual.motion_map = visual.compute_motion_map(visual.luma)

    frozen_output = initial_frame.copy()
    previous_output = initial_frame.copy()
    hold_frame = initial_frame.copy()

    # =====================================================
    # AUDIO TUNING
    # =====================================================
    AUDIO_GAIN = 2.6
    LOW_GAIN = 3.0
    MID_GAIN = 2.2
    HIGH_GAIN = 2.0
    TRANSIENT_GAIN = 3.5

    smoothed = {
        "rms": 0.0,
        "low": 0.0,
        "mid": 0.0,
        "high": 0.0,
        "transient": 0.0,
    }

    prev_low = 0.0
    prev_rms = 0.0

    # =====================================================
    # STATE
    # =====================================================
    last_jump_time = 0.0
    jump_hold_until = 0.0

    JUMP_COOLDOWN = 0.75
    JUMP_HOLD = 0.20
    jump_accumulator = 0.0

    SILENCE_THRESHOLD = 0.020
    SILENCE_HOLD = 0.45
    last_audio_time = time.time()

    # visual state machine
    scene_mode = "calm"
    mode_until = 0.0

    # pulse / glitch
    pulse_energy = 0.0
    glitch_energy = 0.0
    hold_energy = 0.0

    # drift
    drift_x = 0.0
    drift_y = 0.0
    drift_phase = 0.0

    target_fps = 30
    frame_duration = 1.0 / target_fps

    try:
        while True:
            loop_start = time.time()
            now = time.time()
            t = now

            # =========================
            # AUDIO INPUT
            # =========================
            audio_frame = get_latest_audio_frame()
            raw = extractor.extract(audio_frame)

            boosted = {
                "rms": clamp(raw["rms"] * AUDIO_GAIN),
                "low": clamp(raw["low"] * LOW_GAIN),
                "mid": clamp(raw["mid"] * MID_GAIN),
                "high": clamp(raw["high"] * HIGH_GAIN),
                "transient": clamp(raw.get("transient", 0.0) * TRANSIENT_GAIN),
            }

            smoothed["rms"] = smooth_value(smoothed["rms"], boosted["rms"], alpha=0.18)
            smoothed["low"] = smooth_value(smoothed["low"], boosted["low"], alpha=0.16)
            smoothed["mid"] = smooth_value(smoothed["mid"], boosted["mid"], alpha=0.16)
            smoothed["high"] = smooth_value(smoothed["high"], boosted["high"], alpha=0.14)
            smoothed["transient"] = smooth_value(smoothed["transient"], boosted["transient"], alpha=0.22)

            low_rise = max(0.0, smoothed["low"] - prev_low)
            rms_rise = max(0.0, smoothed["rms"] - prev_rms)

            onset_score = (
                smoothed["low"] * 0.55 +
                smoothed["transient"] * 0.95 +
                low_rise * 1.20 +
                rms_rise * 0.55
            )

            prev_low = smoothed["low"]
            prev_rms = smoothed["rms"]

            # =========================
            # SILENCE
            # =========================
            if smoothed["rms"] >= SILENCE_THRESHOLD:
                last_audio_time = now

            no_audio = (now - last_audio_time) > SILENCE_HOLD

            # =========================
            # MODE SELECTION
            # =========================
            if no_audio:
                scene_mode = "silence"
            elif now > mode_until:
                energy = (
                    smoothed["rms"] * 0.4 +
                    smoothed["low"] * 0.3 +
                    smoothed["transient"] * 0.3
                )

                if energy > 0.52:
                    scene_mode = "impact"
                    mode_until = now + random.uniform(0.25, 0.6)
                elif energy > 0.22:
                    scene_mode = "groove"
                    mode_until = now + random.uniform(0.8, 1.8)
                else:
                    scene_mode = "calm"
                    mode_until = now + random.uniform(1.2, 2.5)

            # =========================
            # FREEZE ON SILENCE
            # =========================
            if no_audio:
                idle = blend_frames(frozen_output, previous_output, 0.08)
                idle = apply_gain(idle, 0.985)
                mapped = map_128x128_to_4x64x64_chain(idle, PANEL_ORDER, PANEL_TRANSFORMS)
                pil_img = Image.fromarray(mapped)
                offscreen_canvas.SetImage(pil_img, 0, 0)
                offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

                previous_output = idle.copy()

                elapsed = time.time() - loop_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # =========================
            # JUMP ACCUMULATION
            # =========================
            jump_drive = onset_score * 0.75 + smoothed["low"] * 0.20

            if jump_drive > 0.20:
                jump_accumulator += jump_drive * 0.06
            else:
                jump_accumulator *= 0.92

            jump_accumulator = clamp(jump_accumulator, 0.0, 1.0)

            jump_condition = (
                smoothed["low"] > 0.16 and
                smoothed["transient"] > 0.08 and
                onset_score > 0.26 and
                jump_accumulator > 0.34 and
                now > jump_hold_until and
                (now - last_jump_time) > JUMP_COOLDOWN
            )

            jump_probability = clamp(
                (onset_score - 0.24) * 2.0 + jump_accumulator * 0.7,
                0.0,
                0.92
            )

            do_jump = jump_condition and (random.random() < jump_probability)

            # =========================
            # SOURCE SELECTION
            # =========================
            if do_jump:
                frame_rgb = get_random_preloaded_frame(random_buffer)
                hold_frame = frame_rgb.copy()
                last_jump_time = now
                jump_hold_until = now + JUMP_HOLD
                jump_accumulator *= 0.35

                pulse_energy = 1.0
                glitch_energy = max(glitch_energy, 0.85)
                hold_energy = 1.0
            else:
                frame_rgb = get_next_video_frame(cap, WIDTH, HEIGHT, random_buffer=random_buffer)
                if frame_rgb is None:
                    frame_rgb = visual.base_img.copy()

            # occasional blend with hold frame for "memory"
            memory_mix = clamp(smoothed["mid"] * 0.18 + smoothed["high"] * 0.12)
            frame_rgb = blend_frames(frame_rgb, hold_frame, memory_mix * 0.35)

            # =========================
            # CONTINUOUS MOTION
            # =========================
            drift_phase += 0.035 + smoothed["mid"] * 0.08

            drift_x = np.sin(drift_phase * 0.91) * (1.5 + smoothed["mid"] * 3.0)
            drift_y = np.cos(drift_phase * 0.73) * (1.2 + smoothed["high"] * 2.5)

            if scene_mode == "calm":
                motion_zoom = 1.0 + smoothed["rms"] * 0.03
            elif scene_mode == "groove":
                motion_zoom = 1.0 + smoothed["low"] * 0.06
            else:
                motion_zoom = 1.0 + smoothed["transient"] * 0.12

            frame_rgb = shift_frame(frame_rgb, int(drift_x), int(drift_y))
            frame_rgb = zoom_frame(frame_rgb, motion_zoom)

            # =========================
            # VISUAL ENGINE UPDATE
            # =========================
            visual.base_img = frame_rgb
            visual.luma = visual.compute_luma(frame_rgb)
            visual.edge_map = visual.compute_edge_map(visual.luma)
            visual.motion_map = visual.compute_motion_map(visual.luma)

            visual_features = {
                "rms": smoothed["rms"],
                "low": smoothed["low"],
                "mid": smoothed["mid"],
                "high": smoothed["high"],
                "transient": smoothed["transient"],
            }

            out_frame = visual.update(visual_features)

            # =========================
            # POST FX LAYER
            # =========================
            pulse_energy *= 0.86
            glitch_energy *= 0.80
            hold_energy *= 0.92

            # brightness pulse
            pulse_gain = 1.0 + pulse_energy * (0.18 + smoothed["transient"] * 0.18)
            out_frame = apply_gain(out_frame, pulse_gain)

            # contrast movement
            dynamic_contrast = 1.0 + smoothed["mid"] * 0.25 + smoothed["high"] * 0.18
            out_frame = apply_contrast(out_frame, dynamic_contrast)

            # glitch only when energetic
            if glitch_energy > 0.08 and random.random() < (0.25 + smoothed["transient"] * 0.5):
                out_frame = glitch_slice(out_frame, glitch_energy)

            # RGB split micro effect
            split_amount = int(smoothed["high"] * 3 + glitch_energy * 2)
            if split_amount > 0 and random.random() < 0.35:
                out_frame = color_split(out_frame, split_amount)

            # hold echo on impacts
            if hold_energy > 0.08:
                out_frame = blend_frames(out_frame, hold_frame, hold_energy * 0.18)

            # trail / persistence
            trail_mix = 0.10 + smoothed["rms"] * 0.18
            out_frame = blend_frames(out_frame, previous_output, trail_mix)

            # scene-specific personality
            if scene_mode == "calm":
                out_frame = apply_gain(out_frame, 0.96)
            elif scene_mode == "groove":
                out_frame = apply_contrast(out_frame, 1.06)
            elif scene_mode == "impact":
                out_frame = apply_gain(out_frame, 1.05)
                if random.random() < 0.18:
                    out_frame = glitch_slice(out_frame, 0.35)

            frozen_output = out_frame.copy()
            previous_output = out_frame.copy()

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
                f"MODE:{scene_mode:<7} "
                f"RMS:{smoothed['rms']:.2f} "
                f"LOW:{smoothed['low']:.2f} "
                f"MID:{smoothed['mid']:.2f} "
                f"HIGH:{smoothed['high']:.2f} "
                f"TR:{smoothed['transient']:.2f} "
                f"ON:{onset_score:.2f} "
                f"ACC:{jump_accumulator:.2f} "
                f"P:{pulse_energy:.2f} "
                f"G:{glitch_energy:.2f}    ",
                end="\r"
            )

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
