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
# VIDEO
# =========================================================
def preload_random_frames(video_path, width, height, num_frames=100):
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


def get_next_video_frame(cap, width, height, random_buffer=None, speed=1.0):
    """
    speed:
        1.0 = normale
        <1.0 = slow motion (hold parziale)
        >1.0 = skip frame
    """
    if speed > 1.2:
        skip = int(speed - 1)
        for _ in range(skip):
            cap.read()

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
# FRAME FX
# =========================================================
def blend_frames(a, b, mix):
    mix = clamp(mix)
    return (a.astype(np.float32) * (1.0 - mix) + b.astype(np.float32) * mix).astype(np.uint8)


def apply_drift(frame, amount_x=0, amount_y=0):
    return np.roll(np.roll(frame, amount_y, axis=0), amount_x, axis=1)


def apply_scan_glitch(frame, intensity=0.0):
    out = frame.copy()
    h, w, _ = out.shape

    num_lines = int(intensity * 10)
    for _ in range(num_lines):
        y = random.randint(0, h - 2)
        shift = random.randint(-8, 8)
        out[y:y+2] = np.roll(out[y:y+2], shift, axis=1)

    return out


# =========================================================
# MODES
# =========================================================
MODE_FLOW = "FLOW"
MODE_PULSE = "PULSE"
MODE_CHAOS = "CHAOS"
MODE_FREEZE_DRIFT = "FREEZE_DRIFT"


def choose_mode(energy, transient, silence_amount, chaos):
    if silence_amount > 0.75:
        return MODE_FREEZE_DRIFT
    if chaos > 0.65 and transient > 0.22:
        return MODE_CHAOS
    if energy > 0.38:
        return MODE_PULSE
    return MODE_FLOW


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

    random_buffer = preload_random_frames(VIDEO_PATH, WIDTH, HEIGHT, num_frames=120)

    initial_frame = get_next_video_frame(cap, WIDTH, HEIGHT, random_buffer=random_buffer)
    if initial_frame is None:
        initial_frame = get_random_preloaded_frame(random_buffer)

    visual.base_img = initial_frame.copy()
    visual.luma = visual.compute_luma(initial_frame)
    visual.edge_map = visual.compute_edge_map(visual.luma)
    visual.motion_map = visual.compute_motion_map(visual.luma)

    frozen_output = initial_frame.copy()
    previous_frame = initial_frame.copy()

    # =====================================================
    # AUDIO GAIN
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
    # GLOBAL STATE
    # =====================================================
    energy = 0.0
    chaos = 0.0
    density = 0.0
    silence_amount = 0.0

    mode = MODE_FLOW
    mode_hold_until = 0.0

    # jump memory
    jump_accumulator = 0.0
    last_jump_time = 0.0
    jump_hold_until = 0.0

    JUMP_COOLDOWN = 0.65
    JUMP_HOLD = 0.14

    # micro loop / frame hold
    frame_hold_until = 0.0
    held_frame = None

    # silence
    SILENCE_THRESHOLD = 0.020
    SILENCE_HOLD = 0.55
    last_audio_time = time.time()

    # fps
    target_fps = 30
    frame_duration = 1.0 / target_fps

    frame_counter = 0

    try:
        while True:
            loop_start = time.time()
            now = time.time()
            frame_counter += 1

            # =========================
            # AUDIO
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

            # invece di freeze binario, costruiamo un "silence amount"
            target_silence = 1.0 if no_audio else 0.0
            silence_amount = smooth_value(silence_amount, target_silence, alpha=0.08)

            # =========================
            # GLOBAL STATES
            # =========================
            target_energy = clamp(
                smoothed["rms"] * 0.45 +
                smoothed["low"] * 0.35 +
                smoothed["high"] * 0.10 +
                smoothed["transient"] * 0.25
            )

            energy = smooth_value(energy, target_energy, alpha=0.10)

            target_chaos = clamp(
                smoothed["transient"] * 0.70 +
                low_rise * 0.60 +
                smoothed["high"] * 0.30
            )

            chaos = smooth_value(chaos, target_chaos, alpha=0.08)

            target_density = clamp(
                smoothed["mid"] * 0.45 +
                smoothed["high"] * 0.35 +
                energy * 0.30
            )

            density = smooth_value(density, target_density, alpha=0.08)

            # =========================
            # MODE SELECTION
            # =========================
            if now > mode_hold_until:
                new_mode = choose_mode(energy, smoothed["transient"], silence_amount, chaos)
                if new_mode != mode:
                    mode = new_mode
                    mode_hold_until = now + random.uniform(1.5, 4.0)

            # =========================
            # JUMP ENERGY
            # =========================
            jump_drive = onset_score * 0.75 + smoothed["low"] * 0.20

            if jump_drive > 0.20:
                jump_accumulator += jump_drive * 0.06
            else:
                jump_accumulator *= 0.94

            jump_accumulator = clamp(jump_accumulator, 0.0, 1.0)

            jump_condition = (
                smoothed["low"] > 0.16 and
                smoothed["transient"] > 0.08 and
                onset_score > 0.26 and
                jump_accumulator > 0.32 and
                now > jump_hold_until and
                (now - last_jump_time) > JUMP_COOLDOWN
            )

            jump_probability = clamp(
                (onset_score - 0.24) * 1.8 +
                jump_accumulator * 0.65 +
                chaos * 0.35,
                0.0,
                0.95
            )

            do_jump = jump_condition and (random.random() < jump_probability)

            # =====================================================
            # VIDEO BEHAVIOUR
            # =====================================================
            speed = 1.0
            frame_rgb = None

            if mode == MODE_FLOW:
                speed = 0.9 + energy * 0.8

            elif mode == MODE_PULSE:
                speed = 1.0 + smoothed["transient"] * 2.2

            elif mode == MODE_CHAOS:
                speed = 1.2 + chaos * 2.0

            elif mode == MODE_FREEZE_DRIFT:
                speed = 0.55

            # -------------------------
            # occasional frame hold
            # -------------------------
            hold_chance = 0.0
            if mode == MODE_FLOW:
                hold_chance = 0.01 + (1.0 - energy) * 0.04
            elif mode == MODE_PULSE:
                hold_chance = 0.03 + smoothed["transient"] * 0.08
            elif mode == MODE_CHAOS:
                hold_chance = 0.05 + chaos * 0.10
            elif mode == MODE_FREEZE_DRIFT:
                hold_chance = 0.08

            if now < frame_hold_until and held_frame is not None:
                frame_rgb = held_frame.copy()
            else:
                if random.random() < hold_chance:
                    held_frame = previous_frame.copy()
                    frame_hold_until = now + random.uniform(0.03, 0.14)
                    frame_rgb = held_frame.copy()

            # -------------------------
            # jump / normal fetch
            # -------------------------
            if frame_rgb is None:
                if do_jump:
                    frame_rgb = get_random_preloaded_frame(random_buffer)
                    last_jump_time = now
                    jump_hold_until = now + JUMP_HOLD
                    jump_accumulator *= 0.35
                else:
                    frame_rgb = get_next_video_frame(
                        cap,
                        WIDTH,
                        HEIGHT,
                        random_buffer=random_buffer,
                        speed=speed
                    )

            if frame_rgb is None:
                frame_rgb = previous_frame.copy()

            # -------------------------
            # blend continuity
            # -------------------------
            continuity_mix = 0.0
            if mode == MODE_FLOW:
                continuity_mix = 0.35
            elif mode == MODE_PULSE:
                continuity_mix = 0.18
            elif mode == MODE_CHAOS:
                continuity_mix = 0.08
            elif mode == MODE_FREEZE_DRIFT:
                continuity_mix = 0.45

            frame_rgb = blend_frames(previous_frame, frame_rgb, 1.0 - continuity_mix)

            # -------------------------
            # drift / glitch
            # -------------------------
            drift_x = int(np.sin(frame_counter * 0.05) * (2 + silence_amount * 4))
            drift_y = int(np.cos(frame_counter * 0.03) * (1 + silence_amount * 2))
            frame_rgb = apply_drift(frame_rgb, drift_x, drift_y)

            glitch_amount = clamp(
                chaos * 0.55 +
                smoothed["transient"] * 0.35 +
                (0.25 if mode == MODE_CHAOS else 0.0)
            )
            frame_rgb = apply_scan_glitch(frame_rgb, glitch_amount)

            previous_frame = frame_rgb.copy()

            # =====================================================
            # UPDATE VISUAL ENGINE
            # =====================================================
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

                # nuove feature "musicali / di stato"
                "energy": energy,
                "chaos": chaos,
                "density": density,
                "silence": silence_amount,
                "mode_flow": 1.0 if mode == MODE_FLOW else 0.0,
                "mode_pulse": 1.0 if mode == MODE_PULSE else 0.0,
                "mode_chaos": 1.0 if mode == MODE_CHAOS else 0.0,
                "mode_freeze": 1.0 if mode == MODE_FREEZE_DRIFT else 0.0,
            }

            out_frame = visual.update(visual_features)
            frozen_output = out_frame.copy()

            # =====================================================
            # IF SILENCE: non freeze totale, ma "slow ghost"
            # =====================================================
            if silence_amount > 0.82:
                out_frame = blend_frames(frozen_output, previous_frame, 0.08)

            # =====================================================
            # SEND TO MATRIX
            # =====================================================
            mapped_frame = map_128x128_to_4x64x64_chain(
                out_frame,
                order=PANEL_ORDER,
                transforms=PANEL_TRANSFORMS,
            )

            pil_img = Image.fromarray(mapped_frame)
            offscreen_canvas.SetImage(pil_img, 0, 0)
            offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

            # =====================================================
            # DEBUG
            # =====================================================
            print(
                f"MODE:{mode:<13} "
                f"RMS:{smoothed['rms']:.2f} "
                f"LOW:{smoothed['low']:.2f} "
                f"MID:{smoothed['mid']:.2f} "
                f"HIGH:{smoothed['high']:.2f} "
                f"TR:{smoothed['transient']:.2f} "
                f"EN:{energy:.2f} "
                f"CH:{chaos:.2f} "
                f"DE:{density:.2f} "
                f"SI:{silence_amount:.2f} "
                f"ON:{onset_score:.2f} "
                f"ACC:{jump_accumulator:.2f} ",
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