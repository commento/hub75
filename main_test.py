import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rgbmatrix import RGBMatrix, RGBMatrixOptions
from pathlib import Path

# =========================
# CONFIG
# =========================
WIDTH = 128
HEIGHT = 128

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "base.jpg"

# =========================
# PANEL TRANSFORMS
# =========================
# Se un pannello è ruotato/specchiato, modificalo qui
PANEL_ORDER = ("p3", "p1", "p2", "p4")

PANEL_TRANSFORMS = {
    "p1": {"flip_x": False, "flip_y": False, "rotate": 270},
    "p2": {"flip_x": False, "flip_y": False, "rotate": 90},
    "p3": {"flip_x": False, "flip_y": False, "rotate": 270},
    "p4": {"flip_x": False, "flip_y": False, "rotate": 90},
}


# =========================
# MATRIX SETUP
# =========================
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

    matrix = RGBMatrix(options=options)
    return matrix


# =========================
# PANEL MAPPING
# =========================
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
    order=("p1", "p2", "p3", "p4"),
    transforms=None
):
    """
    Mappa un frame 128x128 RGB in una strip 256x64 RGB per 4 pannelli 64x64 HUB75.

    Layout logico desiderato:
        [ P1 ][ P2 ]
        [ P3 ][ P4 ]

    Chain fisica:
        P1 -> P2 -> P3 -> P4
    """

    if frame_128.shape[0] != 128 or frame_128.shape[1] != 128:
        raise ValueError(f"Expected frame shape (128,128,3), got {frame_128.shape}")

    panels = {
        "p1": frame_128[0:64,   0:64].copy(),     # top-left
        "p2": frame_128[0:64,  64:128].copy(),    # top-right
        "p3": frame_128[64:128, 0:64].copy(),     # bottom-left
        "p4": frame_128[64:128, 64:128].copy(),   # bottom-right
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


# =========================
# IMAGE + TEST OVERLAY
# =========================
def load_base_image(path, width=128, height=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def add_test_overlay(img):
    """
    Disegna:
    - griglia quadranti
    - croce centrale
    - bordi colorati
    - etichette P1/P2/P3/P4
    """
    draw = ImageDraw.Draw(img)

    # colori forti per capire bene i quadranti
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 120, 255)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # linee centrali
    draw.line((64, 0, 64, 127), fill=WHITE, width=2)
    draw.line((0, 64, 127, 64), fill=WHITE, width=2)

    # croce centrale più evidente
    draw.rectangle((60, 60, 68, 68), outline=WHITE, width=2)

    # bordi quadranti
    draw.rectangle((0, 0, 63, 63), outline=RED, width=3)         # P1
    draw.rectangle((64, 0, 127, 63), outline=GREEN, width=3)     # P2
    draw.rectangle((0, 64, 63, 127), outline=BLUE, width=3)      # P3
    draw.rectangle((64, 64, 127, 127), outline=YELLOW, width=3)  # P4

    # testo
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    # box testo leggibile
    def label(x, y, text, color):
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2),
            fill=BLACK
        )
        draw.text((x, y), text, font=font, fill=color)

    label(18, 20, "P1", RED)
    label(82, 20, "P2", GREEN)
    label(18, 84, "P3", BLUE)
    label(82, 84, "P4", YELLOW)

    # numeri di orientamento
    label(4, 4, "TL", WHITE)
    label(104, 4, "TR", WHITE)
    label(4, 112, "BL", WHITE)
    label(104, 112, "BR", WHITE)

    return img


# =========================
# MAIN
# =========================
def main():
    print("Starting panel mapping test...")

    matrix = setup_matrix()
    offscreen_canvas = matrix.CreateFrameCanvas()

    # carica immagine base
    base_img = load_base_image(IMAGE_PATH, WIDTH, HEIGHT)

    # aggiunge overlay di test
    test_img = add_test_overlay(base_img.copy())

    # converte in numpy
    frame_128 = np.array(test_img, dtype=np.uint8)

    # mapping verso strip HUB75
    mapped_frame = map_128x128_to_4x64x64_chain(
        frame_128,
        order=PANEL_ORDER,
        transforms=PANEL_TRANSFORMS
    )

    # invia alla matrice
    output_img = Image.fromarray(mapped_frame)
    offscreen_canvas.SetImage(output_img, 0, 0)
    offscreen_canvas = matrix.SwapOnVSync(offscreen_canvas)

    print("\n=== PANEL TEST ACTIVE ===")
    print(f"ORDER: {PANEL_ORDER}")
    print(f"TRANSFORMS: {PANEL_TRANSFORMS}")
    print("\nExpected logical layout:")
    print("[ P1 ][ P2 ]")
    print("[ P3 ][ P4 ]")
    print("\nPress CTRL+C to quit.")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        matrix.Clear()


if __name__ == "__main__":
    main()
