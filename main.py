# main.py

import pygame
import numpy as np
import cv2
import random

from config import WIDTH, HEIGHT, FPS, SCALE, IMAGE_PATH
from audio_input import start_audio_stream, get_latest_audio_frame
from audio_features import StereoFeatureExtractor
from visual_engine import VisualEngineClean

SHOW_DEBUG = True

def is_black_frame(frame_rgb, threshold=18, dark_ratio=0.92):
    luma = (
        0.299 * frame_rgb[:,:,0] +
        0.587 * frame_rgb[:,:,1] +
        0.114 * frame_rgb[:,:,2]
    )
    dark_pixels = np.mean(luma < threshold)
    return dark_pixels > dark_ratio

def draw_debug_text(screen, font, features):
    lines = [
        f"RMS: {features['rms']:.2f}",
        f"LOW: {features['low']:.2f}",
        f"MID: {features['mid']:.2f}",
        f"HIGH: {features['high']:.2f}",
        "D = debug on/off",
        "ESC = quit"
    ]
    y = 10
    for line in lines:
        surf = font.render(line, True, (255, 255, 255))
        screen.blit(surf, (10, y))
        y += 20

def main():
    global SHOW_DEBUG

    pygame.init()
    screen = pygame.display.set_mode((WIDTH*SCALE, HEIGHT*SCALE))
    pygame.display.set_caption("Edge Reactive Engine V4 Clean")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # avvia audio
    stream = start_audio_stream()
    extractor = StereoFeatureExtractor()
    visual = VisualEngineClean(IMAGE_PATH, WIDTH, HEIGHT)


    cap = cv2.VideoCapture("video.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # soglia kick
    KICK_THRESHOLD = 0.50

    # flag per evitare multi-trigger nello stesso colpo
    kick_triggered = False

    def get_random_frame(cap, total_frames, max_tries=12):
        for _ in range(max_tries):
            frame_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not is_black_frame(frame_rgb):
                return frame_rgb

        # fallback se trova solo nero
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_d:
                        SHOW_DEBUG = not SHOW_DEBUG

            audio_frame = get_latest_audio_frame()
            # features = extractor.extract(audio_frame)

            # frame = visual.update(features)

            # estrai RMS dalla tua pipeline audio
            features = extractor.extract(audio_frame)  # {"rms":..., "low":..., "mid":..., "high":...}
            
            # kick detection
            if features["rms"] > KICK_THRESHOLD and not kick_triggered:
                visual.base_img = get_random_frame(cap, total_frames)
                visual.luma = visual.compute_luma(visual.base_img)
                visual.edge_map = visual.compute_edge_map(visual.luma)
                visual.motion_map = visual.compute_motion_map(visual.luma)
                kick_triggered = True
            elif features["rms"] <= KICK_THRESHOLD:
                kick_triggered = False
                if features["rms"] < 0.001:
                    continue
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                    ret, frame = cap.read()
                frame = cv2.resize(frame, (WIDTH,HEIGHT))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # se troppo nero -> jump random
                if is_black_frame(frame_rgb):
                    frame_rgb = get_random_frame(cap, total_frames)
                visual.base_img = frame_rgb  # aggiorna il frame corrente
                visual.luma = visual.compute_luma(frame_rgb)
                visual.edge_map = visual.compute_edge_map(visual.luma)
                visual.motion_map = visual.compute_motion_map(visual.luma)
            
            # applica edge-reactive
            frame = visual.update(features)

            # prepara superficie pygame
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1,0,2)))
            surface = pygame.transform.scale(surface, (WIDTH*SCALE, HEIGHT*SCALE))
            screen.blit(surface, (0,0))

            if SHOW_DEBUG:
                draw_debug_text(screen, font, features)

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        stream.stop()
        stream.close()
        pygame.quit()

if __name__ == "__main__":
    main()
