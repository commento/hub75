# video_loader.py

import cv2
import numpy as np

def preload_video_frames(video_path, width, height, max_frames=None, step=1):
    """
    Carica il video in RAM come lista di frame RGB uint8 già ridimensionati.
    
    Args:
        video_path (str): path al file video
        width (int): larghezza target
        height (int): altezza target
        max_frames (int|None): massimo numero di frame da caricare
        step (int): prende 1 frame ogni 'step' per alleggerire
    
    Returns:
        list[np.ndarray]: lista di frame shape (H, W, 3), RGB uint8
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] Total frames in file: {total}")

    idx = 0
    loaded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            loaded += 1
            if max_frames is not None and loaded >= max_frames:
                break

        idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Nessun frame caricato dal video.")

    print(f"[VIDEO] Loaded frames in RAM: {len(frames)}")
    print(f"[VIDEO] Approx memory: {len(frames) * width * height * 3 / (1024*1024):.2f} MB")

    return frames