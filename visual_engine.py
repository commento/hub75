import numpy as np
import cv2
from PIL import Image

class VisualEngineClean:
    def __init__(self, image_path, width=64, height=64):
        self.width = width
        self.height = height
        self.base_img = self.load_base_image(image_path)
        self.luma = self.compute_luma(self.base_img)
        self.edge_map = self.compute_edge_map(self.luma)
        self.prev_luma = self.luma.copy()
        self.motion_map = np.zeros_like(self.luma)
        self.time = 0.0

    def get_edge_focus_mask(self):
        edges = np.clip(self.edge_map, 0.0, 1.0)
        edges = self.blur3(edges)
        edges = np.power(edges, 0.85)
        return np.clip(edges, 0.0, 1.0)

    def apply_red_grade(self, img, strength=1.0):
        out = img.astype(np.float32)

        # luma per mantenere un po' di leggibilità
        luma = self.compute_luma(img) / 255.0
        luma = np.expand_dims(luma, axis=2)

        # palette rosso / sangue
        red_boost   = 1.35 + 0.25 * strength
        green_scale = 0.35
        blue_scale  = 0.28

        out[:,:,0] *= red_boost
        out[:,:,1] *= green_scale
        out[:,:,2] *= blue_scale

        # un po' di glow sui chiari
        out[:,:,0] += luma[:,:,0] * 45.0
        out[:,:,1] += luma[:,:,0] * 8.0
        out[:,:,2] += luma[:,:,0] * 4.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def load_base_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def compute_luma(self, img):
        return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.float32)

    def blur3(self, arr):
        padded = np.pad(arr, ((1,1),(1,1)), mode='edge')
        out = (
            padded[:-2,:-2] + padded[:-2,1:-1] + padded[:-2,2:] +
            padded[1:-1,:-2] + padded[1:-1,1:-1] + padded[1:-1,2:] +
            padded[2:,:-2] + padded[2:,1:-1] + padded[2:,2:]
        ) / 9.0
        return out

    def compute_edge_map(self, luma):
        blurred = cv2.GaussianBlur(luma.astype(np.float32), (3, 3), 0)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.GaussianBlur(mag, (3, 3), 0)

        lo = float(np.percentile(mag, 70))
        hi = float(np.percentile(mag, 99))
        mag = (mag - lo) / max(hi - lo, 1e-6)
        mag = np.clip(mag, 0.0, 1.0)
        mag = np.power(mag, 1.15)
        return mag

    def compute_motion_map(self, current_luma):
        diff = np.abs(current_luma - self.prev_luma)
        diff = self.blur3(diff)
        diff = diff / (np.max(diff) + 1e-6)
        self.prev_luma = current_luma.copy()
        return diff

    # =========================================
    # NUOVA MASCHERA: TUTTO CIO' CHE E' STATICO
    # =========================================
    def get_static_field_mask(self):
        # motion basso = statico
        static_mask = 1.0 - np.clip(self.motion_map * 2.8, 0.0, 1.0)

        # leggera esclusione di zone completamente piatte
        local_mean = self.blur3(self.luma)
        texture = np.abs(self.luma - local_mean)
        texture = self.blur3(texture)
        texture = texture / (np.max(texture) + 1e-6)

        # IMPORTANTISSIMO:
        # qui la texture pesa poco, non deve dominare
        field_mask = static_mask * (0.55 + texture * 0.45)

        # morbidezza
        field_mask = self.blur3(field_mask)
        field_mask = np.clip(field_mask, 0.0, 1.0)

        return field_mask

    # =========================================
    # DISPLACEMENT GROSSO SU MASSE STATICHE
    # =========================================
    def static_field_displacement(self, img, amount):
        if amount < 0.02:
            return img.copy()

        static_mask = self.get_static_field_mask()
        edge_mask = self.get_edge_focus_mask()
        mask = np.clip(static_mask * 0.7 + edge_mask * 0.9, 0.0, 1.0)
        out = img.copy()

        amp = 1 + int(amount * 8)

        dx1 = int(np.sin(self.time * 1.7) * amp)
        dy1 = int(np.cos(self.time * 1.3) * amp)

        dx2 = int(np.sin(self.time * 2.4 + 1.2) * amp)
        dy2 = int(np.cos(self.time * 2.1 + 0.7) * amp)

        shifted1 = np.roll(img, shift=(dy1, dx1), axis=(0,1))
        shifted2 = np.roll(img, shift=(dy2, dx2), axis=(0,1))

        # mix dei due spostamenti
        mixed = ((shifted1.astype(np.float32) * 0.5) + (shifted2.astype(np.float32) * 0.5)).astype(np.uint8)

        for c in range(3):
            out[:,:,c] = (
                img[:,:,c] * (1.0 - mask) +
                mixed[:,:,c] * mask
            ).astype(np.uint8)

        return out

    # =========================================
    # RGB SPLIT MOLTO PIU' VISIBILE
    # =========================================
    def static_field_rgb_split(self, img, amount):
        if amount < 0.02:
            return img.copy()

        static_mask = self.get_static_field_mask()
        edge_mask = self.get_edge_focus_mask()
        mask = np.clip(static_mask * 0.45 + edge_mask * 1.15, 0.0, 1.0)
        out = img.copy()

        shift_r_x = int(1 + amount * 6)
        shift_g_y = int(np.sin(self.time * 2.1) * (1 + amount * 5))
        shift_b_x = int(-(1 + amount * 6))

        r = np.roll(img[:,:,0], shift=(0, shift_r_x), axis=(0,1))
        g = np.roll(img[:,:,1], shift=(shift_g_y, 0), axis=(0,1))
        b = np.roll(img[:,:,2], shift=(0, shift_b_x), axis=(0,1))

        out[:,:,0] = (img[:,:,0] * (1.0 - mask) + r * mask).astype(np.uint8)
        out[:,:,1] = (img[:,:,1] * (1.0 - mask) + g * mask).astype(np.uint8)
        out[:,:,2] = (img[:,:,2] * (1.0 - mask) + b * mask).astype(np.uint8)

        return out

    def edge_contour_warp(self, img, amount):
        if amount < 0.02:
            return img.copy()

        edge_mask = self.get_edge_focus_mask()
        if np.max(edge_mask) < 1e-4:
            return img.copy()

        h, w = edge_mask.shape
        y, x = np.indices((h, w), dtype=np.float32)

        phase_x = y * 0.115 + self.time * 3.2
        phase_y = x * 0.095 - self.time * 2.7

        disp_x = np.sin(phase_x) * (0.8 + amount * 3.8)
        disp_y = np.cos(phase_y) * (0.6 + amount * 2.6)

        warp = np.power(edge_mask, 1.35) * np.clip(amount * 1.8, 0.0, 1.0)
        sample_x = np.clip(x + disp_x * warp, 0, w - 1).astype(np.int32)
        sample_y = np.clip(y + disp_y * warp, 0, h - 1).astype(np.int32)

        warped = img[sample_y, sample_x]
        mix = np.expand_dims(np.clip(warp, 0.0, 0.85), axis=2)
        out = img.astype(np.float32) * (1.0 - mix) + warped.astype(np.float32) * mix
        return np.clip(out, 0, 255).astype(np.uint8)

    def static_field_drift(self, img, amount):
        if amount < 0.02:
            return img.copy()

        static_mask = self.get_static_field_mask()
        edge_mask = self.get_edge_focus_mask()
        mask = np.clip(static_mask * (1.0 - edge_mask * 0.9), 0.0, 1.0)
        if np.max(mask) < 1e-4:
            return img.copy()

        dx = int(np.sin(self.time * 1.05) * (1 + amount * 5))
        dy = int(np.cos(self.time * 0.87) * (1 + amount * 4))
        shifted = np.roll(img, shift=(dy, dx), axis=(0, 1))

        mix = np.expand_dims(np.clip(mask * (0.18 + amount * 0.28), 0.0, 0.42), axis=2)
        out = img.astype(np.float32) * (1.0 - mix) + shifted.astype(np.float32) * mix
        return np.clip(out, 0, 255).astype(np.uint8)

    def edge_noise_overlay(self, img, amount):
        if amount < 0.015:
            return img.copy()

        edge_mask = self.get_edge_focus_mask()
        if np.max(edge_mask) < 1e-4:
            return img.copy()

        h, w = edge_mask.shape
        y, x = np.indices((h, w), dtype=np.float32)

        phase_a = x * 0.33 + y * 0.21 + self.time * 11.0
        phase_b = x * -0.17 + y * 0.29 - self.time * 8.0
        wave = np.sin(phase_a) + np.cos(phase_b)
        wave = wave / 2.0

        gate = np.clip(edge_mask * (0.4 + amount * 2.6), 0.0, 1.0)
        signed = wave * gate * (18.0 + amount * 90.0)

        out = img.astype(np.float32)
        out[:, :, 0] += signed * 1.25
        out[:, :, 1] -= signed * 0.35
        out[:, :, 2] += signed * 0.85

        return np.clip(out, 0, 255).astype(np.uint8)

    def edge_glow(self, img, amount):
        if amount < 0.015:
            return img.copy()

        edge_mask = np.expand_dims(self.get_edge_focus_mask(), axis=2)
        glow = edge_mask * (20.0 + amount * 110.0)

        out = img.astype(np.float32)
        out[:, :, 0] += glow[:, :, 0] * 1.1
        out[:, :, 1] += glow[:, :, 0] * 0.18
        out[:, :, 2] += glow[:, :, 0] * 0.32

        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================================
    # COLOR SHIFT SULLE SUPERFICI STATICHE
    # =========================================
    def static_field_color_push(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.astype(np.float32)

        pulse = (0.5 + 0.5 * np.sin(self.time * 3.0)) * amount

        out[:,:,0] += mask * pulse * 180.0
        out[:,:,1] += mask * pulse * 40.0
        out[:,:,2] += mask * pulse * 220.0

        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================================
    # GLUE: PRESERVA LE PARTI IN MOVIMENTO
    # =========================================
    def preserve_moving_areas(self, img):
        moving_mask = np.expand_dims(np.clip(self.motion_map * 3.0, 0.0, 1.0), axis=2)

        out = (
            img.astype(np.float32) * (1.0 - moving_mask * 0.75) +
            self.base_img.astype(np.float32) * (moving_mask * 0.75)
        )

        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================================
    # RITORNO ALLA QUIETE IN SILENZIO
    # =========================================
    def preserve_stillness(self, img, activity):
        if activity > 0.06:
            return img.copy()

        alpha = np.clip((0.06 - activity) / 0.06, 0.0, 1.0) * 0.75
        out = img.astype(np.float32) * (1.0 - alpha) + self.base_img.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================================
    # UPDATE
    # =========================================
    def update(self, features):
        self.time += 0.06

        rms = features["rms"]
        low = features["low"]
        mid = features["mid"]
        high = features["high"]
        transient = features.get("transient", 0.0)
        activity = features.get("activity", rms)

        bass_mid_peak = np.clip(
            max(low, mid) * 0.75 +
            min(1.0, low + mid) * 0.35 +
            rms * 0.45 +
            transient * 0.70,
            0.0,
            1.0,
        )
        peak_drive = np.power(bass_mid_peak, 1.85)

        img = self.base_img.copy()
        
        # MASSA STATICA: displacement vero
        img = self.static_field_displacement(
            img,
            amount=low * 0.38 + mid * 0.32 + peak_drive * 0.14,
        )

        # MASSA STATICA: separazione colore
        img = self.static_field_rgb_split(
            img,
            amount=high * 110 + transient * 160 + peak_drive * 0.06,
        )

        # EDGE: deformazione morbida dei contorni, meno random e più leggibile
        img = self.edge_contour_warp(img, amount=peak_drive * 0.18 + low * 0.08 + mid * 0.05)

        # STATIC FIELD: drift lento sulle masse statiche, ma non sui bordi
        img = self.static_field_drift(img, amount=low * 0.10 + mid * 0.08 + peak_drive * 0.06)

        # EDGE: noise localizzato e meno costante
        img = self.edge_noise_overlay(
            img,
            amount=high * 0.05 + transient * 0.10 + rms * 0.08 + peak_drive * 0.18,
        )

        # EDGE: leggero glow sui contorni, così il soggetto resta leggibile
        img = self.edge_glow(img, amount=mid * 0.12 + high * 0.10 + transient * 0.08 + peak_drive * 0.10)

        # MASSA STATICA: push cromatico
        img = self.static_field_color_push(img, amount=high * 0.04 + mid * 0.05 + peak_drive * 0.04)

        img = self.apply_red_grade(img, strength=1.0)

        # preserva movimento
        img = self.preserve_moving_areas(img)

        # se silenzio, torna leggibile
        img = self.preserve_stillness(img, activity)

        return img
