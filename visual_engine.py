# visual_engine.py

import numpy as np
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
        gx = np.zeros_like(luma)
        gy = np.zeros_like(luma)
        gx[:,1:-1] = luma[:,2:] - luma[:,:-2]
        gy[1:-1,:] = luma[2:,:] - luma[:-2,:]
        mag = np.sqrt(gx*gx + gy*gy)
        mag = self.blur3(mag)
        mag = mag / (np.max(mag) + 1e-6)
        return np.clip(mag, 0.0, 1.0)

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

    def edge_burst_displace(self, img, amount):
        if amount < 0.02:
            return img.copy()

        edge_mask = self.get_edge_focus_mask()
        burst = np.clip(np.power(edge_mask, 0.7) * (amount * 1.4), 0.0, 1.0)
        if np.max(burst) < 1e-4:
            return img.copy()

        x_shift = int(np.sin(self.time * 7.7) * (2 + amount * 14))
        y_shift = int(np.cos(self.time * 5.9) * (1 + amount * 10))
        shifted = np.roll(img, shift=(y_shift, x_shift), axis=(0, 1))

        out = img.astype(np.float32)
        burst = np.expand_dims(burst, axis=2)
        out = out * (1.0 - burst) + shifted.astype(np.float32) * burst
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
    def preserve_stillness(self, img, rms):
        if rms > 0.05:
            return img.copy()

        alpha = np.clip((0.05 - rms) / 0.05, 0.0, 1.0) * 0.75
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

        # EDGE: burst più raro e più pulito, guidato davvero dai picchi
        img = self.edge_burst_displace(img, amount=peak_drive * 0.18 + low * 0.06)

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
        img = self.preserve_stillness(img, rms)

        return img
