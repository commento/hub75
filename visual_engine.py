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

        # memoria visiva
        self.persistence_buffer = self.base_img.copy().astype(np.float32)
        self.previous_output = self.base_img.copy().astype(np.float32)

        # piccolo noise field persistente
        self.noise_field = np.random.rand(self.height, self.width).astype(np.float32)

    # =========================================================
    # IO
    # =========================================================
    def load_base_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    # =========================================================
    # CORE MAPS
    # =========================================================
    def compute_luma(self, img):
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)

    def blur3(self, arr):
        padded = np.pad(arr, ((1, 1), (1, 1)), mode='edge')
        out = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
        return out

    def compute_edge_map(self, luma):
        gx = np.zeros_like(luma)
        gy = np.zeros_like(luma)

        gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
        gy[1:-1, :] = luma[2:, :] - luma[:-2, :]

        mag = np.sqrt(gx * gx + gy * gy)
        mag = self.blur3(mag)
        mag = mag / (np.max(mag) + 1e-6)
        return np.clip(mag, 0.0, 1.0)

    def compute_motion_map(self, current_luma):
        diff = np.abs(current_luma - self.prev_luma)
        diff = self.blur3(diff)
        diff = diff / (np.max(diff) + 1e-6)
        self.prev_luma = current_luma.copy()
        return diff

    # =========================================================
    # SMALL UTILS
    # =========================================================
    def blend(self, a, b, mix):
        mix = np.clip(mix, 0.0, 1.0)
        return (a.astype(np.float32) * (1.0 - mix) + b.astype(np.float32) * mix)

    def clip_img(self, img):
        return np.clip(img, 0, 255).astype(np.uint8)

    # =========================================================
    # STATIC FIELD
    # =========================================================
    def get_static_field_mask(self):
        static_mask = 1.0 - np.clip(self.motion_map * 2.8, 0.0, 1.0)

        local_mean = self.blur3(self.luma)
        texture = np.abs(self.luma - local_mean)
        texture = self.blur3(texture)
        texture = texture / (np.max(texture) + 1e-6)

        field_mask = static_mask * (0.55 + texture * 0.45)
        field_mask = self.blur3(field_mask)
        field_mask = np.clip(field_mask, 0.0, 1.0)
        return field_mask

    # =========================================================
    # STATIC MASS DISPLACEMENT
    # =========================================================
    def static_field_displacement(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.copy()

        amp = 1 + int(amount * 10)

        dx1 = int(np.sin(self.time * 1.7) * amp)
        dy1 = int(np.cos(self.time * 1.3) * amp)

        dx2 = int(np.sin(self.time * 2.4 + 1.2) * amp)
        dy2 = int(np.cos(self.time * 2.1 + 0.7) * amp)

        shifted1 = np.roll(img, shift=(dy1, dx1), axis=(0, 1))
        shifted2 = np.roll(img, shift=(dy2, dx2), axis=(0, 1))
        mixed = ((shifted1.astype(np.float32) * 0.5) + (shifted2.astype(np.float32) * 0.5))

        mask3 = np.expand_dims(mask, axis=2)
        out = img.astype(np.float32) * (1.0 - mask3) + mixed * mask3

        return self.clip_img(out)

    # =========================================================
    # RGB SPLIT
    # =========================================================
    def static_field_rgb_split(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        mask3 = np.expand_dims(mask, axis=2)

        shift_r_x = int(1 + amount * 6)
        shift_g_y = int(np.sin(self.time * 2.1) * (1 + amount * 5))
        shift_b_x = int(-(1 + amount * 6))

        r = np.roll(img[:, :, 0], shift=(0, shift_r_x), axis=(0, 1))
        g = np.roll(img[:, :, 1], shift=(shift_g_y, 0), axis=(0, 1))
        b = np.roll(img[:, :, 2], shift=(0, shift_b_x), axis=(0, 1))

        split = np.stack([r, g, b], axis=2).astype(np.float32)
        out = img.astype(np.float32) * (1.0 - mask3) + split * mask3

        return self.clip_img(out)

    # =========================================================
    # COLOR PUSH
    # =========================================================
    def static_field_color_push(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.astype(np.float32)

        pulse = (0.5 + 0.5 * np.sin(self.time * 3.0)) * amount

        out[:, :, 0] += mask * pulse * 180.0
        out[:, :, 1] += mask * pulse * 35.0
        out[:, :, 2] += mask * pulse * 220.0

        return self.clip_img(out)

    # =========================================================
    # EDGE BLOOM
    # =========================================================
    def apply_edge_bloom(self, img, amount):
        if amount < 0.02:
            return img.copy()

        edge = self.edge_map
        edge = self.blur3(edge)
        edge = self.blur3(edge)

        out = img.astype(np.float32)
        out[:, :, 0] += edge * amount * 140.0
        out[:, :, 1] += edge * amount * 20.0
        out[:, :, 2] += edge * amount * 70.0

        return self.clip_img(out)

    # =========================================================
    # SCAN TEARING
    # =========================================================
    def apply_scan_tearing(self, img, amount):
        if amount < 0.03:
            return img.copy()

        out = img.copy()
        h, w, _ = out.shape

        num_bands = int(1 + amount * 8)

        for _ in range(num_bands):
            y = np.random.randint(0, h - 2)
            band_h = np.random.randint(1, 4)
            shift = np.random.randint(-int(2 + amount * 12), int(2 + amount * 12) + 1)
            out[y:y + band_h] = np.roll(out[y:y + band_h], shift, axis=1)

        return out

    # =========================================================
    # POSTERIZE
    # =========================================================
    def posterize(self, img, levels=6):
        if levels < 2:
            return img.copy()

        out = img.astype(np.float32)
        out = np.floor(out / 255.0 * levels) / levels * 255.0
        return self.clip_img(out)

    # =========================================================
    # NOISE FIELD
    # =========================================================
    def evolve_noise_field(self):
        fresh = np.random.rand(self.height, self.width).astype(np.float32)
        self.noise_field = self.noise_field * 0.92 + fresh * 0.08
        self.noise_field = self.blur3(self.noise_field)
        self.noise_field = self.noise_field / (np.max(self.noise_field) + 1e-6)

    def apply_noise_field(self, img, amount):
        if amount < 0.02:
            return img.copy()

        self.evolve_noise_field()

        out = img.astype(np.float32)
        n = self.noise_field

        out[:, :, 0] += (n - 0.5) * amount * 90.0
        out[:, :, 1] += (n - 0.5) * amount * 25.0
        out[:, :, 2] += (n - 0.5) * amount * 110.0

        return self.clip_img(out)

    # =========================================================
    # RED GRADE
    # =========================================================
    def apply_red_grade(self, img, strength=1.0):
        out = img.astype(np.float32)

        luma = self.compute_luma(img) / 255.0
        luma = np.expand_dims(luma, axis=2)

        red_boost = 1.28 + 0.22 * strength
        green_scale = 0.38
        blue_scale = 0.30

        out[:, :, 0] *= red_boost
        out[:, :, 1] *= green_scale
        out[:, :, 2] *= blue_scale

        out[:, :, 0] += luma[:, :, 0] * 45.0
        out[:, :, 1] += luma[:, :, 0] * 8.0
        out[:, :, 2] += luma[:, :, 0] * 5.0

        return self.clip_img(out)

    # =========================================================
    # PRESERVE MOVEMENT
    # =========================================================
    def preserve_moving_areas(self, img, preserve_amount=0.75):
        moving_mask = np.expand_dims(np.clip(self.motion_map * 3.0, 0.0, 1.0), axis=2)

        out = (
            img.astype(np.float32) * (1.0 - moving_mask * preserve_amount) +
            self.base_img.astype(np.float32) * (moving_mask * preserve_amount)
        )

        return self.clip_img(out)

    # =========================================================
    # SILENCE RETURN
    # =========================================================
    def preserve_stillness(self, img, silence_amount):
        if silence_amount < 0.05:
            return img.copy()

        alpha = np.clip(silence_amount, 0.0, 1.0) * 0.78
        out = img.astype(np.float32) * (1.0 - alpha) + self.base_img.astype(np.float32) * alpha
        return self.clip_img(out)

    # =========================================================
    # PERSISTENCE / GHOST MEMORY
    # =========================================================
    def apply_persistence(self, img, amount, motion_protect=0.65):
        if amount < 0.02:
            self.persistence_buffer = img.astype(np.float32)
            return img.copy()

        moving_mask = np.expand_dims(np.clip(self.motion_map * 2.5, 0.0, 1.0), axis=2)
        static_mask = 1.0 - moving_mask

        persistence_mix = static_mask * amount + moving_mask * (amount * (1.0 - motion_protect))

        self.persistence_buffer = (
            self.persistence_buffer * persistence_mix +
            img.astype(np.float32) * (1.0 - persistence_mix)
        )

        out = self.persistence_buffer.copy()
        return self.clip_img(out)

    # =========================================================
    # UPDATE
    # =========================================================
    def update(self, features):
        self.time += 0.06

        rms = features["rms"]
        low = features["low"]
        mid = features["mid"]
        high = features["high"]
        transient = features.get("transient", 0.0)

        energy = features.get("energy", rms)
        chaos = features.get("chaos", transient)
        density = features.get("density", mid)
        silence = features.get("silence", 0.0)

        mode_flow = features.get("mode_flow", 0.0)
        mode_pulse = features.get("mode_pulse", 0.0)
        mode_chaos = features.get("mode_chaos", 0.0)

        img = self.base_img.copy()

        # =====================================================
        # STRUCTURAL FX
        # =====================================================
        displacement_amount = low * 1.0 + energy * 0.6 + mode_chaos * 0.5
        split_amount = high * 0.8 + transient * 1.0 + chaos * 0.8
        color_push_amount = density * 0.25 + high * 0.15 + mode_pulse * 0.18

        img = self.static_field_displacement(img, amount=displacement_amount)
        img = self.static_field_rgb_split(img, amount=split_amount)
        img = self.static_field_color_push(img, amount=color_push_amount)

        # =====================================================
        # EDGE / STRUCTURE EMPHASIS
        # =====================================================
        edge_amount = density * 0.45 + transient * 0.20 + mode_flow * 0.10
        img = self.apply_edge_bloom(img, amount=edge_amount)

        # =====================================================
        # CHAOS LAYER
        # =====================================================
        tear_amount = chaos * 0.75 + transient * 0.25 + mode_chaos * 0.25
        noise_amount = chaos * 0.28 + density * 0.12 + (1.0 - silence) * 0.04

        img = self.apply_scan_tearing(img, amount=tear_amount)
        img = self.apply_noise_field(img, amount=noise_amount)

        # =====================================================
        # TONALITY
        # =====================================================
        grade_strength = 0.85 + energy * 0.35 + chaos * 0.15
        img = self.apply_red_grade(img, strength=grade_strength)

        # =====================================================
        # POSTERIZATION IN SILENCE
        # =====================================================
        if silence > 0.35:
            levels = int(np.clip(8 - silence * 4.0, 3, 8))
            img = self.posterize(img, levels=levels)

        # =====================================================
        # PRESERVE REAL MOTION
        # =====================================================
        preserve_amount = 0.62 + (1.0 - chaos) * 0.18
        img = self.preserve_moving_areas(img, preserve_amount=preserve_amount)

        # =====================================================
        # PERSISTENCE MEMORY
        # =====================================================
        persistence_amount = (
            0.10 +
            (1.0 - energy) * 0.35 +
            silence * 0.38 +
            mode_flow * 0.08
        )
        img = self.apply_persistence(img, amount=np.clip(persistence_amount, 0.0, 0.92))

        # =====================================================
        # RETURN TO STILLNESS
        # =====================================================
        img = self.preserve_stillness(img, silence_amount=silence)

        self.previous_output = img.astype(np.float32)
        return img