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
        self.prev_output = self.base_img.copy()
        self.time = 0.0

    def apply_red_grade(self, img, strength=1.0):
        out = img.astype(np.float32)

        luma = self.compute_luma(img) / 255.0
        luma = np.expand_dims(luma, axis=2)

        red_boost = 1.35 + 0.25 * strength
        green_scale = 0.35
        blue_scale = 0.28

        out[:, :, 0] *= red_boost
        out[:, :, 1] *= green_scale
        out[:, :, 2] *= blue_scale

        out[:, :, 0] += luma[:, :, 0] * 45.0
        out[:, :, 1] += luma[:, :, 0] * 8.0
        out[:, :, 2] += luma[:, :, 0] * 4.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def load_base_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def compute_luma(self, img):
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)

    def blur3(self, arr):
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
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

    def static_field_displacement(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.copy()

        amp = 1 + int(amount * 8)

        dx1 = int(np.sin(self.time * 1.7) * amp)
        dy1 = int(np.cos(self.time * 1.3) * amp)

        dx2 = int(np.sin(self.time * 2.4 + 1.2) * amp)
        dy2 = int(np.cos(self.time * 2.1 + 0.7) * amp)

        shifted1 = np.roll(img, shift=(dy1, dx1), axis=(0, 1))
        shifted2 = np.roll(img, shift=(dy2, dx2), axis=(0, 1))

        mixed = ((shifted1.astype(np.float32) * 0.5) + (shifted2.astype(np.float32) * 0.5)).astype(np.uint8)

        for c in range(3):
            out[:, :, c] = (
                img[:, :, c] * (1.0 - mask) +
                mixed[:, :, c] * mask
            ).astype(np.uint8)

        return out

    def static_field_rgb_split(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.copy()

        shift_r_x = int(1 + amount * 6)
        shift_g_y = int(np.sin(self.time * 2.1) * (1 + amount * 5))
        shift_b_x = int(-(1 + amount * 6))

        r = np.roll(img[:, :, 0], shift=(0, shift_r_x), axis=(0, 1))
        g = np.roll(img[:, :, 1], shift=(shift_g_y, 0), axis=(0, 1))
        b = np.roll(img[:, :, 2], shift=(0, shift_b_x), axis=(0, 1))

        out[:, :, 0] = (img[:, :, 0] * (1.0 - mask) + r * mask).astype(np.uint8)
        out[:, :, 1] = (img[:, :, 1] * (1.0 - mask) + g * mask).astype(np.uint8)
        out[:, :, 2] = (img[:, :, 2] * (1.0 - mask) + b * mask).astype(np.uint8)

        return out

    def static_field_color_push(self, img, amount):
        if amount < 0.02:
            return img.copy()

        mask = self.get_static_field_mask()
        out = img.astype(np.float32)

        pulse = (0.5 + 0.5 * np.sin(self.time * 3.0)) * amount

        out[:, :, 0] += mask * pulse * 180.0
        out[:, :, 1] += mask * pulse * 40.0
        out[:, :, 2] += mask * pulse * 220.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def preserve_moving_areas(self, img):
        moving_mask = np.expand_dims(np.clip(self.motion_map * 3.0, 0.0, 1.0), axis=2)

        out = (
            img.astype(np.float32) * (1.0 - moving_mask * 0.75) +
            self.base_img.astype(np.float32) * (moving_mask * 0.75)
        )

        return np.clip(out, 0, 255).astype(np.uint8)

    def preserve_stillness(self, img, rms):
        if rms > 0.05:
            return img.copy()

        alpha = np.clip((0.05 - rms) / 0.05, 0.0, 1.0) * 0.75
        out = img.astype(np.float32) * (1.0 - alpha) + self.base_img.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def datamosh_delta_repeat(self, img, amount):
        if amount < 0.10:
            return img.copy()

        prev = self.prev_output.astype(np.float32)
        current = img.astype(np.float32)

        static_mask = self.get_static_field_mask()
        edge_mask = self.edge_map
        persistence = np.clip(static_mask * 0.65 + edge_mask * 0.35, 0.0, 1.0)

        block = 2 if amount > 0.72 else 4
        h, w = persistence.shape
        coarse_h = max(1, h // block)
        coarse_w = max(1, w // block)

        coarse_mask = persistence.reshape(coarse_h, block, coarse_w, block).mean(axis=(1, 3))

        y, x = np.indices((coarse_h, coarse_w), dtype=np.float32)
        phase_a = x * 0.73 + y * 0.41 + self.time * 6.0
        phase_b = x * -0.52 + y * 0.67 - self.time * 4.5

        shift_x = np.rint((np.sin(phase_a) + np.cos(phase_b)) * (0.8 + amount * 4.0)).astype(np.int32)
        shift_y = np.rint((np.cos(phase_a * 0.8) - np.sin(phase_b * 1.1)) * (0.6 + amount * 3.0)).astype(np.int32)

        moshed = current.copy()
        hold_strength = np.clip(0.18 + amount * 0.72, 0.0, 0.95)

        for by in range(coarse_h):
            y0 = by * block
            y1 = min(h, y0 + block)
            for bx in range(coarse_w):
                x0 = bx * block
                x1 = min(w, x0 + block)

                local_mask = coarse_mask[by, bx]
                if local_mask < 0.08:
                    continue

                src_y0 = int(np.clip(y0 + shift_y[by, bx], 0, h - (y1 - y0)))
                src_x0 = int(np.clip(x0 + shift_x[by, bx], 0, w - (x1 - x0)))
                src_y1 = src_y0 + (y1 - y0)
                src_x1 = src_x0 + (x1 - x0)

                repeated = prev[src_y0:src_y1, src_x0:src_x1]
                strength = np.clip(local_mask * hold_strength, 0.0, 1.0)
                moshed[y0:y1, x0:x1] = (
                    current[y0:y1, x0:x1] * (1.0 - strength) +
                    repeated * strength
                )

        return np.clip(moshed, 0, 255).astype(np.uint8)

    def datamosh_pixel_sort_decay(self, img, amount):
        if amount < 0.14:
            return img.copy()

        out = img.astype(np.float32)
        prev = self.prev_output.astype(np.float32)
        edge_mask = np.clip(self.edge_map, 0.0, 1.0)
        static_mask = self.get_static_field_mask()
        collapse = np.clip(edge_mask * 0.55 + static_mask * 0.60, 0.0, 1.0)

        h, w = collapse.shape
        row_step = 1 if amount > 0.7 else 2
        run = max(3, min(10, int(3 + amount * 7)))
        blend = np.clip(0.15 + amount * 0.55, 0.0, 0.92)

        for y in range(0, h, row_step):
            row_mask = collapse[y]
            active = np.where(row_mask > 0.22)[0]
            if len(active) < run:
                continue

            start = active[0]
            end = active[-1]
            for x0 in range(start, end - run + 1, run):
                x1 = min(w, x0 + run)
                local = float(np.mean(row_mask[x0:x1]))
                if local < 0.22:
                    continue

                source = prev[y, x0:x1]
                target = out[y, x0:x1]
                sort_idx = np.argsort(np.sum(source, axis=1))
                reordered = source[sort_idx]
                strength = np.clip(local * blend, 0.0, 1.0)
                out[y, x0:x1] = target * (1.0 - strength) + reordered * strength

        return np.clip(out, 0, 255).astype(np.uint8)

    def update(self, features):
        self.time += 0.06

        rms = features["rms"]
        low = features["low"]
        mid = features["mid"]
        high = features["high"]
        transient = features.get("transient", 0.0)
        mosh_drive = np.clip(
            rms * 0.40 +
            low * 0.38 +
            mid * 0.34 +
            transient * 0.12,
            0.0,
            1.0,
        )
        mosh_drive = np.clip((mosh_drive - 0.58) / 0.16, 0.0, 1.0)

        img = self.base_img.copy()

        img = self.static_field_displacement(img, amount=low * 1.1 + mid * 0.8)
        img = self.static_field_rgb_split(img, amount=high * 1000 + transient * 1000)
        img = self.static_field_color_push(img, amount=high * 0.1 + mid * 0.1)
        img = self.datamosh_delta_repeat(img, amount=mosh_drive)
        img = self.datamosh_pixel_sort_decay(img, amount=mosh_drive)
        img = self.apply_red_grade(img, strength=1.0)
        img = self.preserve_moving_areas(img)
        img = self.preserve_stillness(img, rms)
        self.prev_output = img.copy()

        return img
