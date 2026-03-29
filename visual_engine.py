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

    def load_base_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def compute_luma(self, img):
        return (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.float32)

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

        # meno blur = edge più secchi
        mag = self.blur3(mag)
        mag = mag / (np.max(mag) + 1e-6)

        # edge molto più contrastati
        mag = np.clip((mag - 0.08) * 4.5, 0.0, 1.0)
        return mag

    def compute_motion_map(self, current_luma):
        diff = np.abs(current_luma - self.prev_luma)
        diff = self.blur3(diff)
        diff = diff / (np.max(diff) + 1e-6)
        self.prev_luma = current_luma.copy()
        return diff

    def contour_displacement_static_only(self, img, amount):
        if amount < 0.02:
            return img.copy()

        # displacement MOLTO più visibile su 64x64
        dx = int(np.sin(self.time * 2.2) * (2 + amount * 10000))
        dy = int(np.cos(self.time * 1.7) * (2 + amount * 10000))

        shifted = np.roll(img, shift=(dy, dx), axis=(0,1))

        edge_mask = self.edge_map > 0.18
        static_mask = self.motion_map < 0.18
        final_mask = edge_mask & static_mask

        out = img.copy()
        for c in range(3):
            channel = out[:,:,c]
            channel[final_mask] = shifted[:,:,c][final_mask]
            out[:,:,c] = channel

        return out

    def edge_rgb_glitch_static_only(self, img, amount):
        if amount < 0.02:
            return img.copy()

        out = img.copy()

        edge_mask = self.edge_map > 0.15
        static_mask = self.motion_map < 0.18
        final_mask = edge_mask & static_mask

        # shift aggressivi per 64x64
        shift_r_x = int(1 + amount * 4)
        shift_b_x = int(-(1 + amount * 4))
        shift_g_y = int(np.sin(self.time * 2.8) * (1 + amount * 3))

        r = np.roll(img[:,:,0], shift=(0, shift_r_x), axis=(0,1))
        g = np.roll(img[:,:,1], shift=(shift_g_y, 0), axis=(0,1))
        b = np.roll(img[:,:,2], shift=(0, shift_b_x), axis=(0,1))

        out[:,:,0][final_mask] = r[final_mask]
        out[:,:,1][final_mask] = g[final_mask]
        out[:,:,2][final_mask] = b[final_mask]

        return out

    def edge_color_burn_static_only(self, img, amount):
        if amount < 0.02:
            return img.copy()

        out = img.astype(np.float32)

        edge_mask = self.edge_map > 0.14
        static_mask = self.motion_map < 0.18
        final_mask = edge_mask & static_mask

        # colori edge più estremi
        edge_strength = self.edge_map * amount

        # palette glitch: magenta/cyan/acid green
        red_boost   = edge_strength * 180.0
        green_boost = edge_strength * 80.0
        blue_boost  = edge_strength * 200.0

        out[:,:,0][final_mask] += red_boost[final_mask]
        out[:,:,1][final_mask] += green_boost[final_mask]
        out[:,:,2][final_mask] += blue_boost[final_mask]

        return np.clip(out, 0, 255).astype(np.uint8)

    def edge_inversion_flash_static_only(self, img, amount):
        if amount < 0.03:
            return img.copy()

        edge_mask = self.edge_map > 0.22
        static_mask = self.motion_map < 0.18
        final_mask = edge_mask & static_mask

        out = img.copy()
        inv = 255 - out
        for c in range(3):
            out[:,:,c][final_mask] = (
                out[:,:,c][final_mask] * (1.0 - amount * 0.6) +
                inv[:,:,c][final_mask] * (amount * 0.6)
            ).astype(np.uint8)

        return out

    def preserve_moving_areas(self, img):
        # più movimento = meno glitch
        moving_mask = np.expand_dims(np.clip(self.motion_map * 2.5, 0.0, 1.0), axis=2)
        out = img.astype(np.float32) * (1.0 - moving_mask * 0.45) + self.base_img.astype(np.float32) * (moving_mask * 0.45)
        return np.clip(out, 0, 255).astype(np.uint8)

    def preserve_stillness(self, img, rms):
        if rms > 0.05:
            return img.copy()

        alpha = np.clip((0.05 - rms) / 0.05, 0.0, 1.0) * 0.80
        out = img.astype(np.float32) * (1.0 - alpha) + self.base_img.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def update(self, features):
        self.time += 0.05

        rms = features["rms"]
        low = features["low"]
        mid = features["mid"]
        high = features["high"]
        transient = features.get("transient", 0.0)

        img = self.base_img.copy()

        # 1) displacement molto visibile
        img = self.contour_displacement_static_only(img, amount=mid * 1.4 + low * 0.4)

        # 2) rgb split molto evidente
        img = self.edge_rgb_glitch_static_only(img, amount=high * 1.3 + mid * 0.5)

        # 3) color burn forte sugli edge
        img = self.edge_color_burn_static_only(img, amount=high * 1.1 + transient * 0.8)

        # 4) piccoli flash invertiti sui bordi statici
        img = self.edge_inversion_flash_static_only(img, amount=transient * 0.9)

        # 5) preserva il movimento reale
        img = self.preserve_moving_areas(img)

        # 6) in silenzio torna più leggibile
        img = self.preserve_stillness(img, rms)

        return img