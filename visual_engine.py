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
        self.feedback = self.base_img.copy()
        self.time = 0.0

    def load_base_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def compute_luma(self, img):
        return (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.float32)

    def blur3(self, arr):
        padded = np.pad(arr, ((1,1),(1,1)), mode='edge')
        out = (padded[:-2,:-2]+padded[:-2,1:-1]+padded[:-2,2:] +
               padded[1:-1,:-2]+padded[1:-1,1:-1]+padded[1:-1,2:] +
               padded[2:,:-2]+padded[2:,1:-1]+padded[2:,2:]) / 9.0
        return out

    def compute_edge_map(self, luma):
        gx = np.zeros_like(luma)
        gy = np.zeros_like(luma)
        gx[:,1:-1] = luma[:,2:] - luma[:,:-2]
        gy[1:-1,:] = luma[2:,:] - luma[:-2,:]
        mag = np.sqrt(gx*gx + gy*gy)
        mag = self.blur3(mag)
        mag = mag / (np.max(mag)+1e-6)
        return mag

    def zoom_image(self, img, factor):
        if factor <= 1.001:
            return img.copy()
        h,w = img.shape[:2]
        crop_w = max(1, int(w/factor))
        crop_h = max(1, int(h/factor))
        x0 = (w - crop_w)//2
        y0 = (h - crop_h)//2
        cropped = img[y0:y0+crop_h, x0:x0+crop_w]
        return np.array(Image.fromarray(cropped).resize((w,h), Image.Resampling.NEAREST), dtype=np.uint8)

    def contour_drift(self, img, amount):
        if amount < 0.000001:
            return img.copy()
        dx = int(np.sin(self.time*5)*amount*10000)
        dy = int(np.cos(self.time*8)*amount*10000)
        shifted = np.roll(img, shift=(dy, dx), axis=(0,1))
        mask = self.edge_map > 0.1
        out = img.copy()
        
        for c in range(3):
            channel = out[:,:,c]
            channel[mask] = shifted[:,:,c][mask]
            out[:,:,c] = channel
        return out

    def edge_glow(self, img, amount):
        if amount < 12:
            return img.copy()
        out = img.astype(np.float32)
        glow = np.expand_dims(self.edge_map * amount * 2000.0, axis=2)  # prima era 120
        out += glow
        return np.clip(out,0,255).astype(np.uint8)

    def mass_breathing(self, img, amount):
        if amount < 0.1:
            return img.copy()
        pulse = 1.0 + amount*0.05
        zoomed = self.zoom_image(img, pulse)
        mask = np.expand_dims(np.clip(self.luma/255.0,0.0,1.0),2)
        out = img.astype(np.float32)*(1-mask*0.5*amount) + zoomed.astype(np.float32)*(mask*0.5*amount)
        return np.clip(out,0,255).astype(np.uint8)

    def preserve_stillness(self, img, rms):
        if rms > 0.05:
            return img
        alpha = np.clip((0.05 - rms)/0.05,0,1)*0.9
        out = img.astype(np.float32)*(1-alpha) + self.base_img.astype(np.float32)*alpha
        return np.clip(out,0,255).astype(np.uint8)

    def update(self, features):
        self.time += 0.04
        rms = features["rms"]
        low = features["low"]
        mid = features["mid"]
        high = features["high"]

        img = self.base_img.copy()
        # LOW → pulsazione dei bordi
        img = self.mass_breathing(img, low*0.6)
        # MID → drift contorni
        img = self.contour_drift(img, mid)
        # HIGH → glow contorni
        img = self.edge_glow(img, high*1.2)
        # preserva immagine stabile quando audio basso
        img = self.preserve_stillness(img, rms)

        return img