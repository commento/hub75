# audio_features.py

import numpy as np
from config import SAMPLE_RATE, BLOCK_SIZE, SMOOTH_FAST, SMOOTH_SLOW

class StereoFeatureExtractor:
    def __init__(self):
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)

        self.state = {
            "rms": 0.0,
            "raw_rms": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "balance": 0.0,
            "width": 0.0,
            "transient": 0.0,
            "activity": 0.0,
        }

        self.prev_rms = 0.0
        self.noise_floor = 0.0
        self.band_floor = {
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
        }
        self.band_peak = {
            "low": 1e-4,
            "mid": 1e-4,
            "high": 1e-4,
        }

    def _band_energy(self, spectrum, low_hz, high_hz):
        idx = np.where((self.freqs >= low_hz) & (self.freqs < high_hz))[0]
        if len(idx) == 0:
            return 0.0
        return float(np.mean(spectrum[idx]))

    def _smooth(self, key, target, amount):
        self.state[key] = (1 - amount) * self.state[key] + amount * target

    def _normalize_band(self, key, value, floor_speed=0.012, peak_attack=0.28, peak_release=0.003):
        floor = self.band_floor[key]
        floor = floor * (1.0 - floor_speed) + value * floor_speed
        self.band_floor[key] = floor

        active = max(0.0, value - floor * 1.08)

        peak = self.band_peak[key]
        if active > peak:
            peak = peak * (1.0 - peak_attack) + active * peak_attack
        else:
            peak = peak * (1.0 - peak_release) + active * peak_release
        self.band_peak[key] = max(peak, 1e-4)

        normalized = active / (self.band_peak[key] + 1e-6)
        return float(np.clip(np.power(normalized, 0.85), 0.0, 1.0))

    def extract(self, stereo_frame):
        left = stereo_frame[:, 0]
        right = stereo_frame[:, 1]

        mono = (left + right) * 0.5
        side = (left - right) * 0.5

        rms_raw = float(np.sqrt(np.mean(mono ** 2) + 1e-9))

        self.noise_floor = self.noise_floor * 0.999 + rms_raw * 0.001
        rms_active = max(0.0, rms_raw - self.noise_floor * 1.04)
        raw_rms = np.clip(np.power(rms_raw * 22.0, 0.72), 0.0, 1.0)
        rms = np.clip(np.power(rms_active * 28.0, 0.72), 0.0, 1.0)

        window = np.hanning(len(mono))
        mono_spec = np.abs(np.fft.rfft(mono * window))
        side_spec = np.abs(np.fft.rfft(side * window))

        mono_spec = np.log1p(mono_spec)
        side_spec = np.log1p(side_spec)

        low_raw = self._band_energy(mono_spec, 35, 180)
        mid_raw = self._band_energy(mono_spec, 180, 2400)
        high_raw = self._band_energy(mono_spec, 2400, 12000)

        low = self._normalize_band("low", low_raw)
        mid = self._normalize_band("mid", mid_raw)
        high = self._normalize_band("high", high_raw)

        width = float(np.mean(side_spec) / (np.mean(mono_spec) + 1e-6))
        width = np.clip(width * 1.6, 0.0, 1.0)

        left_rms = float(np.sqrt(np.mean(left ** 2) + 1e-9))
        right_rms = float(np.sqrt(np.mean(right ** 2) + 1e-9))
        balance = np.clip((right_rms - left_rms) * 7.0, -1.0, 1.0)

        spectral_flux = (
            max(0.0, low - self.state["low"]) * 0.8 +
            max(0.0, mid - self.state["mid"]) * 1.0 +
            max(0.0, high - self.state["high"]) * 0.55
        )
        transient = max(0.0, rms - self.prev_rms) * 1.9 + spectral_flux * 1.35
        transient = np.clip(transient, 0.0, 1.0)
        self.prev_rms = rms

        activity = np.clip(
            raw_rms * 0.42 +
            rms * 0.28 +
            low * 0.16 +
            mid * 0.14 +
            transient * 0.26,
            0.0,
            1.0,
        )

        gate = np.clip((raw_rms - 0.010) * 8.0, 0.0, 1.0)
        low *= gate
        mid *= gate
        high *= gate
        width *= gate
        balance *= gate
        transient *= gate

        self._smooth("rms", rms, SMOOTH_FAST)
        self._smooth("raw_rms", raw_rms, 0.32)
        self._smooth("low", low, SMOOTH_FAST)
        self._smooth("mid", mid, SMOOTH_FAST)
        self._smooth("high", high, SMOOTH_FAST)
        self._smooth("balance", balance, SMOOTH_SLOW)
        self._smooth("width", width, SMOOTH_SLOW)
        self._smooth("transient", transient, 0.50)
        self._smooth("activity", activity, 0.28)

        for key in self.state:
            if key != "balance":
                self.state[key] = float(np.clip(self.state[key], 0.0, 1.0))

        return self.state.copy()
