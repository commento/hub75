# audio_features.py

import numpy as np
from config import SAMPLE_RATE, BLOCK_SIZE, SMOOTH_FAST, SMOOTH_SLOW

class StereoFeatureExtractor:
    def __init__(self):
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)

        self.state = {
            "rms": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "balance": 0.0,
            "width": 0.0,
            "transient": 0.0,
        }

        self.prev_rms = 0.0
        self.noise_floor = 0.0

    def _band_energy(self, spectrum, low_hz, high_hz):
        idx = np.where((self.freqs >= low_hz) & (self.freqs < high_hz))[0]
        if len(idx) == 0:
            return 0.0
        return float(np.mean(spectrum[idx]))

    def _smooth(self, key, target, amount):
        self.state[key] = (1 - amount) * self.state[key] + amount * target

    def extract(self, stereo_frame):
        left = stereo_frame[:, 0]
        right = stereo_frame[:, 1]

        mono = (left + right) * 0.5
        side = (left - right) * 0.5

        rms_raw = float(np.sqrt(np.mean(mono ** 2) + 1e-9))

        # adaptive noise floor
        #self.noise_floor = 0.995 * self.noise_floor + 0.005 * rms_raw
        rms = rms_raw #max(0.0, rms_raw - self.noise_floor * 1.15)
        rms = np.clip(rms * 12.0, 0.0, 1.0)

        window = np.hanning(len(mono))
        mono_spec = np.abs(np.fft.rfft(mono * window))
        side_spec = np.abs(np.fft.rfft(side * window))

        mono_spec = np.log1p(mono_spec)
        side_spec = np.log1p(side_spec)

        low = self._band_energy(mono_spec, 55, 180)
        mid = self._band_energy(mono_spec, 180, 2200)
        high = self._band_energy(mono_spec, 2200, 10000)

        width = float(np.mean(side_spec) / (np.mean(mono_spec) + 1e-6))
        width = np.clip(width * 1.6, 0.0, 1.0)

        left_rms = float(np.sqrt(np.mean(left ** 2) + 1e-9))
        right_rms = float(np.sqrt(np.mean(right ** 2) + 1e-9))
        balance = np.clip((right_rms - left_rms) * 7.0, -1.0, 1.0)

        transient = max(0.0, rms - self.prev_rms) * 2.8
        transient = np.clip(transient, 0.0, 1.0)
        self.prev_rms = rms

        norm = max(low, mid, high, 1e-6)
        low /= norm
        mid /= norm
        high /= norm

        gate = np.clip((rms - 0.03) * 8.0, 0.0, 1.0)
        low *= gate
        mid *= gate
        high *= gate
        width *= gate
        balance *= gate
        transient *= gate

        self._smooth("rms", rms, SMOOTH_FAST)
        self._smooth("low", low, SMOOTH_FAST)
        self._smooth("mid", mid, SMOOTH_FAST)
        self._smooth("high", high, SMOOTH_FAST)
        self._smooth("balance", balance, SMOOTH_SLOW)
        self._smooth("width", width, SMOOTH_SLOW)
        self._smooth("transient", transient, 0.50)

        for key in self.state:
            if key != "balance":
                self.state[key] = float(np.clip(self.state[key], 0.0, 1.0))

        return self.state.copy()
