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
            "kick": 0.0,
        }

        self.prev_rms = 0.0
        self.prev_low_energy = 0.0
        self.prev_flux = 0.0

        # adaptive floors
        self.noise_floor = 0.0
        self.low_floor = 0.0
        self.mid_floor = 0.0
        self.high_floor = 0.0

        # slow adaptive peaks
        self.low_peak = 1e-6
        self.mid_peak = 1e-6
        self.high_peak = 1e-6

    # =========================================================
    # HELPERS
    # =========================================================
    def _band_energy(self, spectrum, low_hz, high_hz):
        idx = np.where((self.freqs >= low_hz) & (self.freqs < high_hz))[0]
        if len(idx) == 0:
            return 0.0
        return float(np.mean(spectrum[idx]))

    def _smooth(self, key, target, amount):
        self.state[key] = (1 - amount) * self.state[key] + amount * target

    def _adaptive_normalize(self, value, floor, peak, floor_mult=1.15):
        """
        Normalizza valore rispetto a un floor dinamico e un peak lento.
        """
        value = max(0.0, value - floor * floor_mult)
        norm = value / (peak + 1e-6)
        return float(np.clip(norm, 0.0, 1.0))

    # =========================================================
    # MAIN EXTRACT
    # =========================================================
    def extract(self, stereo_frame):
        left = stereo_frame[:, 0].astype(np.float32)
        right = stereo_frame[:, 1].astype(np.float32)

        mono = (left + right) * 0.5
        side = (left - right) * 0.5

        # -----------------------------------------------------
        # RMS
        # -----------------------------------------------------
        rms_raw = float(np.sqrt(np.mean(mono ** 2) + 1e-9))

        # adaptive noise floor (molto lento)
        self.noise_floor = self.noise_floor * 0.995 + rms_raw * 0.005

        rms_clean = max(0.0, rms_raw - self.noise_floor * 1.10)
        rms = np.clip(rms_clean * 14.0, 0.0, 1.0)

        # -----------------------------------------------------
        # FFT
        # -----------------------------------------------------
        window = np.hanning(len(mono))
        mono_fft = np.fft.rfft(mono * window)
        side_fft = np.fft.rfft(side * window)

        mono_mag = np.abs(mono_fft)
        side_mag = np.abs(side_fft)

        # compressione dolce
        mono_spec = np.log1p(mono_mag * 2.5)
        side_spec = np.log1p(side_mag * 2.5)

        # -----------------------------------------------------
        # BAND ENERGIES (ABSOLUTE)
        # -----------------------------------------------------
        low_raw = self._band_energy(mono_spec, 45, 180)
        mid_raw = self._band_energy(mono_spec, 180, 2200)
        high_raw = self._band_energy(mono_spec, 2200, 10000)

        # adaptive floors
        self.low_floor = self.low_floor * 0.992 + low_raw * 0.008
        self.mid_floor = self.mid_floor * 0.992 + mid_raw * 0.008
        self.high_floor = self.high_floor * 0.992 + high_raw * 0.008

        # adaptive peaks (release lento, attack veloce)
        self.low_peak = max(low_raw, self.low_peak * 0.992)
        self.mid_peak = max(mid_raw, self.mid_peak * 0.992)
        self.high_peak = max(high_raw, self.high_peak * 0.992)

        low = self._adaptive_normalize(low_raw, self.low_floor, self.low_peak, floor_mult=1.10)
        mid = self._adaptive_normalize(mid_raw, self.mid_floor, self.mid_peak, floor_mult=1.10)
        high = self._adaptive_normalize(high_raw, self.high_floor, self.high_peak, floor_mult=1.08)

        # -----------------------------------------------------
        # SPECTRAL FLUX (vero transient driver)
        # -----------------------------------------------------
        low_rise = max(0.0, low_raw - self.prev_low_energy)
        self.prev_low_energy = low_raw

        # flux semplificato sulle bande
        band_flux = (
            max(0.0, low_raw - self.low_floor) * 0.55 +
            max(0.0, mid_raw - self.mid_floor) * 0.25 +
            max(0.0, high_raw - self.high_floor) * 0.20
        )

        # transient: mix di rise RMS + flux banda
        rms_rise = max(0.0, rms - self.prev_rms)
        transient = (
            rms_rise * 0.45 +
            low_rise * 0.85 +
            band_flux * 0.20
        )

        self.prev_rms = rms

        # normalizzazione transient più conservativa
        transient = np.clip(transient * 1.6, 0.0, 1.0)

        # -----------------------------------------------------
        # KICK DETECTOR (molto più utile del tuo random trigger)
        # -----------------------------------------------------
        kick = (
            low * 0.55 +
            transient * 0.95 +
            np.clip(low_rise * 1.2, 0.0, 1.0) * 0.65
        )

        # kick deve essere anche "sostenuto" da energia reale
        kick *= np.clip((rms - 0.04) * 3.2, 0.0, 1.0)
        kick = np.clip(kick, 0.0, 1.0)

        # -----------------------------------------------------
        # STEREO
        # -----------------------------------------------------
        width = float(np.mean(side_spec) / (np.mean(mono_spec) + 1e-6))
        width = np.clip(width * 1.8, 0.0, 1.0)

        left_rms = float(np.sqrt(np.mean(left ** 2) + 1e-9))
        right_rms = float(np.sqrt(np.mean(right ** 2) + 1e-9))
        balance = np.clip((right_rms - left_rms) * 6.0, -1.0, 1.0)

        # -----------------------------------------------------
        # GLOBAL GATE
        # -----------------------------------------------------
        gate = np.clip((rms - 0.025) * 10.0, 0.0, 1.0)

        low *= gate
        mid *= gate
        high *= gate
        width *= gate
        balance *= gate
        transient *= gate
        kick *= gate

        # -----------------------------------------------------
        # SMOOTHING
        # -----------------------------------------------------
        self._smooth("rms", rms, SMOOTH_FAST)
        self._smooth("low", low, 0.22)
        self._smooth("mid", mid, 0.18)
        self._smooth("high", high, 0.16)
        self._smooth("balance", balance, SMOOTH_SLOW)
        self._smooth("width", width, SMOOTH_SLOW)
        self._smooth("transient", transient, 0.28)
        self._smooth("kick", kick, 0.32)

        # clamp
        for key in self.state:
            if key != "balance":
                self.state[key] = float(np.clip(self.state[key], 0.0, 1.0))

        return self.state.copy()