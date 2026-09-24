"""
Signal Preprocessor module for IgniCoal AI.
Implements:
  - Baseline drift removal (low-order polynomial / baseline detrending)
  - Laser trigger pulse blanking/removal (first ~1.2 microseconds)
  - Savitzky-Golay smoothing
  - Wavelet denoising (VisuShrink / BayesShrink soft-thresholding)
  - Noise-only reference signal extraction per signal
  - SNR calculation for raw and processed signals
"""

import numpy as np
from scipy.signal import savgol_filter
import pywt


class SignalPreprocessor:
    def __init__(
        self,
        sampling_rate: float = 50e6,  # 50 MHz (20 ns sampling interval)
        trigger_cutoff_us: float = 1.2, # 1.2 us cutoff for 400ns laser excitation artifact
        sg_window: int = 15,
        sg_order: int = 3,
        wavelet: str = 'sym4',
        wavelet_level: int = 3,
        noise_window_samples: int = 200  # Tail samples for noise reference
    ):
        self.sampling_rate = sampling_rate
        self.trigger_cutoff_us = trigger_cutoff_us
        self.trigger_cutoff_samples = int(trigger_cutoff_us * 1e-6 * sampling_rate) # 60 samples
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.noise_window_samples = noise_window_samples

    def remove_baseline_drift(self, signal: np.ndarray, poly_deg: int = 2) -> np.ndarray:
        """
        Removes baseline offset and low-frequency drift via polynomial fitting.
        """
        x = np.linspace(-1, 1, len(signal))
        poly_coeffs = np.polyfit(x, signal, deg=poly_deg)
        baseline = np.polyval(poly_coeffs, x)
        return signal - baseline

    def blank_trigger_pulse(self, signal: np.ndarray) -> np.ndarray:
        """
        Blanks the early electrical/optical trigger artifact (t <= trigger_cutoff_us)
        by replacing the trigger zone with the pre-acoustic baseline level.
        """
        clean = signal.copy()
        cutoff = min(self.trigger_cutoff_samples, len(signal) // 4)
        # Use baseline level around the cutoff boundary or coda mean
        baseline_val = np.mean(clean[cutoff:cutoff + 30])
        clean[:cutoff] = baseline_val
        return clean

    def savgol_smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies Savitzky-Golay smoothing to remove high-frequency jitter
        while preserving peak sharpness and rise/fall dynamics.
        """
        window = min(self.sg_window, len(signal) - 1)
        if window % 2 == 0:
            window -= 1
        return savgol_filter(signal, window_length=window, polyorder=self.sg_order)

    def wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Performs multiresolution wavelet denoising with universal soft-thresholding.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.wavelet_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            new_coeffs.append(pywt.threshold(d, threshold, mode='soft'))
            
        denoised = pywt.waverec(new_coeffs, self.wavelet)
        return denoised[:len(signal)]

    def process_signal(self, raw_signal: np.ndarray, remove_trigger: bool = True) -> np.ndarray:
        """
        Complete processing pipeline (_proc):
        1. Baseline drift removal
        2. Savitzky-Golay smoothing
        3. Wavelet denoising
        """
        detrended = self.remove_baseline_drift(raw_signal)
        smoothed = self.savgol_smooth(detrended)
        proc_signal = self.wavelet_denoise(smoothed)
        if remove_trigger:
            proc_signal = self.blank_trigger_pulse(proc_signal)
        return proc_signal

    def extract_noise_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Extracts noise-only reference signal on a per-signal basis from the tail region.
        """
        noise_len = min(self.noise_window_samples, len(signal) // 5)
        return signal[-noise_len:]

    def compute_snr(self, signal: np.ndarray, is_raw: bool = True) -> float:
        """
        Computes Signal-to-Noise Ratio (dB) on a per-signal basis.
        Signal power is computed from the acoustic arrival window,
        noise power is computed from the noise reference window.
        """
        noise_ref = self.extract_noise_reference(signal)
        noise_var = np.var(noise_ref)
        if noise_var <= 1e-12:
            noise_var = 1e-12

        # Ignore trigger zone when computing signal peak power
        cutoff = self.trigger_cutoff_samples if is_raw else 0
        active_signal = signal[cutoff:] if len(signal) > cutoff + 10 else signal
        
        ac_signal = active_signal - np.mean(noise_ref)
        peak_amp = np.max(np.abs(ac_signal))
        
        rms_noise = np.sqrt(noise_var)
        if rms_noise > 0 and peak_amp > 0:
            snr = 20 * np.log10(peak_amp / rms_noise)
        else:
            snr = 0.0
        return float(np.clip(snr, -20.0, 80.0))
