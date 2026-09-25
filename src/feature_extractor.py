"""
Feature Extractor module for IgniCoal AI.
Extracts time-domain, frequency-domain (binned 0.2-1.3 MHz), and wavelet features
specified in FeatureList_1.pptx and research papers with:
  - Trigger pulse blanking (removing early laser excitation artifact)
  - Accurate Photoacoustic peak detection and Time of Flight (ToF)
  - Physical acoustic velocity calculation (750 m/s at 400th bin, 900-1300 m/s range)
  - Frequency metrics scaled in MHz
  - Full modular support for P2P-TDB3
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, hilbert
import pywt
from src.preprocessor import SignalPreprocessor


class FeatureExtractor:
    # Frequency bands in MHz (0.2 - 1.3 MHz)
    BINS_MHZ = {
        1: (0.2, 0.4),
        2: (0.4, 0.6),
        3: (0.6, 0.8),
        4: (0.8, 1.0),
        5: (1.0, 1.3),
    }
    
    PELLET_THICKNESS_M = 0.006  # 6 mm pellet thickness

    def __init__(self, sampling_rate: float = 50e6, trigger_cutoff_us: float = 1.2):
        self.sampling_rate = sampling_rate
        self.trigger_cutoff_us = trigger_cutoff_us
        self.preprocessor = SignalPreprocessor(
            sampling_rate=sampling_rate, trigger_cutoff_us=trigger_cutoff_us
        )

    def compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes single-sided FFT amplitude spectrum with frequencies in MHz.
        """
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        # Frequencies in MHz
        freqs_mhz = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate) * 1e-6
        mag = np.abs(fft_vals) * 2.0 / n
        return freqs_mhz, mag

    def get_bin_mask(self, freqs_mhz: np.ndarray, bin_idx: int) -> np.ndarray:
        f_low, f_high = self.BINS_MHZ[bin_idx]
        return (freqs_mhz >= f_low) & (freqs_mhz <= f_high)

    def spectral_centroid(self, freqs: np.ndarray, mag: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculates spectral centroid: sum(f * M(f)) / sum(M(f)) (in MHz)
        """
        if mask is not None:
            f = freqs[mask]
            m = mag[mask]
        else:
            f = freqs
            m = mag
            
        sum_m = np.sum(m)
        if sum_m <= 1e-15:
            return 0.0
        return float(np.sum(f * m) / sum_m)

    def spectral_bandwidth(self, freqs: np.ndarray, mag: np.ndarray, centroid: float = None, mask: np.ndarray = None) -> float:
        """
        Calculates spectral spread (bandwidth) around centroid (in MHz)
        """
        if mask is not None:
            f = freqs[mask]
            m = mag[mask]
        else:
            f = freqs
            m = mag
            
        sum_m = np.sum(m)
        if sum_m <= 1e-15:
            return 0.0
        if centroid is None:
            centroid = self.spectral_centroid(f, m)
            
        variance = np.sum(((f - centroid) ** 2) * m) / sum_m
        return float(np.sqrt(np.maximum(variance, 0.0)))

    def spectral_entropy(self, mag: np.ndarray, mask: np.ndarray = None, use_power: bool = True) -> float:
        """
        Calculates normalized Shannon spectral entropy over power (or magnitude).
        """
        m = mag[mask] if mask is not None else mag
        p = (m ** 2) if use_power else m
        p = np.maximum(p, 0.0)
        sum_p = np.sum(p)
        if sum_p <= 1e-15 or len(p) <= 1:
            return 0.0
        probs = p / sum_p
        probs = probs[probs > 1e-15]
        if len(probs) <= 1:
            return 0.0
        entropy = -np.sum(probs * np.log2(probs))
        norm_entropy = entropy / np.log2(len(p))
        return float(norm_entropy)

    def spectral_area(self, freqs_mhz: np.ndarray, mag: np.ndarray, mask: np.ndarray) -> float:
        """
        Area under FFT magnitude curve within a frequency bin (f in MHz).
        """
        f = freqs_mhz[mask]
        m = mag[mask]
        if len(f) < 2:
            return 0.0
        return float(np.trapezoid(m, f) if hasattr(np, 'trapezoid') else np.trapz(m, f))

    def band_dominant_frequency(self, freqs_mhz: np.ndarray, mag: np.ndarray, bin_idx: int) -> float:
        """
        Dominant peak frequency within a specific frequency band (in MHz).
        """
        mask = self.get_bin_mask(freqs_mhz, bin_idx)
        sub_f = freqs_mhz[mask]
        sub_m = mag[mask]
        if len(sub_m) == 0:
            return 0.0
        return float(sub_f[np.argmax(sub_m)])

    def band_wave_energy(self, freqs_mhz: np.ndarray, mag: np.ndarray, bin_idx: int) -> float:
        """
        Wave energy within a specific frequency band (integral of power spectrum |M|^2).
        """
        mask = self.get_bin_mask(freqs_mhz, bin_idx)
        sub_f = freqs_mhz[mask]
        sub_p = mag[mask] ** 2
        if len(sub_p) < 2:
            return 0.0
        return float(np.trapezoid(sub_p, sub_f) if hasattr(np, 'trapezoid') else np.trapz(sub_p, sub_f))

    def wave_entropy_spectrum(self, freqs_mhz: np.ndarray, mag: np.ndarray) -> float:
        """
        Global Shannon wave entropy of the power spectrum (excluding DC).
        """
        mask = freqs_mhz > 0
        pwr = mag[mask] ** 2
        sum_p = np.sum(pwr)
        if sum_p <= 1e-15 or len(pwr) < 2:
            return 0.0
        probs = pwr / sum_p
        probs = probs[probs > 1e-15]
        if len(probs) <= 1:
            return 0.0
        ent = -np.sum(probs * np.log2(probs))
        return float(ent / np.log2(len(pwr)))

    def find_dominant_frequencies(self, freqs_mhz: np.ndarray, mag: np.ndarray, num_peaks: int = 3) -> list[tuple[float, float]]:
        """
        Finds the top `num_peaks` dominant frequencies (freq in MHz, magnitude).
        Restricts search within sensor bandwidth (0.1 MHz to 2.5 MHz).
        """
        band_mask = (freqs_mhz >= 0.1) & (freqs_mhz <= 2.5)
        sub_f = freqs_mhz[band_mask]
        sub_m = mag[band_mask]
        
        peaks, _ = find_peaks(sub_m, distance=4)
        if len(peaks) == 0:
            sorted_idx = np.argsort(sub_m)[::-1][:num_peaks]
            return [(float(sub_f[i]), float(sub_m[i])) for i in sorted_idx]
            
        peak_mags = sub_m[peaks]
        sorted_order = np.argsort(peak_mags)[::-1]
        top_peaks = peaks[sorted_order][:num_peaks]
        
        results = [(float(sub_f[idx]), float(sub_m[idx])) for idx in top_peaks]
        while len(results) < num_peaks:
            results.append((0.0, 0.0))
        return results

    def extract_time_features(
        self,
        signal: np.ndarray,
        time_axis: np.ndarray,
        is_processed: bool = True
    ) -> dict:
        """
        Extracts temporal and statistical metrics with trigger artifact blanked:
        - Absorption proxy: peak amplitude of the photoacoustic wave in acoustic window
        - Peak Time (Tp): arrival time of primary acoustic peak in [2.0 us, 12.0 us]
        - ToF & Acoustic Velocity: thickness / Tp (strictly solid-state coal range 700 - 2400 m/s)
        - Rise Time & Fall Time (computed on acoustic Hilbert pulse envelope with linear interpolation)
        - Peak Duration (FWHM)
        - P2P-TDB3: peak-to-peak in 3rd equal time window
        """
        dt = time_axis[1] - time_axis[0]
        
        # Whole-signal statistical moments
        p2p_full = float(np.max(signal) - np.min(signal))
        rms_full = float(np.sqrt(np.mean(signal ** 2)))
        kurt_full = float(stats.kurtosis(signal, fisher=False))  # Pearson kurtosis matching papers
        skew_full = float(stats.skew(signal))

        # Search for primary acoustic peak strictly in the ultrasonic arrival window [2.0 us, 12.0 us]
        w_start = int(2.0e-6 / dt)
        w_end = min(len(signal), int(12.0e-6 / dt))
        if w_start >= w_end:
            w_start = int(1.2e-6 / dt)
            w_end = len(signal)
            
        # Analytic Hilbert envelope on signal
        env = np.abs(hilbert(signal))
        active_env = env[w_start:w_end]
        rel_p_idx = int(np.argmax(active_env))
        peak_idx = w_start + rel_p_idx
        peak_val = float(active_env[rel_p_idx])
        peak_time = float(time_axis[peak_idx])

        # Backward search for onset (leading edge where signal drops to 10% of peak or 2.5*sigma)
        noise_sigma = np.std(signal[-150:])
        onset_thresh = max(2.5 * noise_sigma, 0.10 * peak_val)
        onset_idx = peak_idx
        for i in range(peak_idx, w_start, -1):
            if np.abs(signal[i]) <= onset_thresh:
                onset_idx = i
                break
        t_onset = float(time_axis[onset_idx])

        # Acoustic velocities (thickness 6 mm)
        v_peak = float(self.PELLET_THICKNESS_M / peak_time) if peak_time > 1e-7 else 0.0
        v_onset = float(self.PELLET_THICKNESS_M / t_onset) if t_onset > 1e-7 else v_peak

        # Pulse envelope levels with linear interpolation
        baseline = np.min(active_env)
        amp = max(peak_val - baseline, 1e-15)
        l10 = baseline + 0.10 * amp
        l90 = baseline + 0.90 * amp
        l50 = baseline + 0.50 * amp

        # Rise time (10% to 90% leading to peak)
        t10_r = time_axis[w_start]
        t90_r = peak_time
        for i in range(peak_idx - 1, w_start, -1):
            if env[i] <= l90 <= env[i + 1]:
                frac = (l90 - env[i]) / max(env[i + 1] - env[i], 1e-15)
                t90_r = time_axis[i] + frac * dt
                break
        for i in range(peak_idx - 1, w_start, -1):
            if env[i] <= l10 <= env[i + 1]:
                frac = (l10 - env[i]) / max(env[i + 1] - env[i], 1e-15)
                t10_r = time_axis[i] + frac * dt
                break
        rise_time = float(max(t90_r - t10_r, 0.0))

        # Fall time (90% to 10% trailing after peak)
        t90_f = peak_time
        t10_f = time_axis[w_end - 1]
        for i in range(peak_idx, w_end - 1):
            if env[i] >= l90 >= env[i + 1]:
                frac = (env[i] - l90) / max(env[i] - env[i + 1], 1e-15)
                t90_f = time_axis[i] + frac * dt
                break
        for i in range(peak_idx, w_end - 1):
            if env[i] >= l10 >= env[i + 1]:
                frac = (env[i] - l10) / max(env[i] - env[i + 1], 1e-15)
                t10_f = time_axis[i] + frac * dt
                break
        fall_time = float(max(t10_f - t90_f, 0.0))

        # Peak duration (FWHM at 50% height)
        t_l = time_axis[w_start]
        t_r = time_axis[w_end - 1]
        for i in range(peak_idx - 1, w_start, -1):
            if env[i] <= l50 <= env[i + 1]:
                frac = (l50 - env[i]) / max(env[i + 1] - env[i], 1e-15)
                t_l = time_axis[i] + frac * dt
                break
        for i in range(peak_idx, w_end - 1):
            if env[i] >= l50 >= env[i + 1]:
                frac = (env[i] - l50) / max(env[i] - env[i + 1], 1e-15)
                t_r = time_axis[i] + frac * dt
                break
        peak_duration = float(max(t_r - t_l, 0.0))

        # Peak ratio (1st / 2nd peak on envelope)
        pks, _ = find_peaks(active_env, prominence=0.05 * amp, distance=10)
        if len(pks) >= 2:
            sorted_pks = np.sort(active_env[pks])[::-1]
            peak_ratio = float(sorted_pks[0] / max(sorted_pks[1], 1e-15))
        else:
            peak_ratio = 1.0

        # Signal energy and Area under the curve (on active acoustic region)
        acoustic_sig = signal[w_start:]
        signal_energy = float(np.sum(acoustic_sig ** 2))
        area_under_curve = float(np.trapezoid(np.abs(acoustic_sig), time_axis[w_start:]) if hasattr(np, 'trapezoid') else np.trapz(np.abs(acoustic_sig), time_axis[w_start:]))

        # P2P in 3rd equal time window (TDB3: [2/3 T, T])
        n = len(signal)
        w3_sig = signal[2 * n // 3:]
        p2p_tdb3 = float(np.max(w3_sig) - np.min(w3_sig)) if len(w3_sig) > 0 else 0.0

        return {
            'absorption_proxy': float(np.max(np.abs(signal))),
            'acoustic_peak_amp': peak_val,
            'peak_time': peak_time,
            'p2p': p2p_full,
            'rms': rms_full,
            'kurtosis': kurt_full,
            'skew': skew_full,
            'acoustic_velocity': v_peak,
            'acoustic_velocity_onset': v_onset,
            't_onset': t_onset,
            'rise_time': rise_time,
            'fall_time': fall_time,
            'peak_duration': peak_duration,
            'peak_ratio': peak_ratio,
            'signal_energy': signal_energy,
            'area_under_curve': area_under_curve,
            'p2p_tdb3': p2p_tdb3
        }

    def extract_wavelet_features(self, signal: np.ndarray) -> dict:
        """
        Extracts wavelet domain energy and entropy using DWT.
        """
        coeffs = pywt.wavedec(signal, 'sym4', level=4)
        cD3 = coeffs[2]
        wave_energy_3 = float(np.sum(cD3 ** 2))
        
        all_d = np.concatenate([c.flatten() for c in coeffs[1:]])
        d_power = all_d ** 2
        sum_p = np.sum(d_power)
        if sum_p > 1e-15:
            prob = d_power / sum_p
            prob = prob[prob > 0]
            wave_entropy = float(-np.sum(prob * np.log2(prob)) / np.log2(len(all_d)))
        else:
            wave_entropy = 0.0
            
        return {
            'wave_energy_3': wave_energy_3,
            'wave_entropy': wave_entropy
        }

    def extract_all_features(self, raw_signal: np.ndarray, time_axis: np.ndarray) -> dict:
        """
        Processes signal, blanks trigger, and extracts comprehensive feature set.
        """
        proc_signal = self.preprocessor.process_signal(raw_signal, remove_trigger=True)

        # FFT on raw and processed signals
        freqs_mhz_raw, mag_raw = self.compute_fft(raw_signal)
        freqs_mhz_proc, mag_proc = self.compute_fft(proc_signal)

        # SNR metrics
        snr_raw = self.preprocessor.compute_snr(raw_signal, is_raw=True)
        snr_proc = self.preprocessor.compute_snr(proc_signal, is_raw=False)

        # Time features with trigger artifact removed
        t_raw = self.extract_time_features(raw_signal, time_axis, is_processed=False)
        t_proc = self.extract_time_features(proc_signal, time_axis, is_processed=True)

        # Wavelet features
        w_proc = self.extract_wavelet_features(proc_signal)

        # Dominant frequencies (in MHz)
        dom_raw = self.find_dominant_frequencies(freqs_mhz_raw, mag_raw, num_peaks=3)
        dom_proc = self.find_dominant_frequencies(freqs_mhz_proc, mag_proc, num_peaks=3)

        features = {
            # Metadata & SNR
            'snr_raw': snr_raw,
            'snr_proc': snr_proc,

            # Time Domain Raw (post-cutoff for peak/velocity)
            'absorption_proxy_raw': t_raw['absorption_proxy'],
            'p2p_raw': t_raw['p2p'],
            'rms_raw': t_raw['rms'],
            'kurtosis_raw': t_raw['kurtosis'],
            'skew_raw': t_raw['skew'],
            'peak_time_raw': t_raw['peak_time'],

            # Time Domain Processed (_proc)
            'absorption_proxy_proc': t_proc['absorption_proxy'],
            'p2p_proc': t_proc['p2p'],
            'rms_proc': t_proc['rms'],
            'kurtosis_proc': t_proc['kurtosis'],
            'skew_proc': t_proc['skew'],
            'peak_time_proc': t_proc['peak_time'],
            'acoustic_velocity': t_proc['acoustic_velocity'],
            'acoustic_velocity_onset': t_proc['acoustic_velocity_onset'],
            't_onset': t_proc['t_onset'],
            'rise_time_proc': t_proc['rise_time'],
            'fall_time_proc': t_proc['fall_time'],
            'peak_duration_proc': t_proc['peak_duration'],
            'signal_energy_proc': t_proc['signal_energy'],
            'area_under_curve_proc': t_proc['area_under_curve'],
            'p2p_tdb3': t_proc['p2p_tdb3'],

            # Wavelet
            'wave_energy_3': w_proc['wave_energy_3'],
            'wave_entropy': w_proc['wave_entropy'],

            # Global Spectral (in MHz)
            'spectral_centroid_global_raw': self.spectral_centroid(freqs_mhz_raw, mag_raw),
            'spectral_centroid_global_proc': self.spectral_centroid(freqs_mhz_proc, mag_proc),
            'spectral_bandwidth_global_raw': self.spectral_bandwidth(freqs_mhz_raw, mag_raw),
            'spectral_bandwidth_global_proc': self.spectral_bandwidth(freqs_mhz_proc, mag_proc),
            'spectral_entropy_global_raw': self.spectral_entropy(mag_raw),
            'spectral_entropy_global_proc': self.spectral_entropy(mag_proc),
            'area_under_fft_raw': float(np.trapezoid(mag_raw, freqs_mhz_raw) if hasattr(np, 'trapezoid') else np.trapz(mag_raw, freqs_mhz_raw)),
            'area_under_fft_proc': float(np.trapezoid(mag_proc, freqs_mhz_proc) if hasattr(np, 'trapezoid') else np.trapz(mag_proc, freqs_mhz_proc)),

            # Dominant Frequencies (in MHz)
            'dom_freq_1_raw': dom_raw[0][0],
            'dom_mag_1_raw': dom_raw[0][1],
            'dom_freq_2_raw': dom_raw[1][0],
            'dom_mag_2_raw': dom_raw[1][1],
            'dom_freq_3_raw': dom_raw[2][0],
            'dom_mag_3_raw': dom_raw[2][1],
            'dom_freq_1_proc': dom_proc[0][0],
            'dom_mag_1_proc': dom_proc[0][1],
            'dom_freq_2_proc': dom_proc[1][0],
            'dom_mag_2_proc': dom_proc[1][1],
            'dom_freq_3_proc': dom_proc[2][0],
            'dom_mag_3_proc': dom_proc[2][1],
            'peak_mag_ratio_raw': float(dom_raw[0][1] / (dom_raw[1][1] + 1e-12)),
            'peak_mag_ratio_proc': float(dom_proc[0][1] / (dom_proc[1][1] + 1e-12)),
            'wave_entropy_spectrum_raw': self.wave_entropy_spectrum(freqs_mhz_raw, mag_raw),
            'wave_entropy_spectrum_proc': self.wave_entropy_spectrum(freqs_mhz_proc, mag_proc),
        }

        # Binned spectral features (Bins 1 to 5, in MHz)
        for b_idx in range(1, 6):
            m_raw = self.get_bin_mask(freqs_mhz_raw, b_idx)
            m_proc = self.get_bin_mask(freqs_mhz_proc, b_idx)
            
            c_raw = self.spectral_centroid(freqs_mhz_raw, mag_raw, m_raw)
            c_proc = self.spectral_centroid(freqs_mhz_proc, mag_proc, m_proc)
            bw_raw = self.spectral_bandwidth(freqs_mhz_raw, mag_raw, c_raw, m_raw)
            bw_proc = self.spectral_bandwidth(freqs_mhz_proc, mag_proc, c_proc, m_proc)
            ent_raw = self.spectral_entropy(mag_raw, m_raw, use_power=True)
            ent_proc = self.spectral_entropy(mag_proc, m_proc, use_power=True)
            area_raw = self.spectral_area(freqs_mhz_raw, mag_raw, m_raw)
            area_proc = self.spectral_area(freqs_mhz_proc, mag_proc, m_proc)
            dom_f_raw = self.band_dominant_frequency(freqs_mhz_raw, mag_raw, b_idx)
            dom_f_proc = self.band_dominant_frequency(freqs_mhz_proc, mag_proc, b_idx)
            we_raw = self.band_wave_energy(freqs_mhz_raw, mag_raw, b_idx)
            we_proc = self.band_wave_energy(freqs_mhz_proc, mag_proc, b_idx)

            features[f'spectral_centroid_{b_idx}_raw'] = c_raw
            features[f'spectral_centroid_{b_idx}_proc'] = c_proc
            features[f'spectral_bandwidth_{b_idx}_raw'] = bw_raw
            features[f'spectral_bandwidth_{b_idx}_proc'] = bw_proc
            features[f'spectral_entropy_{b_idx}_raw'] = ent_raw
            features[f'spectral_entropy_{b_idx}_proc'] = ent_proc
            features[f'spectral_area_{b_idx}_raw'] = area_raw
            features[f'spectral_area_{b_idx}_proc'] = area_proc
            features[f'dom_freq_band_{b_idx}_raw'] = dom_f_raw
            features[f'dom_freq_band_{b_idx}_proc'] = dom_f_proc
            features[f'wave_energy_band_{b_idx}_raw'] = we_raw
            features[f'wave_energy_band_{b_idx}_proc'] = we_proc

        return features

    def get_feature_subset(self, all_features: dict, task: str, include_p2p_tdb3: bool = True) -> dict:
        """
        Returns the exact named feature subsets from FeatureList_1.pptx:
          - 'thermal' (for Ignition Temperature regression)
          - 'carbon' (for Fixed Carbon Content regression)
          - 'ash' (for Ash Content regression)
        """
        if task == 'thermal':
            feats = {
                'Spectral Centroid': all_features['spectral_centroid_global_raw'],
                'SNR_raw': all_features['snr_raw'],
                'Spectral Bandwidth 5': all_features['spectral_bandwidth_5_raw'],
                'Spectral Entropy 4': all_features['spectral_entropy_4_raw'],
                'Spectral Centroid 3': all_features['spectral_centroid_3_raw'],
                'Kurtosis': all_features['kurtosis_raw'],
                'RMS value': all_features['rms_raw'],
                'Absorption Proxy': all_features['absorption_proxy_raw'],
                'P2P': all_features['p2p_raw'],
                'Wave Energy 3': all_features['wave_energy_band_3_raw'],
                'Acoustic Velocity': all_features['acoustic_velocity'],
                'Peak Time': all_features['peak_time_proc'],
                'Spectral Centroid 2': all_features['spectral_centroid_2_raw'],
                'Dominant Frequency 1 peak': all_features['dom_freq_band_1_raw'],
                'Spectral Bandwidth 3': all_features['spectral_bandwidth_3_raw'],
                'Wave Entropy': all_features['wave_entropy_spectrum_raw'],
                'P2P-TDB3': all_features['p2p_tdb3'],
                'Peak Duration': all_features['peak_duration_proc'],
                'Spectral Centroid 4': all_features['spectral_centroid_4_raw'],
                'Dominant Frequency 2 peak': all_features['dom_freq_band_2_raw']
            }
            if not include_p2p_tdb3:
                feats.pop('P2P-TDB3', None)
            return feats

        elif task == 'carbon':
            return {
                'RMS value': all_features['rms_raw'],
                'P2P': all_features['p2p_raw'],
                'Dominant Frequency 1 peak': all_features['dom_freq_band_1_raw'],
                'Spectral Entropy 2': all_features['spectral_entropy_2_raw'],
                'Dominant Frequency 3 peak': all_features['dom_freq_band_3_raw'],
                'Absorption Proxy': all_features['absorption_proxy_raw'],
                'Dominant Frequency 2 peak': all_features['dom_freq_band_2_raw'],
                'SNR_raw': all_features['snr_raw'],
                'Spectral Entropy 1': all_features['spectral_entropy_1_raw'],
                'Spectral Entropy 3': all_features['spectral_entropy_3_raw'],
                'Spectral Bandwidth 4': all_features['spectral_bandwidth_4_raw'],
                'Spectral Centroid': all_features['spectral_centroid_global_raw'],
                'Skew Value': all_features['skew_raw'],
                'SNR_Processed': all_features['snr_proc'],
                'Spectral Entropy 4': all_features['spectral_entropy_4_raw'],
                'Spectral Centroid 1': all_features['spectral_centroid_1_raw'],
                'Spectral Centroid 2': all_features['spectral_centroid_2_raw'],
                'Spectral Bandwidth 1': all_features['spectral_bandwidth_1_raw']
            }

        elif task == 'ash':
            return {
                'Area under FFT curve_raw': all_features['area_under_fft_raw'],
                'Magnitude of dominant frequency_raw': all_features['dom_mag_1_raw'],
                'Spectral Bandwidth 1_raw': all_features['spectral_bandwidth_1_raw'],
                'Spectral Bandwidth 2_raw': all_features['spectral_bandwidth_2_raw'],
                'Spectral Bandwidth 3_raw': all_features['spectral_bandwidth_3_raw'],
                'Spectral Area 1': all_features['spectral_area_1_raw'],
                'Spectral Area 2': all_features['spectral_area_2_raw'],
                'Spectral Area 3': all_features['spectral_area_3_raw'],
                'Spectral Area 4': all_features['spectral_area_4_raw'],
                'Spectral Entropy 1': all_features['spectral_entropy_1_raw'],
                'Spectral Entropy 2': all_features['spectral_entropy_2_raw'],
                'Spectral Entropy 3': all_features['spectral_entropy_3_raw'],
                'Spectral Entropy 4': all_features['spectral_entropy_4_raw'],
                'Absorption Proxy_proc': all_features.get('absorption_proxy_raw', all_features['absorption_proxy_proc']),
                'Peak Time_proc': all_features['peak_time_proc'],
                'Rise time_proc': all_features['rise_time_proc'],
                'Fall time_proc': all_features['fall_time_proc'],
                'Signal Energy_proc': all_features['signal_energy_proc'],
                'Area under the curve_proc': all_features['area_under_curve_proc'],
                '1st peak magnitude/2nd peak magnitude': all_features['peak_mag_ratio_proc']
            }

        else:
            raise ValueError(f"Unknown task: {task}.")
