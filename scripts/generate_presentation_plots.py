"""
Generate comprehensive, presentation-grade diagnostic plots for IgniCoal AI.
Saves all figures to results/presentation/.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_and_clean_signals
from src.preprocessor import SignalPreprocessor
from src.feature_extractor import FeatureExtractor


def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2


def main():
    print("=" * 60)
    print("Generating Presentation Graphics & Diagnostic Figures")
    print("=" * 60)

    pres_dir = os.path.join(BASE_DIR, 'results', 'presentation')
    os.makedirs(pres_dir, exist_ok=True)
    set_plot_style()

    time_axis, signals = load_and_clean_signals(BASE_DIR, ignore_c1=True)
    prep = SignalPreprocessor()
    fe = FeatureExtractor()

    # -------------------------------------------------------------
    # Plot 1: Waveform Physics: Trigger Artifact vs True PA Wave
    # -------------------------------------------------------------
    print("  [1/4] Generating waveform physics plot...")
    sig_high = [s for s in signals if 'C14_P1_05' in s['signal_id']][0]['signal']
    sig_low = [s for s in signals if 'C4_P1_05' in s['signal_id']][0]['signal']

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    t_us = time_axis * 1e6

    # Sample C14 (High SCS) - Raw
    axes[0, 0].plot(t_us, sig_high * 1000, color='#d62728', lw=1.2, label='Raw Sensor Signal')
    axes[0, 0].axvspan(0, 1.2, color='gray', alpha=0.25, label='Laser Trigger Artifact (0-1.2 µs)')
    axes[0, 0].axvline(0.06, color='black', linestyle='--', alpha=0.7, label='Trigger Spike (60 ns)')
    axes[0, 0].set_title('Sample C14 (High SCS) - Raw Signal with Laser Trigger Spike', fontweight='bold', fontsize=11)
    axes[0, 0].set_xlabel('Time (µs)', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude (mV)', fontweight='bold')
    axes[0, 0].legend(loc='upper right', frameon=True)
    axes[0, 0].set_xlim(0, 26)

    # Sample C14 - Processed (Post-Cutoff)
    proc_high = prep.process_signal(sig_high, remove_trigger=True)
    axes[0, 1].plot(t_us, proc_high * 1000, color='#1f77b4', lw=1.5, label='Processed Acoustic Wave')
    p_idx = 60 + np.argmax(np.abs(proc_high[60:]))
    axes[0, 1].plot(t_us[p_idx], proc_high[p_idx] * 1000, 'ro', markersize=8, label=f'True PA Peak ($T_p = {t_us[p_idx]:.2f}\,\mu s$, $v = {0.006/time_axis[p_idx]:.0f}$ m/s)')
    axes[0, 1].axvline(8.0, color='darkgreen', linestyle=':', lw=2, label='400th Bin ($8.0\,\mu s \\to 750$ m/s)')
    axes[0, 1].set_title('Sample C14 - Trigger Blanked & Processed (Acoustic Arrival at ~8.0 µs)', fontweight='bold', fontsize=11)
    axes[0, 1].set_xlabel('Time (µs)', fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude (mV)', fontweight='bold')
    axes[0, 1].legend(loc='upper right', frameon=True)
    axes[0, 1].set_xlim(0, 26)

    # Sample C4 (Low SCS) - Raw
    axes[1, 0].plot(t_us, sig_low * 1000, color='#d62728', lw=1.2, label='Raw Sensor Signal')
    axes[1, 0].axvspan(0, 1.2, color='gray', alpha=0.25, label='Laser Trigger Artifact')
    axes[1, 0].set_title('Sample C4 (Low SCS) - Raw Signal with Laser Trigger Spike', fontweight='bold', fontsize=11)
    axes[1, 0].set_xlabel('Time (µs)', fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude (mV)', fontweight='bold')
    axes[1, 0].legend(loc='upper right', frameon=True)
    axes[1, 0].set_xlim(0, 26)

    # Sample C4 - Processed (Post-Cutoff)
    proc_low = prep.process_signal(sig_low, remove_trigger=True)
    axes[1, 1].plot(t_us, proc_low * 1000, color='#2ca02c', lw=1.5, label='Processed Acoustic Wave')
    p_idx_low = 60 + np.argmax(np.abs(proc_low[60:]))
    axes[1, 1].plot(t_us[p_idx_low], proc_low[p_idx_low] * 1000, 'ro', markersize=8, label=f'True PA Peak ($T_p = {t_us[p_idx_low]:.2f}\,\mu s$, $v = {0.006/time_axis[p_idx_low]:.0f}$ m/s)')
    axes[1, 1].set_title('Sample C4 - Trigger Blanked & Processed (Faster Arrival at 3.44 µs)', fontweight='bold', fontsize=11)
    axes[1, 1].set_xlabel('Time (µs)', fontweight='bold')
    axes[1, 1].set_ylabel('Amplitude (mV)', fontweight='bold')
    axes[1, 1].legend(loc='upper right', frameon=True)
    axes[1, 1].set_xlim(0, 26)

    plt.tight_layout()
    p1_path = os.path.join(pres_dir, 'waveform_physics_analysis.png')
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print(f"  Saved: {p1_path}")

    # -------------------------------------------------------------
    # Plot 2: Frequency Bins & FFT Spectrum
    # -------------------------------------------------------------
    print("  [2/4] Generating frequency spectrum and binning plot...")
    freqs_mhz, mag = fe.compute_fft(proc_high)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_mhz, mag * 1000, color='#1f77b4', lw=2, label='Photoacoustic Amplitude Spectrum')

    bin_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    bin_names = ['Bin 1\n(0.2-0.4 MHz)', 'Bin 2\n(0.4-0.6 MHz)', 'Bin 3\n(0.6-0.8 MHz)', 'Bin 4\n(0.8-1.0 MHz)', 'Bin 5\n(1.0-1.3 MHz)']
    for i in range(1, 6):
        f_low, f_high = fe.BINS_MHZ[i]
        plt.axvspan(f_low, f_high, color=bin_colors[i-1], alpha=0.35, label=bin_names[i-1])

    # Mark dominant peak
    dom = fe.find_dominant_frequencies(freqs_mhz, mag, num_peaks=2)
    dom_f, dom_m = dom[0]
    plt.plot(dom_f, dom_m * 1000, 'r*', markersize=12, label=f'Dominant Peak: {dom_f:.3f} MHz')

    plt.title('Photoacoustic Sensor FFT Spectrum & Binned Frequency Descriptors (0.2 - 1.3 MHz)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Frequency (MHz)', fontsize=11, fontweight='bold')
    plt.ylabel('Magnitude (a.u.)', fontsize=11, fontweight='bold')
    plt.xlim(0, 2.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    p2_path = os.path.join(pres_dir, 'spectral_decomposition_and_bins.png')
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"  Saved: {p2_path}")

    # -------------------------------------------------------------
    # Plot 3: Acoustic Velocity by Coal Sample & SCS Class
    # -------------------------------------------------------------
    print("  [3/4] Generating acoustic velocity distribution plot...")
    master = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'ignicoal_all_features.csv'))
    
    plt.figure(figsize=(10, 5))
    order = ['Low', 'Moderate', 'High']
    palette = {'Low': '#2ca02c', 'Moderate': '#ff7f0e', 'High': '#d62728'}
    sns.boxplot(data=master, x='scs_label', y='acoustic_velocity', order=order, palette=palette, width=0.45, boxprops=dict(alpha=0.75))
    sns.stripplot(data=master, x='scs_label', y='acoustic_velocity', order=order, color='black', alpha=0.6, jitter=0.15, size=5)
    
    plt.axhline(750, color='darkred', linestyle='--', lw=1.5, label='Mentor 400th Bin Baseline (750 m/s)')
    plt.axhspan(900, 1300, color='lightblue', alpha=0.3, label='Mentor Nominal Coal Range (900-1300 m/s)')

    plt.title('Acoustic Velocity (ToF) Distribution Across SCS Susceptibility Classes\n(Validating TCS Paper 1: Low SCS Coals Have Higher Velocity / High SCS ~750 m/s)', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('SCS Susceptibility Class', fontsize=11, fontweight='bold')
    plt.ylabel('Acoustic Velocity (m/s)', fontsize=11, fontweight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    p3_path = os.path.join(pres_dir, 'acoustic_velocity_vs_scs_physics.png')
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"  Saved: {p3_path}")

    # -------------------------------------------------------------
    # Plot 4: 3-Target Regression Dashboard
    # -------------------------------------------------------------
    print("  [4/4] Generating combined regression dashboard...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tasks = [
        ('Fixed_Carbon', 'Fixed Carbon Content (%)', '%', '#1f77b4'),
        ('Ash_Content', 'Ash Content (%)', '%', '#ff7f0e'),
        ('Ignition_Temperature', 'Ignition Temp (Tign)', '°C', '#2ca02c')
    ]

    file_map = {
        'Fixed_Carbon': 'carbon_features.csv',
        'Ash_Content': 'ash_features.csv',
        'Ignition_Temperature': 'thermal_features.csv'
    }

    for idx, (tname, tlabel, tunit, tcolor) in enumerate(tasks):
        with open(os.path.join(BASE_DIR, 'models', 'saved', f"best_model_{tname}.pkl"), 'rb') as f:
            pkg = pickle.load(f)
        
        df_t = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', file_map[tname]))
        sub_df = df_t[df_t[pkg['target']].notna()]
        X_vals = sub_df[pkg['features']].values
        y_true = sub_df[pkg['target']].values
        
        # Test predictions
        from sklearn.model_selection import train_test_split
        _, X_te, _, y_te = train_test_split(X_vals, y_true, test_size=0.2, random_state=42)
        y_pr = pkg['model'].predict(X_te)
        
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_te, y_pr)
        rmse = np.sqrt(mean_squared_error(y_te, y_pr))

        ax = axes[idx]
        ax.scatter(y_te, y_pr, color=tcolor, edgecolors='k', alpha=0.8, s=55, label='Test Instances')
        mn = min(min(y_te), min(y_pr))
        mx = max(max(y_te), max(y_pr))
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.8, label='1:1 Ideal Parity')
        ax.set_title(f"{tlabel}\nTest R² = {r2*100:.1f}%, RMSE = {rmse:.2f}{tunit}", fontweight='bold', fontsize=11)
        ax.set_xlabel(f"Actual ({tunit})", fontweight='bold')
        ax.set_ylabel(f"Predicted ({tunit})", fontweight='bold')
        ax.legend(frameon=True, loc='upper left')

    plt.tight_layout()
    p4_path = os.path.join(pres_dir, 'regression_dashboard_3x1.png')
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f"  Saved: {p4_path}")

    print("\nPresentation graphics generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
