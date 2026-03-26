#!/usr/bin/env python3
"""
BME Lab 3 - Analysis: Music & Involuntary Muscle Tension (EMG)
BioRadio + Biocapture data

INSTRUCTIONS:
  1. Export your Biocapture recording:
       Open > Recording... > select file > Data > Export to .CSV File
  2. Transfer the CSV to this folder.
  3. Update INPUT_FILE and PHASE_START_S below.
  4. Run:  python3 analyze_emg.py
  5. Figures saved as PNGs for your report.

Dependencies:  pip install numpy scipy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============ CONFIGURATION ============

INPUT_FILE = "lab3_emg_data.csv"

# Column indices in the Biocapture CSV (0-indexed).
# Adjust if your CSV has different columns.
TIME_COL = 0       # Column for timestamps (seconds)
CH1_COL  = 1       # Column for CH1: Forearm EMG
CH2_COL  = 2       # Column for CH2: Trapezius EMG

CH1_NAME = "Forearm"
CH2_NAME = "Trapezius"

# ============ EDIT THESE TIMESTAMPS (seconds from recording start) ============
# Convert your stopwatch times (MM:SS) to total seconds.

PHASE_START_S = [
    0,     # 1  Baseline           (3 min)
    180,   # 2  Recovery 1          (2 min)
    300,   # 3  Calm music          (3 min)
    480,   # 4  Recovery 2          (2 min)
    600,   # 5  Intense music       (3 min)
    780,   # 6  Recovery 3          (2 min)
    900,   # 7  Happy music         (3 min)
    1080,  # 8  Recovery 4          (2 min)
    1200,  # 9  END marker
]

PHASE_NAMES = [
    "Baseline",      "Recovery 1",
    "Calm Music",    "Recovery 2",
    "Intense Music", "Recovery 3",
    "Happy Music",   "Recovery 4",
]

CONDITION_IDX = [0, 2, 4, 6]  # Baseline, Calm, Intense, Happy
CONDITION_COLORS = ["#B0B0B0", "#3399FF", "#FF4D4D", "#FFCC33"]

# EMG processing parameters
EMG_BANDPASS_LO = 20     # Hz - high-pass to remove motion artifact
EMG_BANDPASS_HI = 450    # Hz - low-pass below Nyquist (Fs/2 = 1000 Hz)
RMS_WINDOW_S    = 0.25   # 250 ms sliding window for RMS envelope
# ==============================================================================


def bandpass_filter(signal, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_rms_envelope(signal, window_samples):
    """Sliding-window RMS of a signal."""
    squared = signal ** 2
    # Use a simple moving average of the squared signal, then sqrt
    kernel = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return rms


def main():
    # --- Load data ---
    # Try to read with header; fall back to headerless
    try:
        df = pd.read_csv(INPUT_FILE)
        # If first column looks numeric, it's probably headerless
        if df.columns[0].replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(INPUT_FILE, header=None)
    except Exception:
        df = pd.read_csv(INPUT_FILE, header=None)

    time_s  = df.iloc[:, TIME_COL].values.astype(float)
    ch1_raw = df.iloc[:, CH1_COL].values.astype(float)
    ch2_raw = df.iloc[:, CH2_COL].values.astype(float)

    # Make time start from 0
    time_s = time_s - time_s[0]

    fs = int(round(1.0 / np.median(np.diff(time_s))))
    n_phases = len(PHASE_NAMES)
    print(f"Loaded {len(df)} samples | Fs ~ {fs} Hz | Duration ~ {time_s[-1]:.0f} s")
    print(f"Channels: {CH1_NAME} (col {CH1_COL}), {CH2_NAME} (col {CH2_COL})\n")

    # --- EMG processing ---
    # Bandpass filter (20-450 Hz)
    ch1_filt = bandpass_filter(ch1_raw, EMG_BANDPASS_LO, EMG_BANDPASS_HI, fs)
    ch2_filt = bandpass_filter(ch2_raw, EMG_BANDPASS_LO, EMG_BANDPASS_HI, fs)

    # Full-wave rectification
    ch1_rect = np.abs(ch1_filt)
    ch2_rect = np.abs(ch2_filt)

    # RMS envelope
    win = int(RMS_WINDOW_S * fs)
    ch1_rms = compute_rms_envelope(ch1_filt, win)
    ch2_rms = compute_rms_envelope(ch2_filt, win)

    # --- Per-phase statistics ---
    print(f"{'Phase':<20s}  {CH1_NAME+' RMS (mV)':>16s}  {CH2_NAME+' RMS (mV)':>16s}")
    print("-" * 56)

    mean_rms1 = np.zeros(n_phases)
    std_rms1  = np.zeros(n_phases)
    mean_rms2 = np.zeros(n_phases)
    std_rms2  = np.zeros(n_phases)

    for i in range(n_phases):
        t0 = PHASE_START_S[i]
        t1 = PHASE_START_S[i + 1]
        mask = (time_s >= t0) & (time_s < t1)

        # Convert to mV for display (BioRadio outputs in V)
        rms1_phase = ch1_rms[mask] * 1000
        rms2_phase = ch2_rms[mask] * 1000

        mean_rms1[i] = np.mean(rms1_phase)
        std_rms1[i]  = np.std(rms1_phase)
        mean_rms2[i] = np.mean(rms2_phase)
        std_rms2[i]  = np.std(rms2_phase)

        print(
            f"{PHASE_NAMES[i]:<20s}  "
            f"{mean_rms1[i]:6.3f} +/- {std_rms1[i]:<6.3f}  "
            f"{mean_rms2[i]:6.3f} +/- {std_rms2[i]:<6.3f}"
        )

    # ===== FIGURE 1: Full time-series (RMS envelopes) =====
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    # Downsample for plotting (every 20th point) to keep file size reasonable
    ds = 20
    ax1.plot(time_s[::ds], ch1_rms[::ds] * 1000, "r-", linewidth=0.5)
    ax1.set_ylabel(f"{CH1_NAME} RMS (mV)")
    ax1.set_title(f"{CH1_NAME} EMG - RMS Envelope")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_s[::ds], ch2_rms[::ds] * 1000, "b-", linewidth=0.5)
    ax2.set_ylabel(f"{CH2_NAME} RMS (mV)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"{CH2_NAME} EMG - RMS Envelope")
    ax2.grid(True, alpha=0.3)

    # Phase markers (after plotting so limits are set)
    for ax in [ax1, ax2]:
        ylo, yhi = ax.get_ylim()
        for i in range(n_phases):
            ax.axvline(PHASE_START_S[i], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(
                PHASE_START_S[i] + 5, yhi - 0.05 * (yhi - ylo),
                PHASE_NAMES[i], fontsize=6, rotation=90,
                va="top", ha="left", color="dimgray",
            )

    fig1.tight_layout()
    fig1.savefig("fig1_emg_timeseries.png", dpi=200)
    print("\nSaved fig1_emg_timeseries.png")

    # ===== FIGURE 2: Bar-chart comparison =====
    cond_labels = [PHASE_NAMES[i] for i in CONDITION_IDX]

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax3.bar(range(4), mean_rms1[CONDITION_IDX], color=CONDITION_COLORS,
            edgecolor="black", linewidth=0.5)
    ax3.errorbar(range(4), mean_rms1[CONDITION_IDX], yerr=std_rms1[CONDITION_IDX],
                 fmt="none", ecolor="black", capsize=6)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(cond_labels, rotation=25, ha="right")
    ax3.set_ylabel("Mean RMS (mV)")
    ax3.set_title(f"{CH1_NAME} Muscle Tension")
    ax3.grid(True, axis="y", alpha=0.3)

    ax4.bar(range(4), mean_rms2[CONDITION_IDX], color=CONDITION_COLORS,
            edgecolor="black", linewidth=0.5)
    ax4.errorbar(range(4), mean_rms2[CONDITION_IDX], yerr=std_rms2[CONDITION_IDX],
                 fmt="none", ecolor="black", capsize=6)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(cond_labels, rotation=25, ha="right")
    ax4.set_ylabel("Mean RMS (mV)")
    ax4.set_title(f"{CH2_NAME} Muscle Tension")
    ax4.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("fig2_emg_comparison.png", dpi=200)
    print("Saved fig2_emg_comparison.png")

    # ===== FIGURE 3: Raw EMG waveform snippet (2 seconds from each condition) =====
    fig3, axes = plt.subplots(4, 2, figsize=(12, 8))
    fig3.suptitle("Raw Filtered EMG Snippets (2 s from each condition)", fontsize=12)

    for row, ci in enumerate(CONDITION_IDX):
        t0 = PHASE_START_S[ci] + 60  # 60s into the phase (settled)
        t1 = t0 + 2                  # 2-second window
        mask = (time_s >= t0) & (time_s <= t1)

        axes[row, 0].plot(time_s[mask], ch1_filt[mask] * 1000, "r-", linewidth=0.4)
        axes[row, 0].set_ylabel(f"{PHASE_NAMES[ci]}\n(mV)", fontsize=8)
        axes[row, 0].set_ylim(-0.5, 0.5)  # consistent scale
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title(CH1_NAME)

        axes[row, 1].plot(time_s[mask], ch2_filt[mask] * 1000, "b-", linewidth=0.4)
        axes[row, 1].set_ylim(-0.5, 0.5)
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title(CH2_NAME)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig3.tight_layout()
    fig3.savefig("fig3_emg_snippets.png", dpi=200)
    print("Saved fig3_emg_snippets.png")

    # ===== FIGURE 4: Frequency spectrum per condition =====
    fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle("EMG Power Spectrum by Condition", fontsize=12)

    for ci, color, label in zip(
        CONDITION_IDX, CONDITION_COLORS, cond_labels
    ):
        t0 = PHASE_START_S[ci]
        t1 = PHASE_START_S[ci + 1]
        mask = (time_s >= t0) & (time_s < t1)

        # Compute power spectrum
        for ax, data, name in [
            (ax5, ch1_filt[mask], CH1_NAME),
            (ax6, ch2_filt[mask], CH2_NAME),
        ]:
            n = len(data)
            fft_vals = np.fft.rfft(data)
            power = (np.abs(fft_vals) ** 2) / n
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)

            # Smooth with moving average for cleaner plot
            win_smooth = 50
            if len(power) > win_smooth:
                kernel = np.ones(win_smooth) / win_smooth
                power_smooth = np.convolve(power, kernel, mode="same")
            else:
                power_smooth = power

            freq_mask = (freqs >= 20) & (freqs <= 450)
            ax.semilogy(freqs[freq_mask], power_smooth[freq_mask],
                        color=color, linewidth=0.8, label=label, alpha=0.8)

    for ax, name in [(ax5, CH1_NAME), (ax6, CH2_NAME)]:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (V^2/Hz)")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig("fig4_emg_spectrum.png", dpi=200)
    print("Saved fig4_emg_spectrum.png")

    print("\n=== Analysis complete ===")
    plt.show()


if __name__ == "__main__":
    main()
