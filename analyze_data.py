#!/usr/bin/env python3
"""
BME Lab 3 - Analysis: Music & Autonomic Nervous System

INSTRUCTIONS:
  1. Run acquire_data.py first to produce lab3_raw_data.csv
  2. Fill in PHASE_START_S below with your actual timestamps
  3. Run:  python3 analyze_data.py
  4. Figures are saved as PNGs for your report.

Dependencies:  pip install numpy scipy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ============ EDIT THESE TIMESTAMPS (seconds from recording start) ============
# Write down stopwatch times during the experiment, then enter them here.
# Convert MM:SS to total seconds (e.g., 5:00 -> 300, 10:15 -> 615).

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

# Indices of the 4 conditions to compare in bar charts
CONDITION_IDX = [0, 2, 4, 6]  # Baseline, Calm, Intense, Happy
CONDITION_COLORS = ["#B0B0B0", "#3399FF", "#FF4D4D", "#FFCC33"]

INPUT_FILE = "lab3_raw_data.csv"
# ==============================================================================


def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def gsr_to_conductance(adc):
    """Convert Grove GSR raw ADC to skin conductance (micro-siemens)."""
    adc = np.clip(adc, 0, 510)  # avoid division by zero near 512
    resistance = ((1024 + 2 * adc) * 100000) / np.maximum(512 - adc, 1)
    return 1e6 / resistance


def main():
    # --- Load data ---
    df = pd.read_csv(INPUT_FILE)
    time_s = (df["time_ms"].values - df["time_ms"].values[0]) / 1000.0
    ppg_raw = df["ppg"].values.astype(float)
    gsr_raw = df["gsr"].values.astype(float)
    fs = int(round(1.0 / np.median(np.diff(time_s))))
    print(f"Loaded {len(df)} samples | Fs ~ {fs} Hz | Duration ~ {time_s[-1]:.0f} s\n")

    # --- Convert GSR -> skin conductance ---
    eda_uS = gsr_to_conductance(gsr_raw)

    # --- Filter PPG and detect peaks -> Heart Rate ---
    ppg_filt = bandpass_filter(ppg_raw, 0.5, 5.0, fs)
    min_dist = int(0.4 * fs)  # 150 BPM ceiling
    prominence = 0.3 * np.std(ppg_filt)
    peaks, _ = find_peaks(ppg_filt, distance=min_dist, prominence=prominence)

    peak_times = time_s[peaks]
    rr = np.diff(peak_times)
    inst_hr = 60.0 / rr
    hr_times = peak_times[1:]

    # Remove non-physiological values
    valid = (inst_hr > 40) & (inst_hr < 180)
    inst_hr = inst_hr[valid]
    hr_times = hr_times[valid]

    # --- Per-phase statistics ---
    n_phases = len(PHASE_NAMES)
    mean_hr = np.zeros(n_phases)
    std_hr = np.zeros(n_phases)
    mean_eda = np.zeros(n_phases)
    std_eda = np.zeros(n_phases)
    rmssd = np.zeros(n_phases)

    print(f"{'Phase':<20s}  {'HR (BPM)':>12s}  {'EDA (uS)':>12s}  {'RMSSD (ms)':>12s}")
    print("-" * 60)

    for i in range(n_phases):
        t0 = PHASE_START_S[i]
        t1 = PHASE_START_S[i + 1]

        # Heart rate
        mask_hr = (hr_times >= t0) & (hr_times < t1)
        hr_phase = inst_hr[mask_hr]
        mean_hr[i] = np.mean(hr_phase) if len(hr_phase) > 0 else np.nan
        std_hr[i] = np.std(hr_phase) if len(hr_phase) > 0 else np.nan

        # EDA
        mask_eda = (time_s >= t0) & (time_s < t1)
        eda_phase = eda_uS[mask_eda]
        mean_eda[i] = np.mean(eda_phase)
        std_eda[i] = np.std(eda_phase)

        # HRV (RMSSD)
        if len(hr_phase) > 2:
            rr_ms = 60000.0 / hr_phase  # convert BPM -> RR in ms
            diffs = np.diff(rr_ms)
            rmssd[i] = np.sqrt(np.mean(diffs ** 2))
        else:
            rmssd[i] = np.nan

        print(
            f"{PHASE_NAMES[i]:<20s}  "
            f"{mean_hr[i]:5.1f} +/- {std_hr[i]:<4.1f}  "
            f"{mean_eda[i]:5.2f} +/- {std_eda[i]:<4.2f}  "
            f"{rmssd[i]:5.1f}"
        )

    # ===== FIGURE 1: Full time-series =====
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax1.plot(hr_times, inst_hr, "r.-", markersize=2, linewidth=0.5)
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.set_title("Instantaneous Heart Rate")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_s, eda_uS, "b-", linewidth=0.5)
    ax2.set_ylabel("Skin Conductance (\u03bcS)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Electrodermal Activity")
    ax2.grid(True, alpha=0.3)

    # Add phase markers after data is plotted (so axis limits are set)
    for ax in [ax1, ax2]:
        ylo, yhi = ax.get_ylim()
        for i in range(n_phases):
            ax.axvline(PHASE_START_S[i], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            # Place label near top of plot area
            ax.text(
                PHASE_START_S[i] + 5, yhi - 0.05 * (yhi - ylo),
                PHASE_NAMES[i], fontsize=6, rotation=90,
                va="top", ha="left", color="dimgray",
            )

    fig1.tight_layout()
    fig1.savefig("fig1_timeseries.png", dpi=200)
    print("\nSaved fig1_timeseries.png")

    # ===== FIGURE 2: Bar-chart comparison =====
    cond_labels = [PHASE_NAMES[i] for i in CONDITION_IDX]
    cond_hr = mean_hr[CONDITION_IDX]
    cond_hr_err = std_hr[CONDITION_IDX]
    cond_eda = mean_eda[CONDITION_IDX]
    cond_eda_err = std_eda[CONDITION_IDX]

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax3.bar(range(4), cond_hr, color=CONDITION_COLORS, edgecolor="black", linewidth=0.5)
    ax3.errorbar(range(4), cond_hr, yerr=cond_hr_err, fmt="none", ecolor="black", capsize=6)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(cond_labels, rotation=25, ha="right")
    ax3.set_ylabel("Mean Heart Rate (BPM)")
    ax3.set_title("Heart Rate by Condition")
    ax3.grid(True, axis="y", alpha=0.3)

    bars2 = ax4.bar(range(4), cond_eda, color=CONDITION_COLORS, edgecolor="black", linewidth=0.5)
    ax4.errorbar(range(4), cond_eda, yerr=cond_eda_err, fmt="none", ecolor="black", capsize=6)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(cond_labels, rotation=25, ha="right")
    ax4.set_ylabel("Mean Skin Conductance (\u03bcS)")
    ax4.set_title("EDA by Condition")
    ax4.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("fig2_comparison.png", dpi=200)
    print("Saved fig2_comparison.png")

    # ===== FIGURE 3: PPG waveform snippet =====
    fig3, ax5 = plt.subplots(figsize=(9, 3))
    snip0 = PHASE_START_S[0] + 30  # 30s into baseline
    snip1 = snip0 + 10  # 10-second window
    mask = (time_s >= snip0) & (time_s <= snip1)
    ax5.plot(time_s[mask], ppg_filt[mask], "r-", linewidth=1.2, label="Filtered PPG")

    # Mark detected peaks in this window
    pk_mask = (peak_times >= snip0) & (peak_times <= snip1)
    pk_in_window = peaks[pk_mask]
    ax5.plot(
        time_s[pk_in_window], ppg_filt[pk_in_window],
        "kv", markersize=7, label="Detected peaks",
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Filtered PPG (a.u.)")
    ax5.set_title("PPG Waveform with Detected Peaks (10 s Baseline Snippet)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("fig3_ppg_waveform.png", dpi=200)
    print("Saved fig3_ppg_waveform.png")

    # ===== FIGURE 4: HRV (RMSSD) =====
    fig4, ax6 = plt.subplots(figsize=(6, 3.5))
    cond_rmssd = rmssd[CONDITION_IDX]
    ax6.bar(range(4), cond_rmssd, color=CONDITION_COLORS, edgecolor="black", linewidth=0.5)
    ax6.set_xticks(range(4))
    ax6.set_xticklabels(cond_labels, rotation=25, ha="right")
    ax6.set_ylabel("RMSSD (ms)")
    ax6.set_title("Heart-Rate Variability by Condition")
    ax6.grid(True, axis="y", alpha=0.3)

    fig4.tight_layout()
    fig4.savefig("fig4_hrv.png", dpi=200)
    print("Saved fig4_hrv.png")

    print("\n=== Analysis complete ===")
    plt.show()


if __name__ == "__main__":
    main()
