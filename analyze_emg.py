#!/usr/bin/env python3
"""
BME Lab 3 - Analysis: Pushup Hand Width & Muscle Recruitment (EMG)
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
TIME_COL = 0       # Column for timestamps (seconds)
CH1_COL  = 1       # Column for CH1: Triceps
CH2_COL  = 2       # Column for CH2: Pectoralis

CH1_NAME = "Triceps"
CH2_NAME = "Pectoralis"

# ============ EDIT THESE TIMESTAMPS (seconds from recording start) ============
# Convert your stopwatch times (MM:SS) to total seconds.
# Each pushup set is ~30s.  Adjust the END of each set to when the
# subject actually finished (not the scheduled time).

PHASE_START_S = [
    0,     #  1  Baseline rest         (1 min)
    60,    #  2  Narrow pushups        (~30 s)
    90,    #  3  Rest                  (2 min)
    210,   #  4  Standard pushups      (~30 s)
    240,   #  5  Rest                  (2 min)
    360,   #  6  Wide pushups          (~30 s)
    390,   #  7  Cooldown rest         (1 min)
    450,   #  8  END marker
]

PHASE_NAMES = [
    "Baseline",
    "Narrow",
    "Rest 1",
    "Standard",
    "Rest 2",
    "Wide",
    "Cooldown",
]

# The 3 pushup conditions to compare (indices into PHASE_NAMES)
CONDITION_IDX    = [1, 3, 5]             # Narrow, Standard, Wide
CONDITION_LABELS = ["Narrow", "Standard", "Wide"]
CONDITION_COLORS = ["#FF4D4D", "#3399FF", "#FFCC33"]

# Also include baseline for reference in some plots
ALL_COMPARE_IDX    = [0, 1, 3, 5]
ALL_COMPARE_LABELS = ["Baseline", "Narrow", "Standard", "Wide"]
ALL_COMPARE_COLORS = ["#B0B0B0", "#FF4D4D", "#3399FF", "#FFCC33"]

# EMG processing parameters
EMG_BANDPASS_LO = 20     # Hz
EMG_BANDPASS_HI = 450    # Hz
RMS_WINDOW_S    = 0.25   # 250 ms sliding window
# ==============================================================================


def bandpass_filter(signal, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_rms_envelope(signal, window_samples):
    """Sliding-window RMS."""
    squared = signal ** 2
    kernel = np.ones(window_samples) / window_samples
    return np.sqrt(np.convolve(squared, kernel, mode="same"))


def main():
    # --- Load data ---
    try:
        df = pd.read_csv(INPUT_FILE)
        first_col = str(df.columns[0])
        if first_col.replace(".", "").replace("-", "").replace("e", "").isdigit():
            df = pd.read_csv(INPUT_FILE, header=None)
    except Exception:
        df = pd.read_csv(INPUT_FILE, header=None)

    time_s  = df.iloc[:, TIME_COL].values.astype(float)
    ch1_raw = df.iloc[:, CH1_COL].values.astype(float)
    ch2_raw = df.iloc[:, CH2_COL].values.astype(float)

    time_s = time_s - time_s[0]
    fs = int(round(1.0 / np.median(np.diff(time_s))))
    n_phases = len(PHASE_NAMES)
    print(f"Loaded {len(df)} samples | Fs ~ {fs} Hz | Duration ~ {time_s[-1]:.0f} s")
    print(f"Channels: {CH1_NAME} (col {CH1_COL}), {CH2_NAME} (col {CH2_COL})\n")

    # --- EMG processing ---
    ch1_filt = bandpass_filter(ch1_raw, EMG_BANDPASS_LO, EMG_BANDPASS_HI, fs)
    ch2_filt = bandpass_filter(ch2_raw, EMG_BANDPASS_LO, EMG_BANDPASS_HI, fs)

    win = int(RMS_WINDOW_S * fs)
    ch1_rms = compute_rms_envelope(ch1_filt, win)
    ch2_rms = compute_rms_envelope(ch2_filt, win)

    # --- Per-phase statistics (in mV) ---
    print(f"{'Phase':<16s}  {CH1_NAME+' RMS (mV)':>16s}  {CH2_NAME+' RMS (mV)':>16s}  {'Peak '+CH1_NAME:>14s}  {'Peak '+CH2_NAME:>14s}")
    print("-" * 80)

    mean_rms1 = np.zeros(n_phases)
    std_rms1  = np.zeros(n_phases)
    mean_rms2 = np.zeros(n_phases)
    std_rms2  = np.zeros(n_phases)
    peak_rms1 = np.zeros(n_phases)
    peak_rms2 = np.zeros(n_phases)

    for i in range(n_phases):
        t0 = PHASE_START_S[i]
        t1 = PHASE_START_S[i + 1]
        mask = (time_s >= t0) & (time_s < t1)

        r1 = ch1_rms[mask] * 1000  # mV
        r2 = ch2_rms[mask] * 1000

        mean_rms1[i] = np.mean(r1)
        std_rms1[i]  = np.std(r1)
        mean_rms2[i] = np.mean(r2)
        std_rms2[i]  = np.std(r2)
        peak_rms1[i] = np.max(r1) if len(r1) > 0 else 0
        peak_rms2[i] = np.max(r2) if len(r2) > 0 else 0

        print(
            f"{PHASE_NAMES[i]:<16s}  "
            f"{mean_rms1[i]:6.3f} +/- {std_rms1[i]:<6.3f}  "
            f"{mean_rms2[i]:6.3f} +/- {std_rms2[i]:<6.3f}  "
            f"{peak_rms1[i]:8.3f} mV  "
            f"{peak_rms2[i]:8.3f} mV"
        )

    # ===== FIGURE 1: Full time-series (RMS envelopes) =====
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ds = max(1, len(time_s) // 50000)  # downsample for plotting
    ax1.plot(time_s[::ds], ch1_rms[::ds] * 1000, "r-", linewidth=0.5)
    ax1.set_ylabel(f"{CH1_NAME} RMS (mV)")
    ax1.set_title(f"{CH1_NAME} EMG - RMS Envelope")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_s[::ds], ch2_rms[::ds] * 1000, "b-", linewidth=0.5)
    ax2.set_ylabel(f"{CH2_NAME} RMS (mV)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"{CH2_NAME} EMG - RMS Envelope")
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        ylo, yhi = ax.get_ylim()
        for i in range(n_phases):
            ax.axvline(PHASE_START_S[i], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(
                PHASE_START_S[i] + 2, yhi - 0.05 * (yhi - ylo),
                PHASE_NAMES[i], fontsize=7, rotation=90,
                va="top", ha="left", color="dimgray",
            )

    fig1.tight_layout()
    fig1.savefig("fig1_emg_timeseries.png", dpi=200)
    print("\nSaved fig1_emg_timeseries.png")

    # ===== FIGURE 2: Bar chart -- mean RMS per pushup width =====
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Mean RMS bars
    ax3.bar(range(len(ALL_COMPARE_IDX)),
            mean_rms1[ALL_COMPARE_IDX], color=ALL_COMPARE_COLORS,
            edgecolor="black", linewidth=0.5)
    ax3.errorbar(range(len(ALL_COMPARE_IDX)),
                 mean_rms1[ALL_COMPARE_IDX], yerr=std_rms1[ALL_COMPARE_IDX],
                 fmt="none", ecolor="black", capsize=6)
    ax3.set_xticks(range(len(ALL_COMPARE_IDX)))
    ax3.set_xticklabels(ALL_COMPARE_LABELS, rotation=20, ha="right")
    ax3.set_ylabel("Mean RMS (mV)")
    ax3.set_title(f"{CH1_NAME} Activation")
    ax3.grid(True, axis="y", alpha=0.3)

    ax4.bar(range(len(ALL_COMPARE_IDX)),
            mean_rms2[ALL_COMPARE_IDX], color=ALL_COMPARE_COLORS,
            edgecolor="black", linewidth=0.5)
    ax4.errorbar(range(len(ALL_COMPARE_IDX)),
                 mean_rms2[ALL_COMPARE_IDX], yerr=std_rms2[ALL_COMPARE_IDX],
                 fmt="none", ecolor="black", capsize=6)
    ax4.set_xticks(range(len(ALL_COMPARE_IDX)))
    ax4.set_xticklabels(ALL_COMPARE_LABELS, rotation=20, ha="right")
    ax4.set_ylabel("Mean RMS (mV)")
    ax4.set_title(f"{CH2_NAME} Activation")
    ax4.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("fig2_emg_comparison.png", dpi=200)
    print("Saved fig2_emg_comparison.png")

    # ===== FIGURE 3: Grouped bar -- both muscles side by side =====
    fig3, ax5 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDITION_LABELS))
    width = 0.35

    bars1 = ax5.bar(x - width / 2, mean_rms1[CONDITION_IDX], width,
                    color="#E06060", edgecolor="black", linewidth=0.5,
                    label=CH1_NAME)
    ax5.errorbar(x - width / 2, mean_rms1[CONDITION_IDX],
                 yerr=std_rms1[CONDITION_IDX], fmt="none", ecolor="black", capsize=5)

    bars2 = ax5.bar(x + width / 2, mean_rms2[CONDITION_IDX], width,
                    color="#6090E0", edgecolor="black", linewidth=0.5,
                    label=CH2_NAME)
    ax5.errorbar(x + width / 2, mean_rms2[CONDITION_IDX],
                 yerr=std_rms2[CONDITION_IDX], fmt="none", ecolor="black", capsize=5)

    ax5.set_xticks(x)
    ax5.set_xticklabels(CONDITION_LABELS)
    ax5.set_ylabel("Mean RMS EMG (mV)")
    ax5.set_title("Muscle Recruitment by Pushup Hand Width")
    ax5.legend()
    ax5.grid(True, axis="y", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("fig3_grouped_comparison.png", dpi=200)
    print("Saved fig3_grouped_comparison.png")

    # ===== FIGURE 4: Raw EMG snippets -- one rep from each width =====
    fig4, axes = plt.subplots(3, 2, figsize=(12, 7))
    fig4.suptitle("Raw Filtered EMG During One Pushup Rep (2 s snippet)", fontsize=12)

    for row, ci in enumerate(CONDITION_IDX):
        t0 = PHASE_START_S[ci] + 5  # 5s into the set (mid-rep)
        t1 = t0 + 2
        mask = (time_s >= t0) & (time_s <= t1)

        # Auto-scale based on data in this window
        ch1_snippet = ch1_filt[mask] * 1000
        ch2_snippet = ch2_filt[mask] * 1000
        ylim = max(np.max(np.abs(ch1_snippet)), np.max(np.abs(ch2_snippet))) * 1.2
        ylim = max(ylim, 0.1)  # minimum scale

        axes[row, 0].plot(time_s[mask], ch1_snippet, "r-", linewidth=0.5)
        axes[row, 0].set_ylabel(f"{CONDITION_LABELS[row]}\n{CH1_NAME} (mV)", fontsize=8)
        axes[row, 0].set_ylim(-ylim, ylim)
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title(CH1_NAME)

        axes[row, 1].plot(time_s[mask], ch2_snippet, "b-", linewidth=0.5)
        axes[row, 1].set_ylim(-ylim, ylim)
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title(CH2_NAME)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig4.tight_layout()
    fig4.savefig("fig4_emg_snippets.png", dpi=200)
    print("Saved fig4_emg_snippets.png")

    # ===== FIGURE 5: Peak RMS comparison =====
    fig5, ax6 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDITION_LABELS))

    ax6.bar(x - width / 2, peak_rms1[CONDITION_IDX], width,
            color="#E06060", edgecolor="black", linewidth=0.5, label=CH1_NAME)
    ax6.bar(x + width / 2, peak_rms2[CONDITION_IDX], width,
            color="#6090E0", edgecolor="black", linewidth=0.5, label=CH2_NAME)

    ax6.set_xticks(x)
    ax6.set_xticklabels(CONDITION_LABELS)
    ax6.set_ylabel("Peak RMS EMG (mV)")
    ax6.set_title("Peak Muscle Activation by Pushup Hand Width")
    ax6.legend()
    ax6.grid(True, axis="y", alpha=0.3)

    fig5.tight_layout()
    fig5.savefig("fig5_peak_comparison.png", dpi=200)
    print("Saved fig5_peak_comparison.png")

    print("\n=== Analysis complete ===")
    plt.show()


if __name__ == "__main__":
    main()
