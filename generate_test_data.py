#!/usr/bin/env python3
"""
Generate synthetic PPG + GSR data that mimics a real experiment.
Produces lab3_raw_data.csv with realistic signals for each phase.
"""

import numpy as np
import csv

FS = 100  # Hz, matches Arduino sketch
DURATION_S = 1200  # 20 minutes total

# Phase boundaries (seconds) -- must match analyze_data.py defaults
# EDA_delta is SUBTRACTED from baseline ADC because:
#   lower ADC -> lower skin resistance -> higher conductance (more sweat/arousal)
PHASES = [
    (0, 180, "baseline",  72, 0.0),    # (start, end, label, HR_bpm, EDA_adc_drop)
    (180, 300, "recovery1", 72, 0.0),
    (300, 480, "calm",      68, -0.02),   # slight HR decrease, tiny EDA increase
    (480, 600, "recovery2", 71, 0.0),
    (600, 780, "intense",   82, 0.35),    # HR increase, large EDA increase (sympathetic)
    (780, 900, "recovery3", 75, 0.15),    # recovering
    (900, 1080, "happy",    77, 0.12),    # moderate HR increase, moderate EDA increase
    (1080, 1200, "recovery4", 73, 0.05),
]

BASELINE_HR = 72       # BPM
BASELINE_EDA_ADC = 350 # raw GSR ADC baseline (~middle of range)

np.random.seed(42)

n_samples = DURATION_S * FS
time_ms = np.arange(n_samples) * 10  # 10 ms intervals

# --- Build PPG signal ---
ppg = np.zeros(n_samples)
t = np.arange(n_samples) / FS  # time in seconds

for (t_start, t_end, label, hr_bpm, _) in PHASES:
    mask = (t >= t_start) & (t < t_end)
    freq = hr_bpm / 60.0  # Hz
    # Simulate PPG as a sum of harmonics with some noise
    phase_t = t[mask] - t_start
    # Add slight HR variability (HRV)
    hr_variation = np.cumsum(np.random.randn(mask.sum()) * 0.002)
    inst_freq = freq + hr_variation * 0.1
    phase_angle = np.cumsum(inst_freq / FS) * 2 * np.pi
    ppg_signal = (
        300 * np.sin(phase_angle)
        + 80 * np.sin(2 * phase_angle + 0.3)
        + 30 * np.sin(3 * phase_angle + 0.6)
    )
    ppg[mask] = 512 + ppg_signal + np.random.randn(mask.sum()) * 8

ppg = np.clip(ppg, 0, 1023).astype(int)

# --- Build GSR signal ---
gsr = np.zeros(n_samples)
for (t_start, t_end, label, _, eda_delta) in PHASES:
    mask = (t >= t_start) & (t < t_end)
    duration = t_end - t_start
    phase_t = t[mask] - t_start
    # Slow tonic drift + random SCRs (skin conductance responses)
    # Subtract delta from ADC: lower ADC = lower resistance = higher conductance
    tonic = BASELINE_EDA_ADC - eda_delta * 200 * (1 - np.exp(-phase_t / 30))
    # Add random SCR events (phasic component)
    scr = np.zeros(mask.sum())
    n_scrs = int(duration / 15)  # roughly one SCR every 15s
    if eda_delta > 0:
        n_scrs = int(duration / 8)  # more SCRs during arousal
    for _ in range(n_scrs):
        onset = np.random.randint(0, mask.sum())
        amplitude = np.random.uniform(5, 25)
        rise = 15   # samples (~0.15s rise)
        decay = 200 # samples (~2s decay)
        for j in range(min(decay, mask.sum() - onset)):
            if j < rise:
                scr[onset + j] += amplitude * (j / rise)
            else:
                scr[onset + j] += amplitude * np.exp(-(j - rise) / 80)

    gsr[mask] = tonic + scr + np.random.randn(mask.sum()) * 2

gsr = np.clip(gsr, 0, 510).astype(int)

# --- Write CSV ---
with open("lab3_raw_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ms", "ppg", "gsr"])
    for i in range(n_samples):
        writer.writerow([int(time_ms[i]), int(ppg[i]), int(gsr[i])])

print(f"Generated {n_samples} samples ({DURATION_S}s at {FS}Hz)")
print(f"Saved to lab3_raw_data.csv ({n_samples} rows)")
print(f"\nPhase summary:")
for (t_start, t_end, label, hr, eda_d) in PHASES:
    print(f"  {label:12s}  {t_start:4d}-{t_end:4d}s  target HR={hr} BPM  EDA delta={eda_d:+.2f}")
