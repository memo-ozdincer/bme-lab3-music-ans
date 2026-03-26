#!/usr/bin/env python3
"""
Generate synthetic BioRadio EMG data for pushup hand-width experiment.
Simulates triceps + pectoralis EMG with width-dependent recruitment.
"""

import numpy as np
import csv

FS = 2000
DURATION_S = 450  # ~7.5 minutes

# Phase layout:
#   Baseline 60s -> Narrow 30s -> Rest 120s -> Standard 30s -> Rest 120s -> Wide 30s -> Cooldown 60s
PHASES = [
    # (start, end, label, triceps_amplitude, pec_amplitude)
    # Amplitudes in mV RMS.  Resting ~ 0.01 mV.  Active pushup ~ 0.3-1.5 mV.
    (0,   60,   "baseline",  0.01,  0.01),
    (60,  90,   "narrow",    1.20,  0.40),   # Narrow: high triceps, low pec
    (90,  210,  "rest1",     0.01,  0.01),
    (210, 240,  "standard",  0.80,  0.80),   # Standard: balanced
    (240, 360,  "rest2",     0.01,  0.01),
    (360, 390,  "wide",      0.45,  1.30),   # Wide: low triceps, high pec
    (390, 450,  "cooldown",  0.01,  0.01),
]

np.random.seed(42)

n_samples = DURATION_S * FS
time_s = np.arange(n_samples) / FS
ch1 = np.zeros(n_samples)  # Triceps
ch2 = np.zeros(n_samples)  # Pectoralis

for (t0, t1, label, tri_amp, pec_amp) in PHASES:
    mask = (time_s >= t0) & (time_s < t1)
    n = mask.sum()

    # Convert mV to V
    tri_v = tri_amp / 1000.0
    pec_v = pec_amp / 1000.0

    if tri_amp > 0.05:  # Active pushup phase
        # Simulate individual pushup reps (~5 reps in 30s)
        phase_t = time_s[mask] - t0
        rep_duration = 4.0   # seconds per rep
        rest_between = 2.0   # seconds between reps

        tri_signal = np.random.randn(n) * tri_v * 0.3  # baseline noise
        pec_signal = np.random.randn(n) * pec_v * 0.3

        for rep in range(5):
            rep_start = rep * (rep_duration + rest_between)
            rep_end = rep_start + rep_duration

            # Concentric phase (pushing up): first 2s, higher activation
            conc_mask = (phase_t >= rep_start) & (phase_t < rep_start + 2.0)
            tri_signal[conc_mask] += np.random.randn(conc_mask.sum()) * tri_v * 0.9
            pec_signal[conc_mask] += np.random.randn(conc_mask.sum()) * pec_v * 0.9

            # Eccentric phase (lowering): next 2s, moderate activation
            ecc_mask = (phase_t >= rep_start + 2.0) & (phase_t < rep_end)
            tri_signal[ecc_mask] += np.random.randn(ecc_mask.sum()) * tri_v * 0.6
            pec_signal[ecc_mask] += np.random.randn(ecc_mask.sum()) * pec_v * 0.6

        ch1[mask] = tri_signal
        ch2[mask] = pec_signal
    else:
        # Resting phase
        ch1[mask] = np.random.randn(n) * tri_v
        ch2[mask] = np.random.randn(n) * pec_v

# Add small 60 Hz powerline noise
powerline = 0.5e-6 * np.sin(2 * np.pi * 60 * time_s)
ch1 += powerline
ch2 += powerline * 0.8

# Write CSV
with open("lab3_emg_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "CH1: Triceps", "CH2: Pectoralis"])
    for i in range(n_samples):
        writer.writerow([f"{time_s[i]:.6f}", f"{ch1[i]:.9f}", f"{ch2[i]:.9f}"])

print(f"Generated {n_samples} samples ({DURATION_S}s at {FS} Hz)")
print(f"Saved to lab3_emg_data.csv")
print(f"\nPhase summary:")
for (t0, t1, label, tri, pec) in PHASES:
    print(f"  {label:12s}  {t0:4d}-{t1:4d}s  triceps={tri:.2f} mV  pec={pec:.2f} mV")
