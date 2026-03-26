#!/usr/bin/env python3
"""
Generate synthetic BioRadio EMG data for testing analyze_emg.py.
Simulates forearm + trapezius resting EMG with music-modulated tension.
Outputs lab3_emg_data.csv in Biocapture export format.
"""

import numpy as np
import csv

FS = 2000  # BioRadio sampling rate (2 kHz)
DURATION_S = 1200  # 20 minutes

# Phase boundaries: (start, end, label, forearm_tension_factor, trap_tension_factor)
# Tension factor: 1.0 = baseline resting tone, >1 = increased tension
PHASES = [
    (0,    180,  "baseline",   1.0,  1.0),
    (180,  300,  "recovery1",  1.0,  1.0),
    (300,  480,  "calm",       0.85, 0.80),  # relaxation -> lower tension
    (480,  600,  "recovery2",  0.95, 0.95),
    (600,  780,  "intense",    1.35, 1.50),  # arousal -> higher tension, especially trapezius
    (780,  900,  "recovery3",  1.10, 1.15),
    (900,  1080, "happy",      1.10, 1.12),  # slight increase
    (1080, 1200, "recovery4",  1.02, 1.03),
]

# Baseline resting EMG RMS amplitude (in volts)
# Typical surface EMG resting: 5-30 uV RMS
BASELINE_RMS_V = 0.015e-3  # 15 uV -> 0.000015 V

np.random.seed(42)

n_samples = DURATION_S * FS
time_s = np.arange(n_samples) / FS

ch1 = np.zeros(n_samples)  # Forearm
ch2 = np.zeros(n_samples)  # Trapezius

for (t_start, t_end, label, f_factor, t_factor) in PHASES:
    mask = (time_s >= t_start) & (time_s < t_end)
    n = mask.sum()

    # EMG is modeled as band-limited Gaussian noise (20-450 Hz)
    # with amplitude proportional to tension factor

    # Generate broadband noise
    noise1 = np.random.randn(n)
    noise2 = np.random.randn(n)

    # Scale by tension factor
    ch1[mask] = noise1 * BASELINE_RMS_V * f_factor
    ch2[mask] = noise2 * BASELINE_RMS_V * t_factor

    # Add occasional motor unit action potential bursts (simulates micro-movements)
    n_bursts = int((t_end - t_start) / 20)  # ~1 burst per 20s
    if f_factor > 1.2:
        n_bursts *= 3  # more bursts during high tension
    for _ in range(n_bursts):
        onset = np.random.randint(0, n)
        dur = np.random.randint(50, 200)  # 25-100 ms
        amp = np.random.uniform(2, 5) * BASELINE_RMS_V * f_factor
        end_idx = min(onset + dur, n)
        burst = np.random.randn(end_idx - onset) * amp
        ch1[mask][onset:end_idx] += burst * 0.7

    for _ in range(n_bursts):
        onset = np.random.randint(0, n)
        dur = np.random.randint(50, 200)
        amp = np.random.uniform(2, 5) * BASELINE_RMS_V * t_factor
        end_idx = min(onset + dur, n)
        burst = np.random.randn(end_idx - onset) * amp
        ch2[mask][onset:end_idx] += burst * 0.8

# Add 60 Hz powerline noise (small amount, as BioRadio has some rejection)
t_full = np.arange(n_samples) / FS
powerline = 0.5e-6 * np.sin(2 * np.pi * 60 * t_full)
ch1 += powerline
ch2 += powerline * 0.8

# Write CSV in Biocapture export format
with open("lab3_emg_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "CH1: Forearm", "CH2: Trapezius"])
    for i in range(n_samples):
        writer.writerow([f"{time_s[i]:.6f}", f"{ch1[i]:.9f}", f"{ch2[i]:.9f}"])

print(f"Generated {n_samples} samples ({DURATION_S}s at {FS} Hz)")
print(f"Saved to lab3_emg_data.csv")
print(f"\nPhase summary:")
for (t0, t1, label, ff, tf) in PHASES:
    print(f"  {label:12s}  {t0:4d}-{t1:4d}s  forearm={ff:.2f}x  trap={tf:.2f}x")
