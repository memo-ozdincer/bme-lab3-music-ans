#!/usr/bin/env python3
"""
BME Lab 3 - Serial Data Acquisition from Arduino
Reads PPG + GSR data and saves to CSV with a live plot.

BEFORE RUNNING:
  1. Upload dual_sensor.ino to Arduino via Arduino IDE.
  2. CLOSE the Arduino IDE Serial Monitor.
  3. Install dependencies:  pip install pyserial matplotlib
  4. Update SERIAL_PORT below (check Arduino IDE > Tools > Port).
  5. Run:  python3 acquire_data.py
  6. Press Ctrl+C to stop -- data auto-saves.
"""

import serial
import time
import csv
import sys
import signal
import matplotlib.pyplot as plt
from collections import deque

# =================== CONFIGURATION ===================
SERIAL_PORT = "/dev/cu.usbmodem14101"   # <-- CHANGE THIS
# Mac:     "/dev/cu.usbmodem14101" or similar (check Arduino IDE > Tools > Port)
# Windows: "COM3", "COM4", etc.
BAUD_RATE = 115200
OUTPUT_FILE = "lab3_raw_data.csv"
MAX_DURATION_S = 30 * 60   # 30-minute safety cutoff
PLOT_WINDOW_S = 30          # show last 30 seconds in live plot
# ======================================================

# Storage
all_data = []
plot_times = deque(maxlen=3000)   # ~30s at 100Hz
plot_ppg = deque(maxlen=3000)
plot_gsr = deque(maxlen=3000)


def save_data():
    """Save collected data to CSV."""
    if not all_data:
        print("\nNo data collected.")
        return
    print(f"\nSaving {len(all_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "ppg", "gsr"])
        writer.writerows(all_data)
    print(f"Saved successfully to {OUTPUT_FILE}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    save_data()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"\nERROR: Could not open {SERIAL_PORT}")
        print(f"  {e}")
        print("\nTroubleshooting:")
        print("  - Is the Arduino plugged in?")
        print("  - Is Arduino IDE Serial Monitor closed?")
        print("  - Check the port name in Arduino IDE > Tools > Port")
        sys.exit(1)

    time.sleep(2)  # Wait for Arduino to reset
    ser.readline()  # Discard CSV header from Arduino

    # Set up live plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle("Live Biosignals (close window or Ctrl+C to stop)")

    line_ppg, = ax1.plot([], [], "r-", linewidth=0.8)
    ax1.set_ylabel("PPG (raw ADC)")
    ax1.set_title("PulseSensor")
    ax1.grid(True, alpha=0.3)

    line_gsr, = ax2.plot([], [], "b-", linewidth=0.8)
    ax2.set_ylabel("GSR (raw ADC)")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_title("Electrodermal Activity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    start_time = time.time()
    sample_count = 0
    last_plot_update = 0

    print(f"Recording started at {time.strftime('%H:%M:%S')}")
    print(f"Press Ctrl+C to stop. Data saves to {OUTPUT_FILE}\n")

    try:
        while (time.time() - start_time) < MAX_DURATION_S:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                parts = line.split(",")

                if len(parts) == 3:
                    try:
                        t_ms = int(parts[0])
                        ppg = int(parts[1])
                        gsr = int(parts[2])
                    except ValueError:
                        continue

                    all_data.append([t_ms, ppg, gsr])
                    sample_count += 1
                    elapsed = time.time() - start_time

                    plot_times.append(elapsed)
                    plot_ppg.append(ppg)
                    plot_gsr.append(gsr)

                    # Update live plot every 0.2s
                    if elapsed - last_plot_update > 0.2:
                        last_plot_update = elapsed
                        t_list = list(plot_times)
                        line_ppg.set_data(t_list, list(plot_ppg))
                        line_gsr.set_data(t_list, list(plot_gsr))

                        if t_list:
                            xmin = max(0, t_list[-1] - PLOT_WINDOW_S)
                            xmax = t_list[-1] + 1
                            ax1.set_xlim(xmin, xmax)
                            ax2.set_xlim(xmin, xmax)
                            ax1.set_ylim(
                                min(plot_ppg) - 20, max(plot_ppg) + 20
                            )
                            ax2.set_ylim(
                                min(plot_gsr) - 20, max(plot_gsr) + 20
                            )

                        try:
                            fig.canvas.draw_idle()
                            fig.canvas.flush_events()
                        except Exception:
                            pass  # Plot window closed

                    # Print status every 10s
                    if sample_count % 1000 == 0:
                        print(
                            f"  {elapsed:6.0f}s | {sample_count} samples"
                        )

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        save_data()
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
