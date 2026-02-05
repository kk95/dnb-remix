#!/usr/bin/env python3
"""
Analyze the flute phrase structure in Saathiya's "other" stem
to find the natural loop point (where the trail-off riff ends).
"""

import numpy as np
import soundfile as sf

PROJECT = "/Users/kshitijkarke/Documents/dnb-remix"
SR = 44100
SNAP_S = 307.454
BEAT_S = 60.0 / 88.0  # 0.6818s
BAR_S = 4 * BEAT_S     # 2.7273s

audio, _ = sf.read(f"{PROJECT}/stems/htdemucs_ft/saathiya/other.wav", dtype='float64')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Analyze 0-5 bars from snap point (covers well past 4 bars)
start = int(SNAP_S * SR)
window = int(5 * BAR_S * SR)
region = audio[start:start + window]

# RMS in short windows (50ms = ~1/14 of a beat)
hop = int(0.05 * SR)  # 50ms
rms = []
times = []
for i in range(0, len(region) - hop, hop):
    chunk = region[i:i+hop]
    r = np.sqrt(np.mean(chunk**2))
    rms.append(r)
    times.append(i / SR)

rms = np.array(rms)
times = np.array(times)

# Normalize
rms_norm = rms / np.max(rms) if np.max(rms) > 0 else rms

print("Flute energy from snap point (307.454s)")
print(f"{'Time':>6s} {'Bar':>5s} {'Beat':>5s}  {'RMS':>5s}  Visual")
print("─" * 60)

for i, (t, r) in enumerate(zip(times, rms_norm)):
    bar = t / BAR_S
    beat = (t % BAR_S) / BEAT_S
    bar_int = int(bar) + 1
    beat_frac = beat + 1

    # Only print every 2nd entry (100ms resolution) for readability
    if i % 2 == 0:
        bar_str = f"{bar:.2f}"
        visual = '█' * int(r * 40)
        marker = ""
        # Mark bar boundaries
        if abs(beat) < 0.15 or abs(beat - 4) < 0.15:
            marker = f" ← BAR {bar_int}"
        print(f"{t:6.2f}s {bar_str:>5s} {beat_frac:5.1f}  {r:.3f}  {visual}{marker}")

# Find where energy drops below threshold
threshold = 0.15  # 15% of peak
print(f"\n── Energy drop analysis (threshold={threshold*100:.0f}% of peak) ──")

# Look for sustained drop (not just a momentary dip)
sustained_drop = None
for i in range(len(rms_norm)):
    if rms_norm[i] < threshold:
        # Check if it stays low for at least 200ms (4 entries)
        if i + 4 < len(rms_norm) and all(rms_norm[i:i+4] < threshold):
            sustained_drop = times[i]
            bar = sustained_drop / BAR_S
            print(f"Energy drops below {threshold*100:.0f}% at {sustained_drop:.3f}s "
                  f"(bar {bar:.2f}, {sustained_drop/BEAT_S:.1f} beats from snap)")
            break

if sustained_drop is None:
    print("No sustained energy drop found in 5-bar window")

# Also show energy at exact bar boundaries
print(f"\n── Energy at bar boundaries ──")
for b in range(1, 6):
    t_bar = b * BAR_S
    idx = int(t_bar / 0.05)
    if idx < len(rms_norm):
        print(f"  Bar {b} end ({t_bar:.3f}s): RMS = {rms_norm[idx]:.3f}")

# Show energy at half-bar points too
print(f"\n── Energy at beat boundaries (bars 2-4) ──")
for beat_num in range(8, 17):  # beats 8-16 = bars 2-4
    t_beat = beat_num * BEAT_S
    idx = int(t_beat / 0.05)
    if idx < len(rms_norm):
        bar = beat_num // 4 + 1
        beat_in_bar = beat_num % 4 + 1
        print(f"  Bar {bar} beat {beat_in_bar} ({t_beat:.3f}s): "
              f"RMS = {rms_norm[idx]:.3f}")
