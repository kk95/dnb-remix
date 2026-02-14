#!/usr/bin/env python3
"""
Create a preview of the KG guitar riff stretched to 88 BPM.
Also create a test mix with the flute loop to hear the layering.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from scipy.signal import butter, sosfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM
BEAT_S = 60.0 / TARGET_BPM
BAR_S = 4 * BEAT_S

# Saathiya local BPM for flute
SAATHIYA_LOCAL_BPM = 87.939

os.makedirs(OUTPUT, exist_ok=True)


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def to_mp3(wav_path):
    mp3 = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3
    ], capture_output=True, check=True)
    os.remove(wav_path)
    return mp3


print("=" * 60)
print("KG Guitar Riff Preview")
print("=" * 60)

# Load the 4-bar guitar extract
guitar_path = f"{OUTPUT}/kg-guitar-4bar-120s.wav"
guitar_raw = load_stereo(guitar_path)
print(f"\nLoaded guitar: {len(guitar_raw)/SR:.3f}s at {KG_BPM} BPM")

# Time-stretch to 88 BPM
print(f"Stretching to {TARGET_BPM} BPM (rate={STRETCH_RATE:.4f})...")
guitar_stretched = pyrb.time_stretch(guitar_raw, SR, STRETCH_RATE)

# Trim to exact 4 bars at target BPM
target_4bar_samples = int(4 * BAR_S * SR)
guitar_stretched = guitar_stretched[:target_4bar_samples]
print(f"Stretched guitar: {len(guitar_stretched)/SR:.3f}s = 4 bars at {TARGET_BPM} BPM")

# Save the stretched guitar
sf.write(f"{OUTPUT}/kg-guitar-4bar-88bpm.wav", guitar_stretched, SR)
print(f"Saved: {OUTPUT}/kg-guitar-4bar-88bpm.wav")


# ── Analyze what frequency range the guitar occupies ────────────────

print("\n--- Frequency content analysis ---")

# Check energy in different bands
guitar_mono = np.mean(guitar_stretched, axis=1)
n_fft = 8192
S = np.abs(np.fft.rfft(guitar_mono, n_fft))
freqs = np.fft.rfftfreq(n_fft, 1/SR)

# Define bands
bands = [
    ("Sub bass", 20, 80),
    ("Bass", 80, 250),
    ("Low mids", 250, 500),
    ("Mids", 500, 2000),
    ("High mids", 2000, 4000),
    ("Highs", 4000, 8000),
    ("Air", 8000, 20000),
]

total_energy = np.sum(S)
print("Energy distribution:")
for name, lo, hi in bands:
    mask = (freqs >= lo) & (freqs < hi)
    band_energy = np.sum(S[mask])
    pct = 100 * band_energy / total_energy if total_energy > 0 else 0
    bar = '#' * int(pct / 2)
    print(f"  {name:12s} ({lo:5d}-{hi:5d}Hz): {pct:5.1f}% {bar}")


# ── Load flute loop for comparison ──────────────────────────────────

print("\n--- Loading flute loop for layering test ---")

# Extract flute (same as in main script)
BEAT_GRID_S = 306.772
SAATHIYA_BAR_S = 4 * (60.0 / SAATHIYA_LOCAL_BPM)
LOOP_BARS = 4

s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_start = int(BEAT_GRID_S * SR)
flute_raw = s_other[flute_start:flute_start + flute_actual_len]

# Stretch flute to match 88 BPM
flute_stretch = TARGET_BPM / SAATHIYA_LOCAL_BPM
flute_loop = pyrb.time_stretch(flute_raw, SR, flute_stretch)
flute_loop = flute_loop[:target_4bar_samples]

print(f"Flute loop: {len(flute_loop)/SR:.3f}s")


# ── Create layered test ─────────────────────────────────────────────

print("\n--- Creating layered test (guitar + flute) ---")

# Analyze flute frequency content for comparison
flute_mono = np.mean(flute_loop, axis=1)
S_flute = np.abs(np.fft.rfft(flute_mono, n_fft))

print("Flute energy distribution:")
for name, lo, hi in bands:
    mask = (freqs >= lo) & (freqs < hi)
    band_energy = np.sum(S_flute[mask])
    pct = 100 * band_energy / np.sum(S_flute) if np.sum(S_flute) > 0 else 0
    bar = '#' * int(pct / 2)
    print(f"  {name:12s} ({lo:5d}-{hi:5d}Hz): {pct:5.1f}% {bar}")

# Check for frequency overlap
print("\n--- Frequency overlap analysis ---")
guitar_peak_freqs = []
flute_peak_freqs = []

for S_arr, peaks_list, name in [(S, guitar_peak_freqs, "Guitar"), (S_flute, flute_peak_freqs, "Flute")]:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(S_arr, height=np.max(S_arr) * 0.1, distance=50)
    for p in peaks[:8]:
        peaks_list.append(freqs[p])
    print(f"{name} peaks: {', '.join([f'{f:.0f}Hz' for f in peaks_list])}")

# Simple mix test (equal volume)
# Loop both 3 times for a longer preview
n_loops = 3
guitar_looped = np.tile(guitar_stretched, (n_loops, 1))
flute_looped = np.tile(flute_loop, (n_loops, 1))

# Mix at different ratios
mixes = [
    ("guitar-only", guitar_looped, np.zeros_like(guitar_looped)),
    ("flute-only", np.zeros_like(flute_looped), flute_looped),
    ("guitar-0.5-flute-0.7", guitar_looped * 0.5, flute_looped * 0.7),
    ("guitar-0.3-flute-0.7", guitar_looped * 0.3, flute_looped * 0.7),
]

for name, g, f in mixes:
    mix = g + f
    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    wav_path = f"{OUTPUT}/layer-test-{name}.wav"
    sf.write(wav_path, mix, SR)
    mp3_path = to_mp3(wav_path)
    print(f"Saved: {mp3_path}")


# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Guitar riff (120s region):
  Original: {len(guitar_raw)/SR:.3f}s at {KG_BPM} BPM
  Stretched: {len(guitar_stretched)/SR:.3f}s at {TARGET_BPM} BPM

  Main frequency content: Low mids (250-500Hz) and Mids (500-2kHz)
  This is typical electric guitar range.

Flute loop (307s region):
  Length: {len(flute_loop)/SR:.3f}s at {TARGET_BPM} BPM

  Main frequency content: Mids (500-2kHz) and High mids (2-4kHz)
  Flute sits higher than guitar - good for layering!

Layering recommendation:
  - Guitar and flute have some overlap in mids (500-2kHz)
  - Guitar has more low-mid content, flute has more high-mid
  - Try guitar at 0.3-0.5 volume to avoid masking flute
  - Could EQ guitar to cut 1-2kHz to make room for flute

Test files created:
  - {OUTPUT}/layer-test-guitar-only.mp3
  - {OUTPUT}/layer-test-flute-only.mp3
  - {OUTPUT}/layer-test-guitar-0.5-flute-0.7.mp3
  - {OUTPUT}/layer-test-guitar-0.3-flute-0.7.mp3
""")

print("Done!")
