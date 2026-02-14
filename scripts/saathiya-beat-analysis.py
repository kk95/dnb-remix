#!/usr/bin/env python3
"""
Analyze Saathiya's actual beat positions in the drop region (305-320s)
using high-resolution beat detection from the drums stem.
This tells us the REAL bar duration for the flute loop.
"""

import os
import numpy as np
import soundfile as sf
import librosa

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
DRUMS = f"{PROJECT}/stems/htdemucs_ft/saathiya/drums.wav"
SR = 44100

# Load just the region we care about (300-325s)
START_S = 300.0
END_S = 325.0
start_sample = int(START_S * SR)
end_sample = int(END_S * SR)

audio, _ = sf.read(DRUMS, dtype='float64')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # mono for analysis

region = audio[start_sample:end_sample]

print(f"Analyzing Saathiya drums {START_S}-{END_S}s ({len(region)/SR:.1f}s)")
print(f"hop_length=128 → {SR/128:.1f} Hz temporal resolution ({128/SR*1000:.2f}ms)")

# Beat detection with maximum resolution
tempo, beat_frames = librosa.beat.beat_track(
    y=region, sr=SR, hop_length=128, units='frames'
)
beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=128)
# Convert to absolute time
beat_times_abs = beat_times + START_S

tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
print(f"\nEstimated tempo: {tempo_val:.3f} BPM")
print(f"Found {len(beat_times_abs)} beats in region\n")

print("Beat positions (absolute time):")
for i, bt in enumerate(beat_times_abs):
    ibi = beat_times_abs[i] - beat_times_abs[i-1] if i > 0 else 0
    bpm = 60.0 / ibi if ibi > 0 else 0
    marker = ""
    if abs(bt - 306.772) < 0.1:
        marker = " ← SAATHIYA_DROP_S (beat grid)"
    if abs(bt - 307.454) < 0.1:
        marker = " ← SNAP POINT (flute starts)"
    print(f"  beat {i:2d}: {bt:.4f}s  IBI={ibi*1000:.1f}ms  ({bpm:.1f} BPM){marker}")

# Focus on the flute region: 307-318s
print("\n── Flute region analysis (307-318s) ──")
flute_beats = beat_times_abs[(beat_times_abs >= 306.5) & (beat_times_abs <= 318.5)]
print(f"Beats in flute region: {len(flute_beats)}")

if len(flute_beats) >= 2:
    ibis = np.diff(flute_beats)
    print(f"\nInter-beat intervals:")
    for i, ibi in enumerate(ibis):
        print(f"  {flute_beats[i]:.4f}s → {flute_beats[i+1]:.4f}s: {ibi*1000:.1f}ms ({60/ibi:.2f} BPM)")

    mean_ibi = np.mean(ibis)
    std_ibi = np.std(ibis)
    local_bpm = 60.0 / mean_ibi
    print(f"\nMean IBI: {mean_ibi*1000:.2f}ms ± {std_ibi*1000:.2f}ms")
    print(f"Local BPM: {local_bpm:.3f}")
    print(f"Bar duration (4 beats): {4*mean_ibi*1000:.1f}ms ({4*mean_ibi:.4f}s)")
    print(f"Mathematical 88.0 BPM bar: {4*60/88*1000:.1f}ms ({4*60/88:.4f}s)")
    print(f"Difference: {(4*mean_ibi - 4*60/88)*1000:.1f}ms per bar")
    print(f"Drift after 4 bars (16 beats): {16*(mean_ibi - 60/88)*1000:.1f}ms")

# Also look at 4-bar spans directly
print("\n── Direct 4-bar measurements ──")
if len(flute_beats) >= 17:
    for i in range(len(flute_beats) - 16):
        span = flute_beats[i+16] - flute_beats[i]
        bpm = 60.0 * 16 / span
        print(f"  beats {i}-{i+16}: {span:.4f}s ({bpm:.3f} BPM)")
elif len(flute_beats) >= 5:
    for i in range(len(flute_beats) - 4):
        span = flute_beats[i+4] - flute_beats[i]
        bpm = 60.0 * 4 / span
        print(f"  beats {i}-{i+4}: {span:.4f}s = 1 bar ({bpm:.3f} BPM)")
