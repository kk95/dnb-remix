#!/usr/bin/env python3
"""
Detailed exploration of KG stems - second pass with spectrograms.

Focus on:
1. 90-110s region (different frequency content detected)
2. Looking for one-shot vocal chops
3. Atmospheric pads that could layer under flute
"""

import os
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, spectrogram, find_peaks

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft/kho-gayi"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def to_mono(audio):
    if audio.ndim == 2:
        return (audio[:, 0] + audio[:, 1]) / 2
    return audio


def extract_and_stretch(audio, start_s, end_s):
    """Extract a segment and time-stretch to 88 BPM."""
    start = int(start_s * SR)
    end = int(end_s * SR)
    segment = audio[start:end]
    stretched = pyrb.time_stretch(segment, SR, STRETCH_RATE)
    peak = np.max(np.abs(stretched))
    if peak > 0:
        stretched = stretched * (0.9 / peak)
    return stretched


def save_extract(audio, name):
    """Save extract as WAV."""
    path = f"{OUTPUT}/kg-explore-{name}.wav"
    sf.write(path, audio, SR, subtype='PCM_16')
    print(f"  Saved: {path}")
    return path


# ── Load stems ─────────────────────────────────────────────────────────

print("Loading KG stems...")
kg_other = load_stereo(f"{STEMS}/other.wav")
kg_vocals = load_stereo(f"{STEMS}/vocals.wav")

total_duration = len(kg_other) / SR
print(f"  Duration: {total_duration:.1f}s")


# ── Create spectrogram overview ────────────────────────────────────────

print("\nGenerating spectrograms...")

fig, axes = plt.subplots(2, 1, figsize=(20, 10))

for ax, (audio, title) in zip(axes, [(kg_other, "KG 'other' stem"), (kg_vocals, "KG 'vocals' stem")]):
    mono = to_mono(audio)
    f, t, Sxx = spectrogram(mono, SR, nperseg=4096, noverlap=3072)

    # Log scale for better visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', vmin=-60, vmax=0)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim(0, 8000)

    # Mark key regions
    ax.axvline(x=33.245, color='cyan', linestyle='--', alpha=0.7, label='Drop')
    ax.axvline(x=120, color='green', linestyle='--', alpha=0.7, label='Guitar (SKIP)')

    # Interesting regions to explore
    for region, color in [(22, 'yellow'), (50, 'orange'), (90, 'red'), (140, 'purple')]:
        ax.axvline(x=region, color=color, linestyle=':', alpha=0.5)

    plt.colorbar(im, ax=ax, label='dB')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-explore-spectrograms.png", dpi=150)
print(f"  Saved: {OUTPUT}/kg-explore-spectrograms.png")


# ── Detailed extraction of interesting regions ─────────────────────────

print("\n" + "="*60)
print("DETAILED EXTRACTION - NEW REGIONS")
print("="*60)

extracts_info = []

# 1. Check 90-110s region more closely (detected unique frequency content)
print("\n[1] Mid-song unique texture (95-103s)")
print("    Detected higher frequencies (991Hz, 1314Hz) - different from rest of song")
ext = extract_and_stretch(kg_other, 95, 103)
save_extract(ext, "mid-unique-95-103s")
extracts_info.append({
    "name": "mid-unique-95-103s",
    "original_time": "95-103s",
    "description": "Mid-song region with unique frequency content (991Hz, 1314Hz peaks)",
    "potential_use": "Could be a breakdown synth pad or texture"
})

# 2. Look for vocal "aah" or sustained notes (good for chops)
print("\n[2] Sustained vocal at 2-5s")
print("    Checking intro for sustained vocal note")
ext = extract_and_stretch(kg_vocals, 2, 5)
save_extract(ext, "vocal-sustain-2-5s")
extracts_info.append({
    "name": "vocal-sustain-2-5s",
    "original_time": "2-5s",
    "description": "Very beginning vocal - looking for clean sustained note",
    "potential_use": "One-shot accent or reverb pad"
})

# 3. Check end of song for outro elements (140-160s)
print("\n[3] Late song atmosphere (145-153s)")
print("    Outro region - potential atmospheric pad")
ext = extract_and_stretch(kg_other, 145, 153)
save_extract(ext, "outro-pad-145-153s")
extracts_info.append({
    "name": "outro-pad-145-153s",
    "original_time": "145-153s",
    "description": "Outro region - looking for atmospheric decay/pad",
    "potential_use": "Background pad, layer under flute"
})

# 4. Check around 165-175s (very end)
print("\n[4] Final section (165-173s)")
print("    Very end - potential clean element")
ext = extract_and_stretch(kg_other, 165, 173)
save_extract(ext, "final-165-173s")
extracts_info.append({
    "name": "final-165-173s",
    "original_time": "165-173s",
    "description": "Final section of song",
    "potential_use": "Outro element, clean texture"
})

# 5. Isolate mid frequencies from 90-100s region
print("\n[5] Mid-freq isolated (90-100s, 400-1500Hz)")
sos_bp = butter(4, [400/(SR/2), 1500/(SR/2)], btype='band', output='sos')
other_bp = np.column_stack([
    sosfilt(sos_bp, kg_other[:, 0]),
    sosfilt(sos_bp, kg_other[:, 1])
])
ext = extract_and_stretch(other_bp, 90, 100)
save_extract(ext, "midfreq-isolated-90-100s")
extracts_info.append({
    "name": "midfreq-isolated-90-100s",
    "original_time": "90-100s",
    "description": "Mid-frequency isolated (400-1500Hz) from unique region",
    "potential_use": "Clean melodic element without bass/highs"
})

# 6. Check for vocal chop around 28s (just before drum build peaks)
print("\n[6] Vocal chop pre-silence (28-32s)")
print("    Right before the silence/break")
ext = extract_and_stretch(kg_vocals, 28, 32)
save_extract(ext, "vocal-chop-28-32s")
extracts_info.append({
    "name": "vocal-chop-28-32s",
    "original_time": "28-32s",
    "description": "Vocal right before the 1-second silence",
    "potential_use": "Tension-building vocal chop"
})

# 7. High-frequency shimmer from intro
print("\n[7] High shimmer intro (0-8s, >3000Hz)")
sos_hp = butter(4, 3000/(SR/2), btype='high', output='sos')
other_hp = np.column_stack([
    sosfilt(sos_hp, kg_other[:, 0]),
    sosfilt(sos_hp, kg_other[:, 1])
])
ext = extract_and_stretch(other_hp, 0, 8)
save_extract(ext, "shimmer-intro-0-8s")
extracts_info.append({
    "name": "shimmer-intro-0-8s",
    "original_time": "0-8s",
    "description": "High-frequency content (>3kHz) from intro",
    "potential_use": "Shimmer/air layer, subtle texture"
})

# 8. Check for synth stab pattern around 70-80s
print("\n[8] Synth stabs (70-78s)")
ext = extract_and_stretch(kg_other, 70, 78)
save_extract(ext, "synth-stabs-70-78s")
extracts_info.append({
    "name": "synth-stabs-70-78s",
    "original_time": "70-78s",
    "description": "Checking for rhythmic synth stabs",
    "potential_use": "Rhythmic accent loop"
})


# ── Find transients in vocals (potential chop points) ──────────────────

print("\n" + "="*60)
print("VOCAL TRANSIENT ANALYSIS")
print("="*60)

vocals_mono = to_mono(kg_vocals)

# Calculate energy in short windows
window_ms = 20
window = int(window_ms / 1000 * SR)
hop = window // 4

energy = []
times = []
for i in range(0, len(vocals_mono) - window, hop):
    e = np.sqrt(np.mean(vocals_mono[i:i+window]**2))
    energy.append(e)
    times.append(i / SR)

energy = np.array(energy)
times = np.array(times)

# Find peaks (potential chop points)
peaks, props = find_peaks(energy, height=np.mean(energy) * 2, distance=int(0.5 * SR / hop))

print(f"\nFound {len(peaks)} significant vocal peaks:")
for i, p in enumerate(peaks[:15]):  # Show first 15
    t = times[p]
    if t < 33:  # Only pre-drop
        print(f"  {i+1}. t={t:.2f}s (energy={energy[p]:.4f})")

# Create plot of vocal energy with peaks
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(times, energy, label='Vocal energy')
ax.scatter(times[peaks], energy[peaks], color='red', s=50, label='Peaks')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy')
ax.set_title('KG Vocals - Energy profile with transient peaks')
ax.axvline(x=33.245, color='cyan', linestyle='--', alpha=0.7, label='Drop')
ax.set_xlim(0, 50)  # Focus on pre-drop region
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-explore-vocal-peaks.png", dpi=150)
print(f"\nSaved: {OUTPUT}/kg-explore-vocal-peaks.png")


# ── Summary ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXTRACTION SUMMARY")
print("="*60)

for info in extracts_info:
    print(f"\n  {info['name']}")
    print(f"    Time: {info['original_time']}")
    print(f"    Description: {info['description']}")
    print(f"    Potential use: {info['potential_use']}")

print(f"\n\nAll extracts saved to: {OUTPUT}")
print("\nListen and identify which elements complement the Saathiya flute!")
