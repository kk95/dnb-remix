#!/usr/bin/env python3
"""
Explore KG stems for interesting melodic/textural elements.

Goal: Find 2-3 interesting sounds OTHER than:
- Guitar at 120-130s
- High ding (>1500Hz) from 33.245s drop
- Drums and bass loops from 33.245s

Looking for: textures, pads, vocal chops, melodic elements
"""

import os
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, spectrogram

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft/kho-gayi"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM

os.makedirs(OUTPUT, exist_ok=True)


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def to_mono(audio):
    if audio.ndim == 2:
        return (audio[:, 0] + audio[:, 1]) / 2
    return audio


def rms_envelope(audio, window_ms=50):
    """Calculate RMS envelope for energy detection."""
    mono = to_mono(audio)
    window = int(window_ms / 1000 * SR)
    rms = np.sqrt(np.convolve(mono**2, np.ones(window)/window, mode='same'))
    return rms


def analyze_spectrum(audio, start_s, end_s, name):
    """Analyze frequency content in a time region."""
    start = int(start_s * SR)
    end = int(end_s * SR)
    segment = to_mono(audio[start:end])

    # Spectrogram
    f, t, Sxx = spectrogram(segment, SR, nperseg=2048, noverlap=1024)

    # Find dominant frequencies
    mean_spectrum = np.mean(Sxx, axis=1)
    top_idx = np.argsort(mean_spectrum)[-10:][::-1]
    top_freqs = f[top_idx]

    print(f"\n{name} ({start_s:.1f}s - {end_s:.1f}s):")
    print(f"  Top frequencies: {', '.join([f'{f:.0f}Hz' for f in top_freqs[:5]])}")
    print(f"  Energy: {np.mean(segment**2):.6f}")

    return f, t, Sxx, mean_spectrum


def extract_and_stretch(audio, start_s, end_s, name):
    """Extract a segment and time-stretch to 88 BPM."""
    start = int(start_s * SR)
    end = int(end_s * SR)
    segment = audio[start:end]

    print(f"\nExtracting {name}: {start_s:.2f}s - {end_s:.2f}s ({(end_s-start_s):.2f}s)")

    # Time-stretch to 88 BPM
    stretched = pyrb.time_stretch(segment, SR, STRETCH_RATE)
    print(f"  Original: {len(segment)/SR:.3f}s, Stretched: {len(stretched)/SR:.3f}s")

    # Normalize
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
kg_drums = load_stereo(f"{STEMS}/drums.wav")
kg_bass = load_stereo(f"{STEMS}/bass.wav")

print(f"  other.wav: {len(kg_other)/SR:.1f}s")
print(f"  vocals.wav: {len(kg_vocals)/SR:.1f}s")
print(f"  drums.wav: {len(kg_drums)/SR:.1f}s")
print(f"  bass.wav: {len(kg_bass)/SR:.1f}s")


# ── Analyze energy profile ─────────────────────────────────────────────

print("\n" + "="*60)
print("ENERGY PROFILE ANALYSIS")
print("="*60)

other_rms = rms_envelope(kg_other)
vocals_rms = rms_envelope(kg_vocals)

# Find peaks in "other" stem
time_axis = np.arange(len(other_rms)) / SR

# Plot energy profile
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(time_axis, other_rms, label='Other stem', alpha=0.8)
axes[0].set_title("KG 'other' stem - RMS energy profile")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("RMS")
axes[0].axvline(x=33.245, color='r', linestyle='--', label='Drop (33.2s)')
axes[0].axvline(x=120, color='g', linestyle='--', label='Guitar (120s) - SKIP')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_axis[:len(vocals_rms)], vocals_rms, label='Vocals stem', alpha=0.8, color='orange')
axes[1].set_title("KG 'vocals' stem - RMS energy profile")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("RMS")
axes[1].axvline(x=33.245, color='r', linestyle='--', label='Drop (33.2s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-explore-energy.png", dpi=150)
print(f"\nSaved energy profile: {OUTPUT}/kg-explore-energy.png")


# ── Explore specific regions ───────────────────────────────────────────

print("\n" + "="*60)
print("EXPLORING SPECIFIC REGIONS")
print("="*60)

# Region 1: Pre-drop melodic content (0-22s) - before drums
print("\n--- REGION 1: Pre-drop (0-22s) - melodic content before drums ---")
analyze_spectrum(kg_other, 0, 22, "Other 0-22s")
analyze_spectrum(kg_vocals, 0, 22, "Vocals 0-22s")

# Region 2: Drum build section (23-32s) - looking for textures behind drums
print("\n--- REGION 2: Drum build (23-32s) ---")
analyze_spectrum(kg_other, 23, 32, "Other 23-32s")

# Region 3: Post-drop variations (50-70s) - different textures
print("\n--- REGION 3: Post-drop (50-70s) ---")
analyze_spectrum(kg_other, 50, 70, "Other 50-70s")

# Region 4: Around 90-110s - potential breakdown?
print("\n--- REGION 4: Mid-song (90-110s) ---")
analyze_spectrum(kg_other, 90, 110, "Other 90-110s")

# Region 5: Late song (140-160s) - outro elements?
print("\n--- REGION 5: Late song (140-160s) ---")
analyze_spectrum(kg_other, 140, 160, "Other 140-160s")


# ── Detailed listening extracts ────────────────────────────────────────

print("\n" + "="*60)
print("CREATING LISTENING EXTRACTS")
print("="*60)

extracts = []

# 1. Pre-drop melodic element (around 10-15s) - nice pad/texture
print("\n[EXTRACT 1] Pre-drop melodic pad (10-18s)")
ext1 = extract_and_stretch(kg_other, 10, 18, "pre-drop-pad")
save_extract(ext1, "pre-drop-pad-10-18s")
extracts.append(("pre-drop-pad-10-18s", 10, 18, "Potential atmospheric pad before drums"))

# 2. Vocal texture from intro (5-12s) - see if there's interesting processing
print("\n[EXTRACT 2] Vocal intro texture (5-12s)")
ext2 = extract_and_stretch(kg_vocals, 5, 12, "vocal-intro")
save_extract(ext2, "vocal-intro-5-12s")
extracts.append(("vocal-intro-5-12s", 5, 12, "Intro vocal texture"))

# 3. Post-drop synth stabs (40-48s) - different section
print("\n[EXTRACT 3] Post-drop variation (40-48s)")
ext3 = extract_and_stretch(kg_other, 40, 48, "post-drop-40s")
save_extract(ext3, "post-drop-40-48s")
extracts.append(("post-drop-40-48s", 40, 48, "Post-drop synth variations"))

# 4. Mid-breakdown region (60-68s)
print("\n[EXTRACT 4] Mid-breakdown (60-68s)")
ext4 = extract_and_stretch(kg_other, 60, 68, "mid-breakdown-60s")
save_extract(ext4, "mid-breakdown-60-68s")
extracts.append(("mid-breakdown-60-68s", 60, 68, "Mid-song breakdown region"))

# 5. Low-pass filtered "other" (1-8s) - isolate potential pad
print("\n[EXTRACT 5] Low-freq content (1-8s) - isolating pad")
sos_lp = butter(4, 800 / (SR/2), btype='low', output='sos')
other_lp = np.column_stack([
    sosfilt(sos_lp, kg_other[:, 0]),
    sosfilt(sos_lp, kg_other[:, 1])
])
ext5 = extract_and_stretch(other_lp, 1, 8, "lowfreq-pad")
save_extract(ext5, "lowfreq-pad-1-8s")
extracts.append(("lowfreq-pad-1-8s", 1, 8, "Low-frequency pad isolated (<800Hz)"))

# 6. Vocal chop potential (15-20s) - before drop
print("\n[EXTRACT 6] Vocal pre-drop (15-20s)")
ext6 = extract_and_stretch(kg_vocals, 15, 20, "vocal-pre-drop")
save_extract(ext6, "vocal-pre-drop-15-20s")
extracts.append(("vocal-pre-drop-15-20s", 15, 20, "Vocals just before drums build"))

# 7. Mid-band from 'other' (3-10s) - isolate melodic content
print("\n[EXTRACT 7] Mid-band melodic (3-10s) - 200-2000Hz")
sos_bp = butter(4, [200/(SR/2), 2000/(SR/2)], btype='band', output='sos')
other_bp = np.column_stack([
    sosfilt(sos_bp, kg_other[:, 0]),
    sosfilt(sos_bp, kg_other[:, 1])
])
ext7 = extract_and_stretch(other_bp, 3, 10, "midband-melodic")
save_extract(ext7, "midband-melodic-3-10s")
extracts.append(("midband-melodic-3-10s", 3, 10, "Mid-band melodic content (200-2000Hz)"))

# 8. Check what's at very beginning (0-5s) - intro texture
print("\n[EXTRACT 8] Intro texture (0-5s)")
ext8 = extract_and_stretch(kg_other, 0, 5, "intro-texture")
save_extract(ext8, "intro-texture-0-5s")
extracts.append(("intro-texture-0-5s", 0, 5, "Very beginning texture"))


# ── Summary ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY OF EXTRACTS")
print("="*60)

for name, start, end, desc in extracts:
    print(f"\n  {name}")
    print(f"    Original: {start:.1f}s - {end:.1f}s")
    print(f"    Description: {desc}")
    print(f"    File: {OUTPUT}/kg-explore-{name}.wav")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
Listen to the extracts and identify:
1. Interesting pad/texture elements (could layer under flute)
2. Vocal chops (could use as one-shot accents)
3. Melodic content that complements the Saathiya flute

Files are in: """ + OUTPUT)
