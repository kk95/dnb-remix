#!/usr/bin/env python3
"""
Analyze KG "other" stem around 120s to find guitar riff.

Goals:
1. Visualize the 110-130s region (spectrogram, waveform)
2. Identify melodic content vs noise/artifacts
3. Find exact timing of any guitar riff
4. Check spectral characteristics
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
import librosa
import librosa.display

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values from MEMORY.md
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM

os.makedirs(OUTPUT, exist_ok=True)


def load_stereo(path):
    audio, sr = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio, sr


print("=" * 60)
print("KG 'other' stem analysis: Looking for guitar riff around 120s")
print("=" * 60)

# Load the full KG other stem
kg_other, sr = load_stereo(f"{STEMS}/kho-gayi/other.wav")
print(f"\nLoaded KG other stem: {len(kg_other)/sr:.1f}s at {sr}Hz")

# Region of interest: 110-130s
ROI_START = 110.0
ROI_END = 130.0
roi_start_samp = int(ROI_START * sr)
roi_end_samp = int(ROI_END * sr)

roi_audio = kg_other[roi_start_samp:roi_end_samp]
print(f"Region of interest: {ROI_START}-{ROI_END}s ({len(roi_audio)/sr:.1f}s)")

# Convert to mono for analysis
roi_mono = np.mean(roi_audio, axis=1)

# ── Plot 1: Waveform of the region ──────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Waveform
t = np.linspace(ROI_START, ROI_END, len(roi_mono))
axes[0].plot(t, roi_mono, linewidth=0.5)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("KG 'other' stem: Waveform (110-130s)")
axes[0].set_xlim(ROI_START, ROI_END)
axes[0].grid(True, alpha=0.3)

# Mark beat positions at KG BPM
beat_interval = 60.0 / KG_BPM
first_beat = ROI_START + (beat_interval - (ROI_START % beat_interval))
beats = np.arange(first_beat, ROI_END, beat_interval)
for beat in beats:
    axes[0].axvline(beat, color='red', alpha=0.3, linewidth=0.5)

# ── Plot 2: Spectrogram ─────────────────────────────────────────────

# Use librosa for mel spectrogram
D = librosa.amplitude_to_db(
    np.abs(librosa.stft(roi_mono, n_fft=2048, hop_length=512)),
    ref=np.max
)

# Full spectrogram
img = librosa.display.specshow(
    D, sr=sr, hop_length=512, x_axis='time', y_axis='hz',
    ax=axes[1], cmap='magma'
)
axes[1].set_xlim(0, ROI_END - ROI_START)
axes[1].set_ylim(0, 8000)  # Focus on 0-8kHz where guitar lives
axes[1].set_xlabel("Time (s from 110s)")
axes[1].set_title("Spectrogram (0-8kHz) - Looking for guitar harmonics")
plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

# ── Plot 3: RMS energy over time ────────────────────────────────────

hop = 512
rms = librosa.feature.rms(y=roi_mono, frame_length=2048, hop_length=hop)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + ROI_START

axes[2].plot(rms_times, rms, linewidth=1)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("RMS Energy")
axes[2].set_title("Energy envelope - identifies active regions")
axes[2].set_xlim(ROI_START, ROI_END)
axes[2].grid(True, alpha=0.3)

# Find peaks in RMS (potential riff locations)
from scipy.signal import find_peaks
rms_norm = rms / np.max(rms) if np.max(rms) > 0 else rms
peaks, props = find_peaks(rms_norm, height=0.3, distance=int(sr/hop * 0.5))
peak_times = rms_times[peaks]
print(f"\nRMS peaks (potential riff locations):")
for pt in peak_times:
    print(f"  {pt:.2f}s")

for pt in peak_times:
    axes[2].axvline(pt, color='green', alpha=0.5, linewidth=1)

# ── Plot 4: Spectral centroid (brightness) ──────────────────────────

spectral_centroid = librosa.feature.spectral_centroid(y=roi_mono, sr=sr, hop_length=hop)[0]
sc_times = librosa.frames_to_time(np.arange(len(spectral_centroid)), sr=sr, hop_length=hop) + ROI_START

axes[3].plot(sc_times, spectral_centroid, linewidth=1, color='purple')
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Hz")
axes[3].set_title("Spectral Centroid - guitar typically 200-2000Hz")
axes[3].set_xlim(ROI_START, ROI_END)
axes[3].axhline(200, color='gray', linestyle='--', alpha=0.5, label='Guitar range')
axes[3].axhline(2000, color='gray', linestyle='--', alpha=0.5)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-guitar-analysis-110-130s.png", dpi=150)
print(f"\nSaved: {OUTPUT}/kg-guitar-analysis-110-130s.png")


# ── Detailed analysis of high-energy regions ────────────────────────

print("\n" + "=" * 60)
print("Detailed analysis of active regions")
print("=" * 60)

# Find contiguous regions with significant energy
threshold = 0.2 * np.max(rms)
active = rms > threshold
active_regions = []
in_region = False
region_start = 0

for i, is_active in enumerate(active):
    if is_active and not in_region:
        in_region = True
        region_start = rms_times[i]
    elif not is_active and in_region:
        in_region = False
        region_end = rms_times[i]
        if region_end - region_start > 0.3:  # Only regions > 300ms
            active_regions.append((region_start, region_end))

if in_region:  # Handle region that extends to end
    active_regions.append((region_start, rms_times[-1]))

print(f"\nActive regions (>300ms, energy above {threshold:.4f}):")
for start, end in active_regions:
    duration = end - start
    print(f"  {start:.2f}s - {end:.2f}s ({duration:.2f}s)")

    # Analyze spectral content of this region
    region_start_samp = int((start - ROI_START) * sr)
    region_end_samp = int((end - ROI_START) * sr)
    region_audio = roi_mono[region_start_samp:region_end_samp]

    if len(region_audio) > 0:
        # Get dominant frequencies
        S = np.abs(librosa.stft(region_audio, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        avg_spectrum = np.mean(S, axis=1)

        # Find top 5 frequency peaks
        top_idx = np.argsort(avg_spectrum)[-5:][::-1]
        print(f"    Top frequencies: {', '.join([f'{freqs[i]:.0f}Hz' for i in top_idx])}")


# ── Check if there's guitar-like content (pitch detection) ──────────

print("\n" + "=" * 60)
print("Pitch analysis (looking for guitar notes)")
print("=" * 60)

# Use librosa's pitch detection
pitches, magnitudes = librosa.piptrack(y=roi_mono, sr=sr, hop_length=512)

# Get the most prominent pitch at each time
pitch_track = []
pitch_times = []
for i in range(pitches.shape[1]):
    idx = magnitudes[:, i].argmax()
    if magnitudes[idx, i] > 0.1:  # Only if there's significant energy
        pitch_track.append(pitches[idx, i])
        pitch_times.append(librosa.frames_to_time(i, sr=sr, hop_length=512) + ROI_START)

if pitch_track:
    pitch_track = np.array(pitch_track)
    pitch_times = np.array(pitch_times)

    # Filter to guitar range (80-1000 Hz)
    guitar_mask = (pitch_track > 80) & (pitch_track < 1000)
    guitar_pitches = pitch_track[guitar_mask]
    guitar_times = pitch_times[guitar_mask]

    if len(guitar_pitches) > 0:
        print(f"Found {len(guitar_pitches)} pitch detections in guitar range (80-1000Hz)")

        # Convert to note names
        notes = librosa.hz_to_note(guitar_pitches)
        unique_notes = set(notes)
        print(f"Detected notes: {', '.join(sorted(unique_notes))}")

        # Find time regions with consistent pitches
        print("\nPitch regions:")
        current_start = guitar_times[0]
        current_pitch = guitar_pitches[0]

        for i in range(1, len(guitar_pitches)):
            if guitar_times[i] - guitar_times[i-1] > 0.2 or abs(guitar_pitches[i] - current_pitch) > 50:
                # New region
                if guitar_times[i-1] - current_start > 0.1:
                    note = librosa.hz_to_note(current_pitch)
                    print(f"  {current_start:.2f}s - {guitar_times[i-1]:.2f}s: ~{current_pitch:.0f}Hz ({note})")
                current_start = guitar_times[i]
                current_pitch = guitar_pitches[i]

        # Final region
        if guitar_times[-1] - current_start > 0.1:
            note = librosa.hz_to_note(current_pitch)
            print(f"  {current_start:.2f}s - {guitar_times[-1]:.2f}s: ~{current_pitch:.0f}Hz ({note})")
    else:
        print("No pitches detected in guitar range (80-1000Hz)")
else:
    print("No significant pitches detected")


# ── Save a zoomed plot of the most interesting region ───────────────

# Find the highest energy region
if active_regions:
    best_region = max(active_regions, key=lambda r: r[1] - r[0])
    best_start, best_end = best_region

    # Expand slightly for context
    zoom_start = max(ROI_START, best_start - 1)
    zoom_end = min(ROI_END, best_end + 1)

    print(f"\nMost interesting region: {best_start:.2f}s - {best_end:.2f}s")
    print(f"Creating zoomed analysis for {zoom_start:.2f}s - {zoom_end:.2f}s")

    # Create zoomed plot
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

    zoom_start_samp = int((zoom_start - ROI_START) * sr)
    zoom_end_samp = int((zoom_end - ROI_START) * sr)
    zoom_audio = roi_mono[zoom_start_samp:zoom_end_samp]

    # Waveform
    t_zoom = np.linspace(zoom_start, zoom_end, len(zoom_audio))
    axes2[0].plot(t_zoom, zoom_audio, linewidth=0.5)
    axes2[0].set_xlabel("Time (s)")
    axes2[0].set_ylabel("Amplitude")
    axes2[0].set_title(f"Zoomed waveform: {zoom_start:.1f}s - {zoom_end:.1f}s")
    axes2[0].grid(True, alpha=0.3)

    # Mark beats
    for beat in beats:
        if zoom_start <= beat <= zoom_end:
            axes2[0].axvline(beat, color='red', alpha=0.5, linewidth=1)

    # Spectrogram
    D_zoom = librosa.amplitude_to_db(
        np.abs(librosa.stft(zoom_audio, n_fft=2048, hop_length=256)),
        ref=np.max
    )
    img2 = librosa.display.specshow(
        D_zoom, sr=sr, hop_length=256, x_axis='time', y_axis='hz',
        ax=axes2[1], cmap='magma'
    )
    axes2[1].set_ylim(0, 4000)  # Focus on guitar range
    axes2[1].set_title(f"Spectrogram (0-4kHz) - Guitar range detail")
    plt.colorbar(img2, ax=axes2[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/kg-guitar-zoomed-{zoom_start:.0f}-{zoom_end:.0f}s.png", dpi=150)
    print(f"Saved: {OUTPUT}/kg-guitar-zoomed-{zoom_start:.0f}-{zoom_end:.0f}s.png")


# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Region analyzed: {ROI_START}-{ROI_END}s of KG 'other' stem
Active regions found: {len(active_regions)}

For DnB remix at {TARGET_BPM} BPM:
- KG native BPM: {KG_BPM}
- Stretch rate needed: {STRETCH_RATE:.4f}
- Any 4-bar loop from {KG_BPM} BPM stretched = {4 * 4 * 60.0 / TARGET_BPM:.3f}s

Next steps:
1. Review the saved spectrograms to identify melodic content
2. If guitar riff found, extract and time-stretch to {TARGET_BPM} BPM
3. Layer with flute loop (same timing: start at BEAT_GRID_S)
""")

plt.close('all')
print("\nAnalysis complete!")
