#!/usr/bin/env python3
"""
Deep analysis of KG guitar riff region (~120-130s).

The initial analysis found a 10s active region with guitar-like pitches.
This script does a more detailed examination to:
1. Find exact riff boundaries (start/end of melodic phrases)
2. Analyze harmonic content to confirm it's guitar (not synth/artifact)
3. Check beat alignment for looping
4. Extract and preview the riff
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

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM
BEAT_S = 60.0 / KG_BPM
BAR_S = 4 * BEAT_S

os.makedirs(OUTPUT, exist_ok=True)


def load_stereo(path):
    audio, sr = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio, sr


print("=" * 60)
print("Deep analysis of KG guitar region (~120s)")
print("=" * 60)

# Load the full KG other stem
kg_other, sr = load_stereo(f"{STEMS}/kho-gayi/other.wav")
print(f"Loaded: {len(kg_other)/sr:.1f}s")

# Focus on the active region found: 119.83s - 129.99s
# Expand slightly for context
REGION_START = 118.0
REGION_END = 132.0

region_start_samp = int(REGION_START * sr)
region_end_samp = int(REGION_END * sr)
region = kg_other[region_start_samp:region_end_samp]
region_mono = np.mean(region, axis=1)

print(f"\nFocused region: {REGION_START}-{REGION_END}s ({len(region)/sr:.1f}s)")

# ── Beat grid analysis ──────────────────────────────────────────────

print(f"\n--- Beat grid at {KG_BPM} BPM ---")
print(f"Beat interval: {BEAT_S:.4f}s ({BEAT_S*1000:.1f}ms)")
print(f"Bar length: {BAR_S:.4f}s")

# Find beat positions in this region
# Assume beat 1 is at the song start (0s)
first_beat_in_region = np.ceil(REGION_START / BEAT_S) * BEAT_S
beats_in_region = np.arange(first_beat_in_region, REGION_END, BEAT_S)
bars_in_region = np.arange(
    np.ceil(REGION_START / BAR_S) * BAR_S,
    REGION_END,
    BAR_S
)

print(f"Beats in region: {len(beats_in_region)} (first at {first_beat_in_region:.2f}s)")
print(f"Bar starts: {', '.join([f'{b:.2f}s' for b in bars_in_region])}")

# ── Onset detection ─────────────────────────────────────────────────

print("\n--- Onset detection ---")

# Detect note onsets
onset_frames = librosa.onset.onset_detect(
    y=region_mono, sr=sr, hop_length=512,
    backtrack=True, units='frames'
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512) + REGION_START

print(f"Detected {len(onset_times)} onsets")

# Group onsets by proximity to beats
beat_aligned_onsets = []
off_beat_onsets = []
BEAT_TOLERANCE = 0.05  # 50ms tolerance

for onset in onset_times:
    nearest_beat = beats_in_region[np.argmin(np.abs(beats_in_region - onset))]
    if abs(onset - nearest_beat) < BEAT_TOLERANCE:
        beat_aligned_onsets.append((onset, nearest_beat))
    else:
        off_beat_onsets.append(onset)

print(f"Beat-aligned onsets: {len(beat_aligned_onsets)}")
print(f"Off-beat onsets: {len(off_beat_onsets)}")

# ── Spectral analysis for guitar characteristics ────────────────────

print("\n--- Spectral analysis (guitar characteristics) ---")

# Guitar has:
# 1. Clear harmonic series (fundamental + overtones)
# 2. Attack transient with quick decay
# 3. Typical range: E2 (82Hz) to E5 (659Hz) for standard tuning

# Chromagram to see note content
chroma = librosa.feature.chroma_stft(y=region_mono, sr=sr, hop_length=512)
chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512) + REGION_START

# Average chroma to see dominant notes
avg_chroma = np.mean(chroma, axis=1)
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
dominant_notes = [(note_names[i], avg_chroma[i]) for i in range(12)]
dominant_notes.sort(key=lambda x: x[1], reverse=True)

print("Dominant notes (chromagram analysis):")
for note, energy in dominant_notes[:5]:
    print(f"  {note}: {energy:.3f}")

# ── Find riff boundaries ────────────────────────────────────────────

print("\n--- Finding riff boundaries ---")

# RMS envelope
hop = 512
rms = librosa.feature.rms(y=region_mono, frame_length=2048, hop_length=hop)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + REGION_START

# Smooth RMS for phrase detection
from scipy.ndimage import gaussian_filter1d
rms_smooth = gaussian_filter1d(rms, sigma=10)

# Find where energy rises significantly (phrase starts)
rms_diff = np.diff(rms_smooth)
phrase_threshold = 0.15 * np.max(np.abs(rms_diff))

# Look for phrase boundaries
phrase_starts = []
phrase_ends = []
in_phrase = False
phrase_start_idx = 0

for i, val in enumerate(rms_smooth):
    if not in_phrase and val > 0.2 * np.max(rms_smooth):
        in_phrase = True
        phrase_start_idx = i
    elif in_phrase and val < 0.1 * np.max(rms_smooth):
        in_phrase = False
        phrase_starts.append(rms_times[phrase_start_idx])
        phrase_ends.append(rms_times[i])

if in_phrase:
    phrase_starts.append(rms_times[phrase_start_idx])
    phrase_ends.append(rms_times[-1])

print(f"Detected {len(phrase_starts)} phrases:")
for start, end in zip(phrase_starts, phrase_ends):
    duration = end - start
    bars = duration / BAR_S
    # Find nearest bar for start
    nearest_bar_start = bars_in_region[np.argmin(np.abs(bars_in_region - start))]
    bar_offset = (start - nearest_bar_start) / BEAT_S
    print(f"  {start:.2f}s - {end:.2f}s ({duration:.2f}s = {bars:.1f} bars) [offset: {bar_offset:.2f} beats from bar]")


# ── Harmonic analysis to distinguish guitar from synth/vocals ───────

print("\n--- Harmonic analysis ---")

# Guitar has characteristic harmonics: f, 2f, 3f, 4f...
# Synthesizers often have different harmonic structures

# Analyze a short segment in the middle of the region
seg_start = 122.0
seg_end = 124.0
seg_start_samp = int((seg_start - REGION_START) * sr)
seg_end_samp = int((seg_end - REGION_START) * sr)
seg = region_mono[seg_start_samp:seg_end_samp]

# FFT
n_fft = 8192
S = np.abs(np.fft.rfft(seg, n_fft))
freqs = np.fft.rfftfreq(n_fft, 1/sr)

# Find peaks in spectrum
from scipy.signal import find_peaks
peaks, _ = find_peaks(S, height=np.max(S) * 0.1, distance=20)
peak_freqs = freqs[peaks]
peak_mags = S[peaks]

# Sort by magnitude
sorted_idx = np.argsort(peak_mags)[::-1]
top_peaks = [(peak_freqs[i], peak_mags[i]) for i in sorted_idx[:10]]

print(f"Top spectral peaks ({seg_start}-{seg_end}s):")
for freq, mag in top_peaks:
    note = librosa.hz_to_note(freq) if freq > 20 else "sub"
    print(f"  {freq:.1f}Hz ({note}): {mag:.1f}")

# Check for harmonic relationships
fundamental = top_peaks[0][0]
print(f"\nPotential fundamental: {fundamental:.1f}Hz ({librosa.hz_to_note(fundamental)})")
print("Harmonic check (expected vs found):")
for n in range(2, 6):
    expected = fundamental * n
    # Find nearest peak
    diffs = np.abs(peak_freqs - expected)
    if np.min(diffs) < 30:  # Within 30Hz
        nearest_idx = np.argmin(diffs)
        actual = peak_freqs[nearest_idx]
        print(f"  H{n}: expected {expected:.0f}Hz, found {actual:.0f}Hz (diff: {actual-expected:.0f}Hz)")
    else:
        print(f"  H{n}: expected {expected:.0f}Hz, NOT FOUND")


# ── Create detailed visualization ───────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Waveform with beat grid
t = np.linspace(REGION_START, REGION_END, len(region_mono))
axes[0].plot(t, region_mono, linewidth=0.5, alpha=0.7)
axes[0].set_xlim(REGION_START, REGION_END)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title(f"KG 'other' stem waveform ({REGION_START}-{REGION_END}s) with beat grid")

# Mark bars (thick red)
for bar in bars_in_region:
    axes[0].axvline(bar, color='red', linewidth=1.5, alpha=0.7, label='Bar' if bar == bars_in_region[0] else '')

# Mark beats (thin gray)
for beat in beats_in_region:
    axes[0].axvline(beat, color='gray', linewidth=0.5, alpha=0.5)

# Mark onsets (green)
for onset in onset_times:
    axes[0].axvline(onset, color='green', linewidth=0.5, alpha=0.3)

axes[0].legend(loc='upper right')

# 2. Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(region_mono, n_fft=2048, hop_length=256)), ref=np.max)
img = librosa.display.specshow(D, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=axes[1], cmap='magma')
axes[1].set_ylim(0, 2000)  # Focus on guitar range
axes[1].set_title("Spectrogram (0-2kHz) - Guitar fundamental range")
plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

# 3. Chromagram
img2 = librosa.display.specshow(chroma, sr=sr, hop_length=512, x_axis='time', y_axis='chroma', ax=axes[2], cmap='coolwarm')
axes[2].set_title("Chromagram - Note content over time")
plt.colorbar(img2, ax=axes[2])

# 4. RMS envelope with phrase boundaries
axes[3].plot(rms_times, rms, linewidth=1, alpha=0.5, label='RMS')
axes[3].plot(rms_times, rms_smooth, linewidth=2, label='Smoothed RMS')
axes[3].set_xlim(REGION_START, REGION_END)
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Energy")
axes[3].set_title("RMS envelope with phrase boundaries")

for start, end in zip(phrase_starts, phrase_ends):
    axes[3].axvspan(start, end, alpha=0.3, color='yellow')
    axes[3].axvline(start, color='green', linewidth=1, linestyle='--')
    axes[3].axvline(end, color='red', linewidth=1, linestyle='--')

axes[3].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-guitar-deep-analysis.png", dpi=150)
print(f"\nSaved: {OUTPUT}/kg-guitar-deep-analysis.png")


# ── Extract potential riff segments ─────────────────────────────────

print("\n" + "=" * 60)
print("EXTRACTING POTENTIAL RIFF SEGMENTS")
print("=" * 60)

# Find best loop points (aligned to bar boundaries)
# The active region is ~120-130s

# Find the bar that contains the start of the active region
active_start = 119.83
active_end = 129.99

# Find nearest bar starts
bar_start = bars_in_region[np.argmin(np.abs(bars_in_region - active_start))]
# If the nearest is after active_start, go back one bar
if bar_start > active_start and len(bars_in_region) > 1:
    idx = np.argmin(np.abs(bars_in_region - active_start))
    if idx > 0:
        bar_start = bars_in_region[idx - 1]

print(f"\nActive region: {active_start:.2f}s - {active_end:.2f}s")
print(f"Nearest bar start: {bar_start:.2f}s")

# Try 2-bar and 4-bar extractions
for n_bars in [2, 4]:
    extract_start = bar_start
    extract_end = bar_start + n_bars * BAR_S
    extract_duration = extract_end - extract_start

    # Make sure we're within the audio
    if extract_end > len(kg_other) / sr:
        extract_end = len(kg_other) / sr
        extract_duration = extract_end - extract_start

    start_samp = int(extract_start * sr)
    end_samp = int(extract_end * sr)
    extract = kg_other[start_samp:end_samp]

    # Save the extract
    filename = f"kg-guitar-{n_bars}bar-{extract_start:.0f}s.wav"
    sf.write(f"{OUTPUT}/{filename}", extract, sr)
    print(f"\nExtracted {n_bars}-bar segment: {extract_start:.2f}s - {extract_end:.2f}s ({extract_duration:.3f}s)")
    print(f"  Saved: {OUTPUT}/{filename}")

    # Calculate stretched duration at target BPM
    stretched_duration = extract_duration / STRETCH_RATE
    print(f"  At {TARGET_BPM} BPM: {stretched_duration:.3f}s")


# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

key_notes = [n for n, e in dominant_notes[:5]]
print(f"""
Guitar riff analysis:
  Location: ~{active_start:.1f}s - {active_end:.1f}s in KG 'other' stem
  Duration: ~{active_end - active_start:.1f}s

Harmonic content:
  Dominant notes: {', '.join(key_notes)}
  Potential fundamental: {fundamental:.1f}Hz ({librosa.hz_to_note(fundamental)})

Beat alignment:
  KG BPM: {KG_BPM}
  Nearest bar start: {bar_start:.2f}s
  Phrases detected: {len(phrase_starts)}

Extracted segments:
  - 2-bar: {OUTPUT}/kg-guitar-2bar-{bar_start:.0f}s.wav
  - 4-bar: {OUTPUT}/kg-guitar-4bar-{bar_start:.0f}s.wav

Next: Listen to extracts, time-stretch to {TARGET_BPM} BPM, layer with flute.
""")

plt.close('all')
print("Deep analysis complete!")
