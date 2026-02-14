#!/usr/bin/env python3
"""
Analyze melodic patterns in flute loop and KG guitar loop to find optimal layering strategy.

Creates visualizations showing:
1. Onset strength over time (where are the "hits"?)
2. Energy envelope comparison
3. Spectrograms side-by-side
4. Frequency band analysis
"""

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# Timing constants
TARGET_BPM = 88.0
BEAT_S = 60.0 / TARGET_BPM
BAR_S = 4 * BEAT_S
LOOP_BARS = 4
LOOP_S = LOOP_BARS * BAR_S  # 10.909s at 88 BPM

# Flute source params
BEAT_GRID_S = 306.772
SAATHIYA_LOCAL_BPM = 87.939
SAATHIYA_BAR_S = 4 * (60.0 / SAATHIYA_LOCAL_BPM)

print(f"Loop duration: {LOOP_S:.3f}s = {LOOP_BARS} bars at {TARGET_BPM} BPM")
print(f"Beat duration: {BEAT_S:.3f}s")

# ── Load the two loops ───────────────────────────────────────────────────

print("\n[1] Loading audio files...")

# Load KG guitar (already at 88 BPM)
kg_guitar_path = f"{OUTPUT}/kg-guitar-4bar-123s-88bpm.mp3"
kg_guitar, _ = librosa.load(kg_guitar_path, sr=SR, mono=True)
print(f"  KG guitar: {len(kg_guitar)/SR:.3f}s ({kg_guitar_path})")

# Load flute from Saathiya "other" stem and extract the same region as main script
import pyrubberband as pyrb

s_other, _ = sf.read(f"{STEMS}/saathiya/other.wav", dtype='float64')
if s_other.ndim == 2:
    s_other = s_other.mean(axis=1)  # mono

flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_start = int(BEAT_GRID_S * SR)
flute_raw = s_other[flute_start:flute_start + flute_actual_len]

# Stretch to 88 BPM
flute_stretch = TARGET_BPM / SAATHIYA_LOCAL_BPM
flute_loop = pyrb.time_stretch(flute_raw, SR, flute_stretch)

# Trim to exact loop length
target_loop_samples = int(LOOP_S * SR)
flute_loop = flute_loop[:target_loop_samples]
if len(flute_loop) < target_loop_samples:
    flute_loop = np.pad(flute_loop, (0, target_loop_samples - len(flute_loop)))

print(f"  Flute loop: {len(flute_loop)/SR:.3f}s (from {STEMS}/saathiya/other.wav)")

# Trim guitar to same length if needed
if len(kg_guitar) > target_loop_samples:
    kg_guitar = kg_guitar[:target_loop_samples]
elif len(kg_guitar) < target_loop_samples:
    kg_guitar = np.pad(kg_guitar, (0, target_loop_samples - len(kg_guitar)))

# ── Analysis: Onset Strength ─────────────────────────────────────────────

print("\n[2] Computing onset strength envelopes...")

# Onset detection with finer resolution
hop_length = 256  # ~5.8ms at 44100Hz
onset_flute = librosa.onset.onset_strength(y=flute_loop, sr=SR, hop_length=hop_length)
onset_guitar = librosa.onset.onset_strength(y=kg_guitar, sr=SR, hop_length=hop_length)

# Time axis for onsets
onset_times = librosa.frames_to_time(np.arange(len(onset_flute)), sr=SR, hop_length=hop_length)

# Normalize to 0-1 for comparison
onset_flute_norm = onset_flute / (onset_flute.max() + 1e-8)
onset_guitar_norm = onset_guitar / (onset_guitar.max() + 1e-8)

# ── Analysis: Energy Envelope ────────────────────────────────────────────

print("[3] Computing RMS energy envelopes...")

frame_length = 2048
rms_flute = librosa.feature.rms(y=flute_loop, frame_length=frame_length, hop_length=hop_length)[0]
rms_guitar = librosa.feature.rms(y=kg_guitar, frame_length=frame_length, hop_length=hop_length)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms_flute)), sr=SR, hop_length=hop_length)

rms_flute_norm = rms_flute / (rms_flute.max() + 1e-8)
rms_guitar_norm = rms_guitar / (rms_guitar.max() + 1e-8)

# ── Analysis: Spectrograms ───────────────────────────────────────────────

print("[4] Computing spectrograms...")

n_fft = 4096
D_flute = librosa.amplitude_to_db(np.abs(librosa.stft(flute_loop, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
D_guitar = librosa.amplitude_to_db(np.abs(librosa.stft(kg_guitar, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

spec_times = librosa.frames_to_time(np.arange(D_flute.shape[1]), sr=SR, hop_length=hop_length)
spec_freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

# ── Analysis: Frequency Band Energy ──────────────────────────────────────

print("[5] Computing frequency band energy over time...")

# Define frequency bands
bands = {
    'sub': (20, 80),      # Sub bass
    'bass': (80, 250),    # Bass
    'low_mid': (250, 500), # Low mids
    'mid': (500, 2000),   # Mids
    'high_mid': (2000, 6000), # High mids
    'high': (6000, 16000), # Highs
}

def compute_band_energy(audio, sr, bands, hop_length=256):
    """Compute energy in each frequency band over time."""
    band_energy = {}
    for name, (low, high) in bands.items():
        sos = butter(4, [low / (sr/2), min(high / (sr/2), 0.99)], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        rms = librosa.feature.rms(y=filtered, frame_length=2048, hop_length=hop_length)[0]
        band_energy[name] = rms
    return band_energy

band_energy_flute = compute_band_energy(flute_loop, SR, bands, hop_length)
band_energy_guitar = compute_band_energy(kg_guitar, SR, bands, hop_length)

# ── Analysis: Beat-aligned averages ──────────────────────────────────────

print("[6] Computing beat-aligned onset patterns...")

beat_samples = int(BEAT_S * SR)
beats_per_loop = LOOP_BARS * 4  # 16 beats

# Average onset strength per beat
def beat_aligned_onset(onset, sr, hop_length, beat_s, n_beats):
    """Compute average onset strength per beat."""
    beat_onsets = []
    samples_per_beat = beat_s * sr / hop_length
    for i in range(n_beats):
        start = int(i * samples_per_beat)
        end = int((i + 1) * samples_per_beat)
        if end <= len(onset):
            beat_onsets.append(onset[start:end].mean())
        else:
            beat_onsets.append(0)
    return np.array(beat_onsets)

beat_onsets_flute = beat_aligned_onset(onset_flute, SR, hop_length, BEAT_S, beats_per_loop)
beat_onsets_guitar = beat_aligned_onset(onset_guitar, SR, hop_length, BEAT_S, beats_per_loop)

# ── Create visualizations ────────────────────────────────────────────────

print("\n[7] Creating visualizations...")

# Plot 1: Onset strength comparison with beat grid
fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Add beat markers
beat_times = np.arange(0, LOOP_S, BEAT_S)
bar_times = np.arange(0, LOOP_S, BAR_S)

ax1 = axes1[0]
ax1.plot(onset_times, onset_flute_norm, label='Flute', color='blue', alpha=0.8)
for bt in beat_times:
    ax1.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax1.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax1.set_ylabel('Onset Strength')
ax1.set_title('FLUTE LOOP - Onset Strength (normalized)')
ax1.legend(loc='upper right')
ax1.set_xlim(0, LOOP_S)

ax2 = axes1[1]
ax2.plot(onset_times, onset_guitar_norm, label='Guitar', color='green', alpha=0.8)
for bt in beat_times:
    ax2.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax2.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax2.set_ylabel('Onset Strength')
ax2.set_title('KG GUITAR LOOP - Onset Strength (normalized)')
ax2.legend(loc='upper right')

ax3 = axes1[2]
ax3.fill_between(onset_times, onset_flute_norm, alpha=0.5, color='blue', label='Flute')
ax3.fill_between(onset_times, onset_guitar_norm, alpha=0.5, color='green', label='Guitar')
for bt in beat_times:
    ax3.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax3.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax3.set_ylabel('Onset Strength')
ax3.set_xlabel('Time (s)')
ax3.set_title('OVERLAY - Onset Strength Comparison')
ax3.legend(loc='upper right')

plt.tight_layout()
fig1.savefig(f"{OUTPUT}/layer-analysis-onsets.png", dpi=150)
print(f"  Saved: {OUTPUT}/layer-analysis-onsets.png")

# Plot 2: RMS Energy comparison
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax1 = axes2[0]
ax1.plot(rms_times, rms_flute_norm, label='Flute', color='blue', alpha=0.8)
for bt in beat_times:
    ax1.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax1.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax1.set_ylabel('RMS Energy')
ax1.set_title('FLUTE LOOP - RMS Energy (normalized)')
ax1.legend(loc='upper right')
ax1.set_xlim(0, LOOP_S)

ax2 = axes2[1]
ax2.plot(rms_times, rms_guitar_norm, label='Guitar', color='green', alpha=0.8)
for bt in beat_times:
    ax2.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax2.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax2.set_ylabel('RMS Energy')
ax2.set_title('KG GUITAR LOOP - RMS Energy (normalized)')
ax2.legend(loc='upper right')

ax3 = axes2[2]
ax3.plot(rms_times, rms_flute_norm, label='Flute', color='blue', alpha=0.8, linewidth=2)
ax3.plot(rms_times, rms_guitar_norm, label='Guitar', color='green', alpha=0.8, linewidth=2)
# Add product (overlap indicator)
overlap = rms_flute_norm * rms_guitar_norm
ax3.fill_between(rms_times, overlap * 2, alpha=0.3, color='red', label='Overlap (busy regions)')
for bt in beat_times:
    ax3.axvline(bt, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
for brt in bar_times:
    ax3.axvline(brt, color='red', alpha=0.5, linestyle='--', linewidth=1)
ax3.set_ylabel('RMS Energy')
ax3.set_xlabel('Time (s)')
ax3.set_title('OVERLAY - Energy Comparison (red = both active)')
ax3.legend(loc='upper right')

plt.tight_layout()
fig2.savefig(f"{OUTPUT}/layer-analysis-energy.png", dpi=150)
print(f"  Saved: {OUTPUT}/layer-analysis-energy.png")

# Plot 3: Spectrograms
fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes3[0]
img1 = librosa.display.specshow(D_flute, y_axis='log', x_axis='time', sr=SR, hop_length=hop_length, ax=ax1, cmap='magma')
ax1.set_title('FLUTE LOOP - Spectrogram')
ax1.set_ylabel('Frequency (Hz)')
for brt in bar_times:
    ax1.axvline(brt, color='white', alpha=0.7, linestyle='--', linewidth=1)
fig3.colorbar(img1, ax=ax1, format='%+2.0f dB')

ax2 = axes3[1]
img2 = librosa.display.specshow(D_guitar, y_axis='log', x_axis='time', sr=SR, hop_length=hop_length, ax=ax2, cmap='magma')
ax2.set_title('KG GUITAR LOOP - Spectrogram')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
for brt in bar_times:
    ax2.axvline(brt, color='white', alpha=0.7, linestyle='--', linewidth=1)
fig3.colorbar(img2, ax=ax2, format='%+2.0f dB')

plt.tight_layout()
fig3.savefig(f"{OUTPUT}/layer-analysis-spectrograms.png", dpi=150)
print(f"  Saved: {OUTPUT}/layer-analysis-spectrograms.png")

# Plot 4: Frequency band comparison
fig4, axes4 = plt.subplots(len(bands), 1, figsize=(14, 12), sharex=True)

band_names = list(bands.keys())
colors = ['purple', 'blue', 'cyan', 'green', 'orange', 'red']

for i, (name, color) in enumerate(zip(band_names, colors)):
    ax = axes4[i]
    flute_band = band_energy_flute[name] / (band_energy_flute[name].max() + 1e-8)
    guitar_band = band_energy_guitar[name] / (band_energy_guitar[name].max() + 1e-8)

    ax.plot(rms_times[:len(flute_band)], flute_band, label='Flute', color='blue', alpha=0.7)
    ax.plot(rms_times[:len(guitar_band)], guitar_band, label='Guitar', color='green', alpha=0.7)
    ax.fill_between(rms_times[:len(flute_band)], flute_band, alpha=0.2, color='blue')
    ax.fill_between(rms_times[:len(guitar_band)], guitar_band, alpha=0.2, color='green')

    for brt in bar_times:
        ax.axvline(brt, color='red', alpha=0.3, linestyle='--', linewidth=1)

    low, high = bands[name]
    ax.set_ylabel(f'{name}\n({low}-{high}Hz)')
    ax.set_xlim(0, LOOP_S)
    ax.legend(loc='upper right', fontsize=8)

axes4[-1].set_xlabel('Time (s)')
axes4[0].set_title('FREQUENCY BAND ENERGY COMPARISON')

plt.tight_layout()
fig4.savefig(f"{OUTPUT}/layer-analysis-bands.png", dpi=150)
print(f"  Saved: {OUTPUT}/layer-analysis-bands.png")

# Plot 5: Beat-aligned onset bar chart
fig5, ax5 = plt.subplots(figsize=(14, 6))

x = np.arange(beats_per_loop)
width = 0.35

bars1 = ax5.bar(x - width/2, beat_onsets_flute / (beat_onsets_flute.max() + 1e-8), width, label='Flute', color='blue', alpha=0.7)
bars2 = ax5.bar(x + width/2, beat_onsets_guitar / (beat_onsets_guitar.max() + 1e-8), width, label='Guitar', color='green', alpha=0.7)

# Add bar boundaries
for bar in range(LOOP_BARS):
    ax5.axvline(bar * 4 - 0.5, color='red', alpha=0.5, linestyle='--', linewidth=1)

ax5.set_xlabel('Beat Number (16 beats = 4 bars)')
ax5.set_ylabel('Avg Onset Strength (normalized)')
ax5.set_title('BEAT-ALIGNED ONSET COMPARISON - Where are the busy vs sparse sections?')
ax5.set_xticks(x)
ax5.set_xticklabels([f'{i+1}' for i in range(beats_per_loop)])
ax5.legend()

# Add bar labels
for bar in range(LOOP_BARS):
    ax5.text(bar * 4 + 1.5, 1.05, f'Bar {bar+1}', ha='center', fontweight='bold')

plt.tight_layout()
fig5.savefig(f"{OUTPUT}/layer-analysis-beats.png", dpi=150)
print(f"  Saved: {OUTPUT}/layer-analysis-beats.png")

# ── Numerical Analysis ───────────────────────────────────────────────────

print("\n" + "="*70)
print("NUMERICAL ANALYSIS")
print("="*70)

# Calculate correlation between onset patterns
corr = np.corrcoef(onset_flute_norm, onset_guitar_norm[:len(onset_flute_norm)])[0, 1]
print(f"\nOnset correlation: {corr:.3f}")
print("  (Positive = tend to hit together, Negative = naturally complementary)")

# Per-bar analysis
print(f"\n{'Bar':<6} {'Flute Avg':<12} {'Guitar Avg':<12} {'Complement?':<15}")
print("-" * 50)
for bar in range(LOOP_BARS):
    start_beat = bar * 4
    end_beat = start_beat + 4
    flute_bar_avg = beat_onsets_flute[start_beat:end_beat].mean()
    guitar_bar_avg = beat_onsets_guitar[start_beat:end_beat].mean()

    flute_bar_norm = flute_bar_avg / (beat_onsets_flute.max() + 1e-8)
    guitar_bar_norm = guitar_bar_avg / (beat_onsets_guitar.max() + 1e-8)

    # Check if they complement (one busy, one sparse)
    if flute_bar_norm > 0.6 and guitar_bar_norm < 0.4:
        comp = "Flute dominant"
    elif guitar_bar_norm > 0.6 and flute_bar_norm < 0.4:
        comp = "Guitar dominant"
    elif flute_bar_norm > 0.5 and guitar_bar_norm > 0.5:
        comp = "Both busy!"
    else:
        comp = "Both sparse"

    print(f"Bar {bar+1}:  {flute_bar_norm:.2f}         {guitar_bar_norm:.2f}         {comp}")

# Frequency separation analysis
print("\n\nFREQUENCY SEPARATION ANALYSIS:")
print("-" * 50)
for name in band_names:
    flute_total = band_energy_flute[name].mean()
    guitar_total = band_energy_guitar[name].mean()

    # Normalize relative to each source's total energy
    flute_norm = flute_total / sum(band_energy_flute[n].mean() for n in band_names)
    guitar_norm = guitar_total / sum(band_energy_guitar[n].mean() for n in band_names)

    low, high = bands[name]
    diff = abs(flute_norm - guitar_norm)
    dominant = "FLUTE" if flute_norm > guitar_norm else "GUITAR"

    print(f"  {name:<10} ({low:>5}-{high:>5}Hz): Flute={flute_norm:.2%}, Guitar={guitar_norm:.2%} → {dominant} dominant (diff={diff:.1%})")

# Timing offset analysis
print("\n\nTIMING OFFSET ANALYSIS:")
print("-" * 50)

offsets_beats = [0, 0.5, 1, 1.5, 2]  # offsets in beats
best_offset = 0
best_anti_corr = corr

for offset_beats in offsets_beats:
    offset_samples = int(offset_beats * BEAT_S * SR / hop_length)
    if offset_samples > 0:
        guitar_shifted = np.roll(onset_guitar_norm[:len(onset_flute_norm)], offset_samples)
        shifted_corr = np.corrcoef(onset_flute_norm, guitar_shifted)[0, 1]
    else:
        shifted_corr = corr

    if shifted_corr < best_anti_corr:
        best_anti_corr = shifted_corr
        best_offset = offset_beats

    print(f"  Offset {offset_beats} beats: correlation = {shifted_corr:.3f}")

print(f"\n  Best offset: {best_offset} beats (correlation = {best_anti_corr:.3f})")
print(f"    → {best_offset * BEAT_S * 1000:.0f}ms offset")

# Summary recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

# Find truly sparse regions
sparse_threshold = 0.3
flute_sparse_bars = [bar+1 for bar in range(LOOP_BARS)
                    if beat_onsets_flute[bar*4:(bar+1)*4].mean() / (beat_onsets_flute.max() + 1e-8) < sparse_threshold]
guitar_sparse_bars = [bar+1 for bar in range(LOOP_BARS)
                     if beat_onsets_guitar[bar*4:(bar+1)*4].mean() / (beat_onsets_guitar.max() + 1e-8) < sparse_threshold]

print(f"\n1. CALL-AND-RESPONSE OPTION:")
print(f"   Flute sparse bars: {flute_sparse_bars or 'None'}")
print(f"   Guitar sparse bars: {guitar_sparse_bars or 'None'}")
if flute_sparse_bars:
    print(f"   → Use guitar in flute's sparse bars ({flute_sparse_bars})")
if guitar_sparse_bars:
    print(f"   → Use flute in guitar's sparse bars ({guitar_sparse_bars})")

print(f"\n2. TIMING OFFSET OPTION:")
print(f"   Best anti-correlation at {best_offset} beats offset")
print(f"   → Start guitar {best_offset * BEAT_S * 1000:.0f}ms after flute")

print(f"\n3. FREQUENCY SEPARATION OPTION:")
# Find which bands have best separation
max_sep_band = max(band_names, key=lambda n: abs(
    band_energy_flute[n].mean() / (sum(band_energy_flute[m].mean() for m in band_names) + 1e-8) -
    band_energy_guitar[n].mean() / (sum(band_energy_guitar[m].mean() for m in band_names) + 1e-8)
))
print(f"   Best separation in: {max_sep_band} band")

# Check if guitar is predominantly lower/higher than flute
guitar_bass_energy = (band_energy_guitar['sub'].mean() + band_energy_guitar['bass'].mean()) / \
                     (sum(band_energy_guitar[n].mean() for n in band_names) + 1e-8)
guitar_high_energy = (band_energy_guitar['high_mid'].mean() + band_energy_guitar['high'].mean()) / \
                     (sum(band_energy_guitar[n].mean() for n in band_names) + 1e-8)
flute_bass_energy = (band_energy_flute['sub'].mean() + band_energy_flute['bass'].mean()) / \
                    (sum(band_energy_flute[n].mean() for n in band_names) + 1e-8)
flute_high_energy = (band_energy_flute['high_mid'].mean() + band_energy_flute['high'].mean()) / \
                    (sum(band_energy_flute[n].mean() for n in band_names) + 1e-8)

if guitar_bass_energy > flute_bass_energy * 1.2:
    print("   → Guitar has more bass energy - consider highpassing guitar to let flute bass through")
elif flute_bass_energy > guitar_bass_energy * 1.2:
    print("   → Flute has more bass energy - EQ guitar into higher bands")

if guitar_high_energy > flute_high_energy * 1.2:
    print("   → Guitar has more high energy - they may naturally separate")
elif flute_high_energy > guitar_high_energy * 1.2:
    print("   → Flute has more high energy - guitar can fill low/mid gaps")

print(f"\n4. REPLACE OPTION:")
print(f"   Based on energy analysis:")
for bar in range(LOOP_BARS):
    start_beat = bar * 4
    end_beat = start_beat + 4
    flute_bar_norm = beat_onsets_flute[start_beat:end_beat].mean() / (beat_onsets_flute.max() + 1e-8)
    guitar_bar_norm = beat_onsets_guitar[start_beat:end_beat].mean() / (beat_onsets_guitar.max() + 1e-8)

    if flute_bar_norm < 0.5:
        print(f"   Bar {bar+1}: Flute weak ({flute_bar_norm:.0%}) - could replace with guitar")
    elif guitar_bar_norm < 0.5:
        print(f"   Bar {bar+1}: Guitar weak ({guitar_bar_norm:.0%}) - keep flute only")

print("\n" + "="*70)
print("DONE - Analysis complete!")
print("="*70)
