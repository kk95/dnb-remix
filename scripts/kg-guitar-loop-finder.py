#!/usr/bin/env python3
"""
Find the best loop point for the KG guitar riff.
Try multiple 2-bar and 4-bar segments to find cleanest loop.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import matplotlib.pyplot as plt
import librosa

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM
BEAT_S_KG = 60.0 / KG_BPM
BAR_S_KG = 4 * BEAT_S_KG
BEAT_S_TARGET = 60.0 / TARGET_BPM
BAR_S_TARGET = 4 * BEAT_S_TARGET

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
print("KG Guitar Loop Finder")
print("=" * 60)

kg_other = load_stereo(f"{STEMS}/kho-gayi/other.wav")
print(f"Loaded KG other: {len(kg_other)/SR:.1f}s")

# The guitar riff is active from ~120s to ~130s
# Bar starts in this region (at KG BPM):
# - 120.37s, 122.88s, 125.39s, 127.90s, 130.40s

# Let's try different start points to find the cleanest loop
bar_starts = [120.37, 122.88, 125.39]  # Skip 127.90 as it would go past 130s for 4-bar

print(f"\nKG BPM: {KG_BPM}")
print(f"Bar length at KG BPM: {BAR_S_KG:.4f}s")
print(f"Target BPM: {TARGET_BPM}")
print(f"Bar length at target BPM: {BAR_S_TARGET:.4f}s")

print("\n--- Extracting segments at different bar starts ---")

segments = []
for bar_start in bar_starts:
    for n_bars in [2, 4]:
        bar_end = bar_start + n_bars * BAR_S_KG
        if bar_end > len(kg_other) / SR:
            continue

        start_samp = int(bar_start * SR)
        end_samp = int(bar_end * SR)
        seg = kg_other[start_samp:end_samp]

        # Time-stretch to target BPM
        seg_stretched = pyrb.time_stretch(seg, SR, STRETCH_RATE)

        # Trim to exact length
        target_samples = int(n_bars * BAR_S_TARGET * SR)
        seg_stretched = seg_stretched[:target_samples]

        # Calculate RMS (average loudness)
        rms = np.sqrt(np.mean(seg_stretched ** 2))

        # Calculate spectral flatness (how "noisy" vs "tonal")
        mono = np.mean(seg_stretched, axis=1)
        sf_val = np.mean(librosa.feature.spectral_flatness(y=mono))

        segments.append({
            'start': bar_start,
            'n_bars': n_bars,
            'audio': seg_stretched,
            'rms': rms,
            'spectral_flatness': sf_val,
            'duration': len(seg_stretched) / SR,
        })

        print(f"  {bar_start:.2f}s, {n_bars} bars: RMS={rms:.4f}, SF={sf_val:.4f}, dur={len(seg_stretched)/SR:.3f}s")


# ── Analyze loop quality ────────────────────────────────────────────

print("\n--- Loop quality analysis ---")

# For a good loop, we want:
# 1. High RMS (loud, not empty)
# 2. Low spectral flatness (tonal, not noise)
# 3. Similar start/end energy (smooth loop)

for seg in segments:
    audio = seg['audio']
    # Check start/end similarity
    check_len = int(0.05 * SR)  # 50ms
    start_rms = np.sqrt(np.mean(audio[:check_len] ** 2))
    end_rms = np.sqrt(np.mean(audio[-check_len:] ** 2))
    rms_ratio = min(start_rms, end_rms) / max(start_rms, end_rms) if max(start_rms, end_rms) > 0 else 0

    seg['start_end_ratio'] = rms_ratio
    seg['score'] = seg['rms'] * (1 - seg['spectral_flatness']) * rms_ratio

    print(f"  {seg['start']:.2f}s, {seg['n_bars']} bars: start/end ratio={rms_ratio:.3f}, score={seg['score']:.4f}")


# ── Save the best segments ──────────────────────────────────────────

print("\n--- Saving best segments ---")

# Sort by score
segments.sort(key=lambda x: x['score'], reverse=True)

for i, seg in enumerate(segments[:4]):  # Top 4
    name = f"kg-guitar-{seg['n_bars']}bar-{seg['start']:.0f}s-88bpm"
    wav_path = f"{OUTPUT}/{name}.wav"
    sf.write(wav_path, seg['audio'], SR)
    mp3_path = to_mp3(wav_path)
    print(f"  #{i+1} (score={seg['score']:.4f}): {mp3_path}")


# ── Create comparison figure ────────────────────────────────────────

print("\n--- Creating comparison figure ---")

fig, axes = plt.subplots(len(segments), 2, figsize=(14, 3 * len(segments)))

for i, seg in enumerate(segments):
    audio = seg['audio']
    mono = np.mean(audio, axis=1)
    t = np.linspace(0, len(mono) / SR, len(mono))

    # Waveform
    axes[i, 0].plot(t, mono, linewidth=0.5)
    axes[i, 0].set_title(f"{seg['start']:.2f}s, {seg['n_bars']} bars (score={seg['score']:.4f})")
    axes[i, 0].set_xlabel("Time (s)")
    axes[i, 0].set_ylabel("Amplitude")
    axes[i, 0].set_xlim(0, t[-1])

    # Mark bar boundaries
    for b in range(seg['n_bars'] + 1):
        axes[i, 0].axvline(b * BAR_S_TARGET, color='red', alpha=0.5, linewidth=0.5)

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=2048, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=SR, hop_length=256, x_axis='time', y_axis='hz', ax=axes[i, 1], cmap='magma')
    axes[i, 1].set_ylim(0, 2000)
    axes[i, 1].set_title("Spectrogram (0-2kHz)")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-guitar-loop-comparison.png", dpi=150)
print(f"Saved: {OUTPUT}/kg-guitar-loop-comparison.png")


# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best = segments[0]
print(f"""
Best loop found:
  Start: {best['start']:.2f}s (at KG native BPM)
  Length: {best['n_bars']} bars
  Stretched duration: {best['duration']:.3f}s at {TARGET_BPM} BPM
  Score: {best['score']:.4f}

Key characteristics:
  - RMS (loudness): {best['rms']:.4f}
  - Spectral flatness: {best['spectral_flatness']:.4f} (lower = more tonal)
  - Start/end ratio: {best['start_end_ratio']:.3f} (closer to 1 = smoother loop)

Files saved:
""")

for i, seg in enumerate(segments[:4]):
    print(f"  #{i+1}: kg-guitar-{seg['n_bars']}bar-{seg['start']:.0f}s-88bpm.mp3")

print(f"\nComparison plot: {OUTPUT}/kg-guitar-loop-comparison.png")
print("\nDone!")
