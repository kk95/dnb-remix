#!/usr/bin/env python3
"""
Find ALL instances of the iconic flute riff in the Saathiya "other" stem.
Rank them by loopability and export top candidates as MP3s.
"""

import numpy as np
import soundfile as sf
import librosa
import subprocess
import os
from scipy import signal

# --- Config ---
STEM_PATH = os.path.expanduser("~/Documents/dnb-remix/stems/htdemucs_ft/saathiya/other.wav")
OUT_DIR = os.path.expanduser("~/Documents/dnb-remix/output/constant-stretch")
SR = 44100

# Current reference: 4 bars at local BPM 87.939, starting at BEAT_GRID_S
BEAT_GRID_S = 306.772
LOCAL_BPM = 87.939
BEATS_PER_BAR = 4
NUM_BARS = 4
BEAT_DUR = 60.0 / LOCAL_BPM
BAR_DUR = BEAT_DUR * BEATS_PER_BAR
LOOP_DUR = BAR_DUR * NUM_BARS

print(f"Beat duration: {BEAT_DUR:.4f}s")
print(f"Bar duration:  {BAR_DUR:.4f}s")
print(f"Loop duration: {LOOP_DUR:.4f}s ({LOOP_DUR*1000:.1f}ms)")
print(f"Loop samples:  {int(LOOP_DUR * SR)}")
print()

# --- Load audio ---
print("Loading stem...")
audio, sr = sf.read(STEM_PATH, dtype="float32")
assert sr == SR, f"Expected {SR}, got {sr}"

if audio.ndim == 2:
    mono = audio.mean(axis=1)
else:
    mono = audio.copy()

total_dur = len(mono) / SR
print(f"Stem duration: {total_dur:.1f}s ({len(mono)} samples)")
print()

# --- Extract reference riff ---
ref_start_samp = int(BEAT_GRID_S * SR)
ref_end_samp = ref_start_samp + int(LOOP_DUR * SR)
ref_audio = mono[ref_start_samp:ref_end_samp]

print(f"Reference riff: {BEAT_GRID_S:.3f}s - {BEAT_GRID_S + LOOP_DUR:.3f}s")
print(f"Reference RMS: {np.sqrt(np.mean(ref_audio**2)):.6f}")
print()

# --- Compute chroma features for similarity matching ---
HOP = 512

print("Computing chroma features for entire stem...")
chroma_full = librosa.feature.chroma_cqt(y=mono, sr=SR, hop_length=HOP, n_chroma=12)
chroma_ref = librosa.feature.chroma_cqt(y=ref_audio, sr=SR, hop_length=HOP, n_chroma=12)
ref_frames = chroma_ref.shape[1]

print(f"Full chroma: {chroma_full.shape} ({chroma_full.shape[1]} frames)")
print(f"Ref chroma:  {chroma_ref.shape} ({ref_frames} frames)")
print()

# --- Sliding window chroma correlation ---
print("Computing sliding window chroma similarity...")

ref_flat = chroma_ref.flatten()
ref_flat_norm = ref_flat - ref_flat.mean()
ref_flat_std = np.std(ref_flat)

n_positions = chroma_full.shape[1] - ref_frames + 1
similarities = np.zeros(n_positions)

for i in range(n_positions):
    window = chroma_full[:, i:i + ref_frames].flatten()
    window_norm = window - window.mean()
    window_std = np.std(window_norm)
    if window_std > 0 and ref_flat_std > 0:
        similarities[i] = np.dot(ref_flat_norm, window_norm) / (
            len(ref_flat_norm) * ref_flat_std * window_std
        )
    else:
        similarities[i] = 0.0

times = librosa.frames_to_time(np.arange(n_positions), sr=SR, hop_length=HOP)

print(f"Similarity computed for {n_positions} positions")
print(f"Max similarity: {similarities.max():.4f} at {times[np.argmax(similarities)]:.2f}s")
print()

# --- Find peaks (candidate instances) ---
min_dist_frames = int((LOOP_DUR * 0.7) * SR / HOP)

THRESHOLD = 0.30

peaks, properties = signal.find_peaks(
    similarities,
    height=THRESHOLD,
    distance=min_dist_frames,
    prominence=0.05
)

peak_times = times[peaks]
peak_scores = similarities[peaks]

sort_idx = np.argsort(-peak_scores)
peaks = peaks[sort_idx]
peak_times = peak_times[sort_idx]
peak_scores = peak_scores[sort_idx]

print(f"Found {len(peaks)} candidate instances (threshold={THRESHOLD}):")
for i, (t, s) in enumerate(zip(peak_times, peak_scores)):
    print(f"  {i+1:2d}. t={t:7.2f}s  chroma_sim={s:.4f}")
print()

# --- Analyze each candidate for loopability ---
print("=" * 70)
print("DETAILED LOOPABILITY ANALYSIS")
print("=" * 70)

candidates = []

for i, (peak_frame, t, chroma_score) in enumerate(zip(peaks, peak_times, peak_scores)):
    start_samp = int(t * SR)
    end_samp = start_samp + int(LOOP_DUR * SR)

    if end_samp > len(mono):
        continue

    segment = mono[start_samp:end_samp]
    segment_stereo = audio[start_samp:end_samp] if audio.ndim == 2 else segment

    # 1. Overall RMS energy
    rms = np.sqrt(np.mean(segment**2))

    # 2. Energy envelope (split into 16 chunks)
    n_chunks = 16
    chunk_size = len(segment) // n_chunks
    envelope = np.array([
        np.sqrt(np.mean(segment[j*chunk_size:(j+1)*chunk_size]**2))
        for j in range(n_chunks)
    ])

    # 3. Loop boundary smoothness: compare first 10ms and last 10ms RMS
    boundary_ms = 10
    boundary_samp = int(boundary_ms * SR / 1000)
    rms_start = np.sqrt(np.mean(segment[:boundary_samp]**2))
    rms_end = np.sqrt(np.mean(segment[-boundary_samp:]**2))

    if max(rms_start, rms_end) > 0:
        boundary_ratio = min(rms_start, rms_end) / max(rms_start, rms_end)
    else:
        boundary_ratio = 0.0

    # 4. Wider boundary view (50ms)
    boundary_samp_50 = int(50 * SR / 1000)
    rms_start_50 = np.sqrt(np.mean(segment[:boundary_samp_50]**2))
    rms_end_50 = np.sqrt(np.mean(segment[-boundary_samp_50:]**2))
    if max(rms_start_50, rms_end_50) > 0:
        boundary_ratio_50 = min(rms_start_50, rms_end_50) / max(rms_start_50, rms_end_50)
    else:
        boundary_ratio_50 = 0.0

    # 5. Vocal bleed check: energy in 100-400Hz
    sos_vocal = signal.butter(4, [100, 400], btype='bandpass', fs=SR, output='sos')
    vocal_band = signal.sosfilt(sos_vocal, segment)
    vocal_rms = np.sqrt(np.mean(vocal_band**2))
    vocal_ratio = vocal_rms / rms if rms > 0 else 0.0

    # 6. Flute presence check: energy in 800-4000Hz
    sos_flute = signal.butter(4, [800, 4000], btype='bandpass', fs=SR, output='sos')
    flute_band = signal.sosfilt(sos_flute, segment)
    flute_rms = np.sqrt(np.mean(flute_band**2))
    flute_ratio = flute_rms / rms if rms > 0 else 0.0

    # 7. Energy consistency
    envelope_std = np.std(envelope) / np.mean(envelope) if np.mean(envelope) > 0 else 1.0

    # 8. Sample-level discontinuity at loop point
    sample_jump = abs(float(segment[-1]) - float(segment[0]))

    # --- Composite loopability score ---
    score_chroma = chroma_score
    score_boundary = (boundary_ratio + boundary_ratio_50) / 2
    score_vocal = 1.0 - min(vocal_ratio * 3, 1.0)
    score_energy = 1.0 - min(envelope_std, 1.0)
    score_flute = min(flute_ratio * 2, 1.0)
    score_level = min(rms / 0.01, 1.0)

    loopability = (
        0.30 * score_chroma +
        0.25 * score_boundary +
        0.15 * score_vocal +
        0.10 * score_energy +
        0.10 * score_flute +
        0.10 * score_level
    )

    candidate = {
        'idx': i,
        'time': t,
        'end_time': t + LOOP_DUR,
        'chroma_score': chroma_score,
        'rms': rms,
        'boundary_ratio_10ms': boundary_ratio,
        'boundary_ratio_50ms': boundary_ratio_50,
        'rms_start_10ms': rms_start,
        'rms_end_10ms': rms_end,
        'rms_start_50ms': rms_start_50,
        'rms_end_50ms': rms_end_50,
        'vocal_ratio': vocal_ratio,
        'flute_ratio': flute_ratio,
        'envelope_std': envelope_std,
        'sample_jump': sample_jump,
        'score_chroma': score_chroma,
        'score_boundary': score_boundary,
        'score_vocal': score_vocal,
        'score_energy': score_energy,
        'score_flute': score_flute,
        'score_level': score_level,
        'loopability': loopability,
        'segment': segment,
        'segment_stereo': segment_stereo,
        'envelope': envelope,
    }
    candidates.append(candidate)

candidates.sort(key=lambda c: -c['loopability'])

for rank, c in enumerate(candidates):
    marker = " *** CURRENT ***" if abs(c['time'] - BEAT_GRID_S) < 1.0 else ""
    print(f"\n{'_'*60}")
    print(f"  RANK #{rank+1}: t={c['time']:.2f}s - {c['end_time']:.2f}s{marker}")
    print(f"{'_'*60}")
    print(f"  Loopability Score: {c['loopability']:.4f}")
    print(f"  |-- Chroma match:      {c['score_chroma']:.3f}  (raw: {c['chroma_score']:.4f})")
    print(f"  |-- Boundary smooth:   {c['score_boundary']:.3f}  (10ms: {c['boundary_ratio_10ms']:.3f}, 50ms: {c['boundary_ratio_50ms']:.3f})")
    print(f"  |-- Vocal cleanliness: {c['score_vocal']:.3f}  (vocal ratio: {c['vocal_ratio']:.4f})")
    print(f"  |-- Energy consistency: {c['score_energy']:.3f}  (envelope CV: {c['envelope_std']:.3f})")
    print(f"  |-- Flute presence:    {c['score_flute']:.3f}  (flute ratio: {c['flute_ratio']:.4f})")
    print(f"  |-- Level:             {c['score_level']:.3f}  (RMS: {c['rms']:.6f})")
    print(f"  Boundary: start_RMS={c['rms_start_10ms']:.6f}, end_RMS={c['rms_end_10ms']:.6f}")
    print(f"  Sample jump at loop: {c['sample_jump']:.6f}")

# --- Export top candidates ---
print("\n" + "=" * 70)
print("EXPORTING TOP CANDIDATES")
print("=" * 70)

TOP_N = min(5, len(candidates))

for rank in range(TOP_N):
    c = candidates[rank]
    t_label = f"{c['time']:.1f}"

    # Single instance
    single_path_wav = os.path.join(OUT_DIR, f"flute-candidate-{t_label}s.wav")
    sf.write(single_path_wav, c['segment_stereo'], SR)

    single_path_mp3 = os.path.join(OUT_DIR, f"flute-candidate-{t_label}s.mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", single_path_wav,
        "-codec:a", "libmp3lame", "-b:a", "320k",
        single_path_mp3
    ], capture_output=True)
    os.remove(single_path_wav)

    # 2-loop test (play it twice back-to-back with tiny edge fades)
    seg = c['segment_stereo'].copy()

    fade_samps = int(0.003 * SR)
    fade_in_env = np.linspace(0, 1, fade_samps)
    fade_out_env = np.linspace(1, 0, fade_samps)

    if seg.ndim == 2:
        seg[:fade_samps, 0] *= fade_in_env
        seg[:fade_samps, 1] *= fade_in_env
        seg[-fade_samps:, 0] *= fade_out_env
        seg[-fade_samps:, 1] *= fade_out_env
        double = np.concatenate([seg, seg], axis=0)
    else:
        seg[:fade_samps] *= fade_in_env
        seg[-fade_samps:] *= fade_out_env
        double = np.concatenate([seg, seg])

    loop_path_wav = os.path.join(OUT_DIR, f"flute-loop-test-{t_label}s.wav")
    sf.write(loop_path_wav, double, SR)

    loop_path_mp3 = os.path.join(OUT_DIR, f"flute-loop-test-{t_label}s.mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", loop_path_wav,
        "-codec:a", "libmp3lame", "-b:a", "320k",
        loop_path_mp3
    ], capture_output=True)
    os.remove(loop_path_wav)

    marker = " <-- CURRENT" if abs(c['time'] - BEAT_GRID_S) < 1.0 else ""
    print(f"  #{rank+1}: {single_path_mp3}{marker}")
    print(f"       {loop_path_mp3}")

# --- Final summary ---
print("\n" + "=" * 70)
print("FINAL RANKING SUMMARY")
print("=" * 70)
print(f"{'Rank':<5} {'Time':>8} {'Loopability':>12} {'Chroma':>8} {'Boundary':>9} {'Vocal':>7} {'Notes'}")
print("-" * 70)

for rank, c in enumerate(candidates):
    notes = []
    if abs(c['time'] - BEAT_GRID_S) < 1.0:
        notes.append("CURRENT")
    if abs(c['time'] - 284.9) < 2.0:
        notes.append("~284.9s (known alt)")
    if c['vocal_ratio'] > 0.15:
        notes.append("vocal bleed!")
    if c['boundary_ratio_50ms'] > 0.8:
        notes.append("clean boundary")
    if c['rms'] < 0.001:
        notes.append("very quiet")

    note_str = ", ".join(notes) if notes else ""
    star = " *" if rank < TOP_N else ""
    print(f"  {rank+1:<3} {c['time']:>7.2f}s {c['loopability']:>11.4f} "
          f"{c['chroma_score']:>7.4f} {c['score_boundary']:>8.3f} "
          f"{c['score_vocal']:>6.3f}  {note_str}{star}")

print()
print(f"  * = exported as MP3 (top {TOP_N})")
print(f"  Files in: {OUT_DIR}/")
print(f"  Single: flute-candidate-{{time}}s.mp3")
print(f"  Loop:   flute-loop-test-{{time}}s.mp3")
print()
print("Listen to the loop tests back-to-back to hear which loops most cleanly!")
