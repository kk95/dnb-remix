#!/usr/bin/env python3
"""
Beat-locked drop transition using pyrubberband timemap_stretch.

Fixes the BPM drift problem: instead of a single global stretch (which lands
at ~86.1 BPM instead of 87.6), this maps every KG beat to the Saathiya beat
grid so they stay locked indefinitely.

Flow:
  1. Detect beats in both tracks with librosa
  2. Build a timemap: KG source beat samples → target (Saathiya-tempo) samples
  3. pyrubberband.timemap_stretch each KG stem → pitch-preserved, beat-locked
  4. Create the DROP transition with beat-locked stems
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/beat-locked"
SR = 44100

# Known precise BPMs
SAATHIYA_BPM = 87.593
KG_BPM = 95.703
SAATHIYA_BEAT = 60.0 / SAATHIYA_BPM   # 0.6850s
KG_BEAT = 60.0 / KG_BPM               # 0.6269s
BAR_S = 4 * SAATHIYA_BEAT              # 2.740s

os.makedirs(OUTPUT, exist_ok=True)


def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def load_mono(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def write_wav(path, audio):
    sf.write(path, audio, SR, subtype='PCM_16')
    print(f"  → {path} ({len(audio)/SR:.1f}s)")


def to_mp3(wav_path):
    mp3 = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3
    ], capture_output=True, check=True)
    os.remove(wav_path)
    print(f"  → {mp3}")
    return mp3


# ── Step 1: Detect beats ──────────────────────────────────────────────

log("Step 1: Detecting beats")

# Saathiya beats
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
s_detected_bpm = 60.0 / np.median(np.diff(s_beats))
print(f"  Saathiya detected: {s_detected_bpm:.3f} BPM (expected: {SAATHIYA_BPM})")

# KG beats
kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
_, kg_beat_frames = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats = librosa.frames_to_time(kg_beat_frames, sr=SR)
kg_detected_bpm = 60.0 / np.median(np.diff(kg_beats))

# KG might be detected at double tempo
if kg_detected_bpm > 120:
    kg_detected_bpm /= 2
    kg_beats = kg_beats[::2]

print(f"  Kho Gayi detected: {kg_detected_bpm:.3f} BPM (expected: {KG_BPM})")
print(f"  Saathiya beats: {len(s_beats)}, KG beats: {len(kg_beats)}")


# ── Step 2: Build timemap ─────────────────────────────────────────────

log("Step 2: Building timemap (KG → Saathiya tempo)")

# The timemap maps each source sample position to a target sample position.
# We want: each KG beat lands at Saathiya beat intervals from the start.
#
# Source:  beat[0] at t_kg[0], beat[1] at t_kg[1], ...
# Target:  beat[0] at t_kg[0], beat[1] at t_kg[0] + 1*SAATHIYA_BEAT, ...
#
# The first beat stays put. Each subsequent beat is placed at the
# Saathiya-tempo spacing from the first beat.

kg_audio_len = len(kg_mono)


def build_timemap(beats, audio_len):
    """Build timemap that re-tempos KG beats to Saathiya beat spacing."""
    time_map = []

    # Start: sample 0 maps to sample 0
    time_map.append((0, 0))

    # Map each beat
    first_beat = beats[0]
    for i, beat_time in enumerate(beats):
        src_sample = int(beat_time * SR)
        # Target: first beat stays, subsequent beats at Saathiya spacing
        target_time = first_beat + i * SAATHIYA_BEAT
        tgt_sample = int(target_time * SR)
        time_map.append((src_sample, tgt_sample))

    # End: map last sample
    # Estimate target end based on the stretch of the last segment
    last_src = int(beats[-1] * SR)
    last_tgt = int((first_beat + (len(beats) - 1) * SAATHIYA_BEAT) * SR)
    remaining_src = audio_len - last_src
    # Stretch remaining at the same ratio as the last beat interval
    if len(beats) > 1:
        last_src_interval = beats[-1] - beats[-2]
        ratio = SAATHIYA_BEAT / last_src_interval
    else:
        ratio = SAATHIYA_BEAT / KG_BEAT
    remaining_tgt = int(remaining_src * ratio)
    end_tgt = last_tgt + remaining_tgt

    time_map.append((audio_len, end_tgt))

    # Remove any duplicates or non-monotonic entries
    clean_map = [time_map[0]]
    for entry in time_map[1:]:
        if entry[0] > clean_map[-1][0] and entry[1] > clean_map[-1][1]:
            clean_map.append(entry)

    # Ensure last entry has correct source length
    if clean_map[-1][0] != audio_len:
        # Extrapolate target
        src_delta = audio_len - clean_map[-1][0]
        tgt_delta = int(src_delta * ratio)
        clean_map.append((audio_len, clean_map[-1][1] + tgt_delta))

    return clean_map


time_map = build_timemap(kg_beats, kg_audio_len)
target_duration = time_map[-1][1] / SR
print(f"  Timemap entries: {len(time_map)}")
print(f"  Source duration: {kg_audio_len/SR:.1f}s")
print(f"  Target duration: {target_duration:.1f}s")
print(f"  Effective ratio: {target_duration / (kg_audio_len/SR):.6f}")
print(f"  Expected ratio:  {SAATHIYA_BEAT/KG_BEAT:.6f}")

# Verify: check beat spacing in target
target_beats = []
for src_s, tgt_s in time_map[1:-1]:  # skip start/end
    target_beats.append(tgt_s / SR)
if len(target_beats) > 2:
    diffs = np.diff(target_beats)
    print(f"  Target beat interval: {np.mean(diffs):.4f}s (should be {SAATHIYA_BEAT:.4f}s)")
    print(f"  Target BPM: {60.0/np.mean(diffs):.3f} (should be {SAATHIYA_BPM})")


# ── Step 3: Warp KG stems ─────────────────────────────────────────────

log("Step 3: Warping KG stems with timemap_stretch")

warped_stems = {}
for stem in ['drums', 'bass', 'other']:
    out_path = f"{OUTPUT}/kg-{stem}-warped.wav"
    if os.path.exists(out_path):
        print(f"  {stem}: using cached {out_path}")
        warped_stems[stem] = out_path
        continue

    stem_path = f"{STEMS}/kho-gayi/{stem}.wav"
    audio, _ = sf.read(stem_path, dtype='float64')

    # Build timemap for this stem's length
    stem_len = len(audio)
    stem_map = build_timemap(kg_beats, stem_len)

    print(f"  {stem}: warping {stem_len/SR:.1f}s → {stem_map[-1][1]/SR:.1f}s ...")
    warped = pyrb.timemap_stretch(audio, SR, stem_map)
    sf.write(out_path, warped, SR, subtype='PCM_16')
    warped_stems[stem] = out_path
    print(f"  → {out_path} ({len(warped)/SR:.1f}s)")

# Also warp full KG for beat verification
kg_full_warped_path = f"{OUTPUT}/kg-full-warped.wav"
if not os.path.exists(kg_full_warped_path):
    print(f"  full: warping...")
    warped_full = pyrb.timemap_stretch(kg_mono, SR, time_map)
    sf.write(kg_full_warped_path, warped_full, SR, subtype='PCM_16')
    print(f"  → {kg_full_warped_path}")


# ── Step 4: Verify beats in warped audio ──────────────────────────────

log("Step 4: Verifying beat-lock")

warped_mono = load_mono(kg_full_warped_path)
_, warped_frames = librosa.beat.beat_track(y=warped_mono, sr=SR, units='frames')
warped_beats = librosa.frames_to_time(warped_frames, sr=SR)
warped_bpm = 60.0 / np.median(np.diff(warped_beats))
if warped_bpm > 120:
    warped_bpm /= 2

print(f"  Warped KG BPM: {warped_bpm:.3f} (target: {SAATHIYA_BPM})")
bpm_error = abs(warped_bpm - SAATHIYA_BPM) / SAATHIYA_BPM * 100
print(f"  BPM error: {bpm_error:.2f}%")
if bpm_error < 0.5:
    print(f"  ✓ Beat-lock looks good!")
else:
    print(f"  ⚠ BPM error still significant — may need manual beat grid")


# ── Step 5: Build drop transition with beat-locked stems ──────────────

log("Step 5: Creating drop transition clips")

# Load Saathiya stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

# Load warped KG stems
kg_drums = load_stereo(warped_stems['drums'])
kg_bass = load_stereo(warped_stems['bass'])
kg_other_stem = load_stereo(warped_stems['other'])

# KG rhythm = drums + bass
min_kg = min(len(kg_drums), len(kg_bass))
kg_rhythm = kg_drums[:min_kg] * 0.9 + kg_bass[:min_kg] * 0.75

# Pre-compute filtered KG (for muffled buildup)
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')

print(f"  Computing filtered KG versions...")
kg_muffled = np.column_stack([
    filtfilt(b_lo, a_lo, kg_rhythm[:, 0]),
    filtfilt(b_lo, a_lo, kg_rhythm[:, 1])
])
kg_mid_filtered = np.column_stack([
    filtfilt(b_mid, a_mid, kg_rhythm[:, 0]),
    filtfilt(b_mid, a_mid, kg_rhythm[:, 1])
])

# Saathiya drop point: ~5:07 = 307s
s_drop_target = 307.0
s_drop_idx = np.argmin(np.abs(s_beats - s_drop_target))
s_drop = s_beats[s_drop_idx]
print(f"  Saathiya drop beat: {s_drop:.3f}s")

# Find warped KG beats in the 22-26s range
# After warping, KG beats should be at: first_beat + i * SAATHIYA_BEAT
warped_beat_times = kg_beats[0] + np.arange(len(kg_beats)) * SAATHIYA_BEAT
kg_candidates_range = (22, 27)
kg_candidates = warped_beat_times[
    (warped_beat_times >= kg_candidates_range[0]) &
    (warped_beat_times <= kg_candidates_range[1])
]

print(f"  KG warped beats in {kg_candidates_range[0]}-{kg_candidates_range[1]}s:")
for b in kg_candidates:
    print(f"    {b:.3f}s")

# Also include the known best offset (24.599s)
# Find nearest warped beat to 24.599
best_known = 24.599
nearest_idx = np.argmin(np.abs(warped_beat_times - best_known))
nearest_beat = warped_beat_times[nearest_idx]
print(f"  Nearest warped beat to known best ({best_known}s): {nearest_beat:.3f}s")

# Add it if not already in candidates
if nearest_beat not in kg_candidates:
    kg_candidates = np.append(kg_candidates, nearest_beat)
    kg_candidates.sort()


def extract_kg(src, start, length):
    """Extract a segment from KG audio with boundary clamping."""
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


# Transition params
buildup_bars = 4
ride_bars = 8
pre_context = 8

beat_s = SAATHIYA_BEAT

for ci, kg_drop_beat in enumerate(kg_candidates):
    kg_timeline_offset = s_drop - kg_drop_beat

    buildup_start = s_drop - buildup_bars * BAR_S
    clip_start = max(0, buildup_start - pre_context)
    clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * BAR_S)
    clip_samples = int((clip_end - clip_start) * SR)
    cs = int(clip_start * SR)

    sv = s_vocals[cs:cs+clip_samples]
    so = s_other[cs:cs+clip_samples]
    sd = s_drums[cs:cs+clip_samples]
    sb = s_bass[cs:cs+clip_samples]

    kg_audio_start = clip_start - kg_timeline_offset
    ka_s = int(kg_audio_start * SR)

    kd = extract_kg(kg_rhythm, ka_s, clip_samples)
    kd_muf = extract_kg(kg_muffled, ka_s, clip_samples)
    kd_mid = extract_kg(kg_mid_filtered, ka_s, clip_samples)

    min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))
    sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
    kd, kd_muf, kd_mid = kd[:min_len], kd_muf[:min_len], kd_mid[:min_len]

    # ── Envelopes ──
    t = np.arange(min_len) / SR + clip_start

    kg_vol = np.zeros(min_len)
    kg_filter = np.zeros(min_len)
    s_bass_env = np.ones(min_len)

    for i in range(min_len):
        ti = t[i]
        if ti < buildup_start:
            kg_vol[i] = 0.0
            kg_filter[i] = 0.0
            s_bass_env[i] = 1.0
        elif ti < s_drop:
            p = (ti - buildup_start) / (s_drop - buildup_start)
            kg_vol[i] = 0.15 + 0.35 * p
            kg_filter[i] = p * 0.4
            s_bass_env[i] = 1.0 - 0.5 * p
        else:
            p = min(1.0, (ti - s_drop) / beat_s)
            kg_vol[i] = 0.5 + 0.5 * p
            kg_filter[i] = 0.4 + 0.6 * p
            s_bass_env[i] = 0.5 * (1.0 - p)

    # ── Apply filter blend to KG ──
    kg_with_filter = np.zeros_like(kd)
    for ch in range(2):
        lo = kg_filter <= 0.4
        hi = kg_filter > 0.4
        t_lo = np.clip(kg_filter / 0.4, 0, 1)
        t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
        kg_with_filter[lo, ch] = kd_muf[lo, ch] * (1 - t_lo[lo]) + kd_mid[lo, ch] * t_lo[lo]
        kg_with_filter[hi, ch] = kd_mid[hi, ch] * (1 - t_hi[hi]) + kd[hi, ch] * t_hi[hi]

    # ── Mix ──
    saathiya_mix = (
        sv * 0.85 +
        so * 0.70 +
        sd * 0.40 * s_bass_env[:, np.newaxis] +
        sb * 0.50 * s_bass_env[:, np.newaxis]
    )
    kg_mix = kg_with_filter * kg_vol[:, np.newaxis]
    mix = saathiya_mix + kg_mix

    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    ms_label = int(kg_drop_beat * 1000)
    label = f"drop-{ms_label}ms"
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)

log("ALL DONE!")
print(f"  Beat-locked clips in: {OUTPUT}/")
print(f"  These should NOT drift after the drop point.")
print(f"\n  Listen:  open {OUTPUT}/drop-*.mp3")
