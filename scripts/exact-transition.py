#!/usr/bin/env python3
"""
Recreate the EXACT djay transition.

What the screenshot shows:
  - Saathiya plays its flute ending
  - At ~5:07, the transition begins
  - Kho Gayi enters at its ~0:23-0:24 mark (the beat drop)
  - 8 bars of overlap with crossfade + bass swap + filter
  - Then it's pure Kho Gayi

This script:
  1. Finds the exact beat at Saathiya ~5:07 and KG ~0:23
  2. Aligns those beats
  3. Creates 8-bar crossfade clips
  4. Tries several sub-beat offsets to find the perfect lock
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/exact-transition"
SR = 44100

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
    print(f"  → {mp3}")
    return mp3


# ── Step 1: Precise BPM + Beat Grid ────────────────────────────────────

log("Step 1: Beat grid detection")

# Saathiya
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
s_tempo, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
s_intervals = np.diff(s_beats)
s_bpm = 60.0 / np.median(s_intervals)

print(f"  Saathiya: {s_bpm:.3f} BPM, {len(s_beats)} beats")
print(f"  Beat interval: {np.median(s_intervals):.4f}s")

# Kho Gayi
kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
kg_tempo, kg_beat_frames = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats = librosa.frames_to_time(kg_beat_frames, sr=SR)
kg_intervals = np.diff(kg_beats)
kg_raw_bpm = 60.0 / np.median(kg_intervals)

# librosa might detect double-time — if BPM > 120, halve it
# and keep only every other beat (downbeats)
if kg_raw_bpm > 120:
    kg_bpm = kg_raw_bpm / 2
    kg_beats = kg_beats[::2]  # every other beat = actual beats
    print(f"  KG detected double-time ({kg_raw_bpm:.1f}), using {kg_bpm:.3f} BPM")
else:
    kg_bpm = kg_raw_bpm
    print(f"  KG: {kg_bpm:.3f} BPM")

kg_intervals = np.diff(kg_beats)
print(f"  KG beat interval: {np.median(kg_intervals):.4f}s, {len(kg_beats)} beats")

# Stretch ratio
stretch_ratio = s_bpm / kg_bpm
print(f"\n  Stretch ratio: {kg_bpm:.3f} → {s_bpm:.3f} = {stretch_ratio:.6f}x")

# Bar duration at Saathiya's tempo
bar_s = 4 * (60.0 / s_bpm)
eight_bars_s = 8 * bar_s
print(f"  Bar duration: {bar_s:.3f}s")
print(f"  8 bars: {eight_bars_s:.2f}s")


# ── Step 2: Find the key beats ──────────────────────────────────────────

log("Step 2: Finding alignment beats")

# Saathiya beat closest to 5:07 (307s)
s_target = 307.0
s_beat_idx = np.argmin(np.abs(s_beats - s_target))
s_align_beat = s_beats[s_beat_idx]
print(f"  Saathiya target: {s_target:.0f}s → nearest beat: {s_align_beat:.3f}s (beat #{s_beat_idx})")

# KG beat closest to 0:23-24
# After stretching, KG beats shift by stretch_ratio
kg_beats_stretched = kg_beats / stretch_ratio  # approximate (rubberband does it properly)

kg_target = 23.5  # middle of 23-24 range
kg_beat_idx = np.argmin(np.abs(kg_beats - kg_target))
kg_align_beat_original = kg_beats[kg_beat_idx]
kg_align_beat_stretched = kg_align_beat_original / stretch_ratio  # after stretching
print(f"  KG target: {kg_target:.1f}s → nearest beat: {kg_align_beat_original:.3f}s (beat #{kg_beat_idx})")
print(f"  KG beat after stretch: {kg_align_beat_stretched:.3f}s")

# Also check neighbors
print(f"\n  KG beats near 23-24s (original time):")
for i in range(max(0, kg_beat_idx-3), min(len(kg_beats), kg_beat_idx+4)):
    marker = " ←" if i == kg_beat_idx else ""
    print(f"    beat {i}: {kg_beats[i]:.3f}s (stretched: {kg_beats[i]/stretch_ratio:.3f}s){marker}")


# ── Step 3: Stretch KG audio ────────────────────────────────────────────

log("Step 3: Stretching Kho Gayi to exact tempo")

# Stretch full KG
kg_stretched_path = f"{OUTPUT}/kg-full-stretched.wav"
if not os.path.exists(kg_stretched_path):
    print(f"  Stretching at ratio {stretch_ratio:.6f}...")
    subprocess.run([
        'rubberband', '--tempo', str(stretch_ratio), '--crisp', '3',
        f"{PROJECT}/wavs/kho-gayi.wav", kg_stretched_path
    ], capture_output=True, check=True)
else:
    print(f"  [skip] Already stretched")

# Stretch KG stems
for stem, crisp in [('drums', 3), ('bass', 5), ('other', 5), ('vocals', 5)]:
    out = f"{OUTPUT}/kg-{stem}-stretched.wav"
    if not os.path.exists(out):
        subprocess.run([
            'rubberband', '--tempo', str(stretch_ratio), '--crisp', str(crisp),
            f"{STEMS}/kho-gayi/{stem}.wav", out
        ], capture_output=True, check=True)
        print(f"  → {out}")
    else:
        print(f"  [skip] {out}")

# Verify stretched BPM
kg_stretched_mono = load_mono(kg_stretched_path)
_, kg_s_frames = librosa.beat.beat_track(y=kg_stretched_mono, sr=SR, units='frames')
kg_s_beats = librosa.frames_to_time(kg_s_frames, sr=SR)
if len(kg_s_beats) > 2:
    kg_s_intervals = np.diff(kg_s_beats)
    kg_s_bpm_raw = 60.0 / np.median(kg_s_intervals)
    if kg_s_bpm_raw > 120:
        kg_s_bpm = kg_s_bpm_raw / 2
        kg_s_beats = kg_s_beats[::2]
    else:
        kg_s_bpm = kg_s_bpm_raw
    print(f"\n  Stretched KG BPM: {kg_s_bpm:.3f} (target: {s_bpm:.3f})")
    print(f"  Error: {abs(kg_s_bpm - s_bpm):.3f} BPM ({abs(kg_s_bpm - s_bpm)/s_bpm*100:.2f}%)")

    # Find KG stretched beat closest to 23-24s (in stretched time)
    kg_s_target = kg_align_beat_original / stretch_ratio
    kg_s_idx = np.argmin(np.abs(kg_s_beats - kg_s_target))
    kg_s_align = kg_s_beats[kg_s_idx]
    print(f"  KG stretched beat near {kg_s_target:.1f}s: {kg_s_align:.3f}s (beat #{kg_s_idx})")

    print(f"\n  KG stretched beats near target:")
    for i in range(max(0, kg_s_idx-3), min(len(kg_s_beats), kg_s_idx+4)):
        marker = " ←" if i == kg_s_idx else ""
        print(f"    beat {i}: {kg_s_beats[i]:.3f}s{marker}")


# ── Step 4: Create transition clips ────────────────────────────────────

log("Step 4: Creating transition clips")

# Load audio
s_full = load_stereo(f"{PROJECT}/wavs/saathiya.wav")
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")

kg_full = load_stereo(kg_stretched_path)
kg_drums = load_stereo(f"{OUTPUT}/kg-drums-stretched.wav")
kg_bass = load_stereo(f"{OUTPUT}/kg-bass-stretched.wav")
kg_other_stem = load_stereo(f"{OUTPUT}/kg-other-stretched.wav")

print(f"  Saathiya: {len(s_full)/SR:.1f}s")
print(f"  KG stretched: {len(kg_full)/SR:.1f}s")

# The transition concept:
# Timeline:  [...Saathiya playing...]---[8 bar overlap]---[...KG playing...]
#
# At s_align_beat (5:07), KG is at its kg_s_align beat (~24s stretched)
# The 8-bar overlap is centered on this meeting point
# KG enters 4 bars BEFORE the meeting point (muffled, fading in)
# Saathiya exits 4 bars AFTER the meeting point (fading out)

# We'll try multiple KG beats near the target for fine-tuning
if len(kg_s_beats) > 2:
    candidates_kg = []
    for i in range(max(0, kg_s_idx-2), min(len(kg_s_beats), kg_s_idx+3)):
        candidates_kg.append((i, kg_s_beats[i]))
else:
    # Fallback: use calculated positions
    kg_s_align = kg_align_beat_original / stretch_ratio
    candidates_kg = [(0, kg_s_align + offset) for offset in [-1.37, -0.685, 0, 0.685, 1.37]]

print(f"\n  Creating {len(candidates_kg)} clips with different KG beat alignments...")

for candidate_idx, (kg_beat_num, kg_beat_time) in enumerate(candidates_kg):
    # KG's beat at kg_beat_time should align with Saathiya's beat at s_align_beat
    # So in the output timeline:
    #   - KG audio starts at: s_align_beat - kg_beat_time
    kg_timeline_start = s_align_beat - kg_beat_time  # when KG audio begins in Saathiya's timeline

    # The meeting point is at s_align_beat in the output timeline
    # 8-bar overlap: 4 bars before → 4 bars after the meeting point
    overlap_start = s_align_beat - 4 * bar_s
    overlap_end = s_align_beat + 4 * bar_s

    # Clip window: 15s before overlap → 10s after overlap
    clip_start_t = max(0, overlap_start - 15)
    clip_end_t = min(len(s_full) / SR, overlap_end + 10)

    # === VERSION A: Full tracks (like djay) ===

    clip_len = int((clip_end_t - clip_start_t) * SR)
    s_start = int(clip_start_t * SR)
    s_end = s_start + clip_len

    s_clip = s_full[s_start:s_end]

    # KG clip: figure out which part of KG audio maps to this clip window
    kg_audio_offset = clip_start_t - kg_timeline_start  # position in KG audio
    kg_a_start = int(kg_audio_offset * SR)
    kg_a_end = kg_a_start + clip_len

    # Handle KG audio boundaries (might start before KG audio begins)
    kg_clip = np.zeros((clip_len, 2))
    if kg_a_end > 0 and kg_a_start < len(kg_full):
        src_s = max(0, kg_a_start)
        src_e = min(len(kg_full), kg_a_end)
        dst_s = max(0, -kg_a_start)
        dst_e = dst_s + (src_e - src_s)
        if dst_e <= clip_len and src_e > src_s:
            kg_clip[dst_s:dst_e] = kg_full[src_s:src_e]

    actual_len = min(len(s_clip), len(kg_clip))
    s_clip = s_clip[:actual_len]
    kg_clip = kg_clip[:actual_len]

    # Create crossfade envelopes
    t = np.arange(actual_len) / SR + clip_start_t

    # Saathiya envelope: full during its section, fades out after meeting point
    s_env = np.ones(actual_len)
    # KG envelope: fades in before meeting point, full after
    kg_env = np.zeros(actual_len)

    for i in range(actual_len):
        ti = t[i]
        if ti < overlap_start:
            # Pure Saathiya
            s_env[i] = 1.0
            kg_env[i] = 0.0
        elif ti < s_align_beat:
            # First half of overlap: KG fading in, Saathiya still strong
            p = (ti - overlap_start) / (4 * bar_s)  # 0→1 over 4 bars
            s_env[i] = 1.0
            kg_env[i] = p * 0.7  # KG comes up to 70%
        elif ti < overlap_end:
            # Second half: Saathiya fading out fast, KG taking over
            p = (ti - s_align_beat) / (4 * bar_s)  # 0→1 over 4 bars
            s_env[i] = 1.0 - p  # Saathiya fades to 0
            kg_env[i] = 0.7 + 0.3 * p  # KG goes to 100%
        else:
            # Pure KG
            s_env[i] = 0.0
            kg_env[i] = 1.0

    # Mix with envelopes
    mix_a = s_clip * s_env[:, np.newaxis] + kg_clip * kg_env[:, np.newaxis]
    peak = np.max(np.abs(mix_a))
    if peak > 0.95:
        mix_a *= 0.95 / peak

    label = f"full-kgbeat{kg_beat_num}-{kg_beat_time:.1f}s"
    wav_path = f"{OUTPUT}/transition-{label}.wav"
    write_wav(wav_path, mix_a)
    to_mp3(wav_path)

    # === VERSION B: Stems (bass swap) ===

    # Saathiya: vocals + other (melody/flute) — no bass/drums
    s_voc_clip = s_vocals[s_start:s_start+actual_len]
    s_oth_clip = s_other[s_start:s_start+actual_len]

    # KG stems
    kg_d_clip = np.zeros((actual_len, 2))
    kg_b_clip = np.zeros((actual_len, 2))
    kg_o_clip = np.zeros((actual_len, 2))

    if kg_a_end > 0 and kg_a_start < len(kg_drums):
        src_s = max(0, kg_a_start)
        src_e = min(len(kg_drums), kg_a_end)
        dst_s = max(0, -kg_a_start)
        dst_e = dst_s + (src_e - src_s)
        if dst_e <= actual_len and src_e > src_s:
            kg_d_clip[dst_s:dst_e] = kg_drums[src_s:src_e]
            kg_b_clip[dst_s:dst_e] = kg_bass[src_s:src_e]
            kg_o_clip[dst_s:dst_e] = kg_other_stem[src_s:src_e]

    # KG combined with low-pass filter automation
    kg_stem_combined = kg_d_clip * 0.9 + kg_b_clip * 0.75 + kg_o_clip * 0.15

    # Create filtered versions for the "low pass filter in" effect
    nyq = SR / 2
    b_lo, a_lo = butter(4, min(350 / nyq, 0.99), btype='low')
    b_mid, a_mid = butter(4, min(1800 / nyq, 0.99), btype='low')

    kg_muffled = np.column_stack([
        filtfilt(b_lo, a_lo, kg_stem_combined[:, 0]),
        filtfilt(b_lo, a_lo, kg_stem_combined[:, 1])
    ])
    kg_mid = np.column_stack([
        filtfilt(b_mid, a_mid, kg_stem_combined[:, 0]),
        filtfilt(b_mid, a_mid, kg_stem_combined[:, 1])
    ])

    # Filter blend follows KG envelope (0=muffled, 1=open)
    # Normalize kg_env to 0-1 for filter
    filter_blend = np.clip(kg_env / max(kg_env.max(), 0.01), 0, 1)

    kg_filtered = np.zeros_like(kg_stem_combined)
    for ch in range(2):
        lo = filter_blend <= 0.5
        hi = filter_blend > 0.5
        t_lo = np.clip(filter_blend * 2, 0, 1)
        t_hi = np.clip((filter_blend - 0.5) * 2, 0, 1)
        kg_filtered[lo, ch] = kg_muffled[lo, ch] * (1 - t_lo[lo]) + kg_mid[lo, ch] * t_lo[lo]
        kg_filtered[hi, ch] = kg_mid[hi, ch] * (1 - t_hi[hi]) + kg_stem_combined[hi, ch] * t_hi[hi]

    # Stem mix: Saathiya melody owns highs, KG owns lows
    mix_b = (
        s_voc_clip * 0.85 * s_env[:, np.newaxis] +
        s_oth_clip * 0.70 * s_env[:, np.newaxis] +
        kg_filtered * kg_env[:, np.newaxis]
    )
    peak = np.max(np.abs(mix_b))
    if peak > 0.95:
        mix_b *= 0.95 / peak

    label_b = f"stems-kgbeat{kg_beat_num}-{kg_beat_time:.1f}s"
    wav_path_b = f"{OUTPUT}/transition-{label_b}.wav"
    write_wav(wav_path_b, mix_b)
    to_mp3(wav_path_b)


# ── Summary ─────────────────────────────────────────────────────────────

log("DONE!")
print(f"  Created {len(candidates_kg) * 2} clips ({len(candidates_kg)} full + {len(candidates_kg)} stems)")
print(f"\n  Full-track versions (like djay):")
print(f"    open {OUTPUT}/transition-full-*.mp3")
print(f"\n  Stem versions (bass swap + filter):")
print(f"    open {OUTPUT}/transition-stems-*.mp3")
print(f"\n  The KG beat time in the filename tells you which KG beat")
print(f"  is aligned with Saathiya's 5:07 ({s_align_beat:.3f}s)")
print(f"\n  Listen and pick the one where the beats lock!")
