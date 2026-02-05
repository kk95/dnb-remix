#!/usr/bin/env python3
"""
Recreate the EXACT djay transition from the screenshot.

Saathiya's flute ending → Kho Gayi's beat entry
8 bars overlap, with:
  - Volume: "Fade in fast out"
  - EQ: "Center bass swap" (stems handle this)
  - Effect: "Low pass filter in" on Kho Gayi

Step 1: Detect precise BPMs and beat grids
Step 2: Extract transition sections
Step 3: Tempo-match KG to Saathiya EXACTLY
Step 4: Align on beat grid
Step 5: Create the crossfade with multiple offset candidates
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/transition-tests"
SR = 44100


def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def load_mono(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def write_wav(path, audio, sr=SR):
    sf.write(path, audio, sr, subtype='PCM_16')
    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {path} ({len(audio)/sr:.1f}s, {size_mb:.1f}MB)")


def to_mp3(wav_path):
    mp3 = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3
    ], capture_output=True, check=True)
    print(f"  → {mp3}")
    return mp3


# ── Step 1: Precise BPM & Beat Grid Detection ──────────────────────────

def detect_beats(audio_path, label):
    """Detect precise BPM and beat positions."""
    print(f"\n  Analyzing {label}...")
    y = load_mono(audio_path)

    # Use librosa's beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=SR)

    # Get tempo as scalar
    if hasattr(tempo, '__len__'):
        tempo = tempo[0]

    # Also compute tempo from beat intervals for more precision
    if len(beat_times) > 2:
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        interval_bpm = 60.0 / median_interval
        print(f"  librosa tempo:    {tempo:.2f} BPM")
        print(f"  interval tempo:   {interval_bpm:.2f} BPM")
        print(f"  beat count:       {len(beat_times)} beats")
        print(f"  first beat:       {beat_times[0]:.3f}s")
        print(f"  median interval:  {median_interval:.4f}s")
        return interval_bpm, beat_times, median_interval
    else:
        return tempo, beat_times, 60.0 / tempo


log("Step 1: Precise BPM detection")

saathiya_bpm, saathiya_beats, saathiya_interval = detect_beats(
    f"{PROJECT}/wavs/saathiya.wav", "Saathiya (full)")

kg_bpm, kg_beats, kg_interval = detect_beats(
    f"{PROJECT}/wavs/kho-gayi.wav", "Kho Gayi (full)")

print(f"\n  BPM Summary:")
print(f"    Saathiya: {saathiya_bpm:.3f} BPM (interval: {saathiya_interval:.4f}s)")
print(f"    Kho Gayi: {kg_bpm:.3f} BPM (interval: {kg_interval:.4f}s)")
print(f"    Stretch ratio needed: {saathiya_bpm/kg_bpm:.6f}")


# ── Step 2: Extract the transition sections ─────────────────────────────

log("Step 2: Extracting transition sections")

# From the screenshot: Saathiya's ending flute (around 5:07-end)
# and Kho Gayi's beat entry (around 0:20-0:50)
# But we need more context, so let's take generous windows

# Saathiya: last ~90 seconds (from ~4:00 to end at 5:27)
s_full = load_stereo(f"{PROJECT}/wavs/saathiya.wav")
s_start_s = 240.0  # 4:00
s_section = s_full[int(s_start_s * SR):]
print(f"  Saathiya section: {s_start_s:.0f}s → end ({len(s_section)/SR:.1f}s)")

# Kho Gayi: first ~90 seconds
kg_full = load_stereo(f"{PROJECT}/wavs/kho-gayi.wav")
kg_section = kg_full[:int(90 * SR)]
print(f"  Kho Gayi section: 0s → 90s ({len(kg_section)/SR:.1f}s)")


# ── Step 3: Precise tempo stretch ───────────────────────────────────────

log("Step 3: Stretching Kho Gayi to EXACT Saathiya tempo")

stretch_ratio = saathiya_bpm / kg_bpm
print(f"  Ratio: {kg_bpm:.3f} → {saathiya_bpm:.3f} BPM = {stretch_ratio:.6f}x")

os.makedirs(OUTPUT, exist_ok=True)

# Stretch the full KG WAV with precise ratio
kg_input = f"{PROJECT}/wavs/kho-gayi.wav"
kg_stretched_path = f"{OUTPUT}/kg-stretched-precise.wav"

print(f"  Stretching via rubberband...")
subprocess.run([
    'rubberband', '--tempo', str(stretch_ratio),
    '--crisp', '3',  # balanced for mixed content
    kg_input, kg_stretched_path
], capture_output=True, check=True)

kg_stretched = load_stereo(kg_stretched_path)
print(f"  Stretched KG: {len(kg_stretched)/SR:.1f}s")

# Also stretch individual stems
for stem, crisp in [('drums', 3), ('bass', 5), ('other', 5)]:
    in_path = f"{STEMS}/kho-gayi/{stem}.wav"
    out_path = f"{OUTPUT}/kg-{stem}-precise.wav"
    subprocess.run([
        'rubberband', '--tempo', str(stretch_ratio),
        '--crisp', str(crisp),
        in_path, out_path
    ], capture_output=True, check=True)
    print(f"  → {out_path}")


# ── Step 4: Find beat alignment ─────────────────────────────────────────

log("Step 4: Beat grid alignment")

# Detect beats on the stretched KG
kg_stretched_mono = load_mono(kg_stretched_path)
kg_tempo2, kg_beats2, kg_interval2 = detect_beats(kg_stretched_path, "KG stretched")
print(f"  Stretched KG tempo: {kg_tempo2:.3f} BPM (should be ~{saathiya_bpm:.3f})")

# Find the beat in Saathiya closest to 5:07 (the flute section start)
flute_start = 307.0  # 5:07
nearest_s_beat_idx = np.argmin(np.abs(saathiya_beats - flute_start))
nearest_s_beat = saathiya_beats[nearest_s_beat_idx]
print(f"\n  Flute start: {flute_start:.1f}s")
print(f"  Nearest Saathiya beat: {nearest_s_beat:.3f}s (beat #{nearest_s_beat_idx})")

# The djay transition is 8 bars = 32 beats before the end
# The crossover point (where KG takes over) is roughly in the middle
# So KG's beat entry point should be somewhere in its first 30 seconds

# Find a strong downbeat in KG (look for beat closest to 0:24 as mentioned in handoff)
kg_entry_target = 24.0
nearest_kg_beat_idx = np.argmin(np.abs(kg_beats2 - kg_entry_target))
nearest_kg_beat = kg_beats2[nearest_kg_beat_idx]
print(f"  KG entry target: {kg_entry_target:.1f}s")
print(f"  Nearest KG beat: {nearest_kg_beat:.3f}s (beat #{nearest_kg_beat_idx})")

# For the transition, we want:
# Saathiya plays → at some point, KG starts underneath
# The alignment: saathiya_beat aligns with kg_beat
# So the offset = saathiya_beat_time - kg_beat_time
# (how many seconds into the output timeline KG audio starts)

# But we want to try MULTIPLE alignments around the flute section
# because the "perfect" alignment might be a beat or two off

# Generate clips at different beat offsets around the flute section
print(f"\n  Generating alignment candidates...")
print(f"  Saathiya beat interval: {saathiya_interval:.4f}s")


# ── Step 5: Create transition clips ─────────────────────────────────────

log("Step 5: Creating transition clips with djay-style crossfade")

# Load Saathiya stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")

# Load KG stretched stems
kg_drums = load_stereo(f"{OUTPUT}/kg-drums-precise.wav")
kg_bass = load_stereo(f"{OUTPUT}/kg-bass-precise.wav")
kg_other = load_stereo(f"{OUTPUT}/kg-other-precise.wav")

# Parameters
bar_duration = 4 * saathiya_interval  # seconds per bar
overlap_bars = 8
overlap_s = overlap_bars * bar_duration
overlap_samples = int(overlap_s * SR)

print(f"  Bar duration: {bar_duration:.3f}s")
print(f"  8-bar overlap: {overlap_s:.2f}s ({overlap_samples} samples)")

# The transition in the screenshot:
# - Saathiya's flute ending plays
# - KG beat enters muffled, filter opens, volume swells
# - At the crossover, bass ownership transfers
# - Saathiya fades out quickly after crossover

# Let's try beats around the flute section (5:07)
# Try aligning KG's beat entry at different Saathiya beats near the flute

# Get Saathiya beats near the flute section (4:40 - 5:21)
flute_region_beats = saathiya_beats[
    (saathiya_beats >= 280) & (saathiya_beats <= 321)
]

# For each candidate, KG entry starts at beat - 8 bars
# (KG enters 8 bars before the "drop" point)
# Try every 4th beat (= every bar) for bar-level alignment
candidates = flute_region_beats[::4]  # every bar

print(f"  Testing {len(candidates)} alignment points...")

# The KG entry point: we want the beat drop
# Let's find the first strong beat after the intro
# Look at KG onset strength to find the drop
kg_onset = librosa.onset.onset_strength(y=kg_stretched_mono, sr=SR)
kg_onset_times = librosa.times_like(kg_onset, sr=SR)

# Find beats that are on a bar boundary (every 4 beats)
kg_bar_beats = kg_beats2[::4]
# Find bars near 24s
kg_drop_candidates = kg_bar_beats[(kg_bar_beats >= 20) & (kg_bar_beats <= 28)]
if len(kg_drop_candidates) > 0:
    kg_drop_beat = kg_drop_candidates[0]  # first bar beat near 24s
else:
    kg_drop_beat = nearest_kg_beat
print(f"  KG drop beat: {kg_drop_beat:.3f}s")

# For each Saathiya alignment point, create a 45-second clip:
# - 10s of pure Saathiya before transition
# - 8 bars of crossfade
# - 10s of pure KG after transition

clip_count = 0
for align_beat in candidates:
    # KG starts playing at: align_beat - overlap_s
    # (the transition happens over 8 bars leading to align_beat)
    transition_end = align_beat  # this is where KG fully takes over
    transition_start = transition_end - overlap_s

    # Clip window: 10s before transition → 10s after
    clip_start = max(0, transition_start - 10)
    clip_end = min(len(s_vocals) / SR, transition_end + 10)
    clip_start_sample = int(clip_start * SR)
    clip_end_sample = int(clip_end * SR)
    clip_len = clip_end_sample - clip_start_sample

    # KG audio offset: KG's drop_beat should align with transition_end
    # So KG starts at: transition_start - (kg_drop_beat - overlap_s)
    # Wait, let me think...
    # At transition_start, KG begins to enter
    # At transition_end, KG is fully in
    # KG's audio should be positioned so that kg_drop_beat aligns with transition_end
    kg_audio_start_time = transition_end - kg_drop_beat  # when KG audio "starts" in the timeline

    # Extract Saathiya portion
    s_voc_clip = s_vocals[clip_start_sample:clip_end_sample]
    s_oth_clip = s_other[clip_start_sample:clip_end_sample]

    # Extract KG portion (offset from timeline)
    kg_clip_start_in_audio = clip_start - kg_audio_start_time
    kg_sample_start = int(kg_clip_start_in_audio * SR)
    kg_sample_end = kg_sample_start + clip_len

    # Handle boundaries
    if kg_sample_start < 0 or kg_sample_end > len(kg_drums):
        # Pad with silence where needed
        kg_d = np.zeros((clip_len, 2))
        kg_b = np.zeros((clip_len, 2))
        kg_o = np.zeros((clip_len, 2))

        src_start = max(0, kg_sample_start)
        src_end = min(len(kg_drums), kg_sample_end)
        dst_start = max(0, -kg_sample_start)
        dst_end = dst_start + (src_end - src_start)

        if src_end > src_start and dst_end <= clip_len:
            kg_d[dst_start:dst_end] = kg_drums[src_start:src_end]
            kg_b[dst_start:dst_end] = kg_bass[src_start:src_end]
            kg_o[dst_start:dst_end] = kg_other[src_start:src_end]
    else:
        kg_d = kg_drums[kg_sample_start:kg_sample_end]
        kg_b = kg_bass[kg_sample_start:kg_sample_end]
        kg_o = kg_other[kg_sample_start:kg_sample_end]

    # Ensure all clips are same length
    min_len = min(len(s_voc_clip), len(s_oth_clip), len(kg_d), len(kg_b), clip_len)
    s_voc_clip = s_voc_clip[:min_len]
    s_oth_clip = s_oth_clip[:min_len]
    kg_d = kg_d[:min_len]
    kg_b = kg_b[:min_len]
    kg_o = kg_o[:min_len]

    # ── Create crossfade envelopes ──

    # Time array relative to clip start
    t = np.arange(min_len) / SR + clip_start

    # Transition region
    trans_start_t = transition_start
    trans_end_t = transition_end

    # Saathiya envelope: full → quick fadeout
    # "Fade in fast out" = KG fades in slowly, then Saathiya drops fast
    s_env = np.ones(min_len)
    for i in range(min_len):
        ti = t[i]
        if ti > trans_end_t:
            # Quick fadeout over ~2 bars after crossover
            fadeout_dur = 2 * bar_duration
            progress = min(1.0, (ti - trans_end_t) / fadeout_dur)
            s_env[i] = 1.0 - progress
        elif ti > trans_start_t:
            # Gradual reduction during transition (keep highs, lose bass)
            progress = (ti - trans_start_t) / overlap_s
            s_env[i] = 1.0 - 0.3 * progress  # only drops to 0.7 during overlap

    # KG envelope: silence → muffled fade in → full
    kg_env = np.zeros(min_len)
    for i in range(min_len):
        ti = t[i]
        if ti >= trans_end_t:
            kg_env[i] = 1.0
        elif ti >= trans_start_t:
            progress = (ti - trans_start_t) / overlap_s
            # Slow rise — "fade in" part of "fade in fast out"
            kg_env[i] = progress ** 0.7  # slightly faster than linear

    # Low-pass filter blend (0 = muffled at 350Hz, 1 = fully open)
    filter_blend = np.zeros(min_len)
    for i in range(min_len):
        ti = t[i]
        if ti >= trans_end_t:
            filter_blend[i] = 1.0
        elif ti >= trans_start_t:
            progress = (ti - trans_start_t) / overlap_s
            # Filter opens progressively
            filter_blend[i] = progress

    # ── Apply processing ──

    # KG combined
    kg_combined = kg_d * 0.9 + kg_b * 0.75 + kg_o * 0.2

    # Create muffled version (lowpass at 350Hz)
    from scipy.signal import butter, filtfilt
    nyq = SR / 2
    b_lo, a_lo = butter(4, 350 / nyq, btype='low')
    b_mid, a_mid = butter(4, 2000 / nyq, btype='low')

    kg_muffled = np.column_stack([
        filtfilt(b_lo, a_lo, kg_combined[:, 0]),
        filtfilt(b_lo, a_lo, kg_combined[:, 1])
    ])
    kg_mid = np.column_stack([
        filtfilt(b_mid, a_mid, kg_combined[:, 0]),
        filtfilt(b_mid, a_mid, kg_combined[:, 1])
    ])

    # Blend: muffled → mid → open based on filter envelope
    kg_filtered = np.zeros_like(kg_combined)
    for ch in range(2):
        low_mask = filter_blend <= 0.5
        high_mask = filter_blend > 0.5
        t_low = filter_blend * 2
        t_high = (filter_blend - 0.5) * 2
        kg_filtered[low_mask, ch] = (
            kg_muffled[low_mask, ch] * (1 - t_low[low_mask]) +
            kg_mid[low_mask, ch] * t_low[low_mask]
        )
        kg_filtered[high_mask, ch] = (
            kg_mid[high_mask, ch] * (1 - t_high[high_mask]) +
            kg_combined[high_mask, ch] * t_high[high_mask]
        )

    # Final mix
    mix = (
        s_voc_clip * 0.85 * s_env[:, np.newaxis] +
        s_oth_clip * 0.70 * s_env[:, np.newaxis] +
        kg_filtered * kg_env[:, np.newaxis]
    )

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    # Save
    label = f"{int(align_beat//60)}m{int(align_beat%60):02d}s"
    wav_path = f"{OUTPUT}/transition-at-{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)
    clip_count += 1

print(f"\n  Generated {clip_count} transition clips.")
print(f"  Listen: open {OUTPUT}/transition-at-*.mp3")

# Also create a "raw overlay" version (no effects, just volume)
# at the best-looking alignment point for A/B comparison
log("Bonus: Raw overlay (no filter, just volume crossfade)")

best_align = candidates[len(candidates)//2]  # middle of flute section
label = f"raw-{int(best_align//60)}m{int(best_align%60):02d}s"

# Same extraction as above but simpler mix
transition_end = best_align
transition_start = transition_end - overlap_s
clip_start = max(0, transition_start - 10)
clip_end = min(len(s_vocals) / SR, transition_end + 10)
clip_start_sample = int(clip_start * SR)
clip_end_sample = int(clip_end * SR)
clip_len = clip_end_sample - clip_start_sample

# Use full Saathiya (not stems) for the raw version
s_full_clip = s_full[clip_start_sample:clip_end_sample]

# KG full stretched
kg_audio_start_time = transition_end - kg_drop_beat
kg_clip_start_in_audio = clip_start - kg_audio_start_time
kg_sample_start = int(kg_clip_start_in_audio * SR)
kg_sample_end = kg_sample_start + clip_len

kg_full_stretched = load_stereo(kg_stretched_path)
if kg_sample_start < 0 or kg_sample_end > len(kg_full_stretched):
    kg_clip = np.zeros((clip_len, 2))
    src_start = max(0, kg_sample_start)
    src_end = min(len(kg_full_stretched), kg_sample_end)
    dst_start = max(0, -kg_sample_start)
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start and dst_end <= clip_len:
        kg_clip[dst_start:dst_end] = kg_full_stretched[src_start:src_end]
else:
    kg_clip = kg_full_stretched[kg_sample_start:kg_sample_end]

min_len = min(len(s_full_clip), len(kg_clip))
s_full_clip = s_full_clip[:min_len]
kg_clip = kg_clip[:min_len]

# Simple crossfade
t = np.arange(min_len) / SR + clip_start
s_env = np.ones(min_len)
kg_env = np.zeros(min_len)
for i in range(min_len):
    ti = t[i]
    if ti >= transition_end:
        s_env[i] = max(0, 1.0 - (ti - transition_end) / (2 * bar_duration))
        kg_env[i] = 1.0
    elif ti >= transition_start:
        p = (ti - transition_start) / overlap_s
        kg_env[i] = p
        s_env[i] = 1.0

raw_mix = s_full_clip * s_env[:, np.newaxis] + kg_clip * 0.8 * kg_env[:, np.newaxis]
peak = np.max(np.abs(raw_mix))
if peak > 0.95:
    raw_mix *= 0.95 / peak

wav_path = f"{OUTPUT}/{label}.wav"
write_wav(wav_path, raw_mix)
to_mp3(wav_path)

log("DONE — Listen and compare!")
print(f"  Stem transitions:  open {OUTPUT}/transition-at-*.mp3")
print(f"  Raw overlay:       open {OUTPUT}/{label}.mp3")
print(f"\n  Pick the one where beats lock, then we'll extend it.")
