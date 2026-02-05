#!/usr/bin/env python3
"""
1-minute dreamy flute + DnB remix.

4-bar flute loop (Saathiya "other") + 4-bar KG drums+bass loop,
both time-stretched to exact 88.0 BPM so they stay locked together.

Smooth transition from Saathiya into the loop section (no jarring snap).
"""

import os
import subprocess
from datetime import datetime
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from scipy.signal import butter, filtfilt, sosfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM

# Alignment
KG_DROP_S = 33.245
KG_DROP_STRETCHED_S = KG_DROP_S / STRETCH_RATE
BEAT_GRID_S = 306.772
SNAP_S = 307.454

# Timing at target BPM
BEAT_S = 60.0 / TARGET_BPM
BAR_S = 4 * BEAT_S

# Saathiya actual local tempo
SAATHIYA_LOCAL_BPM = 87.939
SAATHIYA_BAR_S = 4 * (60.0 / SAATHIYA_LOCAL_BPM)

# Loop config
LOOP_BARS = 4
N_FLUTE_LOOPS = 6  # ~65s of looped flute
EDGE_FADE_MS = 3   # 3ms micro-fade at loop edges — prevents clicks, zero overlap

os.makedirs(OUTPUT, exist_ok=True)


def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def extract(src, start, length):
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


def to_mp3(wav_path):
    mp3 = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3
    ], capture_output=True, check=True)
    os.remove(wav_path)
    print(f"  → {mp3}")
    return mp3


def distort_bass(audio, drive=6, sub_cutoff=80):
    """Tanh saturation on mids/highs, clean sub preserved."""
    # Split into sub (<80Hz) and mids/highs
    sos_lo = butter(4, sub_cutoff / (SR / 2), btype='low', output='sos')
    sos_hi = butter(4, sub_cutoff / (SR / 2), btype='high', output='sos')

    result = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        sub = sosfilt(sos_lo, audio[:, ch])
        mids = sosfilt(sos_hi, audio[:, ch])
        # Saturate mids/highs
        mids_dist = np.tanh(drive * mids) / np.tanh(drive)
        result[:, ch] = sub + mids_dist
    return result


def make_loop(audio, n_loops, xf_samples):
    """Loop audio with overlapping crossfade at boundaries.

    Consecutive loops overlap by xf_samples. Equal-power (sqrt) curves
    keep energy constant through the blend — no -6dB dip like linear.
    Total length = (loop_len - xf_samples) * n_loops + xf_samples.
    """
    loop_len = len(audio)
    if xf_samples <= 0 or n_loops <= 1:
        if n_loops == 1:
            return audio.copy()
        return np.tile(audio, (n_loops, 1))

    step = loop_len - xf_samples
    total_len = step * n_loops + xf_samples
    looped = np.zeros((total_len, 2))

    # Equal-power crossfade: sqrt curves satisfy fade_in² + fade_out² = 1
    t = np.linspace(0, 1, xf_samples)
    fade_in = np.sqrt(t)[:, np.newaxis]
    fade_out = np.sqrt(1 - t)[:, np.newaxis]

    for i in range(n_loops):
        offset = i * step
        chunk = audio.copy()
        if i > 0:
            chunk[:xf_samples] *= fade_in
        if i < n_loops - 1:
            chunk[-xf_samples:] *= fade_out
        looped[offset:offset + loop_len] += chunk

    return looped


# ── Step 1: Load stems ────────────────────────────────────────────────

log("Step 1: Load all stems")
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass_st = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums_raw = load_stereo(f"{STEMS}/kho-gayi/drums.wav")
kg_bass_raw = load_stereo(f"{STEMS}/kho-gayi/bass.wav")
print(f"  Saathiya: {len(s_vocals)/SR:.1f}s, KG: {len(kg_drums_raw)/SR:.1f}s")


# ── Step 2: Build KG drum+bass loop ──────────────────────────────────

log("Step 2: Build KG drum+bass 4-bar loop")

# KG instrumental: 33.245s-51s. Best loop region: ~35-47s (stable pattern).
# Extract 4 bars starting from the drop kick.
kg_loop_start_s = KG_DROP_S  # 33.245s — right at the drop
kg_loop_len_s = LOOP_BARS * (4 * 60.0 / KG_BPM)  # 4 bars at KG's native BPM
kg_loop_start = int(kg_loop_start_s * SR)
kg_loop_len = int(kg_loop_len_s * SR)

kg_drums_4bar = kg_drums_raw[kg_loop_start:kg_loop_start + kg_loop_len]
kg_bass_4bar = kg_bass_raw[kg_loop_start:kg_loop_start + kg_loop_len]
print(f"  Extracted {kg_loop_len_s:.3f}s of KG at native {KG_BPM} BPM")

# Time-stretch both to 88.0 BPM
print(f"  Stretching KG drums to {TARGET_BPM} BPM (rate={STRETCH_RATE:.4f})...")
kg_drums_loop = pyrb.time_stretch(kg_drums_4bar, SR, STRETCH_RATE)
print(f"  Stretching KG bass to {TARGET_BPM} BPM...")
kg_bass_loop = pyrb.time_stretch(kg_bass_4bar, SR, STRETCH_RATE)

# Trim to exact 4-bar length at target BPM
target_loop_samples = int(LOOP_BARS * BAR_S * SR)
kg_drums_loop = kg_drums_loop[:target_loop_samples]
kg_bass_loop = kg_bass_loop[:target_loop_samples]
print(f"  KG loop: {target_loop_samples/SR:.4f}s = {LOOP_BARS} bars at {TARGET_BPM} BPM")

# Distort the bass (drive=6 = medium crunch)
BASS_DRIVE = 6
print(f"  Distorting bass (drive={BASS_DRIVE}, clean sub <80Hz)...")
kg_bass_loop = distort_bass(kg_bass_loop, drive=BASS_DRIVE)

# Loop KG drums+bass — micro-fade edges then tile (zero overlap)
edge_fade = int(EDGE_FADE_MS / 1000 * SR)
fade_in = np.linspace(0, 1, edge_fade)[:, np.newaxis]
fade_out = np.linspace(1, 0, edge_fade)[:, np.newaxis]

for loop in [kg_drums_loop, kg_bass_loop]:
    loop[:edge_fade] *= fade_in
    loop[-edge_fade:] *= fade_out

kg_drums_looped = np.tile(kg_drums_loop, (N_FLUTE_LOOPS, 1))
kg_bass_looped = np.tile(kg_bass_loop, (N_FLUTE_LOOPS, 1))
print(f"  KG looped: {len(kg_drums_looped)/SR:.1f}s")


# ── Step 3: Build flute loop ─────────────────────────────────────────

log("Step 3: Build Saathiya flute 4-bar loop")

# Extract from "other" stem at BEAT_GRID (downbeat, 1 beat before SNAP)
flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_start = int(BEAT_GRID_S * SR)
flute_raw = s_other[flute_start:flute_start + flute_actual_len]

# Speed up to match 88.0 BPM exactly (same formula as KG: target/source)
flute_stretch = TARGET_BPM / SAATHIYA_LOCAL_BPM
print(f"  Stretch: {(flute_stretch-1)*100:.3f}% (Saathiya {SAATHIYA_LOCAL_BPM:.3f} → {TARGET_BPM})")
flute_loop = pyrb.time_stretch(flute_raw, SR, flute_stretch)
flute_loop = flute_loop[:target_loop_samples]

# Pad if needed
if len(flute_loop) < target_loop_samples:
    pad = np.zeros((target_loop_samples - len(flute_loop), 2))
    flute_loop = np.vstack([flute_loop, pad])

# Loop flute — micro-fade edges then tile (zero overlap, full phrase intact)
flute_loop[:edge_fade] *= fade_in
flute_loop[-edge_fade:] *= fade_out
flute_looped = np.tile(flute_loop, (N_FLUTE_LOOPS, 1))
print(f"  Flute looped: {len(flute_looped)/SR:.1f}s ({N_FLUTE_LOOPS}x, zero overlap)")


# ── Step 4: Build the 1-minute remix ─────────────────────────────────

log("Step 4: Build 1-minute dreamy remix")

# Structure:
# - 8s pre-context: pure Saathiya
# - 4 bars buildup: KG fades in muffled under Saathiya
# - Snap/transition: 2-bar smooth crossover from Saathiya to loop world
# - Loop section: flute loop + KG drum/bass loop for ~60s
#
# The "loop world" starts at SNAP_S. Total ~75s clip.

loop_duration_s = len(flute_looped) / SR
pre_context_s = 8
buildup_bars = 4
buildup_start = BEAT_GRID_S - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context_s)
clip_end_s = BEAT_GRID_S + loop_duration_s + 4.0  # 4s fadeout tail
clip_samples = int((clip_end_s - clip_start) * SR)
cs = int(clip_start * SR)

print(f"  Clip: {clip_start:.1f}s → {clip_end_s:.1f}s ({clip_samples/SR:.1f}s)")

# Saathiya stems for the intro/buildup section
sv = extract(s_vocals, cs, clip_samples)
so = extract(s_other, cs, clip_samples)
sd = extract(s_drums_st, cs, clip_samples)
sb = extract(s_bass_st, cs, clip_samples)

# Place ALL loops at BEAT_GRID_S (downbeat) — KG + flute on the same downbeat
drop_clip_start = int((BEAT_GRID_S - clip_start) * SR)
flute_loop_clip_start = drop_clip_start

flute_in_clip = np.zeros((clip_samples, 2))
kg_drums_in_clip = np.zeros((clip_samples, 2))
kg_bass_in_clip = np.zeros((clip_samples, 2))

for src, dst, start in [
    (flute_looped, flute_in_clip, flute_loop_clip_start),
    (kg_drums_looped, kg_drums_in_clip, drop_clip_start),
    (kg_bass_looped, kg_bass_in_clip, drop_clip_start),
]:
    end = min(start + len(src), clip_samples)
    copy_len = end - start
    if copy_len > 0:
        dst[start:end] = src[:copy_len]

# Build envelopes — SNAP transition (same as proven "decoupled" approach)
# KG builds in muffled, SNAPS to full at BEAT_GRID_S, bass swap at SNAP_S
t = np.arange(clip_samples) / SR + clip_start

fadeout_start = clip_end_s - 4.0
fadeout_end = clip_end_s

kg_buildup_env = np.zeros(clip_samples)   # muffled KG before drop
kg_loop_env = np.zeros(clip_samples)      # looped KG post-drop
flute_loop_env = np.zeros(clip_samples)   # looped flute post-snap
s_bass_env = np.ones(clip_samples)        # Saathiya bass/drums fade
s_drums_env = np.ones(clip_samples)
s_vocals_env = np.ones(clip_samples)     # Saathiya vocals
s_other_env = np.ones(clip_samples)       # original "other" stem
master_fade = np.ones(clip_samples)

for i in range(clip_samples):
    ti = t[i]

    if ti < buildup_start:
        # Pure Saathiya
        pass
    elif ti < BEAT_GRID_S:
        # Buildup: KG fades in muffled
        p = (ti - buildup_start) / (BEAT_GRID_S - buildup_start)
        kg_buildup_env[i] = 0.15 + 0.35 * p
        s_bass_env[i] = 1.0 - 0.5 * p
    else:
        # DROP at downbeat: KG + flute snap in, ALL Saathiya cut
        kg_loop_env[i] = 1.0
        flute_loop_env[i] = 1.0
        s_vocals_env[i] = 0.0
        s_bass_env[i] = 0.0
        s_drums_env[i] = 0.0
        s_other_env[i] = 0.0

    # Final fadeout
    if ti > fadeout_start:
        fp = (ti - fadeout_start) / (fadeout_end - fadeout_start)
        master_fade[i] = max(0, 1.0 - fp)

# Saathiya mix (with per-stem envelopes)
saathiya_mix = (
    sv * 0.85 * s_vocals_env[:, np.newaxis] +
    so * 0.70 * s_other_env[:, np.newaxis] +
    sd * 0.40 * s_drums_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)

# KG buildup: muffled version from cached stretched stems
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')

kg_stretched_drums = load_stereo(f"{OUTPUT}/kg-drums-stretched.wav")
kg_stretched_bass = load_stereo(f"{OUTPUT}/kg-bass-stretched.wav")
min_kg_s = min(len(kg_stretched_drums), len(kg_stretched_bass))
kg_stretched_rhythm = kg_stretched_drums[:min_kg_s] * 0.9 + kg_stretched_bass[:min_kg_s] * 0.75

kg_timeline_start = BEAT_GRID_S - KG_DROP_STRETCHED_S
kg_sample_offset = int((clip_start - kg_timeline_start) * SR)
kg_buildup_audio = extract(kg_stretched_rhythm, kg_sample_offset, clip_samples)
kg_buildup_muf = np.column_stack([
    filtfilt(b_lo, a_lo, kg_buildup_audio[:, 0]),
    filtfilt(b_lo, a_lo, kg_buildup_audio[:, 1])
])

# Loop world: flute + KG drums + KG bass (separate so we can envelope them)
flute_mix = flute_in_clip * 0.70 * flute_loop_env[:, np.newaxis]
kg_loop_mix = (
    kg_drums_in_clip * 0.85 + kg_bass_in_clip * 0.70
) * kg_loop_env[:, np.newaxis]

# Final mix
mix = (
    saathiya_mix +
    kg_buildup_muf * kg_buildup_env[:, np.newaxis] +
    kg_loop_mix +
    flute_mix
) * master_fade[:, np.newaxis]

# Normalize
peak = np.max(np.abs(mix))
if peak > 0.95:
    mix = mix * (0.95 / peak)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
basename = f"dreamy-flute-dnb-{timestamp}"
wav_path = f"{OUTPUT}/{basename}.wav"
sf.write(wav_path, mix, SR, subtype='PCM_16')
print(f"  Wrote {len(mix)/SR:.1f}s")
mp3_path = to_mp3(wav_path)


# ── Done ──────────────────────────────────────────────────────────────

log("DONE — 1-minute dreamy flute DnB remix")
print(f"""
  Output: {mp3_path}

  Structure:
    {clip_start:.0f}s:  Pure Saathiya
    {buildup_start:.0f}s:  KG builds in muffled
    {BEAT_GRID_S:.0f}s:  DROP — KG snaps to full
    {SNAP_S:.0f}s:  Bass swap + flute loop takes over
    {fadeout_start:.0f}s:  Fadeout

  Loops (both at 88.0 BPM, zero drift):
    Flute: 4-bar, {N_FLUTE_LOOPS}x
    KG drums+bass: 4-bar, {N_FLUTE_LOOPS}x
""")
