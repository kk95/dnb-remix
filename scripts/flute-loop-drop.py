#!/usr/bin/env python3
"""
1-minute dreamy flute + DnB remix.

Full-section KG drums+bass (no looping, no restarts) + 4-bar flute loop
(Saathiya "other" from 290.4s), both time-stretched to 88.0 BPM.

Smooth transition from Saathiya into the drop section.
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
KG_BPM = 95.703                # global KG BPM (used for 4-bar ding loop)
KG_FULL_SECTION_BPM = 95.0    # actual BPM of KG post-drop section (33-93s)
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM
FULL_STRETCH_RATE = TARGET_BPM / KG_FULL_SECTION_BPM

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
N_FLUTE_LOOPS = 6
EDGE_FADE_MS = 3
FLUTE_XF_MS = 134       # circular crossfade at flute loop boundaries
FLUTE_SOURCE_S = 290.4   # best loop boundary (from riff instance scan)

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
    sos_lo = butter(4, sub_cutoff / (SR / 2), btype='low', output='sos')
    sos_hi = butter(4, sub_cutoff / (SR / 2), btype='high', output='sos')

    result = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        sub = sosfilt(sos_lo, audio[:, ch])
        mids = sosfilt(sos_hi, audio[:, ch])
        mids_dist = np.tanh(drive * mids) / np.tanh(drive)
        result[:, ch] = sub + mids_dist
    return result


def make_circular_loop(one_loop, xf_samples):
    """Prepare a loop for seamless tiling via circular crossfade.

    Blends the last xf_samples INTO the first xf_samples using equal-power
    curves. The end is set to match the start so np.tile() is seamless.
    """
    if xf_samples <= 0 or xf_samples > len(one_loop) // 2:
        return one_loop.copy()

    result = one_loop.copy()
    head = one_loop[:xf_samples].copy()
    tail = one_loop[-xf_samples:].copy()

    t = np.linspace(0, 1, xf_samples)[:, np.newaxis]
    fi = np.sqrt(t)       # 0 → 1
    fo = np.sqrt(1 - t)   # 1 → 0

    blend = head * fi + tail * fo
    result[:xf_samples] = blend
    result[-xf_samples:] = blend
    return result


# ── Step 1: Load stems ────────────────────────────────────────────────

log("Step 1: Load all stems")
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass_st = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums_raw = load_stereo(f"{STEMS}/kho-gayi/drums.wav")
kg_bass_raw = load_stereo(f"{STEMS}/kho-gayi/bass.wav")
kg_other_raw = load_stereo(f"{STEMS}/kho-gayi/other.wav")
print(f"  Saathiya: {len(s_vocals)/SR:.1f}s, KG: {len(kg_drums_raw)/SR:.1f}s")


# ── Step 2: Build full KG drums+bass section ─────────────────────────

log("Step 2: Build full KG drums+bass from drop (no looping)")

target_loop_samples = int(LOOP_BARS * BAR_S * SR)
total_needed = target_loop_samples * N_FLUTE_LOOPS

# Extract enough raw audio from the drop to cover the full flute duration
raw_needed = int(total_needed / SR * FULL_STRETCH_RATE * SR) + SR
kg_drop_sample = int(KG_DROP_S * SR)

print(f"  Extracting {raw_needed/SR:.1f}s of raw KG from drop ({KG_DROP_S}s)...")
kg_drums_section = kg_drums_raw[kg_drop_sample:kg_drop_sample + raw_needed]
kg_bass_section = kg_bass_raw[kg_drop_sample:kg_drop_sample + raw_needed]

print(f"  Stretching KG drums to {TARGET_BPM} BPM (rate={FULL_STRETCH_RATE:.4f}, "
      f"source={KG_FULL_SECTION_BPM} BPM)...")
kg_drums_full = pyrb.time_stretch(kg_drums_section, SR, FULL_STRETCH_RATE)

print(f"  Stretching KG bass to {TARGET_BPM} BPM...")
kg_bass_full = pyrb.time_stretch(kg_bass_section, SR, FULL_STRETCH_RATE)

# Distort the bass (drive=6 = medium crunch)
BASS_DRIVE = 6
print(f"  Distorting bass (drive={BASS_DRIVE}, clean sub <80Hz)...")
kg_bass_full = distort_bass(kg_bass_full, drive=BASS_DRIVE)

# Trim to exact needed length
kg_drums_full = kg_drums_full[:total_needed]
kg_bass_full = kg_bass_full[:total_needed]
if len(kg_drums_full) < total_needed:
    kg_drums_full = np.vstack([kg_drums_full,
                                np.zeros((total_needed - len(kg_drums_full), 2))])
if len(kg_bass_full) < total_needed:
    kg_bass_full = np.vstack([kg_bass_full,
                               np.zeros((total_needed - len(kg_bass_full), 2))])

print(f"  Full KG: {total_needed/SR:.1f}s of continuous drums+bass (no loop restarts)")


# ── Step 2b: Build KG ding loop (high-pitched synth from "other" stem) ──

log("Step 2b: Build KG ding loop (>1500Hz from 'other' stem)")

# Ding stays as 4-bar loop — it's subtle and the loop boundary isn't audible
kg_loop_start = int(KG_DROP_S * SR)
kg_loop_len = int(LOOP_BARS * (4 * 60.0 / KG_BPM) * SR)

DING_OFFSET_SAMPLES = int(0.006 * SR)
kg_other_4bar = kg_other_raw[kg_loop_start - DING_OFFSET_SAMPLES:
                              kg_loop_start - DING_OFFSET_SAMPLES + kg_loop_len]

sos_ding = butter(4, 1500 / (SR / 2), btype='high', output='sos')
kg_ding_4bar = np.column_stack([
    sosfilt(sos_ding, kg_other_4bar[:, 0]),
    sosfilt(sos_ding, kg_other_4bar[:, 1])
])

print(f"  Stretching KG ding to {TARGET_BPM} BPM...")
kg_ding_loop = pyrb.time_stretch(kg_ding_4bar, SR, STRETCH_RATE)
kg_ding_loop = kg_ding_loop[:target_loop_samples]

# Loop ding with micro-fades
edge_fade = int(EDGE_FADE_MS / 1000 * SR)
fade_in = np.linspace(0, 1, edge_fade)[:, np.newaxis]
fade_out = np.linspace(1, 0, edge_fade)[:, np.newaxis]

kg_ding_loop[:edge_fade] *= fade_in
kg_ding_loop[-edge_fade:] *= fade_out
kg_ding_looped = np.tile(kg_ding_loop, (N_FLUTE_LOOPS, 1))

print(f"  Ding looped: {len(kg_ding_looped)/SR:.1f}s ({N_FLUTE_LOOPS}x, 4-bar loop)")


# ── Step 3: Build flute loop ─────────────────────────────────────────

log("Step 3: Build Saathiya flute 4-bar loop (from 290.4s)")

flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_start = int(FLUTE_SOURCE_S * SR)
flute_raw = s_other[flute_start:flute_start + flute_actual_len]

flute_stretch = TARGET_BPM / SAATHIYA_LOCAL_BPM
print(f"  Source: {FLUTE_SOURCE_S}s, stretch: {(flute_stretch-1)*100:.3f}%")
flute_loop = pyrb.time_stretch(flute_raw, SR, flute_stretch)
flute_loop = flute_loop[:target_loop_samples]

if len(flute_loop) < target_loop_samples:
    pad = np.zeros((target_loop_samples - len(flute_loop), 2))
    flute_loop = np.vstack([flute_loop, pad])

# Circular crossfade for seamless tiling
xf_samples = int(FLUTE_XF_MS / 1000 * SR)
flute_loop = make_circular_loop(flute_loop, xf_samples)
print(f"  Circular crossfade: {FLUTE_XF_MS}ms ({xf_samples} samples)")

# Edge fades + tile
flute_loop[:edge_fade] *= fade_in
flute_loop[-edge_fade:] *= fade_out
flute_looped = np.tile(flute_loop, (N_FLUTE_LOOPS, 1))
print(f"  Flute looped: {len(flute_looped)/SR:.1f}s ({N_FLUTE_LOOPS}x, {FLUTE_XF_MS}ms xfade)")


# ── Step 4: Build the 1-minute remix ─────────────────────────────────

log("Step 4: Build 1-minute dreamy remix")

loop_duration_s = len(flute_looped) / SR
buildup_bars = 4
buildup_start = BEAT_GRID_S - buildup_bars * BAR_S
clip_start = BEAT_GRID_S - 3 * BAR_S
clip_end_s = BEAT_GRID_S + loop_duration_s + 4.0
clip_samples = int((clip_end_s - clip_start) * SR)
cs = int(clip_start * SR)

print(f"  Clip: {clip_start:.1f}s → {clip_end_s:.1f}s ({clip_samples/SR:.1f}s)")

# Saathiya stems for the intro/buildup section
sv = extract(s_vocals, cs, clip_samples)
so = extract(s_other, cs, clip_samples)
sd = extract(s_drums_st, cs, clip_samples)
sb = extract(s_bass_st, cs, clip_samples)

# Place everything at BEAT_GRID_S (downbeat)
drop_clip_start = int((BEAT_GRID_S - clip_start) * SR)

flute_in_clip = np.zeros((clip_samples, 2))
kg_drums_in_clip = np.zeros((clip_samples, 2))
kg_bass_in_clip = np.zeros((clip_samples, 2))
kg_ding_in_clip = np.zeros((clip_samples, 2))

for src, dst in [
    (flute_looped, flute_in_clip),
    (kg_drums_full, kg_drums_in_clip),
    (kg_bass_full, kg_bass_in_clip),
    (kg_ding_looped, kg_ding_in_clip),
]:
    end = min(drop_clip_start + len(src), clip_samples)
    copy_len = end - drop_clip_start
    if copy_len > 0:
        dst[drop_clip_start:end] = src[:copy_len]

# Build envelopes
t = np.arange(clip_samples) / SR + clip_start

fadeout_start = clip_end_s - 4.0
fadeout_end = clip_end_s

kg_buildup_env = np.zeros(clip_samples)
kg_loop_env = np.zeros(clip_samples)
flute_loop_env = np.zeros(clip_samples)
s_bass_env = np.ones(clip_samples)
s_drums_env = np.ones(clip_samples)
s_vocals_env = np.ones(clip_samples)
s_other_env = np.ones(clip_samples)
master_fade = np.ones(clip_samples)

for i in range(clip_samples):
    ti = t[i]

    if ti < buildup_start:
        pass
    elif ti < BEAT_GRID_S:
        p = (ti - buildup_start) / (BEAT_GRID_S - buildup_start)
        kg_buildup_env[i] = 0.15 + 0.35 * p
        s_bass_env[i] = 1.0 - 0.5 * p
    else:
        kg_loop_env[i] = 1.0
        flute_loop_env[i] = 1.0
        s_vocals_env[i] = 0.0
        s_bass_env[i] = 0.0
        s_drums_env[i] = 0.0
        s_other_env[i] = 0.0

    if ti > fadeout_start:
        fp = (ti - fadeout_start) / (fadeout_end - fadeout_start)
        master_fade[i] = max(0, 1.0 - fp)

# Saathiya mix
saathiya_mix = (
    sv * 0.85 * s_vocals_env[:, np.newaxis] +
    so * 0.70 * s_other_env[:, np.newaxis] +
    sd * 0.40 * s_drums_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)

# KG buildup: muffled version from cached stretched stems
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')

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

# Loop world: flute + KG drums + KG bass + KG ding
flute_mix = flute_in_clip * 0.57 * flute_loop_env[:, np.newaxis]
kg_loop_mix = (
    kg_drums_in_clip * 0.85 + kg_bass_in_clip * 0.50 + kg_ding_in_clip * 0.30
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
    {fadeout_start:.0f}s:  Fadeout

  KG: Full section from drop ({KG_FULL_SECTION_BPM} BPM → {TARGET_BPM} BPM, no loop restarts)
  Flute: 4-bar from {FLUTE_SOURCE_S}s, {N_FLUTE_LOOPS}x, {FLUTE_XF_MS}ms circular xfade
  Flute volume: 0.57
""")
