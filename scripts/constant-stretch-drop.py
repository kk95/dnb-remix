#!/usr/bin/env python3
"""
Constant-stretch drop: single fixed time-stretch of KG to match Saathiya's
local ~88 BPM. No beat-to-beat warping — same approach as DJ software.

WHY THIS IS DIFFERENT from local-grid-drop.py:
- local-grid used pyrb.timemap_stretch (beat-to-beat warp) which maps each
  detected KG beat to each detected Saathiya beat. When beat detection is
  inconsistent, this creates audible tempo fluctuations ("speeds up and down").
- This uses pyrb.time_stretch with a SINGLE constant rate. KG plays at a
  steady 88 BPM throughout — no wobble, no artifacts.

Precise drop onsets from drums-stem onset detection:
- KG drop kick: 24.727s (amplitude peak in KG drums)
- Saathiya drop beat: 306.772s (strongest beat near 307s)
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# BPM values
KG_BPM = 95.703
TARGET_BPM = 88.0  # Saathiya local tempo in drop region (300-325s)
STRETCH_RATE = TARGET_BPM / KG_BPM  # ~0.9195 (slower = longer audio)

# Precise drop onsets (from onset detection on drums stems)
KG_DROP_S = 24.727       # Strong kick in KG drums near drop
SAATHIYA_DROP_S = 306.772  # Strongest beat hit near 307s in Saathiya drums

# After constant stretch, KG drop moves to:
KG_DROP_STRETCHED_S = KG_DROP_S / STRETCH_RATE  # ~26.89s

# Timing constants
BEAT_S = 60.0 / TARGET_BPM   # ~0.6818s
BAR_S = 4 * BEAT_S            # ~2.727s

os.makedirs(OUTPUT, exist_ok=True)


def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def extract(src, start, length):
    """Extract a window from src, zero-padding if out of bounds."""
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


# ── Step 1: Constant time-stretch KG stems ───────────────────────────

log("Step 1: Constant time-stretch KG stems (95.7 → 88.0 BPM)")
print(f"  Rate: {STRETCH_RATE:.6f}")
print(f"  Audio gets {(1/STRETCH_RATE - 1)*100:.1f}% longer")
print(f"  KG drop {KG_DROP_S:.3f}s → {KG_DROP_STRETCHED_S:.3f}s after stretch")

# Only stretch first 55s of KG (covers well past the transition region)
KG_SECTION_END = int(55.0 * SR)

stretched = {}
for stem in ['drums', 'bass']:
    cache_path = f"{OUTPUT}/kg-{stem}-stretched.wav"
    stem_path = f"{STEMS}/kho-gayi/{stem}.wav"
    audio = load_stereo(stem_path)
    audio = audio[:min(len(audio), KG_SECTION_END)]
    print(f"  {stem}: stretching {len(audio)/SR:.1f}s ...")
    s = pyrb.time_stretch(audio, SR, STRETCH_RATE)
    sf.write(cache_path, s, SR, subtype='PCM_16')
    stretched[stem] = s
    print(f"  → {cache_path} ({len(s)/SR:.1f}s)")


# ── Step 2: Load Saathiya stems ──────────────────────────────────────

log("Step 2: Loading Saathiya stems")

s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")
print(f"  Saathiya duration: {len(s_vocals)/SR:.1f}s")


# ── Step 3: Build drop clips ─────────────────────────────────────────

log("Step 3: Building clips at various offsets")

# Clip window
buildup_bars = 4
ride_bars = 8
pre_context_s = 8
buildup_start = SAATHIYA_DROP_S - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context_s)
clip_end = min(len(s_vocals) / SR, SAATHIYA_DROP_S + ride_bars * BAR_S)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

print(f"  Clip: {clip_start:.1f}s → {clip_end:.1f}s ({clip_samples/SR:.1f}s)")
print(f"  Buildup: {buildup_start:.1f}s, Drop: {SAATHIYA_DROP_S:.3f}s")

# Saathiya stems for clip
sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums_st[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

# Filter setup
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')

# Combine KG drums + bass
min_kg = min(len(stretched['drums']), len(stretched['bass']))
kg_rhythm = stretched['drums'][:min_kg] * 0.9 + stretched['bass'][:min_kg] * 0.75

# Offsets: positive = KG beat AFTER Saathiya beat (late)
#          negative = KG beat BEFORE Saathiya beat (early)
offsets_ms = [+30, +15, 0, -15, -30]

for offset_ms in offsets_ms:
    # KG drop should land at this time on the Saathiya timeline
    kg_drop_target = SAATHIYA_DROP_S + offset_ms / 1000.0

    # KG timeline start = where KG's t=0 maps to on Saathiya timeline
    kg_timeline_start = kg_drop_target - KG_DROP_STRETCHED_S

    # Sample offset into KG audio for this clip window
    kg_sample_offset = int((clip_start - kg_timeline_start) * SR)

    kd = extract(kg_rhythm, kg_sample_offset, clip_samples)

    # Pre-filtered versions for buildup
    kd_muf = np.column_stack([filtfilt(b_lo, a_lo, kd[:, 0]),
                               filtfilt(b_lo, a_lo, kd[:, 1])])
    kd_mid = np.column_stack([filtfilt(b_mid, a_mid, kd[:, 0]),
                               filtfilt(b_mid, a_mid, kd[:, 1])])

    min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))
    t = np.arange(min_len) / SR + clip_start

    # Envelopes
    kg_vol = np.zeros(min_len)
    kg_filter = np.zeros(min_len)
    s_bass_env = np.ones(min_len)

    for i in range(min_len):
        ti = t[i]
        if ti < buildup_start:
            # Pure Saathiya, no KG
            kg_vol[i] = 0.0
            kg_filter[i] = 0.0
            s_bass_env[i] = 1.0
        elif ti < SAATHIYA_DROP_S:
            # Buildup: KG fades in muffled
            p = (ti - buildup_start) / (SAATHIYA_DROP_S - buildup_start)
            kg_vol[i] = 0.15 + 0.35 * p
            kg_filter[i] = p * 0.4
            s_bass_env[i] = 1.0 - 0.5 * p
        else:
            # Post-drop: SNAP — KG full in, Saathiya bass/drums out
            p = min(1.0, (ti - SAATHIYA_DROP_S) / BEAT_S)
            kg_vol[i] = 0.5 + 0.5 * p
            kg_filter[i] = 0.4 + 0.6 * p
            s_bass_env[i] = 0.5 * (1.0 - p)

    # Apply filter envelope to KG
    kg_filt = np.zeros_like(kd[:min_len])
    for ch in range(2):
        lo_mask = kg_filter <= 0.4
        hi_mask = kg_filter > 0.4
        t_lo = np.clip(kg_filter / 0.4, 0, 1)
        t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
        kg_filt[lo_mask, ch] = (kd_muf[lo_mask, ch] * (1 - t_lo[lo_mask]) +
                                 kd_mid[lo_mask, ch] * t_lo[lo_mask])
        kg_filt[hi_mask, ch] = (kd_mid[hi_mask, ch] * (1 - t_hi[hi_mask]) +
                                 kd[hi_mask, ch] * t_hi[hi_mask])

    kg_mix = kg_filt[:min_len] * kg_vol[:, np.newaxis]

    saathiya_mix = (
        sv[:min_len] * 0.85 +
        so[:min_len] * 0.70 +
        sd[:min_len] * 0.40 * s_bass_env[:, np.newaxis] +
        sb[:min_len] * 0.50 * s_bass_env[:, np.newaxis]
    )

    mix = saathiya_mix + kg_mix
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    # Label
    if offset_ms == 0:
        label = "cs-0ms"
    elif offset_ms > 0:
        label = f"cs-{offset_ms}ms-late"
    else:
        label = f"cs-{abs(offset_ms)}ms-early"

    wav_path = f"{OUTPUT}/{label}.wav"
    sf.write(wav_path, mix, SR, subtype='PCM_16')
    print(f"  {label}: wrote {len(mix)/SR:.1f}s")
    to_mp3(wav_path)


# ── Done ─────────────────────────────────────────────────────────────

log("DONE — Constant-stretch clips ready")
print(f"  Output: {OUTPUT}/")
print(f"  KG stretched at constant {KG_BPM} → {TARGET_BPM} BPM (no beat-to-beat warp)")
print(f"  KG drop ({KG_DROP_S:.3f}s) aligned to Saathiya beat ({SAATHIYA_DROP_S:.3f}s)")
print()
for ms in offsets_ms:
    if ms == 0:
        print(f"  cs-0ms           — beats exactly aligned")
    elif ms > 0:
        print(f"  cs-{ms}ms-late      — KG {ms}ms BEHIND Saathiya beat")
    else:
        print(f"  cs-{abs(ms)}ms-early     — KG {abs(ms)}ms AHEAD of Saathiya beat")
print(f"\n  open {OUTPUT}/*.mp3")
