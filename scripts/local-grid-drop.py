#!/usr/bin/env python3
"""
Local-Grid Drop: re-warp KG to Saathiya's ACTUAL local beat positions
detected from the drums stem with high resolution (hop=128, 2.9ms).

Key insight from analysis:
- Global BPM: 87.593 (wrong for this region)
- Local BPM around drop: ~88.0
- This causes ~69ms drift over 18s post-drop
- Previous "actual-grid" used full-mix beat detection at low resolution,
  which smoothed to global tempo. This version uses drums-only + hop=128.

Also applies KG timing offset based on user listening tests:
- -50ms: amazing at start (drop alignment perfect)
- -15ms: almost worked for whole section
- Target: -50ms with local-grid (no drift) = best of both
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
OUTPUT = f"{PROJECT}/output/local-grid"
SR = 44100
HOP = 128  # 2.9ms resolution

# Use the LOCAL BPM, not global
LOCAL_BPM = 88.0  # approximate, actual positions from beat detection
SAATHIYA_BEAT = 60.0 / 87.593  # for fallback/boundary
KG_BPM = 95.703
BAR_S = 4 * SAATHIYA_BEAT

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


def extract(src, start, length):
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


# ── Step 1: High-resolution beat detection on drums stem ──────────────

log("Step 1: High-res beat detection (drums stem, hop=128)")

s_drums_mono = load_mono(f"{STEMS}/saathiya/drums.wav")
_, s_beat_frames = librosa.beat.beat_track(
    y=s_drums_mono, sr=SR, hop_length=HOP,
    start_bpm=87.5, tightness=200,
)
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR, hop_length=HOP)
s_local_bpm = 60.0 / np.median(np.diff(s_beats))
print(f"  Detected {len(s_beats)} beats")
print(f"  Median BPM: {s_local_bpm:.3f}")

# Check local BPM around drop
drop_region_beats = s_beats[(s_beats >= 300) & (s_beats <= 325)]
if len(drop_region_beats) > 2:
    local_ibis = np.diff(drop_region_beats)
    local_bpm = 60.0 / np.mean(local_ibis)
    print(f"  Drop-region BPM (300-325s): {local_bpm:.3f}")
    print(f"  Drop-region IBI: {np.mean(local_ibis)*1000:.2f}ms (std: {np.std(local_ibis)*1000:.2f}ms)")
else:
    local_bpm = s_local_bpm

# Find Saathiya drop beat
s_drop_idx = np.argmin(np.abs(s_beats - 307.0))
s_drop = s_beats[s_drop_idx]
print(f"  Drop beat: {s_drop:.3f}s (index {s_drop_idx})")

# Show beats around drop
print(f"\n  Beats around drop:")
for i in range(max(0, s_drop_idx - 4), min(len(s_beats), s_drop_idx + 12)):
    marker = " ← DROP" if i == s_drop_idx else ""
    ibi = (s_beats[i] - s_beats[i-1]) * 1000 if i > 0 else 0
    print(f"    Beat {i}: {s_beats[i]:.3f}s (IBI: {ibi:.1f}ms){marker}")

# ── Step 2: KG beat detection ────────────────────────────────────────

log("Step 2: KG beat detection")

kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
_, kg_beat_frames = librosa.beat.beat_track(
    y=kg_mono, sr=SR, hop_length=HOP,
    start_bpm=95.0, tightness=200,
)
kg_beats = librosa.frames_to_time(kg_beat_frames, sr=SR, hop_length=HOP)
kg_bpm = 60.0 / np.median(np.diff(kg_beats))
if kg_bpm > 120:
    kg_bpm /= 2
    kg_beats = kg_beats[::2]
print(f"  KG beats: {len(kg_beats)}, BPM: {kg_bpm:.3f}")


# ── Step 3: Build timemap using Saathiya's actual beat positions ──────

log("Step 3: Building timemap (KG → Saathiya actual beats)")

# Find the KG beat index that should align with the Saathiya drop
# The previous best alignment was KG at ~24.8s matching Saathiya drop at ~306.8s
# We need to find which KG beat is nearest to 24.8s (in original KG time)
KG_DROP_APPROX = 24.8  # seconds in original KG timeline
kg_drop_idx = np.argmin(np.abs(kg_beats - KG_DROP_APPROX))
print(f"  KG drop beat: index {kg_drop_idx}, time {kg_beats[kg_drop_idx]:.3f}s")


def build_local_timemap(kg_beats_arr, s_beats_arr, kg_drop_idx, s_drop_idx, audio_len):
    """Map each KG beat to the corresponding Saathiya actual beat position.

    Uses Saathiya's real detected beats (which reflect local ~88 BPM tempo)
    instead of a mathematical 87.593 BPM grid.
    """
    time_map = [(0, 0)]

    # For beats outside Saathiya's range, extrapolate using local IBI
    local_ibi = 60.0 / local_bpm  # use the drop-region BPM

    for i, kg_time in enumerate(kg_beats_arr):
        s_idx = s_drop_idx + (i - kg_drop_idx)

        if 0 <= s_idx < len(s_beats_arr):
            target_time = s_beats_arr[s_idx]
        else:
            # Extrapolate from nearest boundary
            if s_idx < 0:
                target_time = s_beats_arr[0] + s_idx * local_ibi
            else:
                overshoot = s_idx - (len(s_beats_arr) - 1)
                target_time = s_beats_arr[-1] + overshoot * local_ibi

        src_sample = int(kg_time * SR)
        tgt_sample = int(target_time * SR)

        if tgt_sample > 0 and src_sample > 0:
            time_map.append((src_sample, tgt_sample))

    # End marker
    if len(kg_beats_arr) > 1:
        last_interval = kg_beats_arr[-1] - kg_beats_arr[-2]
        ratio = local_ibi / last_interval
    else:
        ratio = local_ibi / (60.0 / KG_BPM)
    remaining = audio_len - int(kg_beats_arr[-1] * SR)
    end_tgt = time_map[-1][1] + int(remaining * ratio)
    time_map.append((audio_len, end_tgt))

    # Clean: monotonic only
    clean = [time_map[0]]
    for entry in time_map[1:]:
        if entry[0] > clean[-1][0] and entry[1] > clean[-1][1]:
            clean.append(entry)
    if clean[-1][0] != audio_len:
        delta = audio_len - clean[-1][0]
        clean.append((audio_len, clean[-1][1] + int(delta * ratio)))

    return clean


kg_audio_len = len(kg_mono)
local_map = build_local_timemap(kg_beats, s_beats, kg_drop_idx, s_drop_idx, kg_audio_len)
print(f"  Timemap entries: {len(local_map)}")

# Verify: check target beat intervals around drop
target_times = []
for src, tgt in local_map[1:-1]:
    target_times.append(tgt / SR)
if len(target_times) > 2:
    diffs = np.diff(target_times)
    # Focus on drop region
    tt = np.array(target_times)
    drop_mask = (tt[:-1] >= 300) & (tt[:-1] <= 325)
    if np.sum(drop_mask) > 0:
        local_diffs = diffs[drop_mask]
        print(f"  Drop-region target IBI: {np.mean(local_diffs)*1000:.2f}ms ({60/np.mean(local_diffs):.3f} BPM)")
    print(f"  Overall target IBI: mean={np.mean(diffs)*1000:.2f}ms, std={np.std(diffs)*1000:.2f}ms")

# The KG drop beat in the warped audio will land at s_beats[s_drop_idx]
kg_drop_warped = s_beats[s_drop_idx]
print(f"  KG drop beat in warped audio: {kg_drop_warped:.3f}s")


# ── Step 4: Warp KG stems ────────────────────────────────────────────

log("Step 4: Warping KG stems to Saathiya's local beat grid")

warped_stems = {}
for stem in ['drums', 'bass', 'other']:
    out_path = f"{OUTPUT}/kg-{stem}-local.wav"
    # Always regenerate — don't use stale cache
    stem_path = f"{STEMS}/kho-gayi/{stem}.wav"
    audio, _ = sf.read(stem_path, dtype='float64')
    stem_map = build_local_timemap(kg_beats, s_beats, kg_drop_idx, s_drop_idx, len(audio))
    print(f"  {stem}: warping {len(audio)/SR:.1f}s ...")
    warped = pyrb.timemap_stretch(audio, SR, stem_map)
    sf.write(out_path, warped, SR, subtype='PCM_16')
    warped_stems[stem] = out_path
    print(f"  → {out_path} ({len(warped)/SR:.1f}s)")


# ── Step 5: Verify warped BPM ─────────────────────────────────────────

log("Step 5: Verify warped KG BPM")

# Quick check: load warped drums and check tempo
wkg_mono = load_mono(warped_stems['drums'])
_, wkg_frames = librosa.beat.beat_track(y=wkg_mono, sr=SR, hop_length=HOP)
wkg_beats = librosa.frames_to_time(wkg_frames, sr=SR, hop_length=HOP)
wkg_bpm = 60.0 / np.median(np.diff(wkg_beats))
if wkg_bpm > 120:
    wkg_bpm /= 2
print(f"  Warped KG BPM: {wkg_bpm:.3f} (target: ~{local_bpm:.3f})")

# Check beat alignment around drop
wkg_drop_region = wkg_beats[(wkg_beats >= 300) & (wkg_beats <= 325)]
s_drop_region = s_beats[(s_beats >= 300) & (s_beats <= 325)]
if len(wkg_drop_region) > 0 and len(s_drop_region) > 0:
    # For each warped KG beat, find nearest Saathiya beat
    offsets = []
    for wkb in wkg_drop_region:
        nearest_sb = s_drop_region[np.argmin(np.abs(s_drop_region - wkb))]
        offsets.append((wkb - nearest_sb) * 1000)
    offsets = np.array(offsets)
    print(f"  Drop-region KG-vs-Saathiya beat offsets:")
    print(f"    Mean: {np.mean(offsets):+.1f}ms, Std: {np.std(offsets):.1f}ms")
    print(f"    Max early: {np.min(offsets):+.1f}ms, Max late: {np.max(offsets):+.1f}ms")

del s_drums_mono, kg_mono, wkg_mono


# ── Step 6: Build drop clips at various offsets ──────────────────────

log("Step 6: Building drop clips with local-grid KG")

s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums_w = load_stereo(warped_stems['drums'])
kg_bass_w = load_stereo(warped_stems['bass'])

# Filter setup
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')
beat_s = SAATHIYA_BEAT

buildup_bars = 4
ride_bars = 8
pre_context = 8
buildup_start = s_drop - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context)
clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * BAR_S)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums_st[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

# Offsets: user said -50ms is perfect at start, -15ms almost works throughout
# With local-grid (no drift), -50ms should now work for the whole section
offsets_ms = [0, -15, -30, -50]

for shift_ms in offsets_ms:
    kg_drop_shifted = kg_drop_warped + shift_ms / 1000.0
    kg_tl_offset = s_drop - kg_drop_shifted
    ka_s = int((clip_start - kg_tl_offset) * SR)

    min_kg = min(len(kg_drums_w), len(kg_bass_w))
    kg_rhythm = kg_drums_w[:min_kg] * 0.9 + kg_bass_w[:min_kg] * 0.75
    kd = extract(kg_rhythm, ka_s, clip_samples)

    kd_muf = np.column_stack([filtfilt(b_lo, a_lo, kd[:, 0]),
                               filtfilt(b_lo, a_lo, kd[:, 1])])
    kd_mid = np.column_stack([filtfilt(b_mid, a_mid, kd[:, 0]),
                               filtfilt(b_mid, a_mid, kd[:, 1])])

    min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))

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

    kg_filt = np.zeros_like(kd[:min_len])
    for ch in range(2):
        lo = kg_filter <= 0.4
        hi = kg_filter > 0.4
        t_lo = np.clip(kg_filter / 0.4, 0, 1)
        t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
        kg_filt[lo, ch] = kd_muf[lo, ch] * (1 - t_lo[lo]) + kd_mid[lo, ch] * t_lo[lo]
        kg_filt[hi, ch] = kd_mid[hi, ch] * (1 - t_hi[hi]) + kd[hi, ch] * t_hi[hi]

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

    if shift_ms == 0:
        label = "local-grid-0ms"
    else:
        label = f"local-grid-{abs(shift_ms)}ms-early"
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  local-grid-0ms          — KG on Saathiya's actual beat grid, no offset")
print(f"  local-grid-15ms-early   — KG 15ms earlier (user said 'almost worked')")
print(f"  local-grid-30ms-early   — KG 30ms earlier")
print(f"  local-grid-50ms-early   — KG 50ms earlier (user said 'amazing at start')")
print(f"\n  KEY: These use LOCAL beat positions (~88 BPM), not global (87.593).")
print(f"  The post-drop drift should be GONE. Listen for the whole ride section.")
print(f"\n  open {OUTPUT}/*.mp3")
