#!/usr/bin/env python3
"""
Recreate the djay transition as a DROP.

What actually happens (from screenshot curves):
  - KG starts muffled underneath Saathiya (low volume + lowpass filter)
  - Over ~4 bars, KG builds: filter opens, volume rises
  - At the DROP POINT (~5:07): KG's beat is fully in, bass swaps
  - Saathiya's flute/vocals CONTINUE playing over KG's drums+bass
  - Saathiya's own bass/drums cut out at the swap point
  - After the drop: Saathiya's melody rides over KG's rhythm = THE MAGIC

Timeline:
  [Saathiya full]...[KG muffled buildup ~4bars]...[DROP]...[Saathiya melody + KG beat]
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/drop-transition"
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


# ── Step 1: BPM + Beat Grid ────────────────────────────────────────────

log("Step 1: Beat grid detection")

s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
s_tempo, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
s_bpm = 60.0 / np.median(np.diff(s_beats))
bar_s = 4 * (60.0 / s_bpm)

print(f"  Saathiya: {s_bpm:.3f} BPM, bar = {bar_s:.3f}s")

kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
kg_tempo, kg_beat_frames = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats = librosa.frames_to_time(kg_beat_frames, sr=SR)
kg_raw_bpm = 60.0 / np.median(np.diff(kg_beats))
if kg_raw_bpm > 120:
    kg_bpm = kg_raw_bpm / 2
    kg_beats = kg_beats[::2]
else:
    kg_bpm = kg_raw_bpm

print(f"  Kho Gayi: {kg_bpm:.3f} BPM")

stretch_ratio = s_bpm / kg_bpm
print(f"  Stretch: {stretch_ratio:.6f}x")


# ── Step 2: Stretch KG ─────────────────────────────────────────────────

log("Step 2: Stretching KG stems to Saathiya tempo")

stretched = {}
for stem, crisp in [('drums', 3), ('bass', 5), ('other', 5)]:
    out = f"{OUTPUT}/kg-{stem}.wav"
    if not os.path.exists(out):
        subprocess.run([
            'rubberband', '--tempo', str(stretch_ratio), '--crisp', str(crisp),
            f"{STEMS}/kho-gayi/{stem}.wav", out
        ], capture_output=True, check=True)
    stretched[stem] = out
    print(f"  {stem}: {out}")

# Also stretch full KG for reference
kg_full_path = f"{OUTPUT}/kg-full.wav"
if not os.path.exists(kg_full_path):
    subprocess.run([
        'rubberband', '--tempo', str(stretch_ratio), '--crisp', '3',
        f"{PROJECT}/wavs/kho-gayi.wav", kg_full_path
    ], capture_output=True, check=True)
print(f"  full: {kg_full_path}")


# ── Step 3: Find alignment beats ───────────────────────────────────────

log("Step 3: Finding drop point beats")

# Saathiya beat near 5:07
s_drop_target = 307.0
s_drop_idx = np.argmin(np.abs(s_beats - s_drop_target))
s_drop = s_beats[s_drop_idx]
print(f"  Saathiya drop beat: {s_drop:.3f}s (target: {s_drop_target})")

# KG: detect beats on stretched audio
kg_s_mono = load_mono(kg_full_path)
_, kg_s_frames = librosa.beat.beat_track(y=kg_s_mono, sr=SR, units='frames')
kg_s_beats = librosa.frames_to_time(kg_s_frames, sr=SR)
kg_s_raw_bpm = 60.0 / np.median(np.diff(kg_s_beats))
if kg_s_raw_bpm > 120:
    kg_s_beats = kg_s_beats[::2]

# KG beat near 23-24s (stretched time ≈ original / stretch_ratio)
# Original 23-24s → stretched ~25-26s
kg_target_range = (23, 27)
kg_candidates = kg_s_beats[(kg_s_beats >= kg_target_range[0]) & (kg_s_beats <= kg_target_range[1])]

print(f"  KG stretched beats in {kg_target_range[0]}-{kg_target_range[1]}s range:")
for b in kg_candidates:
    print(f"    {b:.3f}s")

# If no beats found, use original beats scaled
if len(kg_candidates) == 0:
    kg_near_24 = kg_beats[(kg_beats >= 22) & (kg_beats <= 25)]
    kg_candidates = kg_near_24 / stretch_ratio
    print(f"  (fallback from original beats, scaled)")
    for b in kg_candidates:
        print(f"    {b:.3f}s")


# ── Step 4: Create drop transition clips ───────────────────────────────

log("Step 4: Building drop transition clips")

# Load all audio
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums = load_stereo(f"{OUTPUT}/kg-drums.wav")
kg_bass = load_stereo(f"{OUTPUT}/kg-bass.wav")
kg_other_stem = load_stereo(f"{OUTPUT}/kg-other.wav")

print(f"  Loaded all stems")

# Pre-compute KG filtered versions (for the muffled buildup)
kg_rhythm = kg_drums * 0.9 + kg_bass * 0.75
min_stem_len = min(len(kg_drums), len(kg_bass))
kg_rhythm = kg_rhythm[:min_stem_len]

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

# For each KG beat candidate, create a clip
buildup_bars = 4  # KG builds up over 4 bars before the drop
ride_bars = 8     # Let it ride 8 bars after the drop (the good stuff)
pre_context = 8   # 8 seconds of pure Saathiya before buildup starts

for ci, kg_drop_beat in enumerate(kg_candidates):
    # The drop point in the output timeline = s_drop (Saathiya's ~5:07 beat)
    # At this moment, KG's audio is at kg_drop_beat
    # So KG audio start in timeline = s_drop - kg_drop_beat
    kg_timeline_offset = s_drop - kg_drop_beat

    # Clip window
    buildup_start = s_drop - buildup_bars * bar_s  # KG starts entering here
    clip_start = buildup_start - pre_context
    clip_end = s_drop + ride_bars * bar_s

    clip_start = max(0, clip_start)
    clip_end = min(len(s_vocals) / SR, clip_end)

    clip_samples = int((clip_end - clip_start) * SR)
    cs = int(clip_start * SR)
    ce = cs + clip_samples

    # Saathiya stems for this clip
    sv = s_vocals[cs:ce]
    so = s_other[cs:ce]
    sd = s_drums[cs:ce]
    sb = s_bass[cs:ce]

    # KG audio for this clip
    kg_audio_start = clip_start - kg_timeline_offset
    ka_s = int(kg_audio_start * SR)
    ka_e = ka_s + clip_samples

    # Extract KG with boundary handling
    def extract_kg(src, start, length):
        out = np.zeros((length, 2))
        src_s = max(0, start)
        src_e = min(len(src), start + length)
        dst_s = max(0, -start)
        dst_e = dst_s + (src_e - src_s)
        if src_e > src_s and dst_e <= length:
            out[dst_s:dst_e] = src[src_s:src_e]
        return out

    kd = extract_kg(kg_rhythm, ka_s, clip_samples)        # KG drums+bass (full)
    kd_muf = extract_kg(kg_muffled, ka_s, clip_samples)    # KG muffled
    kd_mid = extract_kg(kg_mid_filtered, ka_s, clip_samples)  # KG mid-filtered

    # Trim to common length
    min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))
    sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
    kd, kd_muf, kd_mid = kd[:min_len], kd_muf[:min_len], kd_mid[:min_len]

    # ── Build envelopes ──

    t = np.arange(min_len) / SR + clip_start
    drop_t = s_drop
    buildup_t = buildup_start

    # KG volume envelope: 0 → buildup → DROP to full
    kg_vol = np.zeros(min_len)
    # KG filter: 0=muffled, 1=open
    kg_filter = np.zeros(min_len)
    # Saathiya bass envelope: 1=full bass, 0=bass cut (swap)
    s_bass_env = np.ones(min_len)

    for i in range(min_len):
        ti = t[i]
        if ti < buildup_t:
            # Pure Saathiya, KG silent
            kg_vol[i] = 0.0
            kg_filter[i] = 0.0
            s_bass_env[i] = 1.0
        elif ti < drop_t:
            # Buildup: KG fading in muffled
            p = (ti - buildup_t) / (drop_t - buildup_t)  # 0→1
            kg_vol[i] = 0.15 + 0.35 * p  # 0.15 → 0.5 (still quiet)
            kg_filter[i] = p * 0.4  # filter opens to 40% (still muffled-ish)
            s_bass_env[i] = 1.0 - 0.5 * p  # Saathiya bass starts to dip
        else:
            # POST-DROP: KG full, Saathiya bass gone, flute rides over
            # Quick snap to full over ~1 beat
            beat_s = 60.0 / s_bpm
            p = min(1.0, (ti - drop_t) / beat_s)
            kg_vol[i] = 0.5 + 0.5 * p  # 0.5 → 1.0
            kg_filter[i] = 0.4 + 0.6 * p  # filter fully opens
            s_bass_env[i] = 0.5 * (1.0 - p)  # Saathiya bass cuts to 0

    # ── Apply filter blend to KG ──

    kg_with_filter = np.zeros_like(kd)
    for ch in range(2):
        lo = kg_filter <= 0.4
        hi = kg_filter > 0.4
        # 0→0.4: muffled → mid
        t_lo = np.clip(kg_filter / 0.4, 0, 1)
        # 0.4→1.0: mid → open
        t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)

        kg_with_filter[lo, ch] = (
            kd_muf[lo, ch] * (1 - t_lo[lo]) +
            kd_mid[lo, ch] * t_lo[lo]
        )
        kg_with_filter[hi, ch] = (
            kd_mid[hi, ch] * (1 - t_hi[hi]) +
            kd[hi, ch] * t_hi[hi]
        )

    # ── Final mix ──

    # Saathiya: vocals + other (melody/flute) always present
    # Saathiya: drums + bass fade out at bass swap
    saathiya_mix = (
        sv * 0.85 +
        so * 0.70 +
        sd * 0.40 * s_bass_env[:, np.newaxis] +
        sb * 0.50 * s_bass_env[:, np.newaxis]
    )

    # KG: drums + bass with filter and volume envelope
    kg_mix = kg_with_filter * kg_vol[:, np.newaxis]

    mix = saathiya_mix + kg_mix

    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    label = f"drop-kg{kg_drop_beat:.1f}s"
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)

# ── Also try half-beat offsets between the detected beats ──

log("Step 5: Half-beat offset variants")

beat_s = 60.0 / s_bpm
half_beat = beat_s / 2

# Take the two most likely candidates and add half-beat offsets
if len(kg_candidates) >= 2:
    base_beats = kg_candidates[:2]
else:
    base_beats = kg_candidates

for base in base_beats:
    for offset_name, offset in [("-half", -half_beat), ("+half", half_beat)]:
        kg_drop_beat = base + offset
        kg_timeline_offset = s_drop - kg_drop_beat

        buildup_start = s_drop - buildup_bars * bar_s
        clip_start = max(0, buildup_start - pre_context)
        clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * bar_s)
        clip_samples = int((clip_end - clip_start) * SR)
        cs = int(clip_start * SR)

        sv = s_vocals[cs:cs+clip_samples]
        so = s_other[cs:cs+clip_samples]
        sd = s_drums[cs:cs+clip_samples]
        sb = s_bass[cs:cs+clip_samples]

        kg_audio_start = clip_start - kg_timeline_offset
        ka_s = int(kg_audio_start * SR)

        kd2 = extract_kg(kg_rhythm, ka_s, clip_samples)
        kd_muf2 = extract_kg(kg_muffled, ka_s, clip_samples)
        kd_mid2 = extract_kg(kg_mid_filtered, ka_s, clip_samples)

        min_len = min(len(sv), len(so), len(sd), len(sb), len(kd2))
        sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
        kd2, kd_muf2, kd_mid2 = kd2[:min_len], kd_muf2[:min_len], kd_mid2[:min_len]

        t = np.arange(min_len) / SR + clip_start
        drop_t = s_drop
        buildup_t = buildup_start

        kg_vol = np.zeros(min_len)
        kg_filter = np.zeros(min_len)
        s_bass_env = np.ones(min_len)

        for i in range(min_len):
            ti = t[i]
            if ti < buildup_t:
                kg_vol[i] = 0.0
                kg_filter[i] = 0.0
                s_bass_env[i] = 1.0
            elif ti < drop_t:
                p = (ti - buildup_t) / (drop_t - buildup_t)
                kg_vol[i] = 0.15 + 0.35 * p
                kg_filter[i] = p * 0.4
                s_bass_env[i] = 1.0 - 0.5 * p
            else:
                p = min(1.0, (ti - drop_t) / beat_s)
                kg_vol[i] = 0.5 + 0.5 * p
                kg_filter[i] = 0.4 + 0.6 * p
                s_bass_env[i] = 0.5 * (1.0 - p)

        kg_f2 = np.zeros_like(kd2)
        for ch in range(2):
            lo = kg_filter <= 0.4
            hi = kg_filter > 0.4
            t_lo = np.clip(kg_filter / 0.4, 0, 1)
            t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
            kg_f2[lo, ch] = kd_muf2[lo, ch] * (1 - t_lo[lo]) + kd_mid2[lo, ch] * t_lo[lo]
            kg_f2[hi, ch] = kd_mid2[hi, ch] * (1 - t_hi[hi]) + kd2[hi, ch] * t_hi[hi]

        saathiya_mix = sv*0.85 + so*0.70 + sd*0.40*s_bass_env[:,np.newaxis] + sb*0.50*s_bass_env[:,np.newaxis]
        kg_mix = kg_f2 * kg_vol[:, np.newaxis]
        mix = saathiya_mix + kg_mix
        peak = np.max(np.abs(mix))
        if peak > 0.95:
            mix *= 0.95 / peak

        label = f"drop-kg{kg_drop_beat:.1f}s{offset_name}"
        wav_path = f"{OUTPUT}/{label}.wav"
        write_wav(wav_path, mix)
        to_mp3(wav_path)


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  Each clip: ~8s Saathiya → 4-bar KG buildup → DROP → 8 bars of flute over beat")
print(f"\n  Listen:  open {OUTPUT}/drop-*.mp3")
