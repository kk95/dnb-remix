#!/usr/bin/env python3
"""
Fix the groove mismatch: Saathiya vocals feel rushed after the drop because
KG's electronic beat has different micro-timing than Saathiya's live drums.

Approach 1: Nudge Saathiya stems later post-drop (20ms, 35ms, 50ms)
Approach 2: Re-warp KG to Saathiya's ACTUAL beat positions (not constant grid)
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
WARPED = f"{PROJECT}/output/beat-locked"
OUTPUT = f"{PROJECT}/output/groove-fix"
SR = 44100

SAATHIYA_BPM = 87.593
SAATHIYA_BEAT = 60.0 / SAATHIYA_BPM
KG_BPM = 95.703
BAR_S = 4 * SAATHIYA_BEAT
KG_DROP_BEAT = 24.822

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


# ── Beat detection ────────────────────────────────────────────────────

log("Step 1: Beat detection")

s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)

kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
_, kg_beat_frames = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats = librosa.frames_to_time(kg_beat_frames, sr=SR)
kg_detected_bpm = 60.0 / np.median(np.diff(kg_beats))
if kg_detected_bpm > 120:
    kg_beats = kg_beats[::2]

s_drop = s_beats[np.argmin(np.abs(s_beats - 307.0))]
print(f"  Saathiya beats: {len(s_beats)}, KG beats: {len(kg_beats)}")
print(f"  Drop: {s_drop:.3f}s")


# ── Approach 2: Re-warp KG to Saathiya's actual beat grid ─────────────

log("Step 2: Warping KG to Saathiya's ACTUAL beat positions")

# Instead of mapping KG beats to constant intervals (87.593 BPM grid),
# map them to Saathiya's real beat times — this makes KG follow any
# micro-timing in Saathiya's performance.

# Find the Saathiya beat nearest the drop point
s_drop_idx = np.argmin(np.abs(s_beats - s_drop))

# Find the KG beat we're aligning (24.822s warped ≈ which original KG beat?)
# KG_DROP_BEAT is in warped time. In original time, beat index:
kg_warped_times = kg_beats[0] + np.arange(len(kg_beats)) * SAATHIYA_BEAT
kg_drop_idx = np.argmin(np.abs(kg_warped_times - KG_DROP_BEAT))
print(f"  KG drop beat index: {kg_drop_idx} (original time: {kg_beats[kg_drop_idx]:.3f}s)")
print(f"  Saathiya drop beat index: {s_drop_idx}")


def build_actual_timemap(kg_beats_arr, s_beats_arr, kg_drop_idx, s_drop_idx, audio_len):
    """Map each KG beat to the corresponding Saathiya beat position."""
    time_map = [(0, 0)]

    for i, kg_time in enumerate(kg_beats_arr):
        # Which Saathiya beat does this KG beat map to?
        s_idx = s_drop_idx + (i - kg_drop_idx)

        if s_idx < 0 or s_idx >= len(s_beats_arr):
            # Outside Saathiya's beat range — use constant interval
            target_time = s_beats_arr[s_drop_idx] + (i - kg_drop_idx) * SAATHIYA_BEAT
        else:
            target_time = s_beats_arr[s_idx]

        src_sample = int(kg_time * SR)
        tgt_sample = int(target_time * SR)

        if tgt_sample > 0 and src_sample > 0:
            time_map.append((src_sample, tgt_sample))

    # End marker
    if len(kg_beats_arr) > 1:
        last_src_interval = kg_beats_arr[-1] - kg_beats_arr[-2]
        ratio = SAATHIYA_BEAT / last_src_interval
    else:
        ratio = SAATHIYA_BEAT / (60.0 / KG_BPM)
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
actual_map = build_actual_timemap(kg_beats, s_beats, kg_drop_idx, s_drop_idx, kg_audio_len)
print(f"  Timemap entries: {len(actual_map)}")

# Check: what BPM do the target beats imply?
target_times = []
for src, tgt in actual_map[1:-1]:
    target_times.append(tgt / SR)
if len(target_times) > 2:
    diffs = np.diff(target_times)
    print(f"  Target beat intervals: mean={np.mean(diffs):.4f}s, std={np.std(diffs):.4f}s")
    print(f"  This follows Saathiya's actual micro-timing (not constant grid)")

# Warp KG stems to Saathiya's actual beat grid
actual_warped = {}
for stem in ['drums', 'bass', 'other']:
    out_path = f"{OUTPUT}/kg-{stem}-actual.wav"
    if os.path.exists(out_path):
        print(f"  {stem}: using cached {out_path}")
        actual_warped[stem] = out_path
        continue

    stem_audio, _ = sf.read(f"{STEMS}/kho-gayi/{stem}.wav", dtype='float64')
    stem_map = build_actual_timemap(kg_beats, s_beats, kg_drop_idx, s_drop_idx, len(stem_audio))
    print(f"  {stem}: warping to actual Saathiya grid...")
    warped = pyrb.timemap_stretch(stem_audio, SR, stem_map)
    sf.write(out_path, warped, SR, subtype='PCM_16')
    actual_warped[stem] = out_path
    print(f"  → {out_path} ({len(warped)/SR:.1f}s)")

# Figure out the new KG_DROP_BEAT for the actual-warped audio
# The KG drop beat in the actual-warped audio lands at the Saathiya drop time
# relative to the start of the warped audio. The warped audio starts the same
# way (sample 0 → sample 0), so the KG drop beat target is s_beats[s_drop_idx].
# But we need it relative to the warped KG timeline, which starts at kg_beats[0]'s
# target time.
# Actually: for actual warp, beat[kg_drop_idx] maps to s_beats[s_drop_idx].
# In the warped audio, that's at time s_beats[s_drop_idx].
kg_drop_actual = s_beats[s_drop_idx]
print(f"  KG drop beat in actual-warped audio: {kg_drop_actual:.3f}s")


# ── Load all stems ────────────────────────────────────────────────────

log("Step 3: Loading stems and building clips")

s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

del s_mono, kg_mono  # free memory

# Clip window
buildup_bars = 4
ride_bars = 8
pre_context = 8

buildup_start = s_drop - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context)
clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * BAR_S)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

# Saathiya stems
sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

# Filter setup
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')
beat_s = SAATHIYA_BEAT


def build_clip(kg_d_src, kg_b_src, kg_drop_time, label, saathiya_nudge_ms=0):
    """Build a drop clip with given KG stems and optional Saathiya nudge."""
    kg_tl_offset = s_drop - kg_drop_time
    ka_s = int((clip_start - kg_tl_offset) * SR)

    min_kg_len = min(len(kg_d_src), len(kg_b_src))
    kg_r = kg_d_src[:min_kg_len] * 0.9 + kg_b_src[:min_kg_len] * 0.75
    kd = extract(kg_r, ka_s, clip_samples)

    kd_muf = np.column_stack([filtfilt(b_lo, a_lo, kd[:, 0]), filtfilt(b_lo, a_lo, kd[:, 1])])
    kd_mid = np.column_stack([filtfilt(b_mid, a_mid, kd[:, 0]), filtfilt(b_mid, a_mid, kd[:, 1])])

    min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))

    t = np.arange(min_len) / SR + clip_start

    # SNAP envelopes (same as drop-24822ms)
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

    # Saathiya mix with optional post-drop nudge
    nudge_samples = int(saathiya_nudge_ms * SR / 1000)

    if nudge_samples > 0:
        # After the drop, shift Saathiya stems later by nudge_samples
        drop_sample = int((s_drop - clip_start) * SR)

        def nudge_stem(stem):
            out = np.copy(stem[:min_len])
            if drop_sample < min_len:
                post = stem[drop_sample:min_len]
                # Shift post-drop audio right by nudge_samples
                out[drop_sample:] = 0
                end = min(min_len, drop_sample + nudge_samples + len(post))
                src_end = end - drop_sample - nudge_samples
                if src_end > 0:
                    out[drop_sample + nudge_samples:end] = post[:src_end]
                # Crossfade at the splice point
                fade_len = min(int(SR * 0.005), nudge_samples)
                if fade_len > 0 and drop_sample + fade_len <= min_len:
                    fade = np.linspace(1, 0, fade_len)[:, np.newaxis]
                    out[drop_sample:drop_sample+fade_len] = stem[drop_sample:drop_sample+fade_len] * fade
            return out

        sv_n = nudge_stem(sv)
        so_n = nudge_stem(so)
        sd_n = nudge_stem(sd)
        sb_n = nudge_stem(sb)
    else:
        sv_n = sv[:min_len]
        so_n = so[:min_len]
        sd_n = sd[:min_len]
        sb_n = sb[:min_len]

    saathiya_mix = (
        sv_n * 0.85 +
        so_n * 0.70 +
        sd_n * 0.40 * s_bass_env[:, np.newaxis] +
        sb_n * 0.50 * s_bass_env[:, np.newaxis]
    )

    mix = saathiya_mix + kg_mix
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


# ── Approach 1: Nudge Saathiya with constant-grid KG ──────────────────

log("Approach 1: Nudge Saathiya (constant-grid KG)")

kg_drums_const = load_stereo(f"{WARPED}/kg-drums-warped.wav")
kg_bass_const = load_stereo(f"{WARPED}/kg-bass-warped.wav")

for nudge_ms in [0, 25, 40, 60]:
    label = f"nudge-{nudge_ms}ms"
    if nudge_ms == 0:
        label = "no-nudge-baseline"
    print(f"  Building {label}...")
    build_clip(kg_drums_const, kg_bass_const, KG_DROP_BEAT, label, nudge_ms)


# ── Approach 2: Actual-grid KG (no nudge needed?) ─────────────────────

log("Approach 2: KG warped to Saathiya's actual beat grid")

kg_drums_actual = load_stereo(actual_warped['drums'])
kg_bass_actual = load_stereo(actual_warped['bass'])

build_clip(kg_drums_actual, kg_bass_actual, kg_drop_actual, "actual-grid", 0)

# Also try actual-grid + small nudge in case it helps
for nudge_ms in [25, 40]:
    label = f"actual-grid-nudge-{nudge_ms}ms"
    print(f"  Building {label}...")
    build_clip(kg_drums_actual, kg_bass_actual, kg_drop_actual, label, nudge_ms)


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  no-nudge-baseline   — same as drop-24822ms (reference)")
print(f"  nudge-25ms          — Saathiya pushed 25ms late post-drop")
print(f"  nudge-40ms          — Saathiya pushed 40ms late post-drop")
print(f"  nudge-60ms          — Saathiya pushed 60ms late post-drop")
print(f"  actual-grid         — KG follows Saathiya's real beat positions")
print(f"  actual-grid-nudge-* — actual grid + small nudge")
print(f"\n  open {OUTPUT}/*.mp3")
