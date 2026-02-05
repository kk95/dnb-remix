#!/usr/bin/env python3
"""
Fine-tune KG alignment: shift KG earlier by small amounts so its transients
land ON Saathiya's beats instead of slightly after.

User feedback: "Saathiya comes early" = KG arrives late. Beat detection may
place KG's beat marker slightly after the audible transient.

Uses actual-grid warped KG (Saathiya micro-timing) + KG shifted earlier.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
GROOVE = f"{PROJECT}/output/groove-fix"
OUTPUT = f"{PROJECT}/output/kg-shift"
SR = 44100

SAATHIYA_BPM = 87.593
SAATHIYA_BEAT = 60.0 / SAATHIYA_BPM
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


# ── Setup ─────────────────────────────────────────────────────────────

log("Loading stems")

s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

# Use actual-grid warped KG (follows Saathiya's micro-timing)
kg_drums = load_stereo(f"{GROOVE}/kg-drums-actual.wav")
kg_bass = load_stereo(f"{GROOVE}/kg-bass-actual.wav")

s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
del s_mono

s_drop = s_beats[np.argmin(np.abs(s_beats - 307.0))]

# In actual-grid, KG drop beat lands at s_drop (306.782s)
kg_drop_actual = s_drop

buildup_bars = 4
ride_bars = 8
pre_context = 8
buildup_start = s_drop - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context)
clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * BAR_S)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)
beat_s = SAATHIYA_BEAT

sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')

print(f"  Drop: {s_drop:.3f}s, Clip: {clip_start:.1f}s → {clip_end:.1f}s")

# ── Generate clips at different KG shifts ─────────────────────────────

log("Generating KG-shifted clips")

# Shift KG earlier (negative = earlier arrival)
shifts_ms = [-15, -30, -50, -70]

for shift_ms in shifts_ms:
    shift_samples = int(shift_ms * SR / 1000)

    # KG drop beat shifts earlier
    kg_drop_shifted = kg_drop_actual + shift_ms / 1000.0
    kg_tl_offset = s_drop - kg_drop_shifted
    ka_s = int((clip_start - kg_tl_offset) * SR)

    min_kg = min(len(kg_drums), len(kg_bass))
    kg_rhythm = kg_drums[:min_kg] * 0.9 + kg_bass[:min_kg] * 0.75
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
        kg_filt[lo, ch] = kd_muf[lo, ch] * (1-t_lo[lo]) + kd_mid[lo, ch] * t_lo[lo]
        kg_filt[hi, ch] = kd_mid[hi, ch] * (1-t_hi[hi]) + kd[hi, ch] * t_hi[hi]

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

    label = f"kg-early-{abs(shift_ms)}ms"
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  kg-early-15ms  — KG arrives 15ms earlier (subtle)")
print(f"  kg-early-30ms  — KG arrives 30ms earlier")
print(f"  kg-early-50ms  — KG arrives 50ms earlier")
print(f"  kg-early-70ms  — KG arrives 70ms earlier (aggressive)")
print(f"\n  open {OUTPUT}/*.mp3")
