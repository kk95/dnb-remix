#!/usr/bin/env python3
"""
Refined drop v3 — KG stays STEADY as the rhythmic anchor.

Lesson from v2: ducking KG's volume removes the timing reference and makes
Saathiya feel early/rushing. Solution: keep KG at constant volume post-drop,
apply interplay effects to Saathiya's melody or use filter-only FX on KG.

Variants:
  A) Flipped sidechain — Saathiya flute ducks on KG hits, breathes in gaps
  B) Pure filter cycling — KG timbre sweeps (wavy) but volume stays constant
  C) Combo — filter cycling on KG + flipped sidechain on Saathiya
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
WARPED = f"{PROJECT}/output/beat-locked"
OUTPUT = f"{PROJECT}/output/refined-v3"
SR = 44100

SAATHIYA_BPM = 87.593
SAATHIYA_BEAT = 60.0 / SAATHIYA_BPM
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


def lowpass(audio, cutoff_hz, order=4):
    nyq = SR / 2
    freq = min(cutoff_hz / nyq, 0.99)
    if freq <= 0.01:
        return np.zeros_like(audio)
    b, a = butter(order, freq, btype='low')
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = filtfilt(b, a, audio[:, ch])
    return out


def extract(src, start, length):
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


def get_envelope(mono_audio):
    frame_len = int(SR * 0.02)
    hop = frame_len // 2
    n_frames = len(mono_audio) // hop
    env = np.zeros(len(mono_audio))
    for i in range(n_frames):
        s = i * hop
        e = min(s + frame_len, len(mono_audio))
        env[s:e] = np.maximum(env[s:e], np.sqrt(np.mean(mono_audio[s:e]**2)))
    env = uniform_filter1d(env, size=int(SR * 0.05))
    return env


# ── Load stems ────────────────────────────────────────────────────────

log("Loading stems")

s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums = load_stereo(f"{WARPED}/kg-drums-warped.wav")
kg_bass = load_stereo(f"{WARPED}/kg-bass-warped.wav")

s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
del s_mono

s_drop = s_beats[np.argmin(np.abs(s_beats - 307.0))]
print(f"  Drop: {s_drop:.3f}s")

# ── Clip window (same as drop-24822ms) ────────────────────────────────

buildup_bars = 4
ride_bars = 8
pre_context = 8

kg_timeline_offset = s_drop - KG_DROP_BEAT
buildup_start = s_drop - buildup_bars * BAR_S
clip_start = max(0, buildup_start - pre_context)
clip_end = min(len(s_vocals) / SR, s_drop + ride_bars * BAR_S)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)
ka_s = int((clip_start - kg_timeline_offset) * SR)

sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

min_kg = min(len(kg_drums), len(kg_bass))
kg_rhythm = kg_drums[:min_kg] * 0.9 + kg_bass[:min_kg] * 0.75
kd = extract(kg_rhythm, ka_s, clip_samples)

# Muffled KG for buildup
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')
kd_muf = np.column_stack([filtfilt(b_lo, a_lo, kd[:, 0]), filtfilt(b_lo, a_lo, kd[:, 1])])
kd_mid = np.column_stack([filtfilt(b_mid, a_mid, kd[:, 0]), filtfilt(b_mid, a_mid, kd[:, 1])])

min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))
sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
kd, kd_muf, kd_mid = kd[:min_len], kd_muf[:min_len], kd_mid[:min_len]

t = np.arange(min_len) / SR + clip_start
beat_s = SAATHIYA_BEAT

# ── SNAP envelopes ────────────────────────────────────────────────────

log("Building envelopes")

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

# Filtered KG with snap
kg_with_filter = np.zeros_like(kd)
for ch in range(2):
    lo = kg_filter <= 0.4
    hi = kg_filter > 0.4
    t_lo = np.clip(kg_filter / 0.4, 0, 1)
    t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
    kg_with_filter[lo, ch] = kd_muf[lo, ch] * (1 - t_lo[lo]) + kd_mid[lo, ch] * t_lo[lo]
    kg_with_filter[hi, ch] = kd_mid[hi, ch] * (1 - t_hi[hi]) + kd[hi, ch] * t_hi[hi]

# KG at constant post-drop volume (the anchor)
kg_steady = kg_with_filter * kg_vol[:, np.newaxis]

# KG drum onset envelope (for flipped sidechain)
kg_mono = kd.mean(axis=1)
kg_env = get_envelope(kg_mono)
kg_env = kg_env / (np.max(kg_env) + 1e-10)

print(f"  Clip: {clip_start:.1f}s → {clip_end:.1f}s ({min_len/SR:.1f}s)")


def finalize(mix, label):
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


# ── Variant A: Flipped Sidechain ──────────────────────────────────────

log("Variant A: Flipped Sidechain (flute ducks on KG hits)")

# When KG hits hard, duck Saathiya melody slightly — flute breathes around beat
# KG stays at CONSTANT volume (the anchor)
flute_duck = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < s_drop:
        continue
    # Duck flute proportional to KG envelope
    flute_duck[i] = 1.0 - 0.35 * kg_env[i]

flute_duck = uniform_filter1d(flute_duck, size=int(SR * 0.03))

saathiya_a = (
    sv * 0.85 +
    so * 0.70 * flute_duck[:, np.newaxis] +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)
mix_a = saathiya_a + kg_steady
finalize(mix_a, "A-flipped-sidechain")


# ── Variant B: Pure Filter Cycling ────────────────────────────────────

log("Variant B: Pure Filter Cycling (wavy timbre, constant volume)")

# Pre-compute filtered KG at different cutoffs
kg_lo_1k = lowpass(kd, 1000)
kg_lo_3k = lowpass(kd, 3000)
kg_lo_8k = lowpass(kd, 8000)

# KG post-drop: filter sweeps but volume stays rock-steady
kg_b = np.copy(kg_with_filter)
for i in range(min_len):
    ti = t[i]
    if ti < s_drop:
        continue
    bars_after = (ti - s_drop) / BAR_S
    # Sine sweep: low → high → low, 1 cycle per bar
    osc = 0.5 * (1 + np.sin(2 * np.pi * bars_after))

    # Blend between filtered versions (tonal movement, no volume change)
    if osc < 0.33:
        blend = osc / 0.33
        for ch in range(2):
            kg_b[i, ch] = kg_lo_1k[i, ch] * (1-blend) + kg_lo_3k[i, ch] * blend
    elif osc < 0.66:
        blend = (osc - 0.33) / 0.33
        for ch in range(2):
            kg_b[i, ch] = kg_lo_3k[i, ch] * (1-blend) + kg_lo_8k[i, ch] * blend
    else:
        blend = (osc - 0.66) / 0.34
        for ch in range(2):
            kg_b[i, ch] = kg_lo_8k[i, ch] * (1-blend) + kd[i, ch] * blend

# Apply volume envelope (constant post-drop)
kg_b = kg_b * kg_vol[:, np.newaxis]

saathiya_base = (
    sv * 0.85 +
    so * 0.70 +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)
mix_b = saathiya_base + kg_b
finalize(mix_b, "B-filter-cycling")


# ── Variant C: Combo (filter cycling + flipped sidechain) ─────────────

log("Variant C: Combo (wavy KG + flute ducks on hits)")

# KG: filter cycling (from B) — timbre moves, volume steady
# Saathiya: flipped sidechain (from A) — flute breathes around beat

saathiya_c = (
    sv * 0.85 +
    so * 0.70 * flute_duck[:, np.newaxis] +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)
mix_c = saathiya_c + kg_b  # kg_b has filter cycling from variant B
finalize(mix_c, "C-combo")


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  A-flipped-sidechain — flute breathes around KG hits (KG stays steady)")
print(f"  B-filter-cycling    — wavy filter sweep on KG, no volume change")
print(f"  C-combo             — wavy KG + flute breathing together")
print(f"\n  open {OUTPUT}/*.mp3")
