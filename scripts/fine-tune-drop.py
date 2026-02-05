#!/usr/bin/env python3
"""
Fine-tune the drop alignment between 23.9s and 24.6s KG beat.
Generates clips at every ~0.08s (1/8 beat) increment.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import subprocess
from scipy.signal import butter, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/fine-tune"
SR = 44100

os.makedirs(OUTPUT, exist_ok=True)

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

def to_mp3(wav_path):
    mp3 = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3
    ], capture_output=True, check=True)
    return mp3

# BPM
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
s_bpm = 60.0 / np.median(np.diff(s_beats))
bar_s = 4 * (60.0 / s_bpm)
beat_s = 60.0 / s_bpm

# Saathiya drop at 5:07
s_drop_idx = np.argmin(np.abs(s_beats - 307.0))
s_drop = s_beats[s_drop_idx]

print(f"Saathiya: {s_bpm:.3f} BPM, drop at {s_drop:.3f}s")
print(f"Beat = {beat_s:.4f}s, 1/8 beat = {beat_s/8:.4f}s")

# Reuse stretched stems from drop-transition
DROP_DIR = f"{PROJECT}/output/drop-transition"

# Load stems
print("Loading stems...")
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass_stem = load_stereo(f"{STEMS}/saathiya/bass.wav")

kg_drums = load_stereo(f"{DROP_DIR}/kg-drums.wav")
kg_bass = load_stereo(f"{DROP_DIR}/kg-bass.wav")

min_kg = min(len(kg_drums), len(kg_bass))
kg_rhythm = kg_drums[:min_kg] * 0.9 + kg_bass[:min_kg] * 0.75

nyq = SR / 2
b_lo, a_lo = butter(4, min(300/nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500/nyq, 0.99), btype='low')

print("Computing filtered KG...")
kg_muffled = np.column_stack([filtfilt(b_lo, a_lo, kg_rhythm[:,0]), filtfilt(b_lo, a_lo, kg_rhythm[:,1])])
kg_mid = np.column_stack([filtfilt(b_mid, a_mid, kg_rhythm[:,0]), filtfilt(b_mid, a_mid, kg_rhythm[:,1])])

def extract_kg(src, start, length):
    out = np.zeros((length, 2))
    s = max(0, start)
    e = min(len(src), start + length)
    d = max(0, -start)
    de = d + (e - s)
    if e > s and de <= length:
        out[d:de] = src[s:e]
    return out

# Generate clips from 24.0 to 24.6 in 0.085s steps (~1/8 beat)
step = beat_s / 8
offsets = np.arange(24.0, 24.65, step)

buildup_bars = 4
ride_bars = 8
pre_s = 8

print(f"\nGenerating {len(offsets)} clips from {offsets[0]:.2f}s to {offsets[-1]:.2f}s")
print(f"Step size: {step*1000:.1f}ms (1/8 beat)\n")

for kg_drop_beat in offsets:
    kg_offset = s_drop - kg_drop_beat
    buildup_start = s_drop - buildup_bars * bar_s
    clip_start = max(0, buildup_start - pre_s)
    clip_end = min(len(s_vocals)/SR, s_drop + ride_bars * bar_s)
    clip_samples = int((clip_end - clip_start) * SR)
    cs = int(clip_start * SR)

    sv = s_vocals[cs:cs+clip_samples]
    so = s_other[cs:cs+clip_samples]
    sd = s_drums[cs:cs+clip_samples]
    sb = s_bass_stem[cs:cs+clip_samples]

    ka_s = int((clip_start - kg_offset) * SR)
    kd = extract_kg(kg_rhythm, ka_s, clip_samples)
    km = extract_kg(kg_muffled, ka_s, clip_samples)
    kmd = extract_kg(kg_mid, ka_s, clip_samples)

    ml = min(len(sv), len(so), len(sd), len(sb), len(kd))
    sv, so, sd, sb = sv[:ml], so[:ml], sd[:ml], sb[:ml]
    kd, km, kmd = kd[:ml], km[:ml], kmd[:ml]

    t = np.arange(ml) / SR + clip_start

    kg_vol = np.zeros(ml)
    kg_filt = np.zeros(ml)
    s_bass_env = np.ones(ml)

    for i in range(ml):
        ti = t[i]
        if ti < buildup_start:
            pass
        elif ti < s_drop:
            p = (ti - buildup_start) / (s_drop - buildup_start)
            kg_vol[i] = 0.15 + 0.35 * p
            kg_filt[i] = p * 0.4
            s_bass_env[i] = 1.0 - 0.5 * p
        else:
            p = min(1.0, (ti - s_drop) / beat_s)
            kg_vol[i] = 0.5 + 0.5 * p
            kg_filt[i] = 0.4 + 0.6 * p
            s_bass_env[i] = 0.5 * (1.0 - p)

    kg_f = np.zeros_like(kd)
    for ch in range(2):
        lo = kg_filt <= 0.4
        hi = kg_filt > 0.4
        tl = np.clip(kg_filt/0.4, 0, 1)
        th = np.clip((kg_filt-0.4)/0.6, 0, 1)
        kg_f[lo, ch] = km[lo, ch]*(1-tl[lo]) + kmd[lo, ch]*tl[lo]
        kg_f[hi, ch] = kmd[hi, ch]*(1-th[hi]) + kd[hi, ch]*th[hi]

    mix = sv*0.85 + so*0.70 + sd*0.40*s_bass_env[:,np.newaxis] + sb*0.50*s_bass_env[:,np.newaxis] + kg_f*kg_vol[:,np.newaxis]

    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    ms = int(kg_drop_beat * 1000)
    label = f"drop-{ms}ms"
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    mp3 = to_mp3(wav_path)
    print(f"  {kg_drop_beat:.3f}s → {mp3}")

print(f"\nDone! Listen: open {OUTPUT}/drop-*.mp3")
print("The number in the filename is KG's beat position in milliseconds.")
