#!/usr/bin/env python3
"""
Refined drop v2 — keeps the SNAP envelope from drop-24822ms (which sounded
right), applies bass-flute interplay effects only in the post-drop ride.

The lesson: gradual 8-bar envelopes blurred the rhythmic anchoring and caused
perceived timing drift. The snap at the drop is what makes it feel locked.

Variants (effects applied POST-DROP only):
  A) Sidechain pump — KG bass ducks when flute plays
  B) Call and response — bass/flute trade every half-bar
  C) Filter cycling — rhythmic filter sweep on KG bass per bar
  D) Bass swells — KG bass volume pulses every half-bar
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
OUTPUT = f"{PROJECT}/output/refined-v2"
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
    """Amplitude envelope with smoothing."""
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

# Beat grid
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
del s_mono

s_drop = s_beats[np.argmin(np.abs(s_beats - 307.0))]
print(f"  Drop point: {s_drop:.3f}s")

# ── Same clip window as drop-24822ms ──────────────────────────────────

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

print(f"  Clip: {clip_start:.1f}s → {clip_end:.1f}s ({clip_samples/SR:.1f}s)")
print(f"  Buildup starts: {buildup_start:.1f}s, Drop: {s_drop:.1f}s")

sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

min_kg = min(len(kg_drums), len(kg_bass))
kg_rhythm = kg_drums[:min_kg] * 0.9 + kg_bass[:min_kg] * 0.75

kd = extract(kg_rhythm, ka_s, clip_samples)
kb_raw = extract(kg_bass, ka_s, clip_samples)

# Pre-filter KG for muffled buildup
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500 / nyq, 0.99), btype='low')

kd_muf = np.column_stack([filtfilt(b_lo, a_lo, kd[:, 0]), filtfilt(b_lo, a_lo, kd[:, 1])])
kd_mid = np.column_stack([filtfilt(b_mid, a_mid, kd[:, 0]), filtfilt(b_mid, a_mid, kd[:, 1])])

min_len = min(len(sv), len(so), len(sd), len(sb), len(kd))
sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
kd, kd_muf, kd_mid = kd[:min_len], kd_muf[:min_len], kd_mid[:min_len]
kb_raw = kb_raw[:min_len]

t = np.arange(min_len) / SR + clip_start
beat_s = SAATHIYA_BEAT

# ── SNAP envelopes (same as drop-24822ms) ─────────────────────────────

log("Building SNAP envelopes (same as winning clip)")

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

# Apply filter to KG (same method as original)
kg_with_filter = np.zeros_like(kd)
for ch in range(2):
    lo = kg_filter <= 0.4
    hi = kg_filter > 0.4
    t_lo = np.clip(kg_filter / 0.4, 0, 1)
    t_hi = np.clip((kg_filter - 0.4) / 0.6, 0, 1)
    kg_with_filter[lo, ch] = kd_muf[lo, ch] * (1 - t_lo[lo]) + kd_mid[lo, ch] * t_lo[lo]
    kg_with_filter[hi, ch] = kd_mid[hi, ch] * (1 - t_hi[hi]) + kd[hi, ch] * t_hi[hi]

# Base KG (snap envelope, no extra effects)
kg_base = kg_with_filter * kg_vol[:, np.newaxis]

# Saathiya base
saathiya_base = (
    sv * 0.85 +
    so * 0.70 +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)

# Flute envelope (for sidechain/call-response)
flute_mono = so.mean(axis=1)
flute_env = get_envelope(flute_mono)
flute_env = flute_env / (np.max(flute_env) + 1e-10)

# Post-drop mask: effects only apply after the drop
post_drop = (t >= s_drop).astype(float)
# Smooth the onset so effects don't pop in
onset_smooth = np.minimum(1.0, np.maximum(0.0, (t - s_drop) / (2 * beat_s)))
post_drop = post_drop * onset_smooth


def finalize(mix, label):
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


# ── Variant A: Sidechain Pump (post-drop only) ───────────────────────

log("Variant A: Sidechain Pump (post-drop)")

sidechain = 1.0 - 0.6 * flute_env[:min_len]
sidechain = uniform_filter1d(sidechain, size=int(SR * 0.08))
# Blend: pre-drop = 1.0 (no effect), post-drop = sidechain
sc_blend = 1.0 - post_drop + post_drop * sidechain
kg_a = kg_base * sc_blend[:, np.newaxis]
mix_a = saathiya_base + kg_a
finalize(mix_a, "A-sidechain-pump")


# ── Variant B: Call and Response (post-drop only) ─────────────────────

log("Variant B: Call and Response (post-drop)")

call_response_kg = np.ones(min_len)
call_response_flute = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < s_drop:
        continue
    bars_after = (ti - s_drop) / BAR_S
    half_bar_idx = int(bars_after * 2) % 2
    if half_bar_idx == 0:
        # Flute's turn: duck bass
        call_response_kg[i] = 0.35
    else:
        # Bass's turn: slightly duck flute
        call_response_flute[i] = 0.75

call_response_kg = uniform_filter1d(call_response_kg, size=int(SR * 0.04))
call_response_flute = uniform_filter1d(call_response_flute, size=int(SR * 0.04))

kg_b = kg_base * call_response_kg[:, np.newaxis]
saathiya_b = (
    sv * 0.85 +
    so * 0.70 * call_response_flute[:, np.newaxis] +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)
mix_b = saathiya_b + kg_b
finalize(mix_b, "B-call-response")


# ── Variant C: Filter Cycling (post-drop only) ────────────────────────

log("Variant C: Filter Cycling (post-drop)")

# Pre-compute additional filtered versions for cycling
kg_lo_1k = lowpass(kd, 1000)
kg_lo_3k = lowpass(kd, 3000)
kg_lo_8k = lowpass(kd, 8000)

kg_c = np.copy(kg_base)
for i in range(min_len):
    ti = t[i]
    if ti < s_drop:
        continue
    bars_after = (ti - s_drop) / BAR_S
    # 1 cycle per bar: sweeps low→high→low
    osc = 0.5 * (1 + np.sin(2 * np.pi * bars_after))  # 0→1→0
    # Blend between filtered versions based on oscillator
    if osc < 0.33:
        blend = osc / 0.33
        for ch in range(2):
            kg_c[i, ch] = (kg_lo_1k[i, ch] * (1-blend) + kg_lo_3k[i, ch] * blend) * kg_vol[i]
    elif osc < 0.66:
        blend = (osc - 0.33) / 0.33
        for ch in range(2):
            kg_c[i, ch] = (kg_lo_3k[i, ch] * (1-blend) + kg_lo_8k[i, ch] * blend) * kg_vol[i]
    else:
        blend = (osc - 0.66) / 0.34
        for ch in range(2):
            kg_c[i, ch] = (kg_lo_8k[i, ch] * (1-blend) + kd[i, ch] * blend) * kg_vol[i]

mix_c = saathiya_base + kg_c
finalize(mix_c, "C-filter-cycling")


# ── Variant D: Bass Swells (post-drop only) ───────────────────────────

log("Variant D: Bass Swells (post-drop)")

swell = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < s_drop:
        continue
    bars_after = (ti - s_drop) / BAR_S
    # Pulse every half-bar
    pulse = 0.5 * (1 + np.sin(2 * np.pi * 2 * bars_after - np.pi/2))
    # Swell depth: 0.5 means bass dips to 50% at trough
    swell[i] = 0.5 + 0.5 * pulse

swell = uniform_filter1d(swell, size=int(SR * 0.02))

kg_d = kg_base * swell[:, np.newaxis]
mix_d = saathiya_base + kg_d
finalize(mix_d, "D-bass-swells")


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  All use the SNAP envelope from drop-24822ms (buildup → snap → ride)")
print(f"  Effects only kick in AFTER the drop point")
print(f"\n  open {OUTPUT}/*.mp3")
