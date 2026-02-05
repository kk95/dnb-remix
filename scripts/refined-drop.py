#!/usr/bin/env python3
"""
Refined drop transition — 4 bass-flute interplay variants.

Fixes from Spotify screenshot comparison:
  - KG volume rises gradually over full 8 bars (not snap at midpoint)
  - Low-pass filter opens gradually over full 8 bars
  - Bass swap still at center (bar 4)
  - Saathiya fades out fast at end of transition

Variants:
  A) Sidechain pump — KG bass ducks when flute plays, pushes in gaps
  B) Call and response — bass and flute take turns
  C) Filter cycling — rhythmic filter sweep on KG bass synced to bars
  D) Bass swells — KG bass volume pulses every half-bar
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, hilbert

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
WARPED = f"{PROJECT}/output/beat-locked"
OUTPUT = f"{PROJECT}/output/refined"
SR = 44100

SAATHIYA_BPM = 87.593
SAATHIYA_BEAT = 60.0 / SAATHIYA_BPM
BAR_S = 4 * SAATHIYA_BEAT

# Winning offset
KG_DROP_BEAT = 24.822  # warped KG time that aligns with Saathiya drop

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
    """Extract segment with boundary clamping."""
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


def lowpass(audio, cutoff_hz, order=4):
    """Apply lowpass filter to stereo audio."""
    nyq = SR / 2
    freq = min(cutoff_hz / nyq, 0.99)
    if freq <= 0.01:
        return np.zeros_like(audio)
    b, a = butter(order, freq, btype='low')
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = filtfilt(b, a, audio[:, ch])
    return out


def get_envelope(mono_audio, sr=SR, frame_ms=20):
    """Get amplitude envelope of mono audio."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    n_frames = len(mono_audio) // hop
    env = np.zeros(len(mono_audio))
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(mono_audio))
        env[start:end] = np.maximum(env[start:end], np.sqrt(np.mean(mono_audio[start:end]**2)))
    # Smooth
    from scipy.ndimage import uniform_filter1d
    env = uniform_filter1d(env, size=int(sr * 0.05))
    return env


# ── Load everything ───────────────────────────────────────────────────

log("Loading stems")

# Saathiya stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")  # flute/melody
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

# Beat-locked KG stems
kg_drums = load_stereo(f"{WARPED}/kg-drums-warped.wav")
kg_bass = load_stereo(f"{WARPED}/kg-bass-warped.wav")
kg_other = load_stereo(f"{WARPED}/kg-other-warped.wav")

# Saathiya beat grid
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_beat_frames = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_beat_frames, sr=SR)
del s_mono  # free memory

# Drop point
s_drop_target = 307.0
s_drop_idx = np.argmin(np.abs(s_beats - s_drop_target))
s_drop = s_beats[s_drop_idx]
print(f"  Saathiya drop: {s_drop:.3f}s")
print(f"  KG drop beat: {KG_DROP_BEAT}s")

# Timeline: 8 bars total centered on drop
# Spotify: transition starts 4 bars before drop, ends 4 bars after
trans_bars = 8
half_bars = trans_bars // 2
trans_start = s_drop - half_bars * BAR_S   # 4 bars before drop
trans_end = s_drop + half_bars * BAR_S     # 4 bars after drop

# Clip window: add context before and after
pre_context = 4   # seconds of pure Saathiya before transition
post_context = 4  # seconds after transition ends
clip_start = trans_start - pre_context
clip_end = trans_end + post_context
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

print(f"  Transition: {trans_start:.1f}s → {s_drop:.1f}s (bass swap) → {trans_end:.1f}s")
print(f"  Clip: {clip_start:.1f}s → {clip_end:.1f}s ({clip_samples/SR:.1f}s)")

# KG timeline offset
kg_timeline_offset = s_drop - KG_DROP_BEAT
ka_s = int((clip_start - kg_timeline_offset) * SR)

# Extract stems for clip window
sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]    # flute/melody
sd = s_drums[cs:cs+clip_samples]
sb = s_bass[cs:cs+clip_samples]

kd = extract(kg_drums, ka_s, clip_samples)
kb = extract(kg_bass, ka_s, clip_samples)
ko = extract(kg_other, ka_s, clip_samples)

# KG rhythm (drums + bass combined)
kg_rhythm = kd * 0.9 + kb * 0.75

min_len = min(len(sv), len(so), len(sd), len(sb), len(kd), len(kb))
sv, so, sd, sb = sv[:min_len], so[:min_len], sd[:min_len], sb[:min_len]
kd, kb, ko = kd[:min_len], kb[:min_len], ko[:min_len]
kg_rhythm = kg_rhythm[:min_len]

# Time array for envelopes
t = np.arange(min_len) / SR + clip_start

print(f"  All stems loaded and clipped")


# ── Spotify-accurate base envelopes ──────────────────────────────────

log("Building Spotify-style envelopes")

# These match the screenshot:
# - KG volume: gradual rise over full 8 bars
# - Low-pass filter: gradual open over full 8 bars
# - Bass swap: at center (bar 4 = s_drop)
# - Saathiya volume: full, then fast out at end

kg_vol = np.zeros(min_len)
kg_filter_env = np.zeros(min_len)  # 0=fully muffled, 1=fully open
s_bass_env = np.ones(min_len)
s_vol_env = np.ones(min_len)

for i in range(min_len):
    ti = t[i]
    if ti < trans_start:
        # Pure Saathiya
        kg_vol[i] = 0.0
        kg_filter_env[i] = 0.0
        s_bass_env[i] = 1.0
        s_vol_env[i] = 1.0
    elif ti < trans_end:
        # Full 8-bar transition
        p = (ti - trans_start) / (trans_end - trans_start)  # 0→1 over 8 bars

        # KG volume: gradual S-curve rise
        kg_vol[i] = 0.5 * (1 - np.cos(np.pi * p))  # smooth 0→1

        # Filter: gradual open (300Hz → 20kHz)
        kg_filter_env[i] = p  # linear 0→1

        # Bass swap at center (p=0.5)
        if p < 0.5:
            # Pre-swap: Saathiya bass fading, KG bass rising
            bp = p / 0.5  # 0→1 over first 4 bars
            s_bass_env[i] = 1.0 - 0.8 * bp  # 1.0 → 0.2
        else:
            # Post-swap: Saathiya bass gone
            s_bass_env[i] = 0.0

        # Saathiya melody: stays full, fast out at very end
        if p > 0.85:
            fade_p = (p - 0.85) / 0.15  # 0→1 in last 15%
            s_vol_env[i] = 1.0 - 0.7 * fade_p  # fade to 0.3
        else:
            s_vol_env[i] = 1.0
    else:
        # After transition
        kg_vol[i] = 1.0
        kg_filter_env[i] = 1.0
        s_bass_env[i] = 0.0
        # Saathiya melody continues but quieter
        fade_after = min(1.0, (ti - trans_end) / (post_context * 0.5))
        s_vol_env[i] = 0.3 * (1.0 - fade_after)


def apply_filter_envelope(audio, filter_env, min_hz=300, max_hz=18000):
    """Apply time-varying lowpass filter using crossfading between pre-filtered versions."""
    # Pre-compute filtered versions at several cutoff frequencies
    cutoffs = [300, 600, 1200, 2500, 5000, 10000, 18000]
    filtered = {}
    for c in cutoffs:
        filtered[c] = lowpass(audio, c)
    filtered[99999] = audio.copy()  # unfiltered
    cutoffs.append(99999)

    out = np.zeros_like(audio)
    for i in range(len(audio)):
        fval = filter_env[i]
        # Map 0→1 to frequency range (log scale)
        if fval <= 0:
            out[i] = filtered[300][i]
            continue
        if fval >= 1:
            out[i] = audio[i]
            continue

        target_hz = min_hz * (max_hz / min_hz) ** fval
        # Find surrounding cutoffs
        lower_c = cutoffs[0]
        upper_c = cutoffs[-1]
        for j in range(len(cutoffs) - 1):
            if cutoffs[j] <= target_hz <= cutoffs[j+1]:
                lower_c = cutoffs[j]
                upper_c = cutoffs[j+1]
                break
        # Crossfade between the two
        if upper_c == lower_c:
            blend = 0
        else:
            blend = (target_hz - lower_c) / (upper_c - lower_c)
        out[i] = filtered[lower_c][i] * (1 - blend) + filtered[upper_c][i] * blend

    return out


# Pre-compute filtered KG rhythm at all cutoffs (shared across variants)
print("  Pre-computing filtered KG versions...")
cutoffs_list = [300, 600, 1200, 2500, 5000, 10000, 18000]
kg_filtered_versions = {}
for c in cutoffs_list:
    kg_filtered_versions[c] = lowpass(kg_rhythm, c)
kg_filtered_versions[99999] = kg_rhythm.copy()
cutoffs_all = cutoffs_list + [99999]

# Same for KG bass alone
kb_filtered_versions = {}
for c in cutoffs_list:
    kb_filtered_versions[c] = lowpass(kb, c)
kb_filtered_versions[99999] = kb.copy()


def fast_filter_apply(audio, filter_env, filtered_cache, cutoffs,
                      min_hz=300, max_hz=18000):
    """Apply filter envelope using pre-computed filtered versions (vectorized)."""
    out = np.zeros_like(audio)

    # Map filter_env (0→1) to target frequencies
    target_hz = min_hz * (max_hz / min_hz) ** np.clip(filter_env, 0, 1)

    for j in range(len(cutoffs) - 1):
        lower_c = cutoffs[j]
        upper_c = cutoffs[j+1]
        mask = (target_hz >= lower_c) & (target_hz < upper_c)
        if not np.any(mask):
            continue
        blend = (target_hz[mask] - lower_c) / (upper_c - lower_c)
        for ch in range(audio.shape[1]):
            out[mask, ch] = (
                filtered_cache[lower_c][mask, ch] * (1 - blend) +
                filtered_cache[upper_c][mask, ch] * blend
            )

    # Handle >= max cutoff
    mask = target_hz >= cutoffs[-1]
    if np.any(mask):
        out[mask] = audio[mask]

    # Handle <= 0
    mask = filter_env <= 0
    if np.any(mask):
        out[mask] = filtered_cache[cutoffs[0]][mask]

    return out


print("  Applying base filter to KG rhythm...")
kg_base_filtered = fast_filter_apply(
    kg_rhythm, kg_filter_env, kg_filtered_versions, cutoffs_all
)

# Flute envelope for sidechain/call-response variants
print("  Computing flute envelope...")
flute_mono = so.mean(axis=1)
flute_env = get_envelope(flute_mono)
# Normalize
flute_env = flute_env / (np.max(flute_env) + 1e-10)


# ── Saathiya base mix (shared across all variants) ────────────────────

saathiya_base = (
    sv * 0.85 * s_vol_env[:, np.newaxis] +
    so * 0.70 * s_vol_env[:, np.newaxis] +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)


def finalize(mix, label):
    """Normalize and export."""
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)
    wav_path = f"{OUTPUT}/{label}.wav"
    write_wav(wav_path, mix)
    to_mp3(wav_path)


# ── Variant A: Sidechain Pump ────────────────────────────────────────

log("Variant A: Sidechain Pump")
# KG bass ducks when flute plays, pushes in gaps

# Invert flute envelope: loud flute = quiet bass
sidechain = 1.0 - 0.6 * flute_env  # ducks to 40% when flute is loud
# Smooth the sidechain for a pumping feel
from scipy.ndimage import uniform_filter1d
sidechain = uniform_filter1d(sidechain, size=int(SR * 0.08))

kg_a = kg_base_filtered * kg_vol[:, np.newaxis] * sidechain[:min_len, np.newaxis]
mix_a = saathiya_base + kg_a
finalize(mix_a, "A-sidechain-pump")


# ── Variant B: Call and Response ──────────────────────────────────────

log("Variant B: Call and Response")
# Bass and flute take turns every half-bar

half_bar_samples = int(BAR_S / 2 * SR)
call_response = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < trans_start or ti > trans_end:
        continue
    # Which half-bar are we in?
    bars_in = (ti - trans_start) / BAR_S
    half_bar_idx = int(bars_in * 2) % 2  # 0 or 1
    if half_bar_idx == 0:
        # Flute's turn: duck bass
        call_response[i] = 0.3
    else:
        # Bass's turn: full bass
        call_response[i] = 1.0

# Smooth transitions between turns
call_response = uniform_filter1d(call_response, size=int(SR * 0.05))

kg_b = kg_base_filtered * kg_vol[:, np.newaxis] * call_response[:, np.newaxis]
# Also duck flute during bass turns
flute_duck = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < trans_start or ti > trans_end:
        continue
    bars_in = (ti - trans_start) / BAR_S
    half_bar_idx = int(bars_in * 2) % 2
    if half_bar_idx == 1:
        # Bass's turn: slightly duck flute
        flute_duck[i] = 0.7

flute_duck = uniform_filter1d(flute_duck, size=int(SR * 0.05))

saathiya_b = (
    sv * 0.85 * s_vol_env[:, np.newaxis] +
    so * 0.70 * s_vol_env[:, np.newaxis] * flute_duck[:, np.newaxis] +
    sd * 0.40 * s_bass_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)
mix_b = saathiya_b + kg_b
finalize(mix_b, "B-call-response")


# ── Variant C: Filter Cycling ─────────────────────────────────────────

log("Variant C: Filter Cycling")
# Rhythmic filter sweep on KG bass synced to bars — opens and closes

# Create a cycling filter that oscillates on top of the base filter
cycle_filter = np.copy(kg_filter_env)
for i in range(min_len):
    ti = t[i]
    if ti < trans_start or ti > trans_end + post_context:
        continue
    # Oscillate at 1 cycle per bar
    bars_in = (ti - trans_start) / BAR_S
    osc = 0.5 * (1 + np.sin(2 * np.pi * bars_in))  # 0→1→0 per bar
    # Modulation depth decreases as transition progresses (filter more open later)
    p = (ti - trans_start) / (trans_end - trans_start)
    p = np.clip(p, 0, 1)
    depth = 0.4 * (1 - p)  # more cycling early, less later
    cycle_filter[i] = np.clip(kg_filter_env[i] + depth * (osc - 0.5), 0, 1)

kg_c_filtered = fast_filter_apply(
    kg_rhythm, cycle_filter, kg_filtered_versions, cutoffs_all
)
kg_c = kg_c_filtered * kg_vol[:, np.newaxis]
mix_c = saathiya_base + kg_c
finalize(mix_c, "C-filter-cycling")


# ── Variant D: Bass Swells ────────────────────────────────────────────

log("Variant D: Bass Swells")
# KG bass volume pulses every half-bar — wave-like energy

swell = np.ones(min_len)
for i in range(min_len):
    ti = t[i]
    if ti < trans_start or ti > trans_end + post_context:
        continue
    # Pulse at 2 cycles per bar (every half-bar)
    bars_in = (ti - trans_start) / BAR_S
    pulse = 0.5 * (1 + np.sin(2 * np.pi * 2 * bars_in - np.pi/2))  # starts at trough
    # Depth decreases as we go (more swell early in transition, steady later)
    p = (ti - trans_start) / (trans_end - trans_start)
    p = np.clip(p, 0, 1)
    depth = 0.5 * (1 - 0.6 * p)  # swell range narrows over time
    swell[i] = (1 - depth) + depth * pulse

swell = uniform_filter1d(swell, size=int(SR * 0.02))

kg_d = kg_base_filtered * kg_vol[:, np.newaxis] * swell[:, np.newaxis]
mix_d = saathiya_base + kg_d
finalize(mix_d, "D-bass-swells")


# ── Also generate base Spotify-accurate version (no extra FX) ─────────

log("Base: Spotify-accurate (no extra FX)")
kg_base = kg_base_filtered * kg_vol[:, np.newaxis]
mix_base = saathiya_base + kg_base
finalize(mix_base, "base-spotify-style")


log("ALL DONE!")
print(f"  Clips in: {OUTPUT}/")
print(f"  A-sidechain-pump.mp3  — bass ducks with flute, pushes in gaps")
print(f"  B-call-response.mp3   — bass and flute take turns every half-bar")
print(f"  C-filter-cycling.mp3  — rhythmic filter sweep on bass per bar")
print(f"  D-bass-swells.mp3     — bass volume pulses every half-bar")
print(f"  base-spotify-style.mp3 — clean Spotify-accurate envelopes, no extras")
print(f"\n  open {OUTPUT}/*.mp3")
