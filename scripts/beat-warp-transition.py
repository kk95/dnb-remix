#!/usr/bin/env python3
"""
Beat-warp KG to Saathiya's exact beat grid.

Problem: Global tempo stretch produces ~86.1 BPM instead of 87.6,
causing tracks to drift apart after the alignment point.

Solution: Stretch KG beat-by-beat so every single beat lands
exactly on Saathiya's beat grid. Like elastic audio in a DAW.

Uses the +1/8 beat offset that sounded good at the drop.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, resample
import warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/beat-warped"
SR = 44100

os.makedirs(OUTPUT, exist_ok=True)


def load_stereo(p):
    a, _ = sf.read(p, dtype='float64')
    return a if a.ndim == 2 else np.column_stack([a, a])

def load_mono(p):
    a, _ = sf.read(p, dtype='float64')
    return a.mean(axis=1) if a.ndim > 1 else a

def write_wav(p, a):
    sf.write(p, a, SR, subtype='PCM_16')
    print(f"  → {p} ({len(a)/SR:.1f}s)")

def to_mp3(p):
    mp3 = p.replace('.wav', '.mp3')
    subprocess.run(['ffmpeg','-y','-i',p,'-codec:a','libmp3lame','-b:a','320k',mp3],
                   capture_output=True, check=True)
    print(f"  → {mp3}")
    return mp3


print("="*60)
print("  BEAT-WARP: Lock KG to Saathiya's beat grid")
print("="*60)

# ── Step 1: Detect beat grids ───────────────────────────────────────────

print("\n[1] Detecting beat grids...")

s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_bf = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_bf, sr=SR)
s_bpm = 60.0 / np.median(np.diff(s_beats))
beat_s = 60.0 / s_bpm
bar_s = 4 * beat_s

print(f"  Saathiya: {s_bpm:.3f} BPM, {len(s_beats)} beats")

kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
_, kg_bf = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats = librosa.frames_to_time(kg_bf, sr=SR)
kg_bpm_raw = 60.0 / np.median(np.diff(kg_beats))
if kg_bpm_raw > 120:
    kg_bpm = kg_bpm_raw / 2
    kg_beats = kg_beats[::2]
else:
    kg_bpm = kg_bpm_raw

print(f"  Kho Gayi: {kg_bpm:.3f} BPM, {len(kg_beats)} beats")


# ── Step 2: Establish beat correspondence ───────────────────────────────

print("\n[2] Establishing beat correspondence...")

# From previous testing: +1/8 offset sounded best
# Peak-sync found KG timeline start at 278.386s
# +1/8 beat nudge = shift KG earlier by beat_s/2
# New KG timeline start = 278.386 - beat_s/2
kg_timeline_start = 278.386 - beat_s / 2

# At Saathiya time T, KG is at time: T - kg_timeline_start
# So KG beat at kg_beats[i] in original KG audio
# should land at Saathiya time: kg_timeline_start + kg_beats[i] * (s_bpm / kg_bpm)
# Wait — the beats are in the ORIGINAL KG time. When stretched, they'd be at:
# kg_beats[i] / stretch_ratio... but we're doing beat-by-beat warping, so:

# The correspondence is:
# KG beat i → should land at Saathiya beat j, where
# Saathiya beat j is the beat closest to: kg_timeline_start + kg_beats[i] * (s_bpm / kg_bpm)

# More directly: we want KG's beat grid to align with Saathiya's beat grid
# starting from the alignment point.

# Find which Saathiya beat aligns with which KG beat
# KG beat i (at time kg_beats[i] in original) should stretch to land at
# a Saathiya beat. The target time for KG beat i in the output timeline:
# target_time = kg_timeline_start + kg_beats[i] * (s_bpm / kg_bpm)

# But we want exact Saathiya beat positions, so snap each target to nearest S beat
stretch_approx = s_bpm / kg_bpm  # approximate stretch ratio

kg_target_times = []  # where each KG beat should land in Saathiya's timeline
kg_to_s_beat = {}     # KG beat index → Saathiya beat index

for ki in range(len(kg_beats)):
    # Approximate target position in Saathiya timeline
    approx_target = kg_timeline_start + kg_beats[ki] * (1.0 / stretch_approx)
    # Snap to nearest Saathiya beat
    si = np.argmin(np.abs(s_beats - approx_target))
    kg_target_times.append(s_beats[si])
    kg_to_s_beat[ki] = si

kg_target_times = np.array(kg_target_times)

# Show alignment around the drop region
drop_region_kg = np.where((kg_beats >= 20) & (kg_beats <= 35))[0]
print(f"\n  Beat mapping near KG 20-35s (transition zone):")
for ki in drop_region_kg[:8]:
    si = kg_to_s_beat[ki]
    print(f"    KG beat {ki}: {kg_beats[ki]:.3f}s → Saathiya beat {si}: {s_beats[si]:.3f}s "
          f"(Saathiya time {s_beats[si]:.1f}s = {int(s_beats[si]//60)}:{s_beats[si]%60:04.1f})")


# ── Step 3: Beat-warp KG stems ──────────────────────────────────────────

print("\n[3] Beat-warping KG stems...")

def beat_warp_stereo(audio, source_beats_s, target_beats_s):
    """
    Warp stereo audio so source beats land at target beat times.
    target_beats_s are relative to the OUTPUT audio (starting from 0).
    """
    # Convert target times to be relative to output start
    output_start = target_beats_s[0]
    target_relative = target_beats_s - output_start

    segments_L = []
    segments_R = []

    for i in range(len(source_beats_s) - 1):
        # Source segment
        src_start = int(source_beats_s[i] * SR)
        src_end = int(source_beats_s[i + 1] * SR)

        if src_start >= len(audio) or src_end > len(audio):
            break

        seg_L = audio[src_start:src_end, 0]
        seg_R = audio[src_start:src_end, 1]
        src_samples = len(seg_L)

        if src_samples == 0:
            continue

        # Target duration
        tgt_duration = target_relative[i + 1] - target_relative[i]
        tgt_samples = int(tgt_duration * SR)

        if tgt_samples <= 0 or tgt_samples > src_samples * 3:
            # Sanity check: don't stretch more than 3x
            tgt_samples = src_samples

        # Resample to target length (scipy resample for clean stretching)
        if abs(tgt_samples - src_samples) > 2:  # only if different
            stretched_L = resample(seg_L, tgt_samples)
            stretched_R = resample(seg_R, tgt_samples)
        else:
            stretched_L = seg_L
            stretched_R = seg_R

        segments_L.append(stretched_L)
        segments_R.append(stretched_R)

    result_L = np.concatenate(segments_L)
    result_R = np.concatenate(segments_R)
    return np.column_stack([result_L, result_R])


# Load KG stems (original, un-stretched)
kg_drums_orig = load_stereo(f"{STEMS}/kho-gayi/drums.wav")
kg_bass_orig = load_stereo(f"{STEMS}/kho-gayi/bass.wav")
kg_other_orig = load_stereo(f"{STEMS}/kho-gayi/other.wav")

# We only need the section of KG that covers our transition
# Transition: ~4:40 to ~5:30 in Saathiya time
# In KG time: (280 - kg_timeline_start) to (330 - kg_timeline_start)
# ≈ 2s to 52s in KG time

# Find KG beats that fall in this range (with some padding)
kg_range_start = 0  # start from beginning for safety
kg_range_end = min(len(kg_beats) - 1, len(kg_beats))

# Use all KG beats and their mapped Saathiya target times
source_beat_times = kg_beats[kg_range_start:kg_range_end]
target_beat_times = kg_target_times[kg_range_start:kg_range_end]

print(f"  Warping {len(source_beat_times)} beats...")
print(f"  KG range: {source_beat_times[0]:.1f}s - {source_beat_times[-1]:.1f}s")
print(f"  Target range: {target_beat_times[0]:.1f}s - {target_beat_times[-1]:.1f}s (Saathiya time)")

for name, audio in [('drums', kg_drums_orig), ('bass', kg_bass_orig), ('other', kg_other_orig)]:
    print(f"  Warping KG {name}...")
    warped = beat_warp_stereo(audio, source_beat_times, target_beat_times)
    out_path = f"{OUTPUT}/kg-{name}-warped.wav"
    write_wav(out_path, warped)

# The warped audio starts at target_beat_times[0] in Saathiya's timeline
warped_start_in_saathiya = target_beat_times[0]
print(f"\n  Warped KG starts at Saathiya time: {warped_start_in_saathiya:.3f}s")


# ── Step 4: Build the transition clip ───────────────────────────────────

print("\n[4] Building transition clip with beat-warped KG...")

# Load warped KG stems
kg_d_w = load_stereo(f"{OUTPUT}/kg-drums-warped.wav")
kg_b_w = load_stereo(f"{OUTPUT}/kg-bass-warped.wav")

min_kg_w = min(len(kg_d_w), len(kg_b_w))
kg_rhythm_w = kg_d_w[:min_kg_w] * 0.9 + kg_b_w[:min_kg_w] * 0.75

# Filtered versions
nyq = SR / 2
b_lo, a_lo = butter(4, min(300/nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500/nyq, 0.99), btype='low')

kg_muffled_w = np.column_stack([filtfilt(b_lo, a_lo, kg_rhythm_w[:, 0]),
                                 filtfilt(b_lo, a_lo, kg_rhythm_w[:, 1])])
kg_mid_w = np.column_stack([filtfilt(b_mid, a_mid, kg_rhythm_w[:, 0]),
                             filtfilt(b_mid, a_mid, kg_rhythm_w[:, 1])])

# Load Saathiya stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass = load_stereo(f"{STEMS}/saathiya/bass.wav")

# Clip window: 15s before drop → 20s after drop (longer to verify no drift)
s_drop = 307.0
buildup_bars = 4
clip_start = s_drop - buildup_bars * bar_s - 10
clip_end = min(len(s_vocals) / SR, s_drop + 12 * bar_s)  # 12 bars after = ~33s
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

sv = s_vocals[cs:cs + clip_samples]
so = s_other[cs:cs + clip_samples]
sd = s_drums[cs:cs + clip_samples]
sb = s_bass[cs:cs + clip_samples]

# KG warped audio position: the warped audio starts at warped_start_in_saathiya
# To extract the clip region from warped KG:
kg_w_clip_start = clip_start - warped_start_in_saathiya
ka_s = int(kg_w_clip_start * SR)

def extract(src, start, length):
    out = np.zeros((length, 2))
    s, e = max(0, start), min(len(src), start + length)
    d = max(0, -start)
    if e > s and d + (e - s) <= length:
        out[d:d + (e - s)] = src[s:e]
    return out

kd = extract(kg_rhythm_w, ka_s, clip_samples)
km = extract(kg_muffled_w, ka_s, clip_samples)
kmd = extract(kg_mid_w, ka_s, clip_samples)

ml = min(len(sv), len(so), len(sd), len(sb), len(kd))
sv, so, sd, sb = sv[:ml], so[:ml], sd[:ml], sb[:ml]
kd, km, kmd = kd[:ml], km[:ml], kmd[:ml]

# Envelopes
t = np.arange(ml) / SR + clip_start
buildup_start = s_drop - buildup_bars * bar_s

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

# Apply filter
kg_filtered = np.zeros_like(kd)
for ch in range(2):
    lo = kg_filt <= 0.4
    hi = kg_filt > 0.4
    tl = np.clip(kg_filt / 0.4, 0, 1)
    th = np.clip((kg_filt - 0.4) / 0.6, 0, 1)
    kg_filtered[lo, ch] = km[lo, ch] * (1 - tl[lo]) + kmd[lo, ch] * tl[lo]
    kg_filtered[hi, ch] = kmd[hi, ch] * (1 - th[hi]) + kd[hi, ch] * th[hi]

# Mix
mix = (sv * 0.85 + so * 0.70
       + sd * 0.40 * s_bass_env[:, np.newaxis]
       + sb * 0.50 * s_bass_env[:, np.newaxis]
       + kg_filtered * kg_vol[:, np.newaxis])

peak = np.max(np.abs(mix))
if peak > 0.95:
    mix *= 0.95 / peak

wav_path = f"{OUTPUT}/drop-beat-warped.wav"
write_wav(wav_path, mix)
to_mp3(wav_path)

print(f"\n  Clip duration: {ml/SR:.1f}s")
print(f"  Post-drop riding: {clip_end - s_drop:.1f}s (should stay locked the whole time)")

print(f"\n{'='*60}")
print(f"  DONE!")
print(f"  open {OUTPUT}/drop-beat-warped.mp3")
print(f"{'='*60}")
