#!/usr/bin/env python3
"""
Peak-sync approach: Align KG and Saathiya by matching their actual
drum transient peaks, not BPM math.

1. Cross-correlate drum onset envelopes in the transition region
2. Find the exact sample offset where peaks align best
3. Generate waveform overlay images to visually verify
4. Create the transition clip at the optimal alignment
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy import signal as sig
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/peak-sync"
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


# ── Step 1: Load drum stems & compute onset envelopes ───────────────────

print("Loading drum stems...")
s_drums_mono = load_mono(f"{STEMS}/saathiya/drums.wav")
kg_drums_mono_original = load_mono(f"{STEMS}/kho-gayi/drums.wav")

# Get precise BPMs from full tracks
s_mono = load_mono(f"{PROJECT}/wavs/saathiya.wav")
_, s_bf = librosa.beat.beat_track(y=s_mono, sr=SR, units='frames')
s_beats = librosa.frames_to_time(s_bf, sr=SR)
s_bpm = 60.0 / np.median(np.diff(s_beats))

kg_mono = load_mono(f"{PROJECT}/wavs/kho-gayi.wav")
_, kg_bf = librosa.beat.beat_track(y=kg_mono, sr=SR, units='frames')
kg_beats_raw = librosa.frames_to_time(kg_bf, sr=SR)
kg_bpm_raw = 60.0 / np.median(np.diff(kg_beats_raw))
kg_bpm = kg_bpm_raw / 2 if kg_bpm_raw > 120 else kg_bpm_raw

stretch_ratio = s_bpm / kg_bpm
bar_s = 4 * 60.0 / s_bpm
beat_s = 60.0 / s_bpm

print(f"Saathiya: {s_bpm:.3f} BPM")
print(f"Kho Gayi: {kg_bpm:.3f} BPM")
print(f"Stretch ratio: {stretch_ratio:.6f}")
print(f"Beat: {beat_s:.4f}s, Bar: {bar_s:.3f}s")

# ── Step 2: Stretch KG drums ────────────────────────────────────────────

kg_drums_stretched_path = f"{OUTPUT}/kg-drums-stretched.wav"
# Try slightly adjusted ratio to compensate for rubberband drift
# Use the ratio directly and rely on onset correlation for fine-tuning
subprocess.run([
    'rubberband', '--tempo', str(stretch_ratio), '--crisp', '3',
    f"{STEMS}/kho-gayi/drums.wav", kg_drums_stretched_path
], capture_output=True, check=True)

kg_drums_stretched = load_mono(kg_drums_stretched_path)
print(f"Stretched KG drums: {len(kg_drums_stretched)/SR:.1f}s")

# ── Step 3: Cross-correlate onset envelopes in transition region ────────

print("\nComputing onset envelopes...")
hop = 256  # finer resolution than default 512

# Saathiya: focus on the flute/transition region (4:40 - 5:25)
s_region_start = int(280 * SR)  # 4:40
s_region_end = int(325 * SR)    # 5:25
s_region = s_drums_mono[s_region_start:s_region_end]

s_onset = librosa.onset.onset_strength(y=s_region, sr=SR, hop_length=hop)

# KG: focus on 15-35s (covers the beat entry area)
kg_region_start = int(15 * SR / stretch_ratio * stretch_ratio)  # ~15s in stretched time
kg_region_end = int(35 * SR)    # ~35s in stretched time
kg_region_start_s = 15
kg_region = kg_drums_stretched[int(kg_region_start_s*SR):int(35*SR)]

kg_onset = librosa.onset.onset_strength(y=kg_region, sr=SR, hop_length=hop)

print(f"Saathiya onset envelope: {len(s_onset)} frames ({len(s_region)/SR:.1f}s)")
print(f"KG onset envelope: {len(kg_onset)} frames ({len(kg_region)/SR:.1f}s)")

# Cross-correlate: slide KG across Saathiya and find best alignment
# We want to find: at what point in Saathiya's timeline does KG's 20-30s region
# best align with Saathiya's drum pattern?

correlation = sig.correlate(s_onset, kg_onset, mode='full')

# Convert correlation indices to time offsets
# The offset tells us: s_region starts at frame 0, and the KG region
# overlaps best when KG's start is at this frame offset
center = len(kg_onset) - 1  # zero-lag point

# We know from previous tests that the drop is around 5:07 (307s)
# So KG's ~24s mark should align with Saathiya's ~307s
# In our regions: Saathiya region starts at 280s, KG region starts at 15s
# Expected offset: (307-280) - (24-15) = 27 - 9 = 18s = 18*SR/hop frames
expected_offset_s = (307 - 280) - (24 - kg_region_start_s)
expected_offset_frames = int(expected_offset_s * SR / hop)

# Search within ±5 seconds of expected
search_range_frames = int(5 * SR / hop)
search_center = center + expected_offset_frames
search_start = max(0, search_center - search_range_frames)
search_end = min(len(correlation), search_center + search_range_frames)

best_idx = search_start + np.argmax(correlation[search_start:search_end])
best_offset_frames = best_idx - center
best_offset_s = best_offset_frames * hop / SR

# This offset means: KG region (starting at kg_region_start_s) aligns
# with Saathiya region (starting at 280s) at this offset.
# In absolute terms: KG's 0s = Saathiya's (280 + best_offset_s - kg_region_start_s)
kg_timeline_start = 280 + best_offset_s - kg_region_start_s

# So at Saathiya's 307s, KG is at: 307 - kg_timeline_start
kg_at_307 = 307.0 - kg_timeline_start

print(f"\nCross-correlation result:")
print(f"  Expected offset: {expected_offset_s:.2f}s ({expected_offset_frames} frames)")
print(f"  Best offset: {best_offset_s:.3f}s ({best_offset_frames} frames)")
print(f"  KG timeline start: {kg_timeline_start:.3f}s")
print(f"  At Saathiya 5:07, KG is at: {kg_at_307:.3f}s")
print(f"  Correlation peak: {correlation[best_idx]:.1f}")


# ── Step 4: Generate waveform overlay image ─────────────────────────────

print("\nGenerating waveform overlay...")

# Show 20 seconds around the drop point
viz_start_s = 296  # ~4:56 in Saathiya
viz_end_s = 316    # ~5:16 in Saathiya
viz_s_start = int(viz_start_s * SR)
viz_s_end = int(viz_end_s * SR)

s_viz = s_drums_mono[viz_s_start:viz_s_end]

# KG at the aligned position
kg_viz_start_in_kg = viz_start_s - kg_timeline_start
kg_viz_start_sample = int(kg_viz_start_in_kg * SR)
kg_viz_end_sample = kg_viz_start_sample + len(s_viz)

if kg_viz_start_sample >= 0 and kg_viz_end_sample <= len(kg_drums_stretched):
    kg_viz = kg_drums_stretched[kg_viz_start_sample:kg_viz_end_sample]
else:
    kg_viz = np.zeros(len(s_viz))
    src_s = max(0, kg_viz_start_sample)
    src_e = min(len(kg_drums_stretched), kg_viz_end_sample)
    dst_s = max(0, -kg_viz_start_sample)
    kg_viz[dst_s:dst_s+(src_e-src_s)] = kg_drums_stretched[src_s:src_e]

t_axis = np.linspace(viz_start_s, viz_end_s, len(s_viz))

fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes[0].plot(t_axis, s_viz, color='#4FC3F7', linewidth=0.3, alpha=0.8)
axes[0].set_title('Saathiya Drums', fontsize=10)
axes[0].set_ylabel('Amplitude')

axes[1].plot(t_axis, kg_viz, color='#FFB74D', linewidth=0.3, alpha=0.8)
axes[1].set_title(f'KG Drums (aligned: KG@{kg_at_307:.1f}s = Saathiya@307s)', fontsize=10)
axes[1].set_ylabel('Amplitude')

# Overlay both (normalized)
s_norm = s_viz / (np.max(np.abs(s_viz)) + 1e-10)
kg_norm = kg_viz / (np.max(np.abs(kg_viz)) + 1e-10)
axes[2].plot(t_axis, s_norm, color='#4FC3F7', linewidth=0.3, alpha=0.6, label='Saathiya')
axes[2].plot(t_axis, kg_norm, color='#FFB74D', linewidth=0.3, alpha=0.6, label='KG')
axes[2].set_title('Overlay (peaks should align)', fontsize=10)
axes[2].set_xlabel('Saathiya timeline (seconds)')
axes[2].legend()

# Add beat grid lines
for b in s_beats:
    if viz_start_s <= b <= viz_end_s:
        for ax in axes:
            ax.axvline(x=b, color='white', alpha=0.15, linewidth=0.5)

# Mark the drop point
for ax in axes:
    ax.axvline(x=307, color='#66BB6A', alpha=0.5, linewidth=1, linestyle='--')

plt.tight_layout()
img_path = f"{OUTPUT}/waveform-overlay.png"
plt.savefig(img_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"  → {img_path}")

# Also zoom in on 4 beats around the drop
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 4))
zoom_start = 305
zoom_end = 309
mask = (t_axis >= zoom_start) & (t_axis <= zoom_end)
ax2.plot(t_axis[mask], s_norm[mask], color='#4FC3F7', linewidth=0.5, alpha=0.7, label='Saathiya drums')
ax2.plot(t_axis[mask], kg_norm[mask], color='#FFB74D', linewidth=0.5, alpha=0.7, label='KG drums')
ax2.set_title('Zoomed: 4 beats around drop (305-309s)', fontsize=10)
ax2.set_xlabel('seconds')
ax2.legend()
for b in s_beats:
    if zoom_start <= b <= zoom_end:
        ax2.axvline(x=b, color='white', alpha=0.2, linewidth=0.5)
ax2.axvline(x=307, color='#66BB6A', alpha=0.5, linewidth=1, linestyle='--', label='drop')
plt.tight_layout()
zoom_path = f"{OUTPUT}/waveform-zoom.png"
plt.savefig(zoom_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"  → {zoom_path}")


# ── Step 5: Create the transition clip at optimal alignment ─────────────

print("\nBuilding transition clip at cross-correlation optimal alignment...")

# Load all stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass_st = load_stereo(f"{STEMS}/saathiya/bass.wav")

# Stretch all KG stems
for stem, crisp in [('drums',3), ('bass',5), ('other',5)]:
    out = f"{OUTPUT}/kg-{stem}.wav"
    if not os.path.exists(out):
        subprocess.run([
            'rubberband', '--tempo', str(stretch_ratio), '--crisp', str(crisp),
            f"{STEMS}/kho-gayi/{stem}.wav", out
        ], capture_output=True, check=True)

kg_drums_st = load_stereo(f"{OUTPUT}/kg-drums.wav")
kg_bass_st = load_stereo(f"{OUTPUT}/kg-bass.wav")

min_kg_len = min(len(kg_drums_st), len(kg_bass_st))
kg_rhythm = kg_drums_st[:min_kg_len] * 0.9 + kg_bass_st[:min_kg_len] * 0.75

# Filtered versions
nyq = SR / 2
b_lo, a_lo = butter(4, min(300/nyq, 0.99), btype='low')
b_mid, a_mid = butter(4, min(1500/nyq, 0.99), btype='low')

kg_muffled = np.column_stack([filtfilt(b_lo,a_lo,kg_rhythm[:,0]), filtfilt(b_lo,a_lo,kg_rhythm[:,1])])
kg_mid_f = np.column_stack([filtfilt(b_mid,a_mid,kg_rhythm[:,0]), filtfilt(b_mid,a_mid,kg_rhythm[:,1])])

def extract(src, start, length):
    out = np.zeros((length, 2))
    s, e = max(0, start), min(len(src), start+length)
    d = max(0, -start)
    if e > s and d+(e-s) <= length:
        out[d:d+(e-s)] = src[s:e]
    return out

# Clip: 15s before drop → 15s after drop
s_drop = 307.0
buildup_bars = 4
clip_start = s_drop - buildup_bars * bar_s - 10  # 10s context before buildup
clip_end = min(len(s_vocals)/SR, s_drop + 10 * bar_s)
clip_samples = int((clip_end - clip_start) * SR)
cs = int(clip_start * SR)

sv = s_vocals[cs:cs+clip_samples]
so = s_other[cs:cs+clip_samples]
sd = s_drums_st[cs:cs+clip_samples]
sb = s_bass_st[cs:cs+clip_samples]

ka_s = int((clip_start - kg_timeline_start) * SR)
kd = extract(kg_rhythm, ka_s, clip_samples)
km = extract(kg_muffled, ka_s, clip_samples)
kmd = extract(kg_mid_f, ka_s, clip_samples)

ml = min(len(sv), len(so), len(sd), len(sb), len(kd))
sv,so,sd,sb = sv[:ml],so[:ml],sd[:ml],sb[:ml]
kd,km,kmd = kd[:ml],km[:ml],kmd[:ml]

t = np.arange(ml)/SR + clip_start
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

kg_filtered = np.zeros_like(kd)
for ch in range(2):
    lo = kg_filt <= 0.4
    hi = kg_filt > 0.4
    tl = np.clip(kg_filt/0.4, 0, 1)
    th = np.clip((kg_filt-0.4)/0.6, 0, 1)
    kg_filtered[lo,ch] = km[lo,ch]*(1-tl[lo]) + kmd[lo,ch]*tl[lo]
    kg_filtered[hi,ch] = kmd[hi,ch]*(1-th[hi]) + kd[hi,ch]*th[hi]

mix = (sv*0.85 + so*0.70 + sd*0.40*s_bass_env[:,np.newaxis] + sb*0.50*s_bass_env[:,np.newaxis]
       + kg_filtered * kg_vol[:,np.newaxis])

peak = np.max(np.abs(mix))
if peak > 0.95:
    mix *= 0.95 / peak

wav_path = f"{OUTPUT}/drop-peak-synced.wav"
write_wav(wav_path, mix)
to_mp3(wav_path)

# Also try ±1/16 beat offsets for fine tuning
sixteenth = beat_s / 4
for nudge_name, nudge in [("-1_16", -sixteenth), ("+1_16", sixteenth),
                           ("-1_8", -beat_s/2), ("+1_8", beat_s/2)]:
    adj_kg_start = kg_timeline_start + nudge  # shift KG earlier/later
    ka_s2 = int((clip_start - adj_kg_start) * SR)
    kd2 = extract(kg_rhythm, ka_s2, clip_samples)[:ml]
    km2 = extract(kg_muffled, ka_s2, clip_samples)[:ml]
    kmd2 = extract(kg_mid_f, ka_s2, clip_samples)[:ml]

    kg_f2 = np.zeros_like(kd2)
    for ch in range(2):
        lo = kg_filt <= 0.4
        hi = kg_filt > 0.4
        tl = np.clip(kg_filt/0.4, 0, 1)
        th = np.clip((kg_filt-0.4)/0.6, 0, 1)
        kg_f2[lo,ch] = km2[lo,ch]*(1-tl[lo]) + kmd2[lo,ch]*tl[lo]
        kg_f2[hi,ch] = kmd2[hi,ch]*(1-th[hi]) + kd2[hi,ch]*th[hi]

    mix2 = (sv*0.85 + so*0.70 + sd*0.40*s_bass_env[:,np.newaxis] + sb*0.50*s_bass_env[:,np.newaxis]
            + kg_f2 * kg_vol[:,np.newaxis])
    pk = np.max(np.abs(mix2))
    if pk > 0.95: mix2 *= 0.95/pk

    wp = f"{OUTPUT}/drop-peak-synced{nudge_name}.wav"
    write_wav(wp, mix2)
    to_mp3(wp)


print(f"\n{'─'*60}")
print(f"  DONE!")
print(f"  Waveform overlay: open {OUTPUT}/waveform-overlay.png")
print(f"  Waveform zoom:    open {OUTPUT}/waveform-zoom.png")
print(f"  Best alignment:   open {OUTPUT}/drop-peak-synced.mp3")
print(f"  Fine-tune:        open {OUTPUT}/drop-peak-synced*.mp3")
print(f"  KG at Saathiya 5:07 = KG {kg_at_307:.3f}s")
print(f"{'─'*60}")
