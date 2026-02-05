#!/usr/bin/env python3
"""Analyze flute loop boundaries — energy, waveform, spectral content at start/end."""

import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import matplotlib.pyplot as plt
from scipy.signal import stft

PROJECT = "/Users/kshitijkarke/Documents/dnb-remix"
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# Same constants as flute-loop-drop.py
TARGET_BPM = 88.0
BEAT_S = 60.0 / TARGET_BPM
BAR_S = 4 * BEAT_S
SAATHIYA_LOCAL_BPM = 87.939
SAATHIYA_BAR_S = 4 * (60.0 / SAATHIYA_LOCAL_BPM)
BEAT_GRID_S = 306.772
LOOP_BARS = 4

# Load the "other" stem (flute)
s_other, _ = sf.read(f"{STEMS}/saathiya/other.wav", dtype='float64')
if s_other.ndim == 1:
    s_other = np.column_stack([s_other, s_other])

# Extract raw flute loop (same as main script)
flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_start = int(BEAT_GRID_S * SR)
flute_raw = s_other[flute_start:flute_start + flute_actual_len]

# Stretch to exact 88.0 BPM
target_loop_samples = int(LOOP_BARS * BAR_S * SR)
flute_stretch = (LOOP_BARS * BAR_S) / (LOOP_BARS * SAATHIYA_BAR_S)
flute_loop = pyrb.time_stretch(flute_raw, SR, flute_stretch)
flute_loop = flute_loop[:target_loop_samples]

# Pad if needed
if len(flute_loop) < target_loop_samples:
    pad = np.zeros((target_loop_samples - len(flute_loop), 2))
    flute_loop = np.vstack([flute_loop, pad])

print(f"Flute loop: {len(flute_loop)} samples = {len(flute_loop)/SR:.4f}s")
print(f"  = {LOOP_BARS} bars at {TARGET_BPM} BPM")
print(f"  Beat = {BEAT_S:.4f}s, Bar = {BAR_S:.4f}s")

# Mono for analysis
mono = np.mean(flute_loop, axis=1)

# --- Plot 1: Full loop energy profile ---
fig, axes = plt.subplots(5, 1, figsize=(16, 14))

# Full waveform
ax = axes[0]
t = np.arange(len(mono)) / SR
ax.plot(t, mono, linewidth=0.3, alpha=0.6)
for bar in range(LOOP_BARS + 1):
    ax.axvline(bar * BAR_S, color='red', linewidth=1, alpha=0.7, linestyle='--')
    if bar < LOOP_BARS:
        ax.text(bar * BAR_S + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.3,
                f'Bar {bar+1}', fontsize=9, color='red')
ax.set_title('Full Flute Loop — Waveform')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')

# RMS energy per beat
beat_samples = int(BEAT_S * SR)
n_beats = LOOP_BARS * 4
rms_per_beat = []
for b in range(n_beats):
    start = b * beat_samples
    end = min(start + beat_samples, len(mono))
    chunk = mono[start:end]
    rms = np.sqrt(np.mean(chunk**2))
    rms_per_beat.append(rms)
    print(f"  Beat {b+1:2d} (bar {b//4+1}, beat {b%4+1}): RMS = {rms:.4f}")

ax = axes[1]
ax.bar(range(1, n_beats+1), rms_per_beat, color='steelblue', alpha=0.8)
for bar in range(LOOP_BARS):
    ax.axvline(bar * 4 + 0.5, color='red', linewidth=1, alpha=0.5, linestyle='--')
ax.set_title('RMS Energy Per Beat')
ax.set_xlabel('Beat number')
ax.set_ylabel('RMS')

# --- Plot 2: Zoom on loop boundary (last 2 beats → first 2 beats) ---
zoom_beats = 2
zoom_samples = int(zoom_beats * BEAT_S * SR)

# End of loop
end_region = mono[-zoom_samples:]
# Start of loop
start_region = mono[:zoom_samples]

ax = axes[2]
t_end = np.arange(len(end_region)) / SR
t_start = np.arange(len(start_region)) / SR
ax.plot(t_end, end_region, label='Loop END (last 2 beats)', alpha=0.7)
ax.plot(t_start, start_region, label='Loop START (first 2 beats)', alpha=0.7)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title('Loop Boundary: END vs START overlay')
ax.set_xlabel('Time within region (s)')
ax.legend()

# --- Plot 3: Energy envelope (short-term RMS) ---
window = int(0.05 * SR)  # 50ms windows
hop = window // 2
n_frames = (len(mono) - window) // hop
rms_envelope = np.zeros(n_frames)
for i in range(n_frames):
    start = i * hop
    rms_envelope[i] = np.sqrt(np.mean(mono[start:start+window]**2))

ax = axes[3]
t_env = np.arange(n_frames) * hop / SR
ax.plot(t_env, rms_envelope, linewidth=1)
for bar in range(LOOP_BARS + 1):
    ax.axvline(bar * BAR_S, color='red', linewidth=1, alpha=0.7, linestyle='--')
ax.set_title('Short-term RMS Envelope (50ms window)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS')

# --- Plot 4: What the current crossfade produces ---
# Simulate current make_loop for 2 iterations
xf_samples = int(4 * BEAT_S * SR)  # 4 beats = 1 bar
loop_len = len(flute_loop)

# Current (buggy) approach
total_len = loop_len * 2
looped_buggy = np.zeros(total_len)
fade_in = np.linspace(0, 1, xf_samples)
fade_out = np.linspace(1, 0, xf_samples)
for i in range(2):
    offset = i * loop_len
    end = offset + loop_len
    looped_buggy[offset:end] += mono
    if i > 0:
        xf_start = offset
        xf_end = offset + xf_samples
        looped_buggy[xf_start:xf_end] *= 0
        looped_buggy[xf_start:xf_end] = (
            mono[-xf_samples:] * fade_out +
            mono[:xf_samples] * fade_in
        )

# Proper overlap approach
step = loop_len - xf_samples
total_proper = step * 2 + xf_samples
looped_proper = np.zeros(total_proper)
t_lin = np.linspace(0, 1, xf_samples)
fi = np.sqrt(t_lin)
fo = np.sqrt(1 - t_lin)
for i in range(2):
    offset = i * step
    chunk = mono.copy()
    if i > 0:
        chunk[:xf_samples] *= fi
    if i < 1:
        chunk[-xf_samples:] *= fo
    looped_proper[offset:offset + loop_len] += chunk

ax = axes[4]
# Show region around first loop boundary
boundary_buggy = loop_len
show_range = int(2 * BEAT_S * SR)
region_b = looped_buggy[boundary_buggy - show_range:boundary_buggy + show_range]
t_b = np.arange(len(region_b)) / SR - show_range / SR

boundary_proper = step
region_p = looped_proper[boundary_proper - show_range:boundary_proper + show_range]
t_p = np.arange(len(region_p)) / SR - show_range / SR

ax.plot(t_b, region_b, label='Current (buggy)', alpha=0.6)
ax.plot(t_p, region_p, label='Proper overlap', alpha=0.6)
ax.axvline(0, color='gray', linewidth=1, linestyle=':', label='Loop boundary')
ax.set_title('Loop Boundary Comparison: Current vs Proper Crossfade')
ax.set_xlabel('Time relative to boundary (s)')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT}/flute-loop-analysis.png", dpi=150)
print(f"\nSaved: {OUTPUT}/flute-loop-analysis.png")

# Print key findings
print(f"\n{'='*60}")
print("ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Loop length: {len(mono)/SR:.4f}s ({LOOP_BARS} bars)")
print(f"Crossfade: {xf_samples} samples = {xf_samples/SR:.4f}s = {xf_samples/SR/BEAT_S:.1f} beats")
print(f"\nPer-bar RMS:")
for bar in range(LOOP_BARS):
    bar_rms = rms_per_beat[bar*4:(bar+1)*4]
    avg = np.mean(bar_rms)
    print(f"  Bar {bar+1}: avg RMS = {avg:.4f}  [{', '.join(f'{r:.4f}' for r in bar_rms)}]")
print(f"\nLast beat RMS:  {rms_per_beat[-1]:.4f}")
print(f"First beat RMS: {rms_per_beat[0]:.4f}")
print(f"Ratio (last/first): {rms_per_beat[-1]/max(rms_per_beat[0], 1e-10):.2f}")
print(f"\nLast sample value: {mono[-1]:.6f}")
print(f"First sample value: {mono[0]:.6f}")
print(f"Jump at boundary: {abs(mono[-1] - mono[0]):.6f}")

# Check for silence at loop boundaries
tail_1ms = mono[-int(0.001*SR):]
head_1ms = mono[:int(0.001*SR)]
print(f"\nTail 1ms peak: {np.max(np.abs(tail_1ms)):.6f}")
print(f"Head 1ms peak: {np.max(np.abs(head_1ms)):.6f}")

tail_50ms = mono[-int(0.05*SR):]
head_50ms = mono[:int(0.05*SR)]
print(f"Tail 50ms RMS: {np.sqrt(np.mean(tail_50ms**2)):.6f}")
print(f"Head 50ms RMS: {np.sqrt(np.mean(head_50ms**2)):.6f}")
