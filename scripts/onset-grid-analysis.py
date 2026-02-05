#!/usr/bin/env python3
"""
Onset-vs-Beat-Grid Analysis for Saathiya Post-Drop Region.

Analyzes where vocal and flute (other) onsets land relative to the
beat grid in the post-drop region (306.8s - 325s).

Goal: quantify whether Saathiya's performance is ahead of / behind
the mathematical beat grid, which could explain the "sped up" feel.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft/saathiya"
OUTPUT = f"{PROJECT}/output/analysis"
SR = 44100

# Precise BPM and beat interval
BPM = 87.593
BEAT_INTERVAL = 60.0 / BPM  # 0.68503...s

# Post-drop region
REGION_START = 306.8  # seconds (drop point)
REGION_END = 325.0    # ~8 bars after drop
BAR_DURATION = 4 * BEAT_INTERVAL  # ~2.740s

os.makedirs(OUTPUT, exist_ok=True)


def load_region_mono(path, start_s, end_s):
    """Load a region of audio as mono float64."""
    start_sample = int(start_s * SR)
    end_sample = int(end_s * SR)
    audio, sr = sf.read(path, dtype='float64', start=start_sample, stop=end_sample)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio


def detect_onsets(audio, sr, stem_type="vocals"):
    """Detect onsets with parameters tuned for stem type."""
    if stem_type == "vocals":
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr,
            hop_length=512,
            backtrack=True,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.07,
            wait=4,
        )
    else:
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr,
            hop_length=512,
            backtrack=True,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.05,
            wait=3,
        )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    return onset_times


def build_beat_grid(start_s, end_s, beat_interval):
    """Build a beat grid from start to end."""
    beats = []
    t = start_s
    while t <= end_s:
        beats.append(t)
        t += beat_interval
    return np.array(beats)


def nearest_beat_offset(onset_time, beat_grid):
    """Find the nearest beat and return (nearest_beat, offset_ms).
    Positive offset = onset is AFTER the beat.
    Negative offset = onset is BEFORE the beat.
    """
    diffs = onset_time - beat_grid
    idx = np.argmin(np.abs(diffs))
    nearest = beat_grid[idx]
    offset_ms = (onset_time - nearest) * 1000.0
    return nearest, offset_ms


def analyze_stem(name, audio_path, stem_type):
    """Full analysis for one stem."""
    print(f"\n{'='*70}")
    print(f"  STEM: {name} ({stem_type})")
    print(f"  Region: {REGION_START:.1f}s - {REGION_END:.1f}s")
    print(f"{'='*70}")

    audio = load_region_mono(audio_path, REGION_START, REGION_END)
    duration = len(audio) / SR
    print(f"  Loaded {duration:.3f}s of audio ({len(audio)} samples)")

    onset_times_relative = detect_onsets(audio, SR, stem_type)
    onset_times = onset_times_relative + REGION_START

    print(f"  Detected {len(onset_times)} onsets")

    beat_grid = build_beat_grid(REGION_START, REGION_END, BEAT_INTERVAL)
    print(f"  Beat grid: {len(beat_grid)} beats from {beat_grid[0]:.3f}s to {beat_grid[-1]:.3f}s")

    results = []
    for onset in onset_times:
        nearest_beat, offset_ms = nearest_beat_offset(onset, beat_grid)
        results.append({
            'onset': onset,
            'nearest_beat': nearest_beat,
            'offset_ms': offset_ms,
        })

    print(f"\n  {'Onset Time':>12s}  {'Nearest Beat':>14s}  {'Offset (ms)':>12s}  {'Direction':>10s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*10}")
    for r in results:
        direction = "EARLY" if r['offset_ms'] < 0 else "LATE " if r['offset_ms'] > 0 else "ON   "
        print(f"  {r['onset']:12.3f}s  {r['nearest_beat']:12.3f}s  {r['offset_ms']:+12.1f}  {direction}")

    offsets = [r['offset_ms'] for r in results]
    if offsets:
        offsets_arr = np.array(offsets)
        print(f"\n  STATISTICS:")
        print(f"    Mean offset:   {np.mean(offsets_arr):+.1f} ms")
        print(f"    Std dev:       {np.std(offsets_arr):.1f} ms")
        print(f"    Min (earliest): {np.min(offsets_arr):+.1f} ms")
        print(f"    Max (latest):   {np.max(offsets_arr):+.1f} ms")
        print(f"    Median:        {np.median(offsets_arr):+.1f} ms")
        n_early = np.sum(offsets_arr < -10)
        n_late = np.sum(offsets_arr > 10)
        n_on = np.sum(np.abs(offsets_arr) <= 10)
        print(f"    Early (>10ms): {n_early}/{len(offsets_arr)} ({100*n_early/len(offsets_arr):.0f}%)")
        print(f"    On beat (<=10ms): {n_on}/{len(offsets_arr)} ({100*n_on/len(offsets_arr):.0f}%)")
        print(f"    Late (>10ms):  {n_late}/{len(offsets_arr)} ({100*n_late/len(offsets_arr):.0f}%)")

        mid = len(offsets_arr) // 2
        if mid > 0:
            first_half_mean = np.mean(offsets_arr[:mid])
            second_half_mean = np.mean(offsets_arr[mid:])
            print(f"\n  DRIFT CHECK:")
            print(f"    First half mean:  {first_half_mean:+.1f} ms (onsets 1-{mid})")
            print(f"    Second half mean: {second_half_mean:+.1f} ms (onsets {mid+1}-{len(offsets_arr)})")
            print(f"    Drift:            {second_half_mean - first_half_mean:+.1f} ms (positive = getting later)")
    else:
        print("  No onsets detected!")

    return onset_times, offsets, beat_grid


def plot_analysis(vocals_data, flute_data, beat_grid):
    """Create visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 2, 1]})
    fig.suptitle(
        f'Saathiya Onset-vs-Beat-Grid Analysis\n'
        f'Post-Drop Region: {REGION_START:.1f}s - {REGION_END:.1f}s | BPM: {BPM} | Beat: {BEAT_INTERVAL*1000:.1f}ms',
        fontsize=14, fontweight='bold'
    )

    v_onsets, v_offsets, _ = vocals_data
    f_onsets, f_offsets, _ = flute_data

    # Panel 1: Timeline with beat grid and onsets
    ax1 = axes[0]
    for b in beat_grid:
        ax1.axvline(b, color='gray', alpha=0.3, linewidth=0.5)
    for i, b in enumerate(beat_grid):
        if i % 4 == 0:
            ax1.axvline(b, color='black', alpha=0.6, linewidth=1.5, linestyle='--')

    for onset, offset in zip(v_onsets, v_offsets):
        color = 'red' if offset < -10 else ('blue' if offset > 10 else 'green')
        ax1.plot(onset, 1, 'v', color=color, markersize=8, alpha=0.8)
        ax1.annotate(f'{offset:+.0f}', (onset, 1), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=6, color=color)

    for onset, offset in zip(f_onsets, f_offsets):
        color = 'red' if offset < -10 else ('blue' if offset > 10 else 'green')
        ax1.plot(onset, 0, '^', color=color, markersize=8, alpha=0.8)
        ax1.annotate(f'{offset:+.0f}', (onset, 0), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=6, color=color)

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Flute', 'Vocals'])
    ax1.set_xlim(REGION_START - 0.5, REGION_END + 0.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Onset Positions vs Beat Grid (annotations = offset in ms)')

    early_patch = mpatches.Patch(color='red', label='Early (>10ms)')
    on_patch = mpatches.Patch(color='green', label='On beat (<=10ms)')
    late_patch = mpatches.Patch(color='blue', label='Late (>10ms)')
    ax1.legend(handles=[early_patch, on_patch, late_patch], loc='upper right', fontsize=8)

    # Panel 2: Offset over time
    ax2 = axes[1]
    if len(v_onsets) > 0:
        ax2.plot(v_onsets, v_offsets, 'o-', color='orange', alpha=0.8, label='Vocals', markersize=5)
        if len(v_onsets) > 2:
            z = np.polyfit(v_onsets, v_offsets, 1)
            p = np.poly1d(z)
            ax2.plot(v_onsets, p(v_onsets), '--', color='orange', alpha=0.5,
                     label=f'Vocals trend ({z[0]:+.2f} ms/s)')

    if len(f_onsets) > 0:
        ax2.plot(f_onsets, f_offsets, 's-', color='purple', alpha=0.8, label='Flute', markersize=5)
        if len(f_onsets) > 2:
            z = np.polyfit(f_onsets, f_offsets, 1)
            p = np.poly1d(z)
            ax2.plot(f_onsets, p(f_onsets), '--', color='purple', alpha=0.5,
                     label=f'Flute trend ({z[0]:+.2f} ms/s)')

    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhspan(-10, 10, color='green', alpha=0.1, label='On-beat zone (+-10ms)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Offset from nearest beat (ms)')
    ax2.set_title('Onset Offset Over Time (negative = early, positive = late)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(REGION_START - 0.5, REGION_END + 0.5)

    # Panel 3: Histogram
    ax3 = axes[2]
    bins = np.arange(-BEAT_INTERVAL * 500, BEAT_INTERVAL * 500 + 5, 5)
    if len(v_offsets) > 0:
        ax3.hist(v_offsets, bins=bins, alpha=0.6, color='orange', label='Vocals', edgecolor='black')
    if len(f_offsets) > 0:
        ax3.hist(f_offsets, bins=bins, alpha=0.6, color='purple', label='Flute', edgecolor='black')
    ax3.axvline(0, color='black', linewidth=1.5)
    ax3.axvspan(-10, 10, color='green', alpha=0.1)
    ax3.set_xlabel('Offset from nearest beat (ms)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Onset Offsets')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out_path = f"{OUTPUT}/onset-grid-analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Visualization saved to: {out_path}")
    plt.close()


def main():
    print(f"Saathiya Onset-vs-Beat-Grid Analysis")
    print(f"BPM: {BPM} | Beat interval: {BEAT_INTERVAL*1000:.3f}ms")
    print(f"Region: {REGION_START:.1f}s - {REGION_END:.1f}s ({(REGION_END-REGION_START)/BAR_DURATION:.1f} bars)")

    vocals_path = f"{STEMS}/vocals.wav"
    other_path = f"{STEMS}/other.wav"

    v_onsets, v_offsets, beat_grid = analyze_stem("Vocals", vocals_path, "vocals")
    f_onsets, f_offsets, _ = analyze_stem("Other (Flute)", other_path, "other")

    print(f"\n{'='*70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*70}")
    if v_offsets and f_offsets:
        all_offsets = np.array(v_offsets + f_offsets)
        print(f"  All onsets combined:")
        print(f"    Mean offset:   {np.mean(all_offsets):+.1f} ms")
        print(f"    Std dev:       {np.std(all_offsets):.1f} ms")

        v_mean = np.mean(v_offsets)
        f_mean = np.mean(f_offsets)
        print(f"\n  Vocals mean: {v_mean:+.1f} ms | Flute mean: {f_mean:+.1f} ms")
        if v_mean < -15:
            print(f"  >> VOCALS are consistently EARLY by {abs(v_mean):.0f}ms")
            print(f"  >> This explains the 'sped up' perception!")
        elif v_mean > 15:
            print(f"  >> VOCALS are consistently LATE by {v_mean:.0f}ms")
        else:
            print(f"  >> Vocals are roughly on the beat grid")

        if f_mean < -15:
            print(f"  >> FLUTE is consistently EARLY by {abs(f_mean):.0f}ms")
        elif f_mean > 15:
            print(f"  >> FLUTE is consistently LATE by {f_mean:.0f}ms")
        else:
            print(f"  >> Flute is roughly on the beat grid")

    plot_analysis(
        (v_onsets, v_offsets, beat_grid),
        (f_onsets, f_offsets, beat_grid),
        beat_grid
    )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
