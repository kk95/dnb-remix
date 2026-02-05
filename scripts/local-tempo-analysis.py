#!/usr/bin/env python3
"""
Local Tempo Analysis — Saathiya drums stem around the drop region.

Measures actual inter-beat intervals to detect micro-tempo variation
that could explain the groove mismatch perception.

Hypothesis: Saathiya's actual tempo in 306-325s is slightly faster
than the global 87.593 BPM, causing cumulative drift.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft/saathiya"
OUTPUT = f"{PROJECT}/output/analysis"
SR = 44100

# Global BPM (what we've been assuming)
GLOBAL_BPM = 87.593
GLOBAL_IBI = 60.0 / GLOBAL_BPM  # inter-beat interval

# Analyze a wider region to see tempo trend
REGION_START = 280.0  # well before drop
REGION_END = 340.0    # well after drop
DROP_TIME = 306.8     # the drop point

os.makedirs(OUTPUT, exist_ok=True)


def load_region_mono(path, start_s, end_s):
    start_sample = int(start_s * SR)
    end_sample = int(end_s * SR)
    audio, sr = sf.read(path, dtype='float64', start=start_sample, stop=end_sample)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio


def main():
    drums_path = f"{STEMS}/drums.wav"

    print(f"Local Tempo Analysis — Saathiya Drums")
    print(f"Global BPM: {GLOBAL_BPM} | Global IBI: {GLOBAL_IBI*1000:.3f}ms")
    print(f"Region: {REGION_START:.0f}s - {REGION_END:.0f}s")
    print(f"Drop point: {DROP_TIME:.1f}s")

    # Load audio
    audio = load_region_mono(drums_path, REGION_START, REGION_END)
    print(f"Loaded {len(audio)/SR:.1f}s of drums audio")

    # Beat tracking with librosa
    print("\n--- Beat Tracking ---")
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio, sr=SR,
        start_bpm=87.5,
        units='frames',
        hop_length=512,
        tightness=200,
    )
    beat_times_relative = librosa.frames_to_time(beat_frames, sr=SR, hop_length=512)
    beat_times = beat_times_relative + REGION_START

    print(f"Detected {len(beat_times)} beats")
    if hasattr(tempo, '__len__'):
        print(f"Estimated tempo: {tempo[0]:.3f} BPM")
    else:
        print(f"Estimated tempo: {tempo:.3f} BPM")

    # Calculate inter-beat intervals
    ibis = np.diff(beat_times)
    ibi_ms = ibis * 1000
    local_bpms = 60.0 / ibis

    print(f"\n--- Inter-Beat Intervals ---")
    print(f"{'Beat':>5s}  {'Time':>10s}  {'IBI (ms)':>10s}  {'Local BPM':>10s}  {'vs Global':>10s}")
    print(f"{'-----':>5s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")

    for i in range(len(ibis)):
        t = beat_times[i]
        ibi = ibi_ms[i]
        bpm = local_bpms[i]
        diff = bpm - GLOBAL_BPM
        marker = " <<< DROP" if abs(t - DROP_TIME) < 2 else ""
        print(f"{i+1:5d}  {t:10.3f}  {ibi:10.1f}  {bpm:10.3f}  {diff:+10.3f}{marker}")

    # Analyze pre-drop vs post-drop
    pre_drop_mask = beat_times[:-1] < DROP_TIME
    post_drop_mask = beat_times[:-1] >= DROP_TIME

    pre_ibis = ibis[pre_drop_mask]
    post_ibis = ibis[post_drop_mask]

    print(f"\n--- Regional Summary ---")
    if len(pre_ibis) > 0:
        pre_bpm = 60.0 / np.mean(pre_ibis)
        print(f"Pre-drop  ({np.sum(pre_drop_mask)} beats): Mean IBI = {np.mean(pre_ibis)*1000:.2f}ms, BPM = {pre_bpm:.3f}")
    if len(post_ibis) > 0:
        post_bpm = 60.0 / np.mean(post_ibis)
        print(f"Post-drop ({np.sum(post_drop_mask)} beats): Mean IBI = {np.mean(post_ibis)*1000:.2f}ms, BPM = {post_bpm:.3f}")
        print(f"Difference: {post_bpm - pre_bpm:+.3f} BPM")

    # Cumulative drift from global grid
    print(f"\n--- Cumulative Drift Analysis ---")
    # Build a perfect grid starting from the first detected beat
    perfect_grid = beat_times[0] + np.arange(len(beat_times)) * GLOBAL_IBI
    drift = (beat_times - perfect_grid) * 1000  # ms

    print(f"{'Beat':>5s}  {'Actual':>10s}  {'Perfect':>10s}  {'Drift (ms)':>12s}")
    for i in range(len(beat_times)):
        marker = " <<< DROP" if abs(beat_times[i] - DROP_TIME) < 2 else ""
        print(f"{i+1:5d}  {beat_times[i]:10.3f}  {perfect_grid[i]:10.3f}  {drift[i]:+12.1f}{marker}")

    # Check drift rate in post-drop region
    post_beat_indices = np.where(beat_times >= DROP_TIME)[0]
    if len(post_beat_indices) > 2:
        post_drifts = drift[post_beat_indices]
        post_times = beat_times[post_beat_indices]
        drift_rate = np.polyfit(post_times - post_times[0], post_drifts, 1)
        print(f"\nPost-drop drift rate: {drift_rate[0]:+.2f} ms/s")
        print(f"Over 18s: {drift_rate[0] * 18:+.1f} ms total drift")

        if drift_rate[0] < 0:
            implied_bpm = 60.0 / (GLOBAL_IBI - drift_rate[0]/1000)
            print(f"Implied local BPM: {implied_bpm:.3f} (faster than {GLOBAL_BPM})")
        elif drift_rate[0] > 0:
            implied_bpm = 60.0 / (GLOBAL_IBI + drift_rate[0]/1000)
            print(f"Implied local BPM: {implied_bpm:.3f} (slower than {GLOBAL_BPM})")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle('Saathiya Local Tempo Analysis (Drums Stem)', fontsize=14, fontweight='bold')

    # Panel 1: Local BPM over time
    ax1 = axes[0]
    ax1.plot(beat_times[:-1], local_bpms, 'o-', color='blue', markersize=4)
    ax1.axhline(GLOBAL_BPM, color='red', linewidth=2, linestyle='--', label=f'Global BPM ({GLOBAL_BPM})')
    ax1.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label=f'Drop ({DROP_TIME}s)')
    ax1.set_ylabel('Local BPM')
    ax1.set_title('Beat-by-Beat Local BPM')
    ax1.legend()
    ax1.set_ylim(GLOBAL_BPM - 3, GLOBAL_BPM + 3)

    # Panel 2: IBI over time
    ax2 = axes[1]
    ax2.plot(beat_times[:-1], ibi_ms, 'o-', color='purple', markersize=4)
    ax2.axhline(GLOBAL_IBI * 1000, color='red', linewidth=2, linestyle='--', label=f'Global IBI ({GLOBAL_IBI*1000:.1f}ms)')
    ax2.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label=f'Drop ({DROP_TIME}s)')
    ax2.set_ylabel('Inter-Beat Interval (ms)')
    ax2.set_title('Beat-by-Beat Inter-Beat Interval')
    ax2.legend()

    # Panel 3: Cumulative drift
    ax3 = axes[2]
    ax3.plot(beat_times, drift, 'o-', color='red', markersize=4)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label=f'Drop ({DROP_TIME}s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Drift (ms)')
    ax3.set_title('Cumulative Drift from Global 87.593 BPM Grid')
    ax3.legend()

    plt.tight_layout()
    out_path = f"{OUTPUT}/local-tempo-analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
