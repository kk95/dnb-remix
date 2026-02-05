#!/usr/bin/env python3
"""
Precise Local Tempo Analysis — using onset strength + peak picking
with high time resolution (hop_length=128 = 2.9ms).

Also uses onset_strength_multi for more robust beat detection.
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
HOP = 128  # 2.9ms resolution (vs 11.6ms at 512)

GLOBAL_BPM = 87.593
GLOBAL_IBI = 60.0 / GLOBAL_BPM

REGION_START = 280.0
REGION_END = 340.0
DROP_TIME = 306.8

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
    print(f"Precise Local Tempo Analysis (hop={HOP}, res={HOP/SR*1000:.1f}ms)")
    print(f"Global BPM: {GLOBAL_BPM} | IBI: {GLOBAL_IBI*1000:.3f}ms")
    print(f"Region: {REGION_START:.0f}s - {REGION_END:.0f}s | Drop: {DROP_TIME:.1f}s\n")

    audio = load_region_mono(drums_path, REGION_START, REGION_END)
    print(f"Loaded {len(audio)/SR:.1f}s of drums audio")

    # Method 1: librosa beat_track with small hop
    print("\n=== Method 1: librosa beat_track (hop=128) ===")
    tempo1, beat_frames1 = librosa.beat.beat_track(
        y=audio, sr=SR, hop_length=HOP,
        start_bpm=87.5, tightness=200,
    )
    beat_times1 = librosa.frames_to_time(beat_frames1, sr=SR, hop_length=HOP) + REGION_START
    ibis1 = np.diff(beat_times1) * 1000
    print(f"Beats: {len(beat_times1)}, Tempo est: {tempo1[0] if hasattr(tempo1, '__len__') else tempo1:.3f}")

    # Method 2: Use onset envelope + autocorrelation for local tempo
    # Then pick beat positions from onset strength peaks
    print("\n=== Method 2: Onset Envelope Analysis ===")
    oenv = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=HOP)

    # Local tempo estimation using tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv, sr=SR, hop_length=HOP,
        win_length=400,  # about 11.6s window
    )
    # Get tempo estimate per window using tempogram
    # Compute local tempo by finding dominant period in short windows
    print(f"\n--- Per-Bar Tempo (4-beat windows) ---")
    bar_dur = 4 * GLOBAL_IBI
    for bar_start in np.arange(REGION_START, REGION_END - bar_dur, bar_dur):
        bar_end = bar_start + bar_dur
        s_frame = int((bar_start - REGION_START) * SR / HOP)
        e_frame = int((bar_end - REGION_START) * SR / HOP)
        if e_frame <= len(oenv):
            local_oenv = oenv[s_frame:e_frame]
            if len(local_oenv) > 10:
                local_tempo = librosa.feature.tempo(
                    onset_envelope=local_oenv, sr=SR, hop_length=HOP,
                    start_bpm=85, max_tempo=92,
                )
                t_val = local_tempo[0] if hasattr(local_tempo, '__len__') else local_tempo
                marker = " <<< DROP" if abs(bar_start - DROP_TIME) < bar_dur else ""
                diff = t_val - GLOBAL_BPM
                print(f"  {bar_start:.1f}-{bar_end:.1f}s: {t_val:.3f} BPM ({diff:+.3f}){marker}")

    # Analyze beat intervals with method 1 results
    print(f"\n=== Detailed Beat Analysis (Method 1, hop=128) ===")

    # Filter to reasonable IBIs (skip any doubles or halves)
    expected_ibi = GLOBAL_IBI * 1000
    valid = (ibis1 > expected_ibi * 0.85) & (ibis1 < expected_ibi * 1.15)

    print(f"Total intervals: {len(ibis1)}, Valid (within 15%): {np.sum(valid)}")

    valid_ibis = ibis1[valid]
    valid_times = beat_times1[:-1][valid]

    # Split pre/post drop
    pre_mask = valid_times < DROP_TIME
    post_mask = valid_times >= DROP_TIME

    pre_ibis = valid_ibis[pre_mask]
    post_ibis = valid_ibis[post_mask]

    print(f"\nPre-drop  ({len(pre_ibis)} valid beats):")
    print(f"  Mean IBI: {np.mean(pre_ibis):.2f}ms (BPM: {60000/np.mean(pre_ibis):.3f})")
    print(f"  Std IBI:  {np.std(pre_ibis):.2f}ms")
    print(f"  Min IBI:  {np.min(pre_ibis):.2f}ms (BPM: {60000/np.min(pre_ibis):.3f})")
    print(f"  Max IBI:  {np.max(pre_ibis):.2f}ms (BPM: {60000/np.max(pre_ibis):.3f})")

    print(f"\nPost-drop ({len(post_ibis)} valid beats):")
    print(f"  Mean IBI: {np.mean(post_ibis):.2f}ms (BPM: {60000/np.mean(post_ibis):.3f})")
    print(f"  Std IBI:  {np.std(post_ibis):.2f}ms")
    print(f"  Min IBI:  {np.min(post_ibis):.2f}ms (BPM: {60000/np.min(post_ibis):.3f})")
    print(f"  Max IBI:  {np.max(post_ibis):.2f}ms (BPM: {60000/np.max(post_ibis):.3f})")

    bpm_diff = 60000/np.mean(post_ibis) - 60000/np.mean(pre_ibis)
    print(f"\nPost vs Pre: {bpm_diff:+.3f} BPM")

    # Cumulative drift (using ALL beats from method 1, not just valid)
    perfect = beat_times1[0] + np.arange(len(beat_times1)) * GLOBAL_IBI
    drift = (beat_times1 - perfect) * 1000

    # Focus on near-drop region
    print(f"\n=== Beats Near Drop (305-320s) ===")
    near_mask = (beat_times1 >= 305) & (beat_times1 <= 320)
    near_beats = beat_times1[near_mask]
    near_idx = np.where(near_mask)[0]

    for i, idx in enumerate(near_idx):
        t = beat_times1[idx]
        d = drift[idx]
        ibi = ibis1[idx-1] if idx > 0 and idx-1 < len(ibis1) else 0
        marker = " <<< DROP" if abs(t - DROP_TIME) < 1 else ""
        print(f"  Beat {idx+1:3d}: {t:.3f}s | IBI={ibi:.1f}ms | Drift={d:+.1f}ms{marker}")

    # Post-drop drift rate with high-res data
    post_idx = beat_times1 >= DROP_TIME
    post_drift = drift[post_idx]
    post_bt = beat_times1[post_idx]
    if len(post_bt) > 3:
        z = np.polyfit(post_bt - post_bt[0], post_drift, 1)
        print(f"\n=== Post-Drop Drift Rate ===")
        print(f"  Drift rate: {z[0]:+.3f} ms/s")
        print(f"  Over 18s: {z[0]*18:+.1f} ms")
        local_ibi = GLOBAL_IBI * 1000 + z[0]  # approximate
        print(f"  Implied local IBI: {local_ibi:.2f}ms")
        print(f"  Implied local BPM: {60000/local_ibi:.3f}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'Precise Local Tempo Analysis (hop={HOP}, res={HOP/SR*1000:.1f}ms)', fontsize=14, fontweight='bold')

    # Panel 1: Beat-by-beat local BPM
    ax1 = axes[0]
    local_bpms = 60000.0 / ibis1
    ax1.plot(beat_times1[:-1], local_bpms, 'o-', color='blue', markersize=3)
    ax1.axhline(GLOBAL_BPM, color='red', linewidth=2, linestyle='--', label=f'Global ({GLOBAL_BPM})')
    ax1.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label='Drop')
    ax1.set_ylabel('Local BPM')
    ax1.set_title('Beat-by-Beat Local BPM')
    ax1.legend()
    ax1.set_ylim(GLOBAL_BPM - 3, GLOBAL_BPM + 3)

    # Panel 2: Beat-by-beat IBI
    ax2 = axes[1]
    ax2.plot(beat_times1[:-1], ibis1, 'o-', color='purple', markersize=3)
    ax2.axhline(GLOBAL_IBI*1000, color='red', linewidth=2, linestyle='--', label=f'Global IBI ({GLOBAL_IBI*1000:.1f}ms)')
    ax2.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label='Drop')
    ax2.set_ylabel('IBI (ms)')
    ax2.set_title('Beat-by-Beat Inter-Beat Interval')
    ax2.legend()

    # Panel 3: Cumulative drift
    ax3 = axes[2]
    ax3.plot(beat_times1, drift, 'o-', color='red', markersize=3)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.axvline(DROP_TIME, color='green', linewidth=2, linestyle=':', label='Drop')
    # Fit line to post-drop
    if len(post_bt) > 3:
        fit_line = np.poly1d(z)
        ax3.plot(post_bt, fit_line(post_bt - post_bt[0]), '--', color='orange', linewidth=2,
                 label=f'Post-drop trend ({z[0]:+.2f} ms/s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Drift (ms)')
    ax3.set_title('Cumulative Drift from Global 87.593 BPM Grid')
    ax3.legend()

    plt.tight_layout()
    out_path = f"{OUTPUT}/precise-tempo-analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
