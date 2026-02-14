#!/usr/bin/env python3
"""Detailed analysis of KG buildup region (20-40s) — find the beeping."""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, sosfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft/kho-gayi"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# Load just the buildup region from each stem
START_S = 15
END_S = 45
start_samp = int(START_S * SR)
end_samp = int(END_S * SR)

print(f"Loading KG stems ({START_S}-{END_S}s)...")
stems_stereo = {}
stems_mono = {}
for name in ['other', 'vocals', 'drums', 'bass']:
    audio, _ = sf.read(f"{STEMS}/{name}.wav", dtype='float64',
                        start=start_samp, stop=end_samp)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    stems_stereo[name] = audio
    stems_mono[name] = audio.mean(axis=1)
    print(f"  {name}: {len(audio)/SR:.1f}s")

# Reconstruct full mix for reference
full_mix = sum(stems_stereo.values())

# Detailed spectrogram with high resolution
fig, axes = plt.subplots(5, 1, figsize=(20, 20))

for ax, (name, audio) in zip(axes[:4], stems_mono.items()):
    f, t, Sxx = spectrogram(audio, SR, nperseg=4096, noverlap=3840)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    ax.pcolormesh(t + START_S, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=-70, vmax=-10)
    ax.set_ylabel(f'{name}\nFreq (Hz)')
    ax.set_ylim(500, 8000)
    ax.axvline(x=23, color='lime', alpha=0.7, linewidth=1, label='drums start')
    ax.axvline(x=32, color='yellow', alpha=0.7, linewidth=1, label='silence')
    ax.axvline(x=33.245, color='red', alpha=0.9, linewidth=2, label='DROP')
    ax.legend(loc='upper right', fontsize=7)

# Full mix
mix_mono = full_mix.mean(axis=1)
f, t, Sxx = spectrogram(mix_mono, SR, nperseg=4096, noverlap=3840)
Sxx_db = 10 * np.log10(Sxx + 1e-10)
axes[4].pcolormesh(t + START_S, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=-70, vmax=-10)
axes[4].set_ylabel('FULL MIX\nFreq (Hz)')
axes[4].set_ylim(500, 8000)
axes[4].axvline(x=23, color='lime', alpha=0.7, linewidth=1)
axes[4].axvline(x=32, color='yellow', alpha=0.7, linewidth=1)
axes[4].axvline(x=33.245, color='red', alpha=0.9, linewidth=2)

axes[0].set_title('KG Stems — Buildup Region Spectrogram (500-8000 Hz)')
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-beep-detail.png", dpi=150)
print(f"  Saved: kg-beep-detail.png")

# Extract frequency-isolated clips for listening
print("\n--- Extracting band-isolated clips (buildup region) ---")
bands = {
    'low-mid-500-1500': (500, 1500),
    'mid-1500-3000': (1500, 3000),
    'hi-mid-3000-6000': (3000, 6000),
    'high-6000-12000': (6000, 12000),
}

for band_name, (lo, hi) in bands.items():
    sos_bp = butter(4, [lo / (SR/2), hi / (SR/2)], btype='band', output='sos')
    for stem_name in ['other', 'vocals', 'drums']:
        audio = stems_stereo[stem_name]
        filtered = np.column_stack([
            sosfilt(sos_bp, audio[:, 0]),
            sosfilt(sos_bp, audio[:, 1])
        ])
        peak = np.max(np.abs(filtered))
        if peak > 0.005:
            filtered_norm = filtered * (0.7 / peak)
            path = f"{OUTPUT}/kg-{stem_name}-{band_name}.wav"
            sf.write(path, filtered_norm, SR, subtype='PCM_16')
            print(f"  {stem_name} [{band_name}]: peak={peak:.4f} → saved")
        else:
            print(f"  {stem_name} [{band_name}]: too quiet (peak={peak:.6f})")

# Also save raw clips of each stem in the buildup region
print("\n--- Raw stem clips (buildup region) ---")
for name, audio in stems_stereo.items():
    path = f"{OUTPUT}/kg-{name}-buildup-{START_S}to{END_S}s.wav"
    sf.write(path, audio, SR, subtype='PCM_16')
    print(f"  Saved: {path}")

# Save full mix buildup
sf.write(f"{OUTPUT}/kg-fullmix-buildup-{START_S}to{END_S}s.wav",
         full_mix * 0.7, SR, subtype='PCM_16')
print(f"  Saved: kg-fullmix-buildup-{START_S}to{END_S}s.wav")

print("\nDone! Listen to the band-isolated clips to find the beeping.")
print("Most likely candidates:")
print("  - vocals mid (1500-3000 Hz) or hi-mid (3000-6000 Hz)")
print("  - other mid or hi-mid")
print("  - drums hi-mid (hi-hat pattern that builds)")
