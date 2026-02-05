#!/usr/bin/env python3
"""Find high-pitched beeping/synth in KG — focused on buildup region (20-40s)."""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, sosfilt

PROJECT = "/Users/kshitijkarke/Documents/dnb-remix"
STEMS = f"{PROJECT}/stems/htdemucs_ft/kho-gayi"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# Only load 0-70s of each stem (the interesting part)
LOAD_S = 70
LOAD_SAMPLES = int(LOAD_S * SR)

print("Loading KG stems (first 70s)...")
stems = {}
for name in ['other', 'vocals', 'drums', 'bass']:
    audio, _ = sf.read(f"{STEMS}/{name}.wav", dtype='float64', stop=LOAD_SAMPLES)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # mono for analysis
    stems[name] = audio
    print(f"  {name}: {len(audio)/SR:.1f}s loaded")

# Spectrogram for "other" stem (most likely to have synth beeps)
print("\nComputing spectrogram for 'other' stem...")
fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Full spectrogram of "other"
f, t, Sxx = spectrogram(stems['other'], SR, nperseg=2048, noverlap=1536)
Sxx_db = 10 * np.log10(Sxx + 1e-10)
axes[0].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', vmin=-60, vmax=0)
axes[0].set_ylabel('Freq (Hz)')
axes[0].set_ylim(0, 12000)
axes[0].axhline(y=2000, color='cyan', alpha=0.4, linestyle='--')
axes[0].axhline(y=4000, color='cyan', alpha=0.4, linestyle='--')
axes[0].axvline(x=23, color='lime', alpha=0.5, label='drums build')
axes[0].axvline(x=33.245, color='red', alpha=0.7, label='DROP')
axes[0].legend(loc='upper right')
axes[0].set_title('KG "other" stem — Spectrogram (0-70s)')

# Spectrogram of "vocals" stem
f_v, t_v, Sxx_v = spectrogram(stems['vocals'], SR, nperseg=2048, noverlap=1536)
Sxx_v_db = 10 * np.log10(Sxx_v + 1e-10)
axes[1].pcolormesh(t_v, f_v, Sxx_v_db, shading='gouraud', cmap='magma', vmin=-60, vmax=0)
axes[1].set_ylabel('Freq (Hz)')
axes[1].set_ylim(0, 12000)
axes[1].axvline(x=23, color='lime', alpha=0.5)
axes[1].axvline(x=33.245, color='red', alpha=0.7)
axes[1].set_title('KG "vocals" stem — Spectrogram (0-70s)')

# Spectrogram of "drums" stem (hi-hats can sound beepy)
f_d, t_d, Sxx_d = spectrogram(stems['drums'], SR, nperseg=2048, noverlap=1536)
Sxx_d_db = 10 * np.log10(Sxx_d + 1e-10)
axes[2].pcolormesh(t_d, f_d, Sxx_d_db, shading='gouraud', cmap='magma', vmin=-60, vmax=0)
axes[2].set_ylabel('Freq (Hz)')
axes[2].set_ylim(0, 12000)
axes[2].axvline(x=23, color='lime', alpha=0.5)
axes[2].axvline(x=33.245, color='red', alpha=0.7)
axes[2].set_title('KG "drums" stem — Spectrogram (0-70s)')

for ax in axes:
    ax.set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-beep-spectrograms.png", dpi=150)
print(f"  Saved: kg-beep-spectrograms.png")

# High-freq energy analysis — look for "beeping" patterns
print("\n--- High-freq energy in each stem (>2kHz) ---")
sos_hi = butter(4, 2000 / (SR / 2), btype='high', output='sos')

fig2, axes2 = plt.subplots(4, 1, figsize=(18, 8), sharex=True)

for ax, (name, audio) in zip(axes2, stems.items()):
    hi = sosfilt(sos_hi, audio)

    # RMS in 50ms windows
    win = int(0.05 * SR)
    hop = int(0.025 * SR)
    rms_times = np.arange(0, len(hi) - win, hop) / SR
    rms_vals = np.array([np.sqrt(np.mean(hi[i:i+win]**2))
                         for i in range(0, len(hi) - win, hop)])
    rms_db = 20 * np.log10(rms_vals + 1e-10)

    ax.plot(rms_times, rms_db, linewidth=0.5)
    ax.set_ylabel(f'{name}\ndB')
    ax.set_ylim(-70, 0)
    ax.axvline(x=23, color='lime', alpha=0.5)
    ax.axvline(x=33.245, color='red', alpha=0.7)
    ax.grid(True, alpha=0.3)

    # Find peaks
    strong = rms_times[rms_db > -30]
    if len(strong) > 0:
        print(f"  {name}: high-freq (>2kHz) active at {strong[0]:.1f}s - {strong[-1]:.1f}s")
    else:
        print(f"  {name}: no strong high-freq content")

axes2[0].set_title('KG Stems — High-Frequency Energy (>2kHz)')
axes2[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/kg-beep-energy.png", dpi=150)
print(f"  Saved: kg-beep-energy.png")

# Extract short clips of the high-freq content for listening
# Focus on buildup region (20-35s) in "other" and "vocals"
print("\n--- Extracting high-freq clips for listening ---")
for name in ['other', 'vocals', 'drums']:
    audio_stereo, _ = sf.read(f"{STEMS}/{name}.wav", dtype='float64',
                               start=int(20*SR), stop=int(40*SR))
    if audio_stereo.ndim == 1:
        audio_stereo = np.column_stack([audio_stereo, audio_stereo])

    # Highpass filter at 2kHz
    sos_hi_stereo = butter(4, 2000 / (SR / 2), btype='high', output='sos')
    hi_stereo = np.column_stack([
        sosfilt(sos_hi_stereo, audio_stereo[:, 0]),
        sosfilt(sos_hi_stereo, audio_stereo[:, 1])
    ])

    # Normalize
    peak = np.max(np.abs(hi_stereo))
    if peak > 0.01:
        hi_stereo = hi_stereo * (0.8 / peak)
        outpath = f"{OUTPUT}/kg-{name}-highfreq-20to40s.wav"
        sf.write(outpath, hi_stereo, SR, subtype='PCM_16')
        print(f"  Saved: kg-{name}-highfreq-20to40s.wav (peak={peak:.4f})")
    else:
        print(f"  {name}: too quiet above 2kHz, skipping")

# Also extract the raw "other" stem 20-40s for reference
audio_raw, _ = sf.read(f"{STEMS}/other.wav", dtype='float64',
                        start=int(20*SR), stop=int(40*SR))
sf.write(f"{OUTPUT}/kg-other-raw-20to40s.wav", audio_raw, SR, subtype='PCM_16')
print(f"  Saved: kg-other-raw-20to40s.wav")

print("\nDone! Check the spectrograms and listen to the extracted clips.")
