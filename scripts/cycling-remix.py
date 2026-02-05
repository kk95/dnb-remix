#!/usr/bin/env python3
"""
Cycling Transition Remix: Saathiya x Kho Gayi
Replicates the djay "transition magic" across the full song.

Pattern (from djay screenshot):
  - Volume:  KG beat swells during flute sections, recedes during vocals
  - EQ:      "Center bass swap" — inherent from stem separation
  - Filter:  KG enters muffled (lowpass), opens up during flute sections

Usage:
  source ../.venv/bin/activate
  python cycling-remix.py [--phase N] [--preview-only]
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import soundfile as sf
import librosa
from scipy import signal as sig

# ── Config ──────────────────────────────────────────────────────────────

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output"
SR = 44100

SAATHIYA_BPM = 88.0
KG_BPM = 95.0

# Flute-prominent sections in Saathiya (seconds)
FLUTE_SECTIONS = [
    (22, 42),     # 0:22 - 0:42  (20s)
    (62, 81),     # 1:02 - 1:21  (20s)
    (198, 206),   # 3:18 - 3:26  (8s)
    (280, 299),   # 4:40 - 4:59  (19s)
    (307, 321),   # 5:07 - 5:21  (14s)
]

# Transition parameters (inspired by djay 8-bar overlap)
TRANSITION_BARS = 4        # bars to ramp in/out
KG_VERSE_LEVEL = 0.20      # KG volume during vocal sections (subtle presence)
KG_FLUTE_LEVEL = 0.95      # KG volume during flute sections (full energy)
SAATHIYA_VOCAL_VOL = 0.85  # Saathiya vocals level
SAATHIYA_OTHER_VOL = 0.70  # Saathiya other (flute/melody) level
KG_DRUMS_VOL = 0.90        # KG drums level (before envelope)
KG_BASS_VOL = 0.75         # KG bass level (before envelope)

# Filter parameters for KG
FILTER_CLOSED_HZ = 350     # Muffled (verse)
FILTER_OPEN_HZ = 10000     # Open (flute drop)


# ── Helpers ─────────────────────────────────────────────────────────────

def log(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def load_stereo(path):
    """Load WAV as stereo float64."""
    audio, sr = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def load_mono(path):
    """Load WAV as mono float64."""
    audio, sr = sf.read(path, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def write_wav(path, audio, sr=SR):
    """Write audio to WAV (16-bit)."""
    sf.write(path, audio, sr, subtype='PCM_16')
    print(f"  → {path} ({len(audio)/sr:.1f}s)")


def to_mp3(wav_path):
    """Convert WAV to 320kbps MP3."""
    mp3_path = wav_path.replace('.wav', '.mp3')
    subprocess.run([
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame', '-b:a', '320k', mp3_path
    ], capture_output=True, check=True)
    print(f"  → {mp3_path}")
    return mp3_path


def rubberband_stretch(input_path, output_path, ratio, crisp=5):
    """Time-stretch using rubberband."""
    subprocess.run([
        'rubberband', '--tempo', str(ratio), '--crisp', str(crisp),
        input_path, output_path
    ], capture_output=True, check=True)


def pad_or_trim(audio, target_length):
    """Pad with silence or trim to exact length."""
    if len(audio) < target_length:
        if audio.ndim == 2:
            pad = np.zeros((target_length - len(audio), audio.shape[1]))
        else:
            pad = np.zeros(target_length - len(audio))
        return np.concatenate([audio, pad])
    return audio[:target_length]


def loop_to_length(audio, target_length):
    """Loop audio to fill target_length samples."""
    if len(audio) >= target_length:
        return audio[:target_length]
    repeats = int(np.ceil(target_length / len(audio)))
    looped = np.tile(audio, (repeats, 1) if audio.ndim == 2 else repeats)
    return looped[:target_length]


def apply_lowpass(audio, cutoff_hz, sr=SR, order=4):
    """Apply Butterworth lowpass filter."""
    nyq = sr / 2
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    if norm_cutoff < 0.01:
        return np.zeros_like(audio)
    b, a = sig.butter(order, norm_cutoff, btype='low')
    if audio.ndim == 2:
        return np.column_stack([
            sig.filtfilt(b, a, audio[:, 0]),
            sig.filtfilt(b, a, audio[:, 1])
        ])
    return sig.filtfilt(b, a, audio)


def apply_highpass(audio, cutoff_hz, sr=SR, order=4):
    """Apply Butterworth highpass filter."""
    nyq = sr / 2
    norm_cutoff = max(cutoff_hz / nyq, 0.01)
    b, a = sig.butter(order, norm_cutoff, btype='high')
    if audio.ndim == 2:
        return np.column_stack([
            sig.filtfilt(b, a, audio[:, 0]),
            sig.filtfilt(b, a, audio[:, 1])
        ])
    return sig.filtfilt(b, a, audio)


# ── Phase 1: Prepare Stems ─────────────────────────────────────────────

def phase1_prepare():
    """Stretch Kho Gayi stems to 88 BPM."""
    log("Phase 1: Preparing stems (KG → 88 BPM)")

    ratio = SAATHIYA_BPM / KG_BPM  # 88/95 = 0.9263 (slow down)
    stretched_dir = f"{PROJECT}/stretched"
    os.makedirs(stretched_dir, exist_ok=True)

    stems_to_stretch = {
        'drums': 3,   # crisp=3 for percussion
        'bass': 5,    # crisp=5 for tonal
        'other': 5,
    }

    for stem, crisp in stems_to_stretch.items():
        input_path = f"{STEMS}/kho-gayi/{stem}.wav"
        output_path = f"{stretched_dir}/kg-{stem}-88bpm.wav"

        if os.path.exists(output_path):
            print(f"  [skip] {output_path} already exists")
            continue

        print(f"  Stretching kho-gayi/{stem}.wav → 88 BPM (ratio={ratio:.4f}, crisp={crisp})")
        rubberband_stretch(input_path, output_path, ratio, crisp)
        print(f"  → {output_path}")

    print("\n  Phase 1 complete.")
    return stretched_dir


# ── Phase 2: Beat Alignment ────────────────────────────────────────────

def phase2_align(stretched_dir):
    """Find optimal beat offset using cross-correlation."""
    log("Phase 2: Beat alignment via cross-correlation")

    # Load both drum tracks as mono
    saathiya_drums = load_mono(f"{STEMS}/saathiya/drums.wav")
    kg_drums = load_mono(f"{stretched_dir}/kg-drums-88bpm.wav")

    print(f"  Saathiya drums: {len(saathiya_drums)/SR:.1f}s")
    print(f"  KG drums (88bpm): {len(kg_drums)/SR:.1f}s")

    # Compute onset strength envelopes
    hop = 512
    print("  Computing onset envelopes...")
    env_s = librosa.onset.onset_strength(y=saathiya_drums, sr=SR, hop_length=hop)
    env_k = librosa.onset.onset_strength(y=kg_drums, sr=SR, hop_length=hop)

    # Cross-correlate: find where KG best aligns with Saathiya
    # We only need to search within one bar (2.727s at 88 BPM)
    beat_period_s = 60.0 / SAATHIYA_BPM
    bar_s = 4 * beat_period_s
    search_frames = int(2 * bar_s * SR / hop)  # search ±2 bars

    print(f"  Cross-correlating (search range: ±{2*bar_s:.1f}s)...")
    correlation = np.correlate(env_s[:len(env_k)], env_k, mode='full')

    center = len(env_k) - 1
    search_start = max(0, center - search_frames)
    search_end = min(len(correlation), center + search_frames)

    best_idx = search_start + np.argmax(correlation[search_start:search_end])
    offset_frames = best_idx - center
    offset_samples = offset_frames * hop
    offset_seconds = offset_samples / SR

    print(f"\n  Best alignment offset: {offset_seconds:+.3f}s ({offset_samples:+d} samples)")
    print(f"  (positive = KG starts later, negative = KG starts earlier)")

    # Quantize to nearest beat for cleaner alignment
    beat_samples = int(beat_period_s * SR)
    quantized_offset = round(offset_samples / beat_samples) * beat_samples
    quantized_seconds = quantized_offset / SR

    print(f"  Quantized to beat grid: {quantized_seconds:+.3f}s ({quantized_offset:+d} samples)")

    return quantized_offset


# ── Phase 3: Build Cycling Remix ────────────────────────────────────────

def phase3_remix(stretched_dir, beat_offset):
    """Create the cycling transition remix."""
    log("Phase 3: Building cycling transition remix")

    # Load Saathiya stems
    print("  Loading Saathiya vocals...")
    s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
    print("  Loading Saathiya other (flute/melody)...")
    s_other = load_stereo(f"{STEMS}/saathiya/other.wav")

    # Load stretched KG stems
    print("  Loading KG drums (88bpm)...")
    kg_drums = load_stereo(f"{stretched_dir}/kg-drums-88bpm.wav")
    print("  Loading KG bass (88bpm)...")
    kg_bass = load_stereo(f"{stretched_dir}/kg-bass-88bpm.wav")

    # Target length = Saathiya length
    target_len = len(s_vocals)
    print(f"\n  Target duration: {target_len/SR:.1f}s ({target_len} samples)")

    # Apply beat offset to KG (shift or trim)
    if beat_offset > 0:
        # KG starts later: prepend silence
        silence = np.zeros((beat_offset, 2))
        kg_drums = np.concatenate([silence, kg_drums])
        kg_bass = np.concatenate([silence, kg_bass])
    elif beat_offset < 0:
        # KG starts earlier: trim the beginning
        trim = abs(beat_offset)
        kg_drums = kg_drums[trim:]
        kg_bass = kg_bass[trim:]

    # Loop KG to match Saathiya length
    print("  Looping KG stems to match Saathiya length...")
    kg_drums = loop_to_length(kg_drums, target_len)
    kg_bass = loop_to_length(kg_bass, target_len)

    # ── Create envelopes ──

    print("  Creating volume & filter envelopes...")
    beat_period_s = 60.0 / SAATHIYA_BPM
    bar_s = 4 * beat_period_s
    transition_s = TRANSITION_BARS * bar_s
    transition_samples = int(transition_s * SR)

    # Volume envelope for KG (swells during flute, recedes during vocals)
    kg_envelope = np.full(target_len, KG_VERSE_LEVEL)

    for start_s, end_s in FLUTE_SECTIONS:
        start = int(start_s * SR)
        end = int(end_s * SR)

        # Ramp up (transition in)
        ramp_start = max(0, start - transition_samples)
        ramp_len = start - ramp_start
        if ramp_len > 0:
            ramp = np.linspace(KG_VERSE_LEVEL, KG_FLUTE_LEVEL, ramp_len)
            kg_envelope[ramp_start:start] = ramp

        # Full level during flute
        kg_envelope[start:min(end, target_len)] = KG_FLUTE_LEVEL

        # Ramp down (transition out)
        ramp_end = min(target_len, end + transition_samples)
        ramp_len = ramp_end - end
        if ramp_len > 0 and end < target_len:
            ramp = np.linspace(KG_FLUTE_LEVEL, KG_VERSE_LEVEL, ramp_len)
            kg_envelope[end:ramp_end] = ramp

    # Filter envelope: cutoff frequency over time
    filter_envelope = np.full(target_len, float(FILTER_CLOSED_HZ))

    for start_s, end_s in FLUTE_SECTIONS:
        start = int(start_s * SR)
        end = int(end_s * SR)

        ramp_start = max(0, start - transition_samples)
        ramp_len = start - ramp_start
        if ramp_len > 0:
            ramp = np.linspace(FILTER_CLOSED_HZ, FILTER_OPEN_HZ, ramp_len)
            filter_envelope[ramp_start:start] = ramp

        filter_envelope[start:min(end, target_len)] = FILTER_OPEN_HZ

        ramp_end = min(target_len, end + transition_samples)
        ramp_len = ramp_end - end
        if ramp_len > 0 and end < target_len:
            ramp = np.linspace(FILTER_OPEN_HZ, FILTER_CLOSED_HZ, ramp_len)
            filter_envelope[end:ramp_end] = ramp

    # ── Apply filter to KG ──
    # Pre-render filtered and unfiltered, then crossfade based on envelope

    print("  Applying low-pass filter automation to KG...")
    kg_combined = kg_drums * KG_DRUMS_VOL + kg_bass * KG_BASS_VOL

    # Create filtered version (muffled)
    kg_filtered = apply_lowpass(kg_combined, FILTER_CLOSED_HZ)

    # Create blend factor from filter envelope (0 = filtered, 1 = open)
    blend = (filter_envelope - FILTER_CLOSED_HZ) / (FILTER_OPEN_HZ - FILTER_CLOSED_HZ)
    blend = np.clip(blend, 0, 1)

    # Also create a mid-filtered version for smoother transitions
    kg_mid_filtered = apply_lowpass(kg_combined, 1500)

    # Three-way blend: closed → mid → open
    kg_with_filter = np.zeros_like(kg_combined)
    for ch in range(2):
        # Below 0.5 blend: mix between closed and mid
        # Above 0.5 blend: mix between mid and open
        low_mask = blend <= 0.5
        high_mask = blend > 0.5

        t_low = blend * 2  # 0→1 in the first half
        t_high = (blend - 0.5) * 2  # 0→1 in the second half

        kg_with_filter[low_mask, ch] = (
            kg_filtered[low_mask, ch] * (1 - t_low[low_mask]) +
            kg_mid_filtered[low_mask, ch] * t_low[low_mask]
        )
        kg_with_filter[high_mask, ch] = (
            kg_mid_filtered[high_mask, ch] * (1 - t_high[high_mask]) +
            kg_combined[high_mask, ch] * t_high[high_mask]
        )

    # ── Apply volume envelope to KG ──
    print("  Applying volume envelope...")
    kg_final = kg_with_filter * kg_envelope[:, np.newaxis]

    # ── Process Saathiya stems ──
    print("  Highpassing Saathiya stems (remove bass below 120Hz)...")
    s_vocals_hp = apply_highpass(s_vocals, 80)   # gentle highpass on vocals
    s_other_hp = apply_highpass(s_other, 120)     # stronger on other (remove bass rumble)

    # ── Final mix ──
    print("  Mixing stems...")
    mix = (
        s_vocals_hp * SAATHIYA_VOCAL_VOL +
        s_other_hp * SAATHIYA_OTHER_VOL +
        kg_final
    )

    # Soft-clip to prevent harsh clipping
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        print(f"  Normalizing (peak was {peak:.2f})")
        mix = mix * (0.95 / peak)

    # Write output
    os.makedirs(OUTPUT, exist_ok=True)
    output_path = f"{OUTPUT}/cycling-remix-88bpm.wav"
    write_wav(output_path, mix)

    return output_path, mix, kg_envelope


# ── Phase 4: Preview Clips ─────────────────────────────────────────────

def phase4_previews(remix_path, mix_audio):
    """Generate preview clips at each flute section."""
    log("Phase 4: Generating preview clips")

    os.makedirs(OUTPUT, exist_ok=True)

    # Preview: 10s before flute → through flute → 5s after
    for i, (start_s, end_s) in enumerate(FLUTE_SECTIONS):
        preview_start = max(0, start_s - 10)
        preview_end = min(len(mix_audio)/SR, end_s + 5)
        duration = preview_end - preview_start

        start_sample = int(preview_start * SR)
        end_sample = int(preview_end * SR)
        clip = mix_audio[start_sample:end_sample]

        label = f"{int(start_s//60)}m{int(start_s%60):02d}"
        wav_path = f"{OUTPUT}/preview-cycling-flute{i+1}-{label}.wav"
        write_wav(wav_path, clip)
        to_mp3(wav_path)

    # Also generate a full first-minute preview
    one_min = mix_audio[:int(60 * SR)]
    wav_path = f"{OUTPUT}/preview-cycling-first60s.wav"
    write_wav(wav_path, one_min)
    to_mp3(wav_path)

    print("\n  All previews generated. Listen with: open output/preview-cycling-*.mp3")


# ── Phase 5: 170 BPM DnB Version ───────────────────────────────────────

def phase5_dnb(remix_path):
    """Speed up the remix to 170 BPM."""
    log("Phase 5: Creating 170 BPM DnB version")

    ratio = 170.0 / SAATHIYA_BPM  # 1.932x
    dnb_path = f"{OUTPUT}/cycling-remix-170bpm-dnb.wav"

    print(f"  Stretching {ratio:.3f}x (88 → 170 BPM)...")
    rubberband_stretch(remix_path, dnb_path, ratio, crisp=5)

    dnb_audio = load_stereo(dnb_path)
    print(f"  DnB version: {len(dnb_audio)/SR:.1f}s")

    to_mp3(dnb_path)
    print("\n  DnB version ready: open output/cycling-remix-170bpm-dnb.mp3")

    return dnb_path


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cycling Transition Remix")
    parser.add_argument('--phase', type=int, help='Run specific phase (1-5)')
    parser.add_argument('--preview-only', action='store_true',
                        help='Only generate preview clips (phases 1-4, skip DnB)')
    parser.add_argument('--offset', type=float, default=None,
                        help='Manual beat offset in seconds (skip cross-correlation)')
    args = parser.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    if args.phase and args.phase == 1:
        phase1_prepare()
        return

    if args.phase and args.phase == 2:
        stretched_dir = f"{PROJECT}/stretched"
        phase2_align(stretched_dir)
        return

    # Full pipeline
    log("CYCLING TRANSITION REMIX: Saathiya x Kho Gayi")
    print("  Replicating djay transition pattern across the full song")
    print(f"  Flute sections: {len(FLUTE_SECTIONS)} detected")
    print(f"  Transition ramp: {TRANSITION_BARS} bars")

    # Phase 1
    stretched_dir = phase1_prepare()

    # Phase 2
    if args.offset is not None:
        beat_offset = int(args.offset * SR)
        print(f"\n  Using manual offset: {args.offset:+.3f}s ({beat_offset:+d} samples)")
    else:
        beat_offset = phase2_align(stretched_dir)

    # Phase 3
    remix_path, mix_audio, envelope = phase3_remix(stretched_dir, beat_offset)

    # Phase 4
    phase4_previews(remix_path, mix_audio)

    # Phase 5
    if not args.preview_only:
        phase5_dnb(remix_path)

    log("ALL DONE!")
    print("  Listen to previews:")
    print("    open output/preview-cycling-*.mp3")
    print("  Full remix:")
    print("    open output/cycling-remix-88bpm.mp3")
    if not args.preview_only:
        print("  DnB version:")
        print("    open output/cycling-remix-170bpm-dnb.mp3")


if __name__ == '__main__':
    main()
