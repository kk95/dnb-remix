#!/usr/bin/env python3
"""
DnB Remix Tuner — interactive Gradio UI.

Three control sections:
  1. Flute Loop — source position, circular crossfade, volume
  2. KG Drums  — "4-bar Loop" (tight, 5 restarts) or "Full Section" (no restarts)
  3. Ding Build-In — ramp envelope controls

Pre-computes all stems once at startup, then each Generate
is pure numpy (<1s).
"""

import os
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import gradio as gr
from scipy.signal import butter, sosfilt, filtfilt

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUTPUT = f"{PROJECT}/output/constant-stretch"
SR = 44100

# ── BPM / timing constants ──────────────────────────────────────────
KG_BPM = 95.703
TARGET_BPM = 88.0
STRETCH_RATE = TARGET_BPM / KG_BPM
KG_DROP_S = 33.245
KG_DROP_STRETCHED_S = KG_DROP_S / STRETCH_RATE
BEAT_GRID_S = 306.772
SNAP_S = 307.454
BEAT_S = 60.0 / TARGET_BPM
BAR_S = 4 * BEAT_S
SAATHIYA_LOCAL_BPM = 87.939
SAATHIYA_BAR_S = 4 * (60.0 / SAATHIYA_LOCAL_BPM)
LOOP_BARS = 4
N_FLUTE_LOOPS = 6
EDGE_FADE_MS = 3

target_loop_samples = int(LOOP_BARS * BAR_S * SR)
edge_fade = int(EDGE_FADE_MS / 1000 * SR)

# Flute source candidates (seconds into Saathiya "other" stem)
FLUTE_SOURCES = {
    "290.4s (Best boundary)": 290.4,
    "306.8s (Original / drop)": 306.772,
    "72.2s (Strongest boundary)": 72.2,
    "61.3s (Alternate)": 61.3,
}


def load_stereo(path):
    audio, _ = sf.read(path, dtype='float64')
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def extract(src, start, length):
    out = np.zeros((length, 2))
    src_s = max(0, start)
    src_e = min(len(src), start + length)
    dst_s = max(0, -start)
    dst_e = dst_s + (src_e - src_s)
    if src_e > src_s and dst_e <= length:
        out[dst_s:dst_e] = src[src_s:src_e]
    return out


def distort_bass(audio, drive=6, sub_cutoff=80):
    sos_lo = butter(4, sub_cutoff / (SR / 2), btype='low', output='sos')
    sos_hi = butter(4, sub_cutoff / (SR / 2), btype='high', output='sos')
    result = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        sub = sosfilt(sos_lo, audio[:, ch])
        mids = sosfilt(sos_hi, audio[:, ch])
        mids_dist = np.tanh(drive * mids) / np.tanh(drive)
        result[:, ch] = sub + mids_dist
    return result


def make_circular_loop(one_loop, xf_samples):
    """Prepare a loop for seamless tiling via circular crossfade.

    Blends the last xf_samples INTO the first xf_samples using equal-power
    curves. The end is set to match the start so np.tile() is seamless.
    """
    if xf_samples <= 0 or xf_samples > len(one_loop) // 2:
        return one_loop.copy()

    result = one_loop.copy()
    head = one_loop[:xf_samples].copy()
    tail = one_loop[-xf_samples:].copy()

    t = np.linspace(0, 1, xf_samples)[:, np.newaxis]
    fi = np.sqrt(t)       # 0 → 1
    fo = np.sqrt(1 - t)   # 1 → 0

    blend = head * fi + tail * fo
    result[:xf_samples] = blend
    result[-xf_samples:] = blend
    return result


# ── Pre-compute (runs once at startup) ──────────────────────────────

print("Loading and processing stems (one-time)...")

# Load raw stems
s_vocals = load_stereo(f"{STEMS}/saathiya/vocals.wav")
s_other = load_stereo(f"{STEMS}/saathiya/other.wav")
s_drums_st = load_stereo(f"{STEMS}/saathiya/drums.wav")
s_bass_st = load_stereo(f"{STEMS}/saathiya/bass.wav")
kg_drums_raw = load_stereo(f"{STEMS}/kho-gayi/drums.wav")
kg_bass_raw = load_stereo(f"{STEMS}/kho-gayi/bass.wav")
kg_other_raw = load_stereo(f"{STEMS}/kho-gayi/other.wav")
print("  Stems loaded.")

# ── KG loops (clean, no modifications yet) ──────────────────────────

kg_loop_start_s = KG_DROP_S
kg_loop_len_s = LOOP_BARS * (4 * 60.0 / KG_BPM)
kg_loop_start = int(kg_loop_start_s * SR)
kg_loop_len = int(kg_loop_len_s * SR)

kg_drums_4bar = kg_drums_raw[kg_loop_start:kg_loop_start + kg_loop_len]
kg_bass_4bar = kg_bass_raw[kg_loop_start:kg_loop_start + kg_loop_len]

print("  Stretching KG drums...")
kg_drums_clean = pyrb.time_stretch(kg_drums_4bar, SR, STRETCH_RATE)[:target_loop_samples]
print("  Stretching KG bass...")
kg_bass_loop = pyrb.time_stretch(kg_bass_4bar, SR, STRETCH_RATE)[:target_loop_samples]
kg_bass_loop = distort_bass(kg_bass_loop, drive=6)

# KG ding
DING_OFFSET_SAMPLES = int(0.006 * SR)
kg_other_4bar = kg_other_raw[kg_loop_start - DING_OFFSET_SAMPLES:
                              kg_loop_start - DING_OFFSET_SAMPLES + kg_loop_len]
sos_ding = butter(4, 1500 / (SR / 2), btype='high', output='sos')
kg_ding_4bar = np.column_stack([
    sosfilt(sos_ding, kg_other_4bar[:, 0]),
    sosfilt(sos_ding, kg_other_4bar[:, 1])
])
print("  Stretching KG ding...")
kg_ding_loop = pyrb.time_stretch(kg_ding_4bar, SR, STRETCH_RATE)[:target_loop_samples]

# Pre-compute kick and riser (used conditionally in callback)
KICK_LEN_MS = 80
kick_len = int(KICK_LEN_MS / 1000 * SR)
kick = kg_drums_clean[:kick_len].copy()
kick_rev = kick[::-1].copy()
kick_rev *= np.linspace(0, 1, len(kick_rev))[:, np.newaxis] * 0.3

riser_len = int(BEAT_S * SR)
np.random.seed(42)
noise = np.random.randn(riser_len, 2) * 0.15
sos_riser = butter(4, [1000 / (SR / 2), 8000 / (SR / 2)], btype='band', output='sos')
noise_filtered = np.column_stack([sosfilt(sos_riser, noise[:, 0]),
                                   sosfilt(sos_riser, noise[:, 1])])
riser_env = np.linspace(0, 1, riser_len) ** 2
noise_riser = noise_filtered * riser_env[:, np.newaxis] * 0.12

ATTACK_SOFT_MS = 15
attack_samples = int(ATTACK_SOFT_MS / 1000 * SR)
attack_env = np.linspace(0.3, 1.0, attack_samples)[:, np.newaxis]

# ── Full KG section (no looping — just stretch the whole post-drop) ──

# The full post-drop section (33-93s) measures at ~95.0 BPM, NOT the global 95.703.
# Using the wrong rate causes ~0.7% drift = ~450ms over 60s (flute runs ahead).
KG_FULL_SECTION_BPM = 95.0
FULL_STRETCH_RATE = TARGET_BPM / KG_FULL_SECTION_BPM

# We need enough stretched audio to cover the flute loop duration
total_needed = target_loop_samples * N_FLUTE_LOOPS
raw_needed = int(total_needed / SR * FULL_STRETCH_RATE * SR) + SR  # +1s margin
kg_drop_sample = int(KG_DROP_S * SR)

print(f"  Stretching full KG drums ({raw_needed/SR:.1f}s raw → ~{total_needed/SR:.1f}s, "
      f"rate={FULL_STRETCH_RATE:.4f} for {KG_FULL_SECTION_BPM} BPM)...")
kg_drums_full_raw = kg_drums_raw[kg_drop_sample:kg_drop_sample + raw_needed]
kg_drums_full = pyrb.time_stretch(kg_drums_full_raw, SR, FULL_STRETCH_RATE)

print(f"  Stretching full KG bass...")
kg_bass_full_raw = kg_bass_raw[kg_drop_sample:kg_drop_sample + raw_needed]
kg_bass_full = pyrb.time_stretch(kg_bass_full_raw, SR, FULL_STRETCH_RATE)
kg_bass_full = distort_bass(kg_bass_full, drive=6)

# Trim to exact needed length
kg_drums_full = kg_drums_full[:total_needed]
kg_bass_full = kg_bass_full[:total_needed]
if len(kg_drums_full) < total_needed:
    kg_drums_full = np.vstack([kg_drums_full,
                                np.zeros((total_needed - len(kg_drums_full), 2))])
if len(kg_bass_full) < total_needed:
    kg_bass_full = np.vstack([kg_bass_full,
                               np.zeros((total_needed - len(kg_bass_full), 2))])
print(f"  Full KG sections: {len(kg_drums_full)/SR:.1f}s (drums), "
      f"{len(kg_bass_full)/SR:.1f}s (bass) — no looping needed")

# ── Multiple flute sources ──────────────────────────────────────────

flute_actual_len = int(LOOP_BARS * SAATHIYA_BAR_S * SR)
flute_stretch = TARGET_BPM / SAATHIYA_LOCAL_BPM

flute_loops = {}
for label, src_s in FLUTE_SOURCES.items():
    start = int(src_s * SR)
    raw = s_other[start:start + flute_actual_len]
    if len(raw) < flute_actual_len:
        raw = np.vstack([raw, np.zeros((flute_actual_len - len(raw), 2))])
    print(f"  Stretching flute from {src_s}s...")
    loop = pyrb.time_stretch(raw, SR, flute_stretch)[:target_loop_samples]
    if len(loop) < target_loop_samples:
        loop = np.vstack([loop, np.zeros((target_loop_samples - len(loop), 2))])
    flute_loops[label] = loop

print(f"  {len(flute_loops)} flute sources ready.")

# ── Clip timeline (fixed) ───────────────────────────────────────────

loop_duration_samples = target_loop_samples * N_FLUTE_LOOPS
buildup_bars = 4
buildup_start = BEAT_GRID_S - buildup_bars * BAR_S
clip_start = BEAT_GRID_S - 3 * BAR_S
clip_end_s = BEAT_GRID_S + loop_duration_samples / SR + 4.0
clip_samples = int((clip_end_s - clip_start) * SR)
cs = int(clip_start * SR)
drop_clip_start = int((BEAT_GRID_S - clip_start) * SR)

# Saathiya stems in clip
sv = extract(s_vocals, cs, clip_samples)
so = extract(s_other, cs, clip_samples)
sd = extract(s_drums_st, cs, clip_samples)
sb = extract(s_bass_st, cs, clip_samples)

# Pre-compute envelopes (don't change)
t = np.arange(clip_samples) / SR + clip_start
fadeout_start = clip_end_s - 4.0

kg_buildup_env = np.zeros(clip_samples)
kg_loop_env = np.zeros(clip_samples)
flute_loop_env = np.zeros(clip_samples)
s_bass_env = np.ones(clip_samples)
s_drums_env = np.ones(clip_samples)
s_vocals_env = np.ones(clip_samples)
s_other_env = np.ones(clip_samples)
master_fade = np.ones(clip_samples)

for i in range(clip_samples):
    ti = t[i]
    if ti < buildup_start:
        pass
    elif ti < BEAT_GRID_S:
        p = (ti - buildup_start) / (BEAT_GRID_S - buildup_start)
        kg_buildup_env[i] = 0.15 + 0.35 * p
        s_bass_env[i] = 1.0 - 0.5 * p
    else:
        kg_loop_env[i] = 1.0
        flute_loop_env[i] = 1.0
        s_vocals_env[i] = 0.0
        s_bass_env[i] = 0.0
        s_drums_env[i] = 0.0
        s_other_env[i] = 0.0
    if ti > fadeout_start:
        fp = (ti - fadeout_start) / (clip_end_s - fadeout_start)
        master_fade[i] = max(0, 1.0 - fp)

# Saathiya mix (static)
saathiya_mix = (
    sv * 0.85 * s_vocals_env[:, np.newaxis] +
    so * 0.70 * s_other_env[:, np.newaxis] +
    sd * 0.40 * s_drums_env[:, np.newaxis] +
    sb * 0.50 * s_bass_env[:, np.newaxis]
)

# KG buildup muffled (static)
nyq = SR / 2
b_lo, a_lo = butter(4, min(300 / nyq, 0.99), btype='low')
kg_stretched_drums = load_stereo(f"{OUTPUT}/kg-drums-stretched.wav")
kg_stretched_bass = load_stereo(f"{OUTPUT}/kg-bass-stretched.wav")
min_kg_s = min(len(kg_stretched_drums), len(kg_stretched_bass))
kg_stretched_rhythm = kg_stretched_drums[:min_kg_s] * 0.9 + kg_stretched_bass[:min_kg_s] * 0.75
kg_timeline_start = BEAT_GRID_S - KG_DROP_STRETCHED_S
kg_sample_offset = int((clip_start - kg_timeline_start) * SR)
kg_buildup_audio = extract(kg_stretched_rhythm, kg_sample_offset, clip_samples)
kg_buildup_muf = np.column_stack([
    filtfilt(b_lo, a_lo, kg_buildup_audio[:, 0]),
    filtfilt(b_lo, a_lo, kg_buildup_audio[:, 1])
])

# Static part of mix (everything that doesn't change)
static_mix = saathiya_mix + kg_buildup_muf * kg_buildup_env[:, np.newaxis]

print(f"  Pre-computation done. Clip: {clip_samples/SR:.1f}s")
print(f"  Each loop = {target_loop_samples/SR:.2f}s ({LOOP_BARS} bars)")
print(f"  Total loops: {N_FLUTE_LOOPS}")

# Edge fades for tiling
fade_in = np.linspace(0, 1, edge_fade)[:, np.newaxis]
fade_out = np.linspace(1, 0, edge_fade)[:, np.newaxis]

# Bass looped (static — no dynamic controls for bass)
BASS_EDGE_MS = 15
bass_edge = int(BASS_EDGE_MS / 1000 * SR)
kg_bass_loop[:bass_edge] *= np.linspace(0, 1, bass_edge)[:, np.newaxis]
kg_bass_loop[-bass_edge:] *= np.linspace(1, 0, bass_edge)[:, np.newaxis]
kg_bass_looped = np.tile(kg_bass_loop, (N_FLUTE_LOOPS, 1))

# Ding looped (flat — envelope applied in callback)
kg_ding_loop[:edge_fade] *= fade_in
kg_ding_loop[-edge_fade:] *= fade_out
kg_ding_looped = np.tile(kg_ding_loop, (N_FLUTE_LOOPS, 1))


# ── Gradio callback ─────────────────────────────────────────────────

def generate_mix(
    # Flute controls
    flute_source, flute_xf_ms, flute_vol,
    # KG mode + drum controls
    kg_mode, add_kick_tail, add_riser, use_soft_attack,
    # Ding controls
    start_vol, end_vol, ramp_loops, curve_type, ding_vol_scale,
):
    """Build the full mix with current settings. Runs in <0.5s."""

    # ── 1. Flute loop ───────────────────────────────────────────────
    flute_one = flute_loops[flute_source].copy()
    xf_samples = int(flute_xf_ms / 1000 * SR)
    if xf_samples > 0:
        flute_one = make_circular_loop(flute_one, xf_samples)
    # Edge fades + tile
    flute_one[:edge_fade] *= fade_in
    flute_one[-edge_fade:] *= fade_out
    flute_tiled = np.tile(flute_one, (N_FLUTE_LOOPS, 1))

    # ── 2. KG drums + bass ──────────────────────────────────────────
    use_full = (kg_mode == "Full Section (no drum restarts)")

    if use_full:
        # Full continuous KG — no loop boundaries at all
        drums_tiled = kg_drums_full.copy()
        bass_tiled = kg_bass_full.copy()
    else:
        # 4-bar loop with optional modifications
        drums = kg_drums_clean.copy()
        if add_kick_tail:
            drums[-kick_len:] += kick_rev
        if add_riser:
            drums[-riser_len:] += noise_riser

        drums_tiled = drums.copy()
        for i in range(1, N_FLUTE_LOOPS):
            loop_copy = drums.copy()
            if use_soft_attack:
                loop_copy[:attack_samples] *= attack_env
            drums_tiled = np.vstack([drums_tiled, loop_copy])

        for i in range(N_FLUTE_LOOPS):
            s = i * target_loop_samples
            if i > 0:
                drums_tiled[s:s + edge_fade] *= fade_in
            if i < N_FLUTE_LOOPS - 1:
                e = s + target_loop_samples
                drums_tiled[e - edge_fade:e] *= fade_out

        bass_tiled = kg_bass_looped

    # ── 3. Place loops in clip ───────────────────────────────────────
    flute_in_clip = np.zeros((clip_samples, 2))
    drums_in_clip = np.zeros((clip_samples, 2))
    bass_in_clip = np.zeros((clip_samples, 2))
    ding_in_clip = np.zeros((clip_samples, 2))

    for src, dst in [
        (flute_tiled, flute_in_clip),
        (drums_tiled, drums_in_clip),
        (bass_tiled, bass_in_clip),
        (kg_ding_looped, ding_in_clip),
    ]:
        end = min(drop_clip_start + len(src), clip_samples)
        copy_len = end - drop_clip_start
        if copy_len > 0:
            dst[drop_clip_start:end] = src[:copy_len]

    # ── 4. Ding envelope ─────────────────────────────────────────────
    ding_env = np.zeros(clip_samples)
    ramp_samples = int(ramp_loops * target_loop_samples)

    for i in range(clip_samples):
        if t[i] < BEAT_GRID_S:
            continue
        loop_pos = i - drop_clip_start
        if loop_pos < 0:
            continue

        if loop_pos < ramp_samples:
            p = loop_pos / ramp_samples
            if curve_type == "Linear":
                vol = start_vol + (end_vol - start_vol) * p
            elif curve_type == "Exponential (slow start)":
                vol = start_vol + (end_vol - start_vol) * (p ** 2.5)
            elif curve_type == "Logarithmic (fast start)":
                vol = start_vol + (end_vol - start_vol) * (1 - (1 - p) ** 2.5)
            elif curve_type == "S-curve":
                s = p * p * (3 - 2 * p)
                vol = start_vol + (end_vol - start_vol) * s
            else:
                vol = end_vol
        else:
            vol = end_vol

        ding_env[i] = vol * ding_vol_scale

    # ── 5. Final mix ─────────────────────────────────────────────────
    loop_mix = (
        flute_in_clip * flute_vol * flute_loop_env[:, np.newaxis] +
        drums_in_clip * 0.85 * kg_loop_env[:, np.newaxis] +
        bass_in_clip * 0.50 * kg_loop_env[:, np.newaxis] +
        ding_in_clip * ding_env[:, np.newaxis] * kg_loop_env[:, np.newaxis]
    )

    final = (static_mix + loop_mix) * master_fade[:, np.newaxis]

    peak = np.max(np.abs(final))
    if peak > 0.95:
        final = final * (0.95 / peak)

    final_16 = (final * 32767).astype(np.int16)
    return (SR, final_16)


# ── Build UI ─────────────────────────────────────────────────────────

loop_dur_s = target_loop_samples / SR

with gr.Blocks(title="DnB Remix Tuner") as app:
    gr.Markdown("# DnB Remix Tuner")
    gr.Markdown(
        "Fine-tune the loop feel and ding build-in. "
        f"Each loop = **{loop_dur_s:.2f}s** ({LOOP_BARS} bars at {TARGET_BPM} BPM). "
        f"**{N_FLUTE_LOOPS} loops** total ({N_FLUTE_LOOPS * loop_dur_s:.0f}s)."
    )

    with gr.Row():
        # ── Left column: controls ────────────────────────────────────
        with gr.Column(scale=1):

            # -- Flute Loop --
            with gr.Accordion("Flute Loop", open=True):
                gr.Markdown(
                    "The flute is a 4-bar melodic phrase from Saathiya. "
                    "Different source positions in the song have different "
                    "loop boundary smoothness."
                )
                flute_source = gr.Dropdown(
                    choices=list(FLUTE_SOURCES.keys()),
                    value=list(FLUTE_SOURCES.keys())[0],
                    label="Source Position",
                    info="Where in the Saathiya track to grab the flute phrase. "
                         "290.4s has the smoothest loop boundary."
                )
                flute_xf_ms = gr.Slider(
                    0, 1000, value=500, step=25,
                    label="Crossfade (ms)",
                    info="Blends the end of each loop into the start of the next. "
                         "0 = hard cut (you'll hear the restart), "
                         "500 = smooth blend, 1000 = very blended."
                )
                flute_vol = gr.Slider(
                    0.0, 1.0, value=0.70, step=0.05,
                    label="Flute Volume",
                    info="How loud the flute is in the mix. "
                         "0.70 is the current default."
                )

            # -- KG Drums --
            with gr.Accordion("KG Drums", open=True):
                kg_full_dur = total_needed / SR
                kg_mode = gr.Radio(
                    ["4-bar Loop (tight, restarts every ~11s)",
                     "Full Section (no drum restarts)"],
                    value="Full Section (no drum restarts)",
                    label="KG Mode",
                    info=f"Full Section uses {kg_full_dur:.0f}s of continuous KG drums "
                         f"from the instrumental drop — no looping, no restarts. "
                         f"4-bar Loop repeats an 11s pattern ({N_FLUTE_LOOPS} times)."
                )
                gr.Markdown(
                    "**4-bar Loop options** *(ignored in Full Section mode)*:"
                )
                add_kick_tail = gr.Checkbox(
                    value=True, label="Reversed Kick Tail",
                    info="A reversed copy of the kick drum (80ms) at the end of "
                         "each loop. Creates anticipation before the next downbeat."
                )
                add_riser = gr.Checkbox(
                    value=True, label="Noise Riser",
                    info="A filtered noise sweep (1 beat long) building up at the "
                         "end of each loop. Adds excitement but also makes you "
                         "'hear' the loop boundary."
                )
                use_soft_attack = gr.Checkbox(
                    value=True, label="Soft Attack (loops 2+)",
                    info="The first loop hits full volume instantly. Loops 2+ "
                         "fade in over 15ms (30%→100%) to soften the 'cue point' feel."
                )

            # -- Ding Build-In --
            with gr.Accordion("Ding Build-In", open=False):
                gr.Markdown(
                    "The **ding** is a high-pitched metallic hit from Kho Gayi. "
                    "These controls let you gradually bring it in after the drop."
                )
                start_vol = gr.Slider(
                    0.0, 0.5, value=0.0, step=0.01,
                    label="Start Volume",
                    info="Ding volume right at the drop. "
                         "0 = silent at first, 0.3 = already noticeable."
                )
                end_vol = gr.Slider(
                    0.05, 0.6, value=0.30, step=0.01,
                    label="End Volume",
                    info="Ding volume after the ramp finishes."
                )
                ramp_loops = gr.Slider(
                    0.5, 6.0, value=3.0, step=0.25,
                    label="Ramp Duration (loops)",
                    info=f"How many loops to ramp over. "
                         f"3 loops = ~{3*loop_dur_s:.0f}s of gradual build."
                )
                curve_type = gr.Radio(
                    ["Linear", "Exponential (slow start)",
                     "Logarithmic (fast start)", "S-curve"],
                    value="Linear", label="Ramp Curve",
                    info="Linear = steady increase. Exponential = quiet then rises fast. "
                         "Logarithmic = loud quickly then levels off. S-curve = smooth."
                )
                ding_vol_scale = gr.Slider(
                    0.5, 2.0, value=1.0, step=0.05,
                    label="Overall Ding Level",
                    info="Master volume for the ding. 1.0 = normal."
                )

        # ── Right column: output ─────────────────────────────────────
        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Preview", type="numpy")
            gen_btn = gr.Button("Generate", variant="primary", size="lg")
            gr.Markdown(
                "*Adjust controls, then click **Generate**. "
                "Listen for the loop restart — does it feel like a "
                "continuous groove or a start-stop pattern?*"
            )

    all_inputs = [
        flute_source, flute_xf_ms, flute_vol,
        kg_mode, add_kick_tail, add_riser, use_soft_attack,
        start_vol, end_vol, ramp_loops, curve_type, ding_vol_scale,
    ]

    gen_btn.click(fn=generate_mix, inputs=all_inputs, outputs=audio_out)
    app.load(fn=generate_mix, inputs=all_inputs, outputs=audio_out)

print("\nLaunching Gradio UI...")
app.launch(share=False)
