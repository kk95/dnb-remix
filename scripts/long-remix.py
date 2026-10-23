#!/usr/bin/env python3
"""
Long-form Saathiya x Kho Gayi remix (~3:00) with switchable options.

Both songs sit on fixed grids, which is what makes a long arrangement possible:
  Saathiya: 88.000 BPM for the whole song (±6ms), bar 0 at 1.317s,
            bar 112 = the old BEAT_GRID_S (306.772s). Plays at native speed.
  Kho Gayi: 95.000 BPM in both drops. Drop 1 at 33.245s, drop 2 at 140.061s,
            8-bar build before each. Stretched once by 88/95 (R3 engine).

Arrangements are lists of sections (see VARIANTS), built from:
  drop_a   KG drop 2, half-time. "hanste rehna" lands on the drop and once more before
           the end; the last bar low-passes closed while KG's guitar layer rings on
           and fades out under the next hook's first two bars
  hook     a Saathiya vocal section that starts at the top of a line (see HOOKS);
           "creep" brings KG's drums in muffled over its last bars (WARN);
           "teaser" brings the next drop's groove in muffled on a vocal line and
           pulls it out for the last bar, so Saathiya's bassline leads in alone
  drop_b   the closer: KG drop 1 kick and tops as 176 two-step, rolling into a fade
Each drop takes a flute plan of Saathiya "other" bars, chosen to continue
from where the section before it leaves the song.

Lyric timings come from scripts/transcribe-lyrics.py (Whisper on the vocal stem),
with cut points snapped to the gaps between words by scripts/find-adlibs.py.
Earlier rounds' scripts are kept next to their renders in output/long/long-remix-r*.py.

Usage: python scripts/long-remix.py [--wav] [variant ...]   (no args = render all)
       --wav also writes a 24-bit WAV master next to the MP3, for sending.
"""
import os
import sys
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt, sosfiltfilt

import dnb_kit as kit

PROJECT = os.path.expanduser("~/Documents/dnb-remix")
STEMS = f"{PROJECT}/stems/htdemucs_ft"
OUT = f"{PROJECT}/output/long"
SR = kit.SR

BPM = 88.0
BAR = 4 * 60 / BPM
S_BAR0 = 306.772 - 112 * BAR
KG_TIME_RATIO = 95.0 / BPM          # stretched KG is this many times longer
KG_DROP1, KG_DROP2 = 33.245, 140.061
TARGET_LUFS = -12.0                 # same loudness for every variant, so A/B is fair

# Mix levels carried over from the approved Day 3 mix
G = dict(s_vox=0.85, s_other=0.70, s_drums=0.40, s_bass=0.50,
         kg_drums=0.85, kg_bass=0.50, ding=0.30, flute=0.57, adlib=0.75)

# Saathiya phrase cut points in seconds (snapped by find-adlibs.py)
PHRASES = {
    "hanste-rehna": (134.615, 136.514),     # the "madham madham" take, cut on the "h"
}

# Saathiya bar range + where the vocal line actually starts (may be a pickup before it)
HOOKS = {
    "ek-hansi": dict(bars=(87, 102), vocal=(240.655, 278.4),
                     desc='"teri ek hansi ke badle, meri ye zameen le le..." into the mukhda'),
    "antara-2": dict(bars=(58, 72), vocal=(158.430, 197.8),
                     desc='second antara, "kabhi neele aasmaan pe..." from its pickup'),
}
WARN = {                            # bars before the hook's end where KG starts creeping in
    "antara-2": 72 - 65.48,         # from "teri baali hil gayi hai" (179.886s = bar 65.48)
}
GULP_FROM = {                       # Saathiya time where KG's drums jump in under the hook
    "ek-hansi": 258.741,            # the second "Saathiya" of the mukhda (bar 94.39, 0:20)
}
OUTRO_FADE_S = 4.0                  # same fade length as the published mix

VARIANTS = {
    "r9": dict(sections=[("hook", "ek-hansi", "teaser"), ("drop_a", [(102, 12), (106, 4)]),
                         ("hook", "antara-2", "creep"), ("drop_b", [(72, 4), (8, 8), (106, 8)])],
               desc="Muffled drop A groove from the second \"Saathiya\", bassline alone into drop A."),
}
PATTERN = "twostep"
SIDECHAIN = 0.35


def load(song, stem):
    a, _ = sf.read(f"{STEMS}/{song}/{stem}.wav", dtype="float64")
    return a if a.ndim == 2 else np.column_stack([a, a])


def stretched_kg(stem):
    """KG stem stretched 95 -> 88 BPM with the R3 engine, cached on disk."""
    path = f"{OUT}/cache/kg-{stem}-88.wav"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["rubberband", "-3", "-q", "-t", f"{KG_TIME_RATIO:.10f}",
                        f"{STEMS}/kho-gayi/{stem}.wav", path], check=True)
    a, _ = sf.read(path, dtype="float64")
    return a


def at(bar):
    return int(round(bar * BAR * SR))


def s_bar(n):
    return int(round((S_BAR0 + n * BAR) * SR))


def kg_bar(drop, m):
    return int(round((drop * KG_TIME_RATIO + m * BAR) * SR))


def place(bus, src, src0, dst_bar, bars, gain=1.0, xf=0.02):
    """Copy `bars` bars of src (from sample src0) onto bus at dst_bar.

    The clip fades in over xf seconds of pre-roll before its downbeat and fades
    out over its last xf seconds, so back-to-back clips crossfade just ahead of
    each downbeat and the downbeat hit itself stays intact.
    """
    x, n, d0 = int(xf * SR), at(bars), at(dst_bar)
    seg = np.zeros((n + x, 2))
    a = max(src0 - x, 0)
    chunk = src[a:src0 + n]
    seg[a - (src0 - x):a - (src0 - x) + len(chunk)] = chunk
    ramp = np.sqrt(np.linspace(0, 1, x))[:, None]
    seg[:x] *= ramp
    seg[-x:] *= ramp[::-1]
    if d0 - x < 0:                  # a clip at bar 0 has no room for pre-roll
        seg, d0 = seg[x - d0:], x
    bus[d0 - x:d0 - x + len(seg)] += seg[:len(bus) - (d0 - x)] * gain


def put_line(bus, src, t0, t1, src_bar, dst_bar, gain, fade_in=0.025):
    """Place Saathiya audio from t0..t1 seconds so bar src_bar lands on dst_bar.

    Keeps the line's position inside the bar, so pickups stay pickups.
    """
    seg = src[int(t0 * SR):int(t1 * SR)].copy()
    fi, fo = int(fade_in * SR), int(0.06 * SR)
    seg[:fi] *= np.linspace(0, 1, fi)[:, None]
    seg[-fo:] *= np.linspace(1, 0, fo)[:, None]
    d = int(round((t0 - S_BAR0 - src_bar * BAR + dst_bar * BAR) * SR))
    bus[d:d + len(seg)] += seg * gain


def adlib(bus, S, name, land_bar):
    """Fire a phrase so it lands on land_bar the way it lands in the song."""
    t0, t1 = PHRASES[name]
    put_line(bus, S["vocals"], t0, t1, round((t0 - S_BAR0) / BAR), land_bar, G["adlib"])


def ramp(bus, bar0, bars, g0, g1):
    a, b = at(bar0), at(bar0 + bars)
    bus[a:b] *= np.linspace(g0, g1, b - a)[:, None]


def filt(x, kind, hz):
    return sosfiltfilt(butter(4, hz, btype=kind, fs=SR, output="sos"), x, axis=0)


def distort(x, drive):
    """Tanh saturation above 80 Hz, clean sub below (Day 1 recipe)."""
    if drive <= 0:
        return x
    lo = sosfilt(butter(4, 80, btype="low", fs=SR, output="sos"), x, axis=0)
    hi = sosfilt(butter(4, 80, btype="high", fs=SR, output="sos"), x, axis=0)
    return lo + np.tanh(drive * hi) / np.tanh(drive)


def instruments(buses, S, src_bar, dst_bar, bars, drums=False):
    stems = [("other", "s_other"), ("bass", "s_bass")] + ([("drums", "s_drums")] if drums else [])
    for stem, key in stems:
        place(buses["sat"], S[stem], s_bar(src_bar), dst_bar, bars, G[key])


def sweep(bus, bar0, bars, cutoffs=(6000, 3000, 1500, 800, 400, 250)):
    """Low-pass that closes over `bars`: blends dry -> each cutoff in turn.

    The filters are zero-phase, so neighbouring versions blend without smearing.
    """
    a, b = at(bar0), at(bar0 + bars)
    versions = [bus[a:b].copy()] + [filt(bus[a:b], "low", hz) for hz in cutoffs]
    pos = np.linspace(0, len(versions) - 1, b - a)
    bus[a:b] = sum(v * np.clip(1 - np.abs(pos - i), 0, 1)[:, None] for i, v in enumerate(versions))


def drop(buses, K, kind, kg_drop, segments, dst_bar):
    """Drums + bass + ding for one drop. segments = [(kg_bar_from, bars), ...]."""
    pos = dst_bar
    for m, bars in segments:
        if kind == "half":
            place(buses["drums"], K["drums"], kg_bar(kg_drop, m), pos, bars, G["kg_drums"], xf=0.01)
        else:
            place(buses["drums"], K["drums_top"], kg_bar(kg_drop, m), pos, bars, 0.55, xf=0.01)
        place(buses["bass"], K["bass"], kg_bar(kg_drop, m), pos, bars, G["kg_bass"], xf=0.01)
        place(buses["ding"], K["ding"], kg_bar(kg_drop, m), pos, bars, G["ding"], xf=0.01)
        pos += bars
    ramp(buses["ding"], dst_bar, 8, 0.0, 1.0)
    if kind == "176":
        kit.render_hits(buses["drums"], kit.pattern_hits(PATTERN, dst_bar, pos - dst_bar, BAR), K["kit"])
    return pos


def flute(buses, S, plan, bar):
    for src, n in plan:
        place(buses["flute"], S["other"], s_bar(src), bar, n, G["flute"])
        bar += n


def drop_a(buses, S, K, bar, plan):
    end = drop(buses, K, "half", KG_DROP2, [(0, 8), (0, 8)], bar)
    flute(buses, S, plan, bar)
    for land in [bar, bar + 12]:
        adlib(buses["vox"], S, "hanste-rehna", land)
    for b in ["drums", "bass", "flute"]:
        sweep(buses[b], end - 1, 1)
    # KG's guitar layer carries on past the drop and fades under the next section
    place(buses["ding"], K["ding"], kg_bar(KG_DROP2, 8), end, 2, G["ding"])
    ramp(buses["ding"], end, 2, 1.0, 0.0)
    return end


def hook(buses, S, K, bar, name, drums=None):
    """drums: None, "creep" or "teaser" (KG drums only: KG bass would fight
    Saathiya's bassline under a hook)."""
    (b0, b1), (t0, t1) = HOOKS[name]["bars"], HOOKS[name]["vocal"]
    instruments(buses, S, b0, bar, b1 - b0)
    put_line(buses["vox"], S["vocals"], t0, t1, b0, bar, G["s_vox"])
    end = bar + b1 - b0
    kg = np.zeros_like(buses["build"])
    if drums == "creep":
        bars = WARN[name]
        whole = int(np.ceil(bars))  # KG's build has 8 bars of drums before its drop 2
        place(kg, K["drums"], kg_bar(KG_DROP2, -whole), end - whole, whole)
        kg[:at(end - bars)] = 0
        ramp(kg, end - bars, bars, 0.3, 1.8)
        kg = filt(kg, "low", 800)   # muffled, but the snare still reads
    elif drums == "teaser":
        start = bar + (GULP_FROM[name] - S_BAR0) / BAR - b0
        first, stop = int(start), end - 1
        # The drop A groove itself, muffled under the voice (KG's own build is
        # still near-silent this far before its drop)
        place(kg, K["drums"], kg_bar(KG_DROP2, 0), first, stop - first, 0.45)
        kg = filt(kg, "low", 1500)
        kg[:at(start)] = 0
        swell = at(0.25)            # one-beat swell in, so it gulps in rather than clicks
        kg[at(start):at(start) + swell] *= np.linspace(0, 1, swell)[:, None]
    buses["build"] += kg
    return end


def drop_b(buses, S, K, bar, plan):
    """The closing 176 drop. KG drop 1 bars 0-11, 4-7 (no second crash), 8-11."""
    end = drop(buses, K, "176", KG_DROP1, [(0, 12), (4, 4), (8, 4)], bar)
    flute(buses, S, plan, bar)
    adlib(buses["vox"], S, "hanste-rehna", end - 2)
    return end


def render(name, v, S, K, lossless=False):
    n = at(100)
    buses = {b: np.zeros((n, 2)) for b in ["sat", "vox", "build", "drums", "bass", "ding", "flute"]}
    blocks = dict(drop_a=drop_a, hook=hook, drop_b=drop_b)
    bar, ducked = 0, np.zeros(n, dtype=bool)
    for sec in v["sections"]:
        kind, *args = (sec,) if isinstance(sec, str) else sec
        start, bar = bar, blocks[kind](buses, S, K, bar, *args)
        if kind == "drop_b":
            ducked[at(start):at(bar)] = True

    # Sidechain only the 176 drop, so drop A sounds the same in every variant
    duck = np.where(ducked, kit.sidechain(buses["drums"], SIDECHAIN), 1.0)
    for b in ["bass", "flute", "ding"]:
        buses[b] *= duck[:, None]
    mix = sum(buses.values())[:at(bar)]
    n_fade = int(OUTRO_FADE_S * SR)
    mix[-n_fade:] *= np.linspace(1, 0, n_fade)[:, None]
    n_in = at(0.25)             # one beat, so a mid-song opening doesn't start cold
    mix[:n_in] *= np.linspace(0, 1, n_in)[:, None]
    return master(mix, name, lossless)


def master(mix, name, lossless=False):
    meter = pyln.Meter(SR)
    mix = mix * 10 ** ((TARGET_LUFS - meter.integrated_loudness(mix)) / 20)
    raw, mp3 = f"{OUT}/{name}-raw.wav", f"{OUT}/{name}.mp3"
    sf.write(raw, mix.astype(np.float32), SR, subtype="FLOAT")
    # Brickwall at -1 dBFS: the old mix only peak-normalized, so its drops sat at -14.8 LUFS
    limiter = ["-af", "alimiter=limit=0.891:attack=3:release=60:level=disabled"]
    outputs = [["-codec:a", "libmp3lame", "-b:a", "320k", mp3]]
    if lossless:                # 44.1 kHz is the stems' rate, so 24-bit is the only upgrade left
        outputs.append(["-codec:a", "pcm_s24le", f"{OUT}/{name}-master.wav"])
    for out in outputs:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, *limiter, *out], check=True)
    os.remove(raw)
    return mp3


def prepare():
    S = {s: load("saathiya", s) for s in ["vocals", "other", "drums", "bass"]}
    K = {s: stretched_kg(s) for s in ["drums", "bass", "other"]}
    K["ding"] = filt(K["other"], "high", 1500)
    K["drums_top"] = filt(K["drums"], "high", 6000)
    K["bass"] = distort(K["bass"], 6)
    K["kit"] = kit.build_kit(load("kho-gayi", "drums"), KG_DROP1)
    return S, K


if __name__ == "__main__":
    lossless = "--wav" in sys.argv
    names = [a for a in sys.argv[1:] if a != "--wav"] or list(VARIANTS)
    S, K = prepare()
    for name in names:
        print(f"  {name}: {render(name, VARIANTS[name], S, K, lossless)}")
