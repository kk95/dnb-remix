"""
176 BPM drum kit for long-remix.py, built from Kho Gayi's own drums.

KG's drop repeats a 2-beat cell. In every cell the 2nd sixteenth is a kick
with no snare on it and the 3rd is a snare with no kick, so averaging those
slots across a dozen cells gives clean one-shots with the bleed averaged out.
Hats are synthesized noise (KG's hats never play alone).

At 88 BPM one bar is two 176 BPM bars, so patterns use 32 steps per 88-bar.
"""
import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
KG_SIXTEENTH = 60 / 95.0 / 4
KG_CELL = 8 * KG_SIXTEENTH

LEVEL = dict(kick=0.85, snare=0.80, hat=0.30, open=0.22)
BODY = 0.75                         # kick body level, relative to the sampled kick's peak

PATTERNS = {
    # step: velocity, per 176 bar (16 steps). Kick on 1 and the "and" of 3.
    "twostep": dict(kick={0: 1.0, 10: 0.9}, snare={4: 1.0, 12: 1.0, 7: 0.18},
                    hat={2: 0.8, 6: 0.8, 10: 0.8, 14: 0.8, 0: 0.4, 4: 0.4, 8: 0.4, 12: 0.4},
                    open={14: 0.5}),
}


def _onset_aligned(mono, times, length, search=0.015):
    """Cut `length` samples at each time, aligned to the hit's actual onset."""
    cuts = []
    for t in times:
        a = int((t - search) * SR)
        w = np.abs(mono[a:a + int(2 * search * SR)])
        onset = a + int(np.argmax(w > 0.5 * w.max())) - int(0.002 * SR)
        cuts.append(mono[onset:onset + length])
    return np.mean(cuts, axis=0)


def _tail_fade(x, ms=15):
    n = int(ms / 1000 * SR)
    x = x.copy()
    x[-n:] *= np.linspace(1, 0, n)
    return x


def _noise_hit(decay, lo, seed=0):
    n = int(decay * 6 * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    sos = butter(4, lo, btype="high", fs=SR, output="sos")
    x = sosfilt(sos, noise) * np.exp(-t / decay)
    x[:int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return x / np.abs(x).max()


def build_kit(kg_drums_raw, drop_s, cells=12):
    mono = kg_drums_raw.mean(axis=1)
    starts = drop_s + KG_CELL * np.arange(cells)
    kick = _onset_aligned(mono, starts + KG_SIXTEENTH, int(0.180 * SR))
    snare = _onset_aligned(mono, starts + 2 * KG_SIXTEENTH, int(0.220 * SR))
    # The kick before each snare rings into it; keep only the snare's body
    snare = sosfilt(butter(2, 150, btype="high", fs=SR, output="sos"), snare)
    peak = np.abs(kick).max()
    # KG's kick is mostly click above 120 Hz; a sine body falling 110 -> 50 Hz gives it
    # the 60-120 Hz weight a 176 drop needs (the half-time drop gets it from KG's dense kicks)
    t = np.arange(len(kick)) / SR
    freq = 50 + 60 * np.exp(-t / 0.03)
    kick = kick + np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.035) * peak * BODY
    return dict(kick=_tail_fade(kick), snare=_tail_fade(snare),
                hat=_noise_hit(0.012, 8000) * peak, open=_noise_hit(0.07, 7000, seed=1) * peak)


def pattern_hits(pattern, bar0, bars, bar_s):
    """(sample, sound, velocity) hits; every 4th bar ends on a short snare fill."""
    step = bar_s / 32
    p = PATTERNS[pattern]
    hits = []
    for b in range(bars):
        for half in range(2):
            base = (bar0 + b) * bar_s + half * 16 * step
            fill = half == 1 and b % 4 == 3
            for sound, steps in p.items():
                if fill and sound == "snare":
                    steps = {4: 1.0, 10: 0.5, 12: 1.0, 13: 0.6, 14: 0.8, 15: 0.9}
                for s, vel in steps.items():
                    hits.append((int(round((base + s * step) * SR)), sound, vel))
    return hits


def render_hits(bus, hits, kit, gain=1.0):
    for pos, sound, vel in hits:
        x = kit[sound] * vel * gain * LEVEL[sound]
        end = min(pos + len(x), len(bus))
        bus[pos:end] += x[:end - pos, None]


def sidechain(drums_bus, depth, block_ms=1, release_s=0.15):
    """Gain curve that dips on every kick in the drums bus (low band only)."""
    lo = sosfilt(butter(4, 120, btype="low", fs=SR, output="sos"), drums_bus.mean(axis=1))
    blk = int(block_ms / 1000 * SR)
    n = len(lo) // blk
    peaks = np.abs(lo[:n * blk]).reshape(n, blk).max(axis=1)
    env = np.empty(n)
    fall = np.exp(-block_ms / 1000 / release_s)
    level = 0.0
    for i, p in enumerate(peaks):
        level = p if p > level else level * fall
        env[i] = level
    ref = np.percentile(env[env > 0], 99) if np.any(env > 0) else 1.0
    duck = 1 - depth * np.clip(env / ref, 0, 1)
    out = np.ones(len(drums_bus))
    out[:n * blk] = np.repeat(duck, blk)
    return out
