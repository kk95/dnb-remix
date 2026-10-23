"""Snap Whisper's rough phrase times to the real gaps in the vocal stem; export dry auditions."""
import os, subprocess
import numpy as np, soundfile as sf

ST = os.path.expanduser("~/Documents/dnb-remix/stems/htdemucs_ft/saathiya/vocals.wav")
OUT = os.path.expanduser("~/Documents/dnb-remix/output/long/adlibs")
BAR = 4 * 60 / 88
G0 = 306.772 - 112 * BAR
os.makedirs(OUT, exist_ok=True)

# name: (whisper start of first word, whisper start of last word)
PHRASES = {
    "hanste-rehna":               (134.62, 135.50),   # the "madham madham" line, sung legato (the one used)
    "hanste-rehna-mausam":        (129.05, 130.38),   # the "mausam mausam" line: real silence before "h"
    "madham-madham-hanste-rehna": (131.68, 135.50),
    "mausam-mausam-hanste-rehna": (125.66, 130.38),
    "saathiya":                   (136.26, 138.20),
    "phir-se-hans-de":            (107.24, 109.40),
    "sona-sona":                  (104.40, 106.24),
    "garam-garam-ujla-dhuaan":    (300.78, 302.74),
    "naram-naram-ujla-dhuaan":    (303.88, 305.50),
    "meri-zameen-mera-aasmaan":   (246.80, 254.72),
    "kabhi-shaam-khil-gayi-hai":  (190.46, 194.98),
    "chalo-ghoomne-chalein-hum":  (163.46, 167.98),
    "saathiya-hum-saathiya":      (317.60, 321.62),
    "sun-ke-humne-saari-pee-li-hansi": (153.76, 156.50),
}

# Hand-checked starts where the snap lands on the wrong side of a legato join
# pyin on both lines: the word's vowel, a hummed "m" (centroid ~600 Hz), then the breathy "h".
# "mausam" ends at 128.99 with silence until 129.06; "madham" hums straight into "h" at 134.62.
MANUAL_START = {"hanste-rehna": 134.615, "hanste-rehna-mausam": 129.055}

v, sr = sf.read(ST)
mono = v.mean(axis=1)
win = int(0.01 * sr)
env_db = 20 * np.log10(np.sqrt(np.convolve(mono ** 2, np.ones(win) / win, "same")) + 1e-9)


def snap_start(t):
    a, b = int((t - 0.35) * sr), int((t + 0.2) * sr)
    gap = a + int(np.argmin(env_db[a:b]))
    rise = gap + int(np.argmax(env_db[gap:b + int(0.3 * sr)] > env_db[gap] + 15))
    return max(gap, rise - int(0.03 * sr)) / sr


def snap_end(t_last):
    a = int((t_last + 0.25) * sr)
    seg = env_db[a:a + int(2.5 * sr)]
    quiet = seg < -42
    run = int(0.08 * sr)
    for i in range(len(seg) - run):
        if quiet[i:i + run].all():
            return (a + i) / sr + 0.05
    return (a + int(np.argmin(seg))) / sr


for name, (t0, tl) in PHRASES.items():
    s, e = MANUAL_START.get(name) or snap_start(t0), snap_end(tl)
    clip = v[int(s * sr):int(e * sr)].copy()
    body = np.percentile(env_db[int(s * sr):int(e * sr)], 90)
    legato = env_db[int(s * sr)] - body > -20     # no silence before this word
    f = int((0.025 if legato else 0.008) * sr)
    clip[:f] *= np.linspace(0, 1, f)[:, None]
    clip[-int(0.04 * sr):] *= np.linspace(1, 0, int(0.04 * sr))[:, None]
    clip /= np.abs(clip).max() / 0.7
    wav = f"{OUT}/{name}.wav"
    sf.write(wav, clip, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "256k", wav[:-4] + ".mp3"], check=True)
    os.remove(wav)
    b = (s - G0) / BAR
    print(f'    "{name}": ({s:.3f}, {e:.3f}),   # bar {int(b)} beat {(b % 1) * 4 + 1:.2f}, {e - s:.2f}s')

# sanity: level at each cut vs the phrase's loud parts (want cuts well below)
for name, (t0, tl) in PHRASES.items():
    s, e = snap_start(t0), snap_end(tl)
    body = np.percentile(env_db[int(s * sr):int(e * sr)], 90)
    print(f"  {name:28s} start {env_db[int(s*sr)] - body:6.1f} dB   end {env_db[int(e*sr)] - body:6.1f} dB  (vs phrase body)")
