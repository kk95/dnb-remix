## Handoff: Long Version (Round 9)

### State

The long version (2:57) is done for now: `samples/17-final-long-version.mp3`, rendered by
`scripts/long-remix.py` (variant `r9`). The February 1:17 track and its scripts are unchanged.

Render: `python scripts/long-remix.py` (writes `output/long/r9.mp3`). The project `.venv` is broken
(Homebrew dropped Python 3.12); any env with numpy, scipy, soundfile, librosa, pyloudnorm works, plus
the `rubberband` and `ffmpeg` CLIs. `mlx-whisper` only for re-transcribing lyrics.

### Facts that make the long version possible

- Saathiya is exactly 88.000 BPM for the whole song (one straight grid, ±6ms). Bar n starts at
  1.317 + n × 2.7273s; bar 112 = the old BEAT_GRID_S (306.772s). It plays at native speed.
- Kho Gayi is 95.0 BPM in both drops (95.703 was a beat-tracker artifact). Drop 1 at 33.245s (12 bars),
  drop 2 at 140.061s (8 bars), 8-bar build before each, drumless breakdown ~64-120s. Its grid shifts
  ~80ms across the breakdown, so each drop has its own anchor.
- Keys: Saathiya ~B major, Kho Gayi ~E major (chroma estimate, Camelot-adjacent).
- Lyrics: `output/long/lyrics.json` (Whisper word timestamps, rough Devanagari, timings ±0.2s).

### Arrangement (r9)

Opening hook "teri ek hansi ke badle" (Saathiya bars 87-101) with KG's drop groove teasing in muffled
from the second "Saathiya" and pulling out for the last bar → half-time drop A (KG drop 2, "hanste
rehna" on the drop and once more) → antara 2 (bars 58-71), KG guitar fading under its start, KG drums
creeping back from "teri baali" → 176 two-step drop B → 4s fade.

### Mastering (applies to every render)

- Reference: the Kho Gayi release (-9.8 LUFS, drops -9.4 / -7.7). Ours: -11.1 LUFS, drops -9.1 / -10.1,
  true peak -1.1 dBTP, every octave within ~4 dB of Kho Gayi's drops.
- Bass distortion splits at 250 Hz (was 80): tanh(6x) boosted 80-250 Hz ~14 dB, a boom 13 dB over the reference.
- Saathiya's MP3 is mono (L/R correlation 1.00); a decorrelated stereo room on its parts gives width.
- Hooks sit 4 dB down (HOOK_GAIN) so drops stay 4-6 dB above them.
- The 176 drop needs its own low end (one kick every other beat): the kit kick gets a sine body
  (110 -> 50 Hz, 35 ms decay) and drop B's bass a +4 dB bell at 55-110 Hz. A 70 ms body pumped the
  mix ~10 dB per kick ("shock absorbing"); 35 ms + sidechain 0.25 brought it to ~7.6 dB.
- `--wav` writes the 24-bit master; the release asset (tag `long-version`) is that file.

### Listening notes that still apply

- Hooks must start at the top of a lyric line; check `lyrics.json` and the vocal stem's gaps.
- "Hanste rehna" is the "madham madham" take, cut at 134.615s on the "h". The hum at the end of
  "madham" runs straight into it; earlier cuts at 134.233 and 133.900 were audibly wrong.
- No snare rolls in builds (sound "pointed"); shakers only inside drops; vocals over 176 drums sound off.
- Kho Gayi's melodic parts (guitar) are in a different key: fine as a fading tail, not as a layer under Sonu.
- Preferred outro: drop keeps rolling into a fade, like the February mix.

### Open ideas (not started)

- Synth "Reese" bass following Saathiya's own bass notes, so the low end is always in key.
- Newer stem separation (RoFormer family) for less bleed in the flute and vocals.
- Other lyric adlibs cut and ready in `output/long/adlibs/` ("phir se hans de", "garam garam ujla dhuaan", ...).
