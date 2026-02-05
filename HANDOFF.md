## Handoff: DnB Remix — Drop Timing + Distorted Bass

### Project Location
`~/Documents/dnb-remix/`
MEMORY: `~/.claude/projects/-Users-kshitijkarke-Documents-dnb-remix/memory/MEMORY.md`

### What's Done (This Session)

1. **Added distorted bass** — tanh saturation (drive=6, medium crunch) on mids/highs, clean sub (<80Hz) preserved. Applied directly to the KG bass loop in `flute-loop-drop.py`.

2. **Fixed Saathiya vocals masking the drop** — Saathiya vocals had NO envelope (0.85 volume through entire clip, including over the KG drop). Added `s_vocals_env` that cuts all Saathiya stems (vocals, other, drums, bass) at the drop point. This was a major bug — 1.55 of Saathiya audio was competing with KG.

3. **Hardened the snap transition** — KG now at full volume (1.0) instantly at the drop. Previously ramped from 0.5→1.0 over 682ms.

4. **Analyzed djay reference recording** (`VID-20251029-WA0010.mp4`):
   - Extracted audio, did frequency band analysis, cross-correlation, beat tracking
   - djay settings: "Fade in fade out" + "Center bass swap" + "None" effect, 8 bars
   - Both tracks beat-synced at ~88 BPM (avg IBI 680.5ms)
   - Cross-correlation: flute and KG patterns land on the SAME beat
   - Key insight: djay beat-syncs both tracks so their downbeats align

5. **Fixed phase alignment (LATEST)** — SNAP_S (307.454) is beat 2 of the Saathiya bar. KG loop starts on its beat 1 (the drop kick). Placing KG at SNAP_S = KG downbeat on Saathiya off-beat = misaligned. Fixed by moving EVERYTHING to BEAT_GRID_S (306.772) — the actual Saathiya downbeat. Flute extraction also starts from BEAT_GRID_S now (1 beat earlier than before).

### Current State: `dreamy-flute-dnb.mp3`
- **88s clip**: 8s Saathiya intro → muffled KG buildup (4 bars) → SNAP drop → 65s flute + DnB loop → 4s fadeout
- **ALL loops start at BEAT_GRID_S** (306.772, the downbeat) — KG drums, distorted KG bass, flute
- **ALL Saathiya stems cut instantly** at the drop (vocals, drums, bass, other)
- Both KG and flute at exactly 88.0 BPM, zero drift
- **User has NOT listened to this latest version yet** — generated right before handoff request

### Drop Timing Evolution (This Session)
User kept saying KG felt late. Root causes found and fixed:
1. KG volume ramp 0.5→1.0 over 682ms → fixed (instant snap)
2. Saathiya vocals (0.85) + other (0.70) masking drop → fixed (vocal envelope)
3. KG + flute starting at SNAP_S (beat 2) instead of downbeat → fixed (BEAT_GRID_S)
4. Decoupled alignment (KG at BEAT_GRID, flute at SNAP) abandoned → both at BEAT_GRID now

### What's Next
1. **Listen to latest `dreamy-flute-dnb.mp3`** — Does the downbeat alignment fix the timing?
2. **If timing still off**: Try rotating KG loop phase, or adjust BEAT_GRID_S
3. **Distorted bass level** — Currently drive=6. Adjust if needed (3=light, 12=heavy)
4. **Export stems for FL Studio** — Individual loops at correct alignment
5. **Speed up to 170 BPM** for actual DnB tempo

### Key Context
- **Constant stretch ONLY** — `pyrb.time_stretch(audio, SR, 0.9195)`. Never beat-to-beat warp.
- **ALL loops at BEAT_GRID_S** (306.772s) — the Saathiya downbeat. NO more decoupled timing.
- **Flute extracted from BEAT_GRID_S** (was SNAP_S before). First beat of flute loop is 1 beat before the flute melody fully starts.
- **ALL Saathiya stems cut at drop** — vocals, drums, bass, other all have envelopes that go to 0.
- **Distorted bass** — drive=6, tanh saturation on mids/highs, clean sub (<80Hz)
- **Saathiya actual local BPM: 87.939**. Flute stretched 0.069% to match KG at 88.0.

### Reference Analysis Files
```
reference/djay-recording.wav              # Extracted audio from screen recording
reference/djay-transition.png             # Original screenshot
reference/frames/frame_*.png              # Video frames showing djay UI
output/constant-stretch/
  djay-recording-analysis.png             # Full waveform + frequency bands
  djay-transition-zoom.png                # Zoomed transition zone (10-20s)
  djay-xcorr-analysis.png                 # Cross-correlation: KG vs flute
  djay-beat-grid.png                      # Beat tracking with pattern positions
  beat-alignment-analysis.png             # Saathiya drum transients at drop
  waveform-alignment.png                  # Saathiya vs KG waveform comparison
  kg-loop-onset-check.png                 # KG loop onset latency (0.5ms — fine)
```

### Key Files
```
scripts/flute-loop-drop.py           # ★ CURRENT: looped flute + looped KG DnB
scripts/flute-phrase-analysis.py     # Flute energy analysis
scripts/saathiya-beat-analysis.py    # Beat detection (proved 87.939 BPM locally)
scripts/constant-stretch-drop.py     # Previous: non-looped alignment experiments

output/constant-stretch/
  dreamy-flute-dnb.mp3               # ★ LATEST: 88s flute + DnB, distorted bass, downbeat aligned
  drop-decoupled.mp3                 # Previous best (non-looped)
  dist-{light,medium,heavy}.mp3      # Distorted bass variants (separate clips)
  kg-drums-stretched.wav             # Cached stretched stems (for buildup)
  kg-bass-stretched.wav

stems/htdemucs_ft/                   # Demucs stems (saathiya + kho-gayi)
reference/                           # djay screenshot + screen recording + frames
```

### Open Questions
- Does the downbeat-aligned version fix the "KG feels late" issue?
- Is drive=6 the right amount of bass distortion?
- Ready for FL Studio stem export?
