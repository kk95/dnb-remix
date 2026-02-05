## Handoff: DnB Remix — KG Guitar Layer + Flute Loop Polish

### Project Location
`~/Documents/dnb-remix/`
MEMORY: `~/.claude/projects/-Users-kshitijkarke-Documents-dnb-remix/memory/MEMORY.md`
Repo: `https://github.com/kk95/dnb-remix` (private, kk95 account)

### Branch & Files
Branch: `main` (clean, all committed and pushed)
Latest commit: `a5f6cd1` — Fix flute stretch rate + zero-overlap looping

### What's Done (This Session)

1. **Fixed flute stretch rate (BUG)** — `flute_stretch` was computed as `BAR_S / SAATHIYA_BAR_S` = 0.99931, which SLOWED the flute instead of speeding it up. Fixed to `TARGET_BPM / SAATHIYA_LOCAL_BPM` = 1.000694. This was causing the flute to drift relative to KG.

2. **Replaced crossfade looping with zero-overlap tiling** — Tested overlapping crossfade (1-beat, 2-beat, 15ms) — all created audible artifacts ("waiting to press restart" feel). Final approach: `np.tile()` with 3ms edge fades. No overlap, no dip, full phrase intact.

3. **Tested and rejected round effect** — 2-layer flute (offset by 2 bars) made it too busy. User pointed out bars 3-4 have DIFFERENT notes (not silence/trail-off) — overlapping them with bars 1-2 muddied both melodies.

4. **Tested and rejected energy compensation** — Volume ramp on bars 3-4 (1.07x) created perceived "slowing down" at the end of each loop.

5. **Analyzed flute riff across the full Saathiya song** — Same riff at 284.9s has 10% stronger end notes but also has vocal bleed in the "other" stem. At 306.8s (current extraction) the stems are clean. Notes are F#, B, C# in bar 4 at both positions.

6. **Initialized git repo** — Private repo at github.com/kk95/dnb-remix. Uses noreply email. Audio/video/stems/output all gitignored.

### What's Next

1. **Add KG guitar from ~2 minutes in** — User wants the guitar riff from KG around 120s to play alongside the flute. Need to:
   - Find the guitar in KG stems (check `stems/htdemucs_ft/kho-gayi/other.wav` around 120s)
   - Extract a loop, time-stretch to 88.0 BPM (same as KG drums/bass)
   - Layer it with the flute — figure out how they complement each other
   - The guitar might help the flute loop restart feel more natural (fills the transition)

2. **Flute loop restart still slightly audible** — The phrase boundary is still noticeable. The guitar layer may help mask this. If not, consider:
   - Trying a different extraction point (284.9s has stronger ending but vocal bleed)
   - Subtle reverb tail at the loop boundary
   - Slightly adjusting the loop start/end point within the bar

3. **Speed up to 170 BPM** for actual DnB tempo (currently 88.0 = half-time)

4. **Export stems for FL Studio** — Individual loops at correct alignment

### Key Context

- **Drop timing is GOOD** — user confirmed "that beat drop timing is appropriate (right after saathiya singing stops)"
- **Flute stretch = `TARGET_BPM / SAATHIYA_LOCAL_BPM`** — same formula as KG. Was inverted for months.
- **Zero-overlap looping** — `np.tile()` + 3ms edge fades. Do NOT use overlapping crossfades.
- **Bars 3-4 of flute have DIFFERENT NOTES** — not trail-off/silence. Don't try to "fill" them.
- **Volume ramps create perceived tempo change** — even 7% ramp felt like "slowing down"
- **pyrubberband rate = speed ratio**: rate < 1 = slower, rate > 1 = faster
- **ALL loops start at BEAT_GRID_S (306.772s)** — the Saathiya downbeat
- **Constant stretch ONLY** — never beat-to-beat warp
- Read MEMORY.md for full BPM values, song structure, and lesson history

### Key Files
```
scripts/flute-loop-drop.py           # ★ CURRENT: looped flute + KG DnB remix
stems/htdemucs_ft/kho-gayi/other.wav # Check ~120s for guitar riff
stems/htdemucs_ft/saathiya/other.wav # Flute source (extracted from 306.772s)
output/constant-stretch/
  dreamy-flute-dnb-*.mp3             # Latest renders (timestamped)
  flute-pattern-comparison.png       # Flute riff comparison across song
  flute-beat-aligned-comparison.png  # Beat-aligned 4-bar comparison
  flute-last-bars-detail.png         # Spectral comparison of end notes
```

### Analysis Plots (This Session)
```
output/constant-stretch/
  flute-pattern-comparison.png       # Energy correlation of flute riff at different song positions
  flute-beat-aligned-comparison.png  # Beat-aligned extraction candidates with per-bar RMS
  flute-last-bars-detail.png         # Spectral + chroma comparison: 306.8s vs 284.9s last bars
```

### Open Questions
- What does the KG guitar sound like at ~120s? Is it in the "other" stem or somewhere else?
- Will the guitar layer help mask the flute loop restart?
- Ready for 170 BPM speed-up or still polishing at 88?
