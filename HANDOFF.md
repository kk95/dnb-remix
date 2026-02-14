## Handoff: Loop Smoothness Fix + Ding Fine-Tuning

### Branch & Files

Branch: `main` (clean, up to date with origin)
Modified (unstaged): `scripts/flute-loop-drop.py` (+82 lines — ding layer, loop smoothing from prior sessions, NOT yet committed)
New (untracked): `scripts/ding-tuner.py` (Gradio UI for ding ramp tuning), `scripts/find-flute-riff-instances.py` (flute riff scanner)

### What's Done (This Session)

1. **Built Gradio ding tuner** (`scripts/ding-tuner.py`) — interactive UI with sliders for ding build-in (start vol, end vol, ramp duration, curve shape, master level). Pre-computes stems at startup, generates audio in <1s. Has accessible tooltips explaining each control for non-musicians. Needs flute loop controls added (see below).

2. **Analyzed djay video (1)** — extracted frames to `reference/frames-v2/`. Found complete transition settings:
   - Volume: "Fade in fade out" (Custom curves)
   - EQ: **"Center bass swap"** — swaps low-end between tracks at midpoint
   - Effect: "None"
   - Duration: 8 bars
   - **Key insight: djay does NOT loop the flute** — plays the continuous original track

3. **Scanned entire Saathiya "other" stem for flute riff instances** — found 20 via chroma correlation. Current extraction at 306.772s has the WORST loop boundary (0.064 score) despite perfect chroma match. Best candidate: **290.4s** (0.649 boundary, 0.81 chroma). Exported top 5 as `flute-candidate-{time}s.mp3` + `flute-loop-test-{time}s.mp3`.

4. **Diagnosed loop boundary energy jump** — 44% energy spike at every loop restart (end RMS 0.011 → start RMS 0.019). Visible in `output/constant-stretch/djay-vs-ours-290.4s.png`.

5. **Tested multiple loop approaches**:
   - ✅ 4-bar from 290.4s with 500ms circular crossfade — best so far (`test-xf-500ms.mp3`)
   - ❌ 2-bar loop (bars 1-2 only) — loses iconic phrase
   - ❌ djay-style continuous — only 10s of flute after drop, too short
   - ❌ Circular xfade 300ms — still jerky
   - ❌ Longer xfades (1beat, 2beat, 1bar) — lose melodic detail

### What's Next (PRIORITY: Fix Start-Stop Loop Feel)

1. **Build interactive loop tuner** — the user needs knob-like control to fine-tune the loop boundary. Parameters to expose:
   - Circular crossfade duration (0-1000ms) — currently 500ms is best
   - Flute source position (dropdown: 290.4s, 72.2s, 61.3s, etc.)
   - **KG drum loop smoothing** — the reversed kick tail + noise riser at end of each drum loop + soft attack on loops 2+ creates build→drop→build cycle that ALSO contributes to start-stop feel
   - Option to remove/reduce the reversed kick tail and noise riser
   - Flute volume (currently 0.70)
   - All ding controls already in `ding-tuner.py`

   **Approach**: Extend `scripts/ding-tuner.py` to add a "Loop Smoothness" section. Pre-compute flute loops from multiple sources at startup. The callback applies crossfade + drum modifications dynamically.

2. **Investigate KG drum loop boundary** — user suspects drums also contribute to start-stop feel. The drum loop has:
   - Reversed kick (80ms) added to end of each iteration
   - Noise riser (1 beat = 682ms) at end of each iteration
   - Soft attack (0.3→1.0 over 15ms) on loops 2+
   - 3ms edge fades
   These create an audible build→release→build cycle. Try: removing the riser/kick-tail, or applying circular crossfade to drums too.

3. **Commit pending changes** — `flute-loop-drop.py` still uncommitted

4. **Once loop is nailed**: Speed up to 170 BPM for actual DnB tempo

5. **Export stems for FL Studio**

### Key Context

- **KG drums + flute BOTH contribute to start-stop feel** — not just one or the other. The drum loop's reversed kick tail + noise riser + soft attack creates a mini "build-up" at every loop boundary that reinforces the flute's natural energy decay.
- **290.4s is the best flute source** — user confirmed it loops well and sounds right
- **500ms circular crossfade is the right amount** — user said "sounds right"
- **2-bar loops DON'T WORK** — the full 4-bar phrase IS the hook, cutting it loses the magic
- **User wants music-accessible UI** — hover tooltips explaining concepts in plain language, not musical jargon
- **User prefers listening over plots** — always generate audio, not just analysis
- **Gradio is installed** in the venv (v6.5.1) — `gr.Blocks(theme=...)` is deprecated, pass theme to `launch()`
- **Background agents can't use Bash** — permissions only allow `Bash(source:*)` and `Bash(python:*)`, background agents can't prompt so they get auto-denied. Run scripts from main context.
- All existing MEMORY.md context still applies (BPM values, phase alignment, etc.)

### Key Files

```
scripts/flute-loop-drop.py           # ★ MAIN (modified, uncommitted — has ding + smoothing)
scripts/ding-tuner.py                # ★ Gradio UI (ding controls + accessible tooltips, needs loop controls)
scripts/find-flute-riff-instances.py # Flute riff scanner (already run, results exported)
output/constant-stretch/
  test-xf-500ms.mp3                  # ★ BEST SO FAR — 4-bar 290.4s, 500ms circular xfade
  test-xf-{1beat,2beat,1bar}*.mp3    # Other crossfade durations
  test-2bar-*-xf500.mp3              # 2-bar loops (rejected — incomplete phrase)
  test-flute-from-{290.4,72.2}s.mp3  # Full mix with different flute sources
  test-djay-style-continuous.mp3     # No-loop version (too short)
  flute-candidate-*.mp3              # Raw flute extracts from different positions
  flute-loop-test-*.mp3              # 2-loop audition files
  djay-vs-ours-290.4s.png            # Visual comparison of loop boundaries
  dreamy-flute-dnb-20260205-160936.mp3 # Previous best render (before this session)
reference/
  frames-v2/                         # djay video (1) frames — shows transition UI settings
stems/htdemucs_ft/                   # Demucs stems
```

### Mix Levels (Current)

```python
kg_drums: 0.85
kg_bass: 0.50 (distorted, drive=6)
kg_ding: 0.30 (with continuous ramp over 3 loops)
flute: 0.70
saathiya_vocals: 0.85 (cut at drop)
saathiya_other: 0.70 (cut at drop)
saathiya_drums: 0.40 (cut at drop)
saathiya_bass: 0.50 (fades during buildup)
```

### Open Issues

- [ ] Uncommitted changes in `flute-loop-drop.py` — ding layer + loop smoothing
- [ ] **Loop start-stop feel** — main unsolved problem. Both flute AND KG drums contribute. User wants interactive tuning.
- [ ] Ding ramp needs fine-tuning (Gradio UI built but not yet tuned by user)
- [ ] `ding-tuner.py` needs loop smoothness controls added (crossfade duration, drum boundary options, flute source selector)
- [ ] Non-melodic KG textures not yet explored (percussion, risers, vocal chops)
- [ ] KG guitar confirmed to be in DIFFERENT SCALE — don't layer melodic KG content with flute
