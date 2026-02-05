# Saathiya x Kho Gayi — A Bollywood DnB Remix, Built Entirely Through Prompting

A Bollywood drum & bass remix created through conversational collaboration with [Claude Code](https://docs.anthropic.com/en/docs/claude-code). No DAW. No manual audio editing. Just natural language prompts, iterative listening, and Claude writing Python scripts to process audio.

The result: Saathiya's dreamy flute melody layered over Kho Gayi's drum break, beat-synced at 88 BPM (half-time DnB), with a distorted bass layer and a snap transition from vocals to drop.

**Final result:**

https://github.com/user-attachments/assets/placeholder

<audio controls src="samples/12-final-dreamy-flute-dnb.mp3"></audio>

[Download: Final remix (3.0 MB)](samples/12-final-dreamy-flute-dnb.mp3)

## The Idea

Spotify has a beta feature that auto-mixes songs in a playlist — beat-syncing transitions between tracks. I had Saathiya and Kho Gayi back to back, and the transition it created was magic: **Saathiya's dreamy flute ending flowing straight into Kho Gayi's beat drop**. The songs are in different tempos (87.6 vs 95.7 BPM), but Spotify's beat-sync made them lock together.

I screen-recorded the transition in [djay](https://www.algoriddim.com/djay-app) to study it more closely — what settings, how many bars of overlap, where exactly the beats aligned. Then I wanted to turn that transition into an actual produced remix — but I don't use a DAW. I wanted to see how far I could get with just Claude and Python.

## How It Worked

The entire project was built through **iterative prompting** across multiple Claude Code sessions. The workflow looked like this:

1. I describe what I want (or what sounds wrong)
2. Claude writes/modifies a Python script
3. The script runs, produces an MP3
4. I listen, give feedback ("the drop feels 2 seconds late", "the bass needs more grit")
5. Repeat

My ears were the quality control. Claude handled all the signal processing, beat analysis, time-stretching, stem manipulation, and mix engineering — but every creative decision came from listening and prompting.

## The Journey (26 Scripts, 254 Renders)

### Phase 1: Source Separation

Started by asking Claude to separate both songs into stems using [Demucs](https://github.com/facebookresearch/demucs) (vocals, drums, bass, other). This gave us isolated flute, drums, and bass to work with independently.

### Phase 2: Recreating the djay Transition

I screen-recorded my djay transition and shared it as a reference. Claude analyzed the video frame-by-frame to extract djay's settings: "Fade in fade out" + "Center bass swap", 8-bar overlap. Then wrote scripts to recreate it programmatically.

<audio controls src="samples/01-early-transition-attempt.mp3"></audio>

[Listen: Early transition attempt (1.6 MB)](samples/01-early-transition-attempt.mp3) — First programmatic recreation of the djay transition. Beat alignment is rough — you can hear the tempos fighting each other.

**Scripts:** `recreate-transition.py`, `exact-transition.py`, `cycling-remix.py`

### Phase 3: The Beat Alignment Problem

The two songs are at different tempos. Early attempts used beat-to-beat warping (detecting every beat in both songs and mapping them one-to-one). This caused audible **tempo wobble** — the track would speed up and slow down slightly on every beat.

I'd prompt things like:
> "It sounds like it speeds up and slows down — like someone is tapping the tempo button"

Claude would analyze why, propose a fix, generate a new render. We went through ~10 different alignment strategies:

<audio controls src="samples/02-beat-warped-wobble.mp3"></audio>

[Listen: Beat-warped version (1.6 MB)](samples/02-beat-warped-wobble.mp3) — The wobble problem. Each beat is individually time-mapped, creating audible speed fluctuations.

<audio controls src="samples/03-beat-locked-drop.mp3"></audio>

[Listen: Beat-locked drop (1.5 MB)](samples/03-beat-locked-drop.mp3) — Improved but still inconsistent. Beat detection errors cause micro-timing shifts.

**Scripts:** `beat-locked-drop.py`, `beat-warp-transition.py`, `peak-sync.py`, `groove-fix.py`, `local-grid-drop.py`

### Phase 4: The Constant Stretch Breakthrough

After many failed attempts at beat-to-beat warping, I described how djay actually works — it just plays both tracks at the same BPM, constantly. Claude had the insight:

> "DJ software uses a single constant time-stretch, not beat-to-beat warping. One fixed rate, no wobble."

This was the turning point. A single `pyrubberband.time_stretch()` call with rate `88.0 / 95.703 = 0.9195` solved what 10 scripts couldn't.

<audio controls src="samples/04-constant-stretch-breakthrough.mp3"></audio>

[Listen: Constant stretch (1.5 MB)](samples/04-constant-stretch-breakthrough.mp3) — Night and day difference. Steady tempo throughout, no wobble. The drop timing is still off, but the foundation is solid.

**Script:** `constant-stretch-drop.py`

### Phase 5: Finding the Exact Drop Point

With tempo solved, the drop timing was wrong. I'd say things like:
> "The beat drop feels about 2 seconds late — it should hit right when the vocals stop"

Claude ran onset detection analysis on the drum stems, identified the exact downbeat at **306.772 seconds** in Saathiya, and the KG drop kick at **33.245 seconds**. We discovered that Saathiya's local BPM in the drop region (87.939) differs from its global average (87.593) — it's a live recording.

<audio controls src="samples/05-refined-spotify-style.mp3"></audio>

[Listen: Refined Spotify-style transition (1.1 MB)](samples/05-refined-spotify-style.mp3) — Drop timing is closer. Smooth crossfade like Spotify/djay would do it. But the transition still feels... late.

**Scripts:** `onset-grid-analysis.py`, `local-tempo-analysis.py`, `precise-tempo-analysis.py`, `fine-tune-drop.py`, `kg-shift.py`

### Phase 6: Drop Feel — Snap vs. Smooth

Even with correct timing, the transition *felt* wrong. I kept saying the drop was "late" even when analysis showed it was on the beat. We discovered two problems:

1. **Saathiya's vocals had no envelope** — they played at 0.85 volume right through the drop, burying KG under 1.5 seconds of extra vocal audio
2. **KG had a volume ramp** (0.5 → 1.0 over one beat) which made the drop feel gradual instead of instant

The fix: **snap transition**. All Saathiya stems cut to zero at the downbeat. KG enters at full volume instantly. No ramp, no crossfade.

> "SNAP transitions > smooth crossovers. The S-curve feels late."

<audio controls src="samples/06-first-looped-drop.mp3"></audio>

[Listen: First looped drop with snap (1.6 MB)](samples/06-first-looped-drop.mp3) — KG drums hit instantly at the downbeat. No more "late" feeling.

**Scripts:** `refined-drop.py`, `refined-drop-v2.py`, `refined-drop-v3.py`, `drop-transition.py`

### Phase 7: The Flute Loop

With the drop working, I wanted the flute to loop indefinitely over the KG beat. This became its own saga:

- **How many bars?** Tested 2-bar, 3-bar, and 4-bar loops. 4 bars captured the full musical phrase.
- **Crossfade looping** created an audible dip at the boundary — "sounds like someone waiting to press restart"
- **Round effect** (2-layer offset) was too cluttered — bars 3-4 have their own distinct melody (F#, B, C#), not silence
- **Energy compensation** (boosting bars 3-4 by 7%) created a perceived tempo slowdown
- **Zero-overlap tiling** with 3ms micro-fades: the cleanest result

We also discovered the **flute stretch was inverted** — the formula was producing 0.99931 (slowing the flute down) instead of 1.000694 (speeding it up). A 0.14% error that accumulated into audible drift over 60 seconds of looping.

<audio controls src="samples/07-flute-loop-2bar.mp3"></audio>

[Listen: 2-bar flute loop (3.1 MB)](samples/07-flute-loop-2bar.mp3) — Too short. The riff feels like it's on repeat immediately.

<audio controls src="samples/08-flute-loop-4bar.mp3"></audio>

[Listen: 4-bar flute loop (3.3 MB)](samples/08-flute-loop-4bar.mp3) — Full phrase. Bars 1-2 are the main riff, bars 3-4 have a melodic variation. Much more natural.

**Scripts:** `flute-loop-drop.py` (the final script), `analyze-flute-loop.py`, `flute-phrase-analysis.py`

### Phase 8: Distorted Bass

I wanted the KG bass to have more character — "gritty, but keep the sub clean." Claude implemented tanh saturation on mids/highs (>80 Hz) while preserving the clean sub-bass below. We tested three drive levels:

<audio controls src="samples/09-bass-dist-light.mp3"></audio>

[Listen: Light distortion (1.6 MB)](samples/09-bass-dist-light.mp3) — Subtle warmth. Almost too clean.

<audio controls src="samples/10-bass-dist-medium.mp3"></audio>

[Listen: Medium distortion (1.6 MB)](samples/10-bass-dist-medium.mp3) — The sweet spot. Grit on the mids, clean sub preserved.

<audio controls src="samples/11-bass-dist-heavy.mp3"></audio>

[Listen: Heavy distortion (1.6 MB)](samples/11-bass-dist-heavy.mp3) — Too aggressive. Overpowers the flute.

**Integrated into:** `flute-loop-drop.py`

## What Claude Built

The final script ([flute-loop-drop.py](scripts/flute-loop-drop.py)) produces an 88-second remix:

```text
0:00  Pure Saathiya — last vocal phrase
0:08  KG builds in muffled (low-pass filtered)
0:11  DROP — KG drums + bass snap to full, flute loop begins
1:07  Fadeout
```

Under the hood:

- **Stem separation** via Demucs (htdemucs_ft model)
- **Time-stretching** via pyrubberband (constant rate, pitch-preserved)
- **4-bar loops** tiled with zero-overlap and 3ms edge fades
- **Bass distortion** via tanh saturation with clean sub preservation
- **Per-stem volume envelopes** for the snap transition
- **Muffled KG buildup** using a 300 Hz low-pass filter

## By the Numbers

| Metric | Count |
| ------ | ----- |
| Python scripts written | 26 |
| Total lines of code | ~7,000 |
| Audio renders generated | 254 |
| Analysis plots | 23 |
| Approaches tried and abandoned | ~12 |
| Key bugs found through listening | 3 (stretch inversion, missing envelope, phase misalignment) |

## Key Lessons About Prompting for Audio

**Your ears are irreplaceable.** Claude can write sophisticated signal processing code, but it can't hear the output. Every breakthrough came from me listening and describing what was wrong in musical terms ("the drop feels late", "it sounds like it speeds up and slows down", "the bass needs more grit").

**Describe the problem, not the solution.** When I said "the drop feels 2 seconds late," Claude investigated onset timing, envelope shapes, and phase alignment to find the actual cause (vocals masking the drop). When I prescribed solutions ("add a crossfade"), they often made things worse.

**Iterate fast, render often.** Each script takes ~10 seconds to run. We'd generate 5-10 renders per conversation, listen to each one, and adjust. The 254 output files represent the real cost of getting audio right.

**AI doesn't replace musical intuition.** Claude caught a 0.069% tempo drift through analysis — but only after I said "the flute is drifting." The inverted stretch formula was mathematically correct but musically backwards, and only human ears noticed.

## Session Management

This project spanned multiple Claude Code sessions. I used Claude's **memory system** and **session handoffs** to maintain context:

- **MEMORY.md**: Persistent notes on BPM values, timing decisions, what worked/failed, and lessons learned. Updated after each session.
- **HANDOFF.md**: End-of-session document capturing current state, what's next, and critical context for the next session to pick up seamlessly.

This meant each new session could start productive immediately instead of re-discovering decisions from scratch.

## Project Structure

```text
samples/                          # Curated audio for this README
  01-early-transition-attempt.mp3 # Phase 2: first programmatic transition
  02-beat-warped-wobble.mp3       # Phase 3: the tempo wobble problem
  03-beat-locked-drop.mp3         # Phase 3: improved but still inconsistent
  04-constant-stretch-breakthrough.mp3  # Phase 4: the turning point
  05-refined-spotify-style.mp3    # Phase 5: smooth crossfade (still felt late)
  06-first-looped-drop.mp3       # Phase 6: snap transition
  07-flute-loop-2bar.mp3         # Phase 7: too short
  08-flute-loop-4bar.mp3         # Phase 7: full phrase
  09-bass-dist-light.mp3         # Phase 8: subtle
  10-bass-dist-medium.mp3        # Phase 8: sweet spot
  11-bass-dist-heavy.mp3         # Phase 8: too much
  12-final-dreamy-flute-dnb.mp3  # The final remix

scripts/
  flute-loop-drop.py              # Current: 1-minute flute + DnB remix
  constant-stretch-drop.py        # The breakthrough: constant time-stretch
  beat-locked-drop.py             # Abandoned: beat-to-beat warp (wobble)
  recreate-transition.py          # Early: djay transition recreation
  ... (22 more scripts)           # Each represents a different approach or analysis

stems/htdemucs_ft/                # Demucs-separated stems (gitignored)
  saathiya/{vocals,drums,bass,other}.wav
  kho-gayi/{vocals,drums,bass,other}.wav

output/                           # 254 audio renders + 23 analysis plots (gitignored)
  constant-stretch/               # Latest approach (45 MP3s, 12 PNGs)
  beat-locked/                    # Beat-warp experiments
  refined/ (v1, v2, v3)          # Drop refinement iterations
  fine-tune/                      # Millisecond-level timing tests
  ...

reference/
  djay-recording.wav              # Extracted audio from djay screen recording
  frames/                         # 52 video frames of djay UI during transition
```

## Tools Used

- **Claude Code** — all scripting, analysis, and audio engineering through conversation
- **Python** — numpy, scipy, soundfile, librosa, pyrubberband, matplotlib
- **Demucs** (htdemucs_ft) — source separation
- **ffmpeg** — audio format conversion
- **My ears** — quality control

## Status

The core remix works. Still exploring:

- Adding KG guitar from ~2 minutes in as a new layer
- Polishing the flute loop restart boundary
- 170 BPM speed-up for full DnB tempo
- Exporting stems for FL Studio
