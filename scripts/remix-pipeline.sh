#!/bin/bash
# ============================================================
# DnB Remix Pipeline
# Saathiya (88 BPM, 12A/Db minor) x Kho Gayi (95 BPM, 12B/Db major)
# Target: 170 BPM
# ============================================================
#
# Prerequisites:
#   1. Run install-tools.sh first
#   2. Activate venv:  source ../.venv/bin/activate
#   3. Place your source audio files (see INPUTS below)
#
# What this script does (in order):
#   Step 1: Convert input files to WAV (consistent format)
#   Step 2: Separate stems via demucs (vocals, drums, bass, other)
#   Step 3: Time-stretch stems to 170 BPM via rubberband
#   Step 4: Mix stems together via sox
#
# You can run individual steps by passing a step number:
#   ./remix-pipeline.sh 3    # only run step 3
# ============================================================

set -e

# ---------- CONFIG (edit these) ----------

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# INPUT: Put your audio files here (mp3, flac, wav, m4a — anything ffmpeg reads)
SAATHIYA_INPUT="$PROJECT_DIR/saathiya.mp3"       # ← your Saathiya file
KHOGANYI_INPUT="$PROJECT_DIR/kho-gayi.mp3"       # ← your Kho Gayi file

# BPM values (already known)
SAATHIYA_BPM=88
KHOGANYI_BPM=95
TARGET_BPM=170

# Output directories
WAVS_DIR="$PROJECT_DIR/wavs"
STEMS_DIR="$PROJECT_DIR/stems"
STRETCHED_DIR="$PROJECT_DIR/stretched"
OUTPUT_DIR="$PROJECT_DIR/output"

# Which step to run (empty = all)
STEP="${1:-all}"

# ---------- HELPERS ----------

log() { echo -e "\n===  $1  ===\n"; }

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: File not found: $1"
        echo "Place your audio file there, or edit the path in this script."
        exit 1
    fi
}

# ---------- STEP 1: Convert to WAV ----------

step1() {
    log "Step 1: Converting inputs to WAV (44.1kHz, 16-bit)"
    mkdir -p "$WAVS_DIR"

    check_file "$SAATHIYA_INPUT"
    check_file "$KHOGANYI_INPUT"

    ffmpeg -y -i "$SAATHIYA_INPUT" -ar 44100 -ac 2 -sample_fmt s16 "$WAVS_DIR/saathiya.wav"
    ffmpeg -y -i "$KHOGANYI_INPUT" -ar 44100 -ac 2 -sample_fmt s16 "$WAVS_DIR/kho-gayi.wav"

    echo "Output: $WAVS_DIR/saathiya.wav"
    echo "Output: $WAVS_DIR/kho-gayi.wav"
}

# ---------- STEP 2: Stem Separation ----------

step2() {
    log "Step 2: Separating stems with demucs (htdemucs_ft model)"
    echo "This will take a few minutes per track..."
    echo "(CPU is slow — if you have an NVIDIA GPU, it'll use CUDA automatically)"
    echo ""

    mkdir -p "$STEMS_DIR"

    # htdemucs_ft = fine-tuned model, best quality (4x slower but worth it)
    # Output: stems/{htdemucs_ft}/{trackname}/{vocals,drums,bass,other}.wav
    demucs -n htdemucs_ft -o "$STEMS_DIR" "$WAVS_DIR/saathiya.wav"
    demucs -n htdemucs_ft -o "$STEMS_DIR" "$WAVS_DIR/kho-gayi.wav"

    echo ""
    echo "Stems created:"
    echo "  Saathiya: $STEMS_DIR/htdemucs_ft/saathiya/"
    ls "$STEMS_DIR/htdemucs_ft/saathiya/"
    echo ""
    echo "  Kho Gayi: $STEMS_DIR/htdemucs_ft/kho-gayi/"
    ls "$STEMS_DIR/htdemucs_ft/kho-gayi/"
}

# ---------- STEP 3: Time-Stretch to 170 BPM ----------

step3() {
    log "Step 3: Time-stretching stems to $TARGET_BPM BPM"
    mkdir -p "$STRETCHED_DIR"

    DEMUCS_MODEL="htdemucs_ft"
    SAATHIYA_STEMS="$STEMS_DIR/$DEMUCS_MODEL/saathiya"
    KHOGANYI_STEMS="$STEMS_DIR/$DEMUCS_MODEL/kho-gayi"

    # Stretch ratio = original_BPM / target_BPM
    # rubberband --tempo uses the ratio: faster = >1, slower = <1
    # To go from 88 BPM → 170 BPM, we speed up by 170/88 = 1.9318x
    # To go from 95 BPM → 170 BPM, we speed up by 170/95 = 1.7895x
    SAATHIYA_RATIO=$(echo "scale=6; $TARGET_BPM / $SAATHIYA_BPM" | bc)
    KHOGANYI_RATIO=$(echo "scale=6; $TARGET_BPM / $KHOGANYI_BPM" | bc)

    echo "Saathiya stretch ratio: ${SAATHIYA_BPM} → ${TARGET_BPM} BPM = ${SAATHIYA_RATIO}x"
    echo "Kho Gayi stretch ratio: ${KHOGANYI_BPM} → ${TARGET_BPM} BPM = ${KHOGANYI_RATIO}x"
    echo ""

    # Stretch Saathiya stems (vocals + other/melody — the ones you want)
    for stem in vocals other; do
        echo "Stretching saathiya/$stem.wav ..."
        rubberband \
            --tempo "$SAATHIYA_RATIO" \
            --crisp 5 \
            "$SAATHIYA_STEMS/$stem.wav" \
            "$STRETCHED_DIR/saathiya-${stem}-${TARGET_BPM}bpm.wav"
    done

    # Also stretch the full Saathiya (useful for reference or bootleg approach)
    echo "Stretching full saathiya.wav ..."
    rubberband \
        --tempo "$SAATHIYA_RATIO" \
        --crisp 5 \
        "$WAVS_DIR/saathiya.wav" \
        "$STRETCHED_DIR/saathiya-full-${TARGET_BPM}bpm.wav"

    # Stretch Kho Gayi drums (the rhythmic reference)
    echo "Stretching kho-gayi/drums.wav ..."
    rubberband \
        --tempo "$KHOGANYI_RATIO" \
        --crisp 3 \
        "$KHOGANYI_STEMS/drums.wav" \
        "$STRETCHED_DIR/khoganyi-drums-${TARGET_BPM}bpm.wav"

    # Stretch Kho Gayi other elements too (might be useful)
    for stem in bass other vocals; do
        if [[ -f "$KHOGANYI_STEMS/$stem.wav" ]]; then
            echo "Stretching kho-gayi/$stem.wav ..."
            rubberband \
                --tempo "$KHOGANYI_RATIO" \
                --crisp 5 \
                "$KHOGANYI_STEMS/$stem.wav" \
                "$STRETCHED_DIR/khoganyi-${stem}-${TARGET_BPM}bpm.wav"
        fi
    done

    echo ""
    echo "Stretched files:"
    ls -lh "$STRETCHED_DIR/"
}

# ---------- STEP 4: Mix stems together ----------

step4() {
    log "Step 4: Mixing stems into draft remix"
    mkdir -p "$OUTPUT_DIR"

    S_VOCALS="$STRETCHED_DIR/saathiya-vocals-${TARGET_BPM}bpm.wav"
    S_MELODY="$STRETCHED_DIR/saathiya-other-${TARGET_BPM}bpm.wav"
    K_DRUMS="$STRETCHED_DIR/khoganyi-drums-${TARGET_BPM}bpm.wav"

    # ---------- Mix 1: Saathiya melody + Kho Gayi drums ----------
    # This is the core concept: Saathiya's musicality + Kho Gayi's rhythm
    echo "Creating Mix 1: Saathiya melody + Kho Gayi drums..."
    sox -m \
        -v 0.7 "$S_MELODY" \
        -v 1.0 "$K_DRUMS" \
        "$OUTPUT_DIR/mix1-melody-plus-drums.wav"

    # ---------- Mix 2: Full combo (vocals + melody + drums) ----------
    echo "Creating Mix 2: Saathiya vocals + melody + Kho Gayi drums..."
    sox -m \
        -v 0.8 "$S_VOCALS" \
        -v 0.6 "$S_MELODY" \
        -v 1.0 "$K_DRUMS" \
        "$OUTPUT_DIR/mix2-full-combo.wav"

    # ---------- Mix 3: Vocals only + drums (clean, minimal) ----------
    echo "Creating Mix 3: Saathiya vocals + Kho Gayi drums..."
    sox -m \
        -v 0.9 "$S_VOCALS" \
        -v 1.0 "$K_DRUMS" \
        "$OUTPUT_DIR/mix3-vocals-plus-drums.wav"

    # ---------- Convert outputs to mp3 for easy listening ----------
    echo ""
    echo "Converting to mp3 for easy preview..."
    for wav in "$OUTPUT_DIR"/mix*.wav; do
        mp3="${wav%.wav}.mp3"
        ffmpeg -y -i "$wav" -codec:a libmp3lame -b:a 320k "$mp3" 2>/dev/null
    done

    echo ""
    echo "========================================"
    echo "  Draft mixes ready!"
    echo "========================================"
    echo ""
    echo "Listen to these and pick your favorite starting point:"
    echo ""
    ls -lh "$OUTPUT_DIR"/*.mp3
    echo ""
    echo "  mix1 = Saathiya melody + Kho Gayi drums"
    echo "  mix2 = Saathiya vocals + melody + Kho Gayi drums"
    echo "  mix3 = Saathiya vocals + Kho Gayi drums (minimal)"
    echo ""
    echo "Next: Open the .wav files in FL Studio to add your own"
    echo "drums, bass, effects, and arrangement."
}

# ---------- RUN ----------

case "$STEP" in
    1) step1 ;;
    2) step2 ;;
    3) step3 ;;
    4) step4 ;;
    all)
        step1
        step2
        step3
        step4
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|all]"
        echo "  1 = Convert to WAV"
        echo "  2 = Separate stems (demucs)"
        echo "  3 = Time-stretch to $TARGET_BPM BPM"
        echo "  4 = Mix stems together"
        echo "  all = Run everything (default)"
        ;;
esac
