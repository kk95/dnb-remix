#!/bin/bash
# Install CLI audio tools for DnB remix pipeline (macOS)
set -e

echo "Installing audio CLI tools via Homebrew..."
brew list ffmpeg      &>/dev/null || brew install ffmpeg
brew list sox         &>/dev/null || brew install sox
brew list rubberband  &>/dev/null || brew install rubberband

echo ""
echo "Setting up Python env for demucs (AI stem separation)..."
VENV="$(cd "$(dirname "$0")/.." && pwd)/.venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --quiet demucs

echo ""
echo "Done! Tools installed:"
echo "  ffmpeg      — format conversion"
echo "  sox         — audio mixing & effects"
echo "  rubberband  — time-stretch & pitch-shift"
echo "  demucs      — AI stem separation"
echo ""
echo "Activate the venv before running the pipeline:"
echo "  source $VENV/bin/activate"
