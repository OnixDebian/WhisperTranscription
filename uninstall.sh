#!/usr/bin/env bash
set -euo pipefail
TARGET_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
rm -f "${TARGET_DIR}/whisper-transcription-arch.desktop"
rm -f "${HOME}/.local/bin/whisper-transcription"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${TARGET_DIR}" || true
fi
echo "Uninstalled launcher and .desktop entry."
echo "Note: Python packages, model cache (~/.cache/whisper) and settings (~/.config/whisper-transcription-arch) are kept."
