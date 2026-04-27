#!/usr/bin/env bash
# Install Whisper Transcription for Arch.
# - registers a .desktop entry visible to Walker, rofi, GNOME, KDE, etc.
# - does NOT install Python dependencies (Arch: pacman -S python-pyqt6 ffmpeg; pip: openai-whisper).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PY="${PROJECT_DIR}/whisper_transcription.py"
DESKTOP_SRC="${PROJECT_DIR}/whisper-transcription-arch.desktop"
TARGET_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
TARGET_FILE="${TARGET_DIR}/whisper-transcription-arch.desktop"
LAUNCHER="${HOME}/.local/bin/whisper-transcription"

echo "Project dir: ${PROJECT_DIR}"

mkdir -p "${TARGET_DIR}" "${HOME}/.local/bin"
chmod +x "${APP_PY}"

# Create launcher in PATH
cat > "${LAUNCHER}" <<EOF
#!/usr/bin/env bash
exec /usr/bin/python "${APP_PY}" "\$@"
EOF
chmod +x "${LAUNCHER}"
echo "Launcher: ${LAUNCHER}"

# Render Exec=, install .desktop
sed "s|__EXEC__|${LAUNCHER}|g" "${DESKTOP_SRC}" > "${TARGET_FILE}"
chmod 644 "${TARGET_FILE}"
echo "Installed: ${TARGET_FILE}"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${TARGET_DIR}" || true
fi

echo
echo "Done. Open Walker and search 'Whisper Transcription'."
echo "Tip: 'walker --query \"Whisper\"' to verify."
