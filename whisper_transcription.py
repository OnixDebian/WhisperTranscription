#!/usr/bin/env python3
"""Whisper Transcription for Arch — simple PyQt6 GUI for OpenAI Whisper.

Features:
- Drag & drop or file picker for audio/video
- Model picker rendered as cards with download status, size, RAM estimate
- CPU thread cap (range derived from the running system)
- Live system RAM display; warns if the picked model exceeds available memory
- Progress while transcribing (runs in a worker thread)
- Save result to .txt / .srt / .vtt / .json or copy to clipboard
- Settings dialog: download/delete cached models
- Theme palette read from Omarchy (~/.config/omarchy/current/theme/colors.toml)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "Whisper Transcription"
APP_ID = "whisper-transcription-arch"
CONFIG_DIR = Path.home() / ".config" / APP_ID
CONFIG_FILE = CONFIG_DIR / "settings.json"
WHISPER_CACHE = Path.home() / ".cache" / "whisper"
THEME_COLORS_PATH = Path.home() / ".config/omarchy/current/theme/colors.toml"

# (model_id, approx download MB, approx peak RAM in GB)
MODELS: list[tuple[str, int, float]] = [
    ("tiny", 75, 1.0),
    ("tiny.en", 75, 1.0),
    ("base", 142, 1.0),
    ("base.en", 142, 1.0),
    ("small", 466, 2.0),
    ("small.en", 466, 2.0),
    ("medium", 1500, 5.0),
    ("medium.en", 1500, 5.0),
    ("large-v3", 2900, 10.0),
    ("turbo", 1500, 6.0),
]

# Heuristic seconds-of-CPU per second-of-audio at ~4 threads on a modern
# x86 box. Used to drive a smooth time-based estimate of the transcription
# progress bar — whisper itself only ticks tqdm at 30-second clip
# boundaries, so on a single-clip recording the bar would otherwise stay
# at 0% the whole time.
# Tuned to slightly UNDER-estimate (bar reaches 99 % a bit before whisper
# is actually done) — feels much better than the bar lagging at ~60 %
# while transcription has already finished.
MODEL_RT_FACTOR: dict[str, float] = {
    "tiny": 0.05,
    "base": 0.2,
    "small": 0.5,
    "medium": 1.2,
    "large-v3": 3.0,
    "turbo": 0.7,
}

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".aiff", ".mp4", ".mkv", ".webm", ".mov",
    ".avi", ".mpeg", ".mpg", ".3gp", ".ts",
}

# --- Settings -----------------------------------------------------------

def load_settings() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


# --- Model cache helpers ------------------------------------------------

def model_cache_path(name: str) -> Path:
    return WHISPER_CACHE / f"{name}.pt"


def is_model_downloaded(name: str) -> bool:
    return model_cache_path(name).exists()


def model_disk_size(name: str) -> int:
    p = model_cache_path(name)
    return p.stat().st_size if p.exists() else 0


def human_size(num: int) -> str:
    f = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{int(f)} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


# --- System info --------------------------------------------------------

def cpu_count() -> int:
    return os.cpu_count() or 1


def _meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of kB values."""
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                info[key] = int(parts[0])  # kB
    except Exception:
        pass
    return info


def total_ram_gb() -> float:
    return _meminfo().get("MemTotal", 0) / 1024 / 1024


def available_ram_gb() -> float:
    info = _meminfo()
    if "MemAvailable" in info:
        return info["MemAvailable"] / 1024 / 1024
    return (info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0)) / 1024 / 1024


def audio_duration_seconds(path: str) -> float:
    """Return the duration of `path` in seconds via ffprobe, or 0 on error."""
    if not path or not shutil.which("ffprobe"):
        return 0.0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5,
        )
        return float(result.stdout.strip() or 0)
    except Exception:
        return 0.0


# --- Theme --------------------------------------------------------------

DEFAULT_THEME = {
    "background": "#1a1b26",
    "foreground": "#a9b1d6",
    "accent": "#7aa2f7",
    "selection_foreground": "#1a1b26",
    "color0": "#32344a",
    "color1": "#f7768e",
    "color2": "#9ece6a",
    "color3": "#e0af68",
    "color4": "#7aa2f7",
    "color8": "#444b6a",
}


def load_theme() -> dict:
    """Read Omarchy theme colors; fall back to Tokyo Night defaults."""
    theme = dict(DEFAULT_THEME)
    if not THEME_COLORS_PATH.exists():
        return theme
    try:
        import tomllib
        with open(THEME_COLORS_PATH, "rb") as f:
            data = tomllib.load(f)
        for k, v in data.items():
            if isinstance(v, str) and v.startswith("#"):
                theme[k] = v
    except Exception:
        pass
    return theme


RUNTIME_DIR = Path.home() / ".cache" / APP_ID
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _arrow_svg_path(direction: str, color: str, suffix: str = "") -> str:
    """Write a tiny triangle SVG to the runtime cache and return its path.

    Qt 6 stylesheets accept `image: url(/abs/path.svg)` reliably; data:
    URLs ride a different code path and were rendering blank in our
    earlier attempt. File-based references just work.
    """
    if direction == "up":
        pts = "4,1 0,5 8,5"
    else:
        pts = "0,0 8,0 4,4"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5">'
        f'<polygon points="{pts}" fill="{color}"/></svg>'
    )
    safe_color = color.lstrip("#")
    path = RUNTIME_DIR / f"arrow-{direction}{suffix}-{safe_color}.svg"
    path.write_text(svg)
    return str(path)


def _check_svg_path(color: str) -> str:
    """Tick-shaped SVG used inside the checked-state QCheckBox indicator."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
        f'viewBox="0 0 14 14" fill="none">'
        f'<polyline points="3,7 6,10 11,4" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )
    safe_color = color.lstrip("#")
    path = RUNTIME_DIR / f"check-{safe_color}.svg"
    path.write_text(svg)
    return str(path)


def build_stylesheet(t: dict) -> str:
    bg = t["background"]
    fg = t["foreground"]
    accent = t.get("accent") or t.get("color4", "#7aa2f7")
    # NOTE: theme's `selection_foreground` can be a light color (e.g.
    # #c0caf5 in Tokyo Night) which is fine for highlighted text on a
    # dark selection, but for our primary button (light blue accent
    # background) it gives near-zero contrast. Force a dark on-accent
    # text so 'Start transcription' is always legible.
    sel_fg = t.get("selection_foreground", bg)
    on_accent = bg  # always dark — used for buttons/menus painted with `accent`.
    border = t.get("color8", "#444b6a")          # structural lines only
    muted = t.get("color7", "#787c99")           # readable secondary text
    surface = t.get("color0", "#32344a")
    surface_hi = t.get("color8", "#444b6a")      # hover surface
    danger = t.get("color1", "#f7768e")
    success = t.get("color2", "#9ece6a")
    arrow_up = _arrow_svg_path("up", fg)
    arrow_down = _arrow_svg_path("down", fg)
    arrow_up_hot = _arrow_svg_path("up", on_accent, "-hot")
    arrow_down_hot = _arrow_svg_path("down", on_accent, "-hot")
    check_mark = _check_svg_path(on_accent)
    return f"""
    QMainWindow, QDialog {{
        background-color: {bg};
        color: {fg};
        font-size: 10pt;
    }}
    /* Cascade text colour and font to all widgets, but DO NOT cascade
       a background-color — that would override our transparent slider
       and other widgets that paint just their sub-controls. */
    QWidget {{
        color: {fg};
        font-size: 10pt;
    }}
    QLabel {{ background: transparent; color: {fg}; }}
    QLabel[role="muted"] {{ color: {muted}; }}
    QLabel[role="warning"] {{ color: {danger}; font-weight: 600; }}
    QGroupBox {{
        border: 1px solid {border};
        border-radius: 8px;
        margin-top: 14px;
        padding: 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {accent};
    }}

    /* Buttons — terminal-bright text, clear hover */
    QPushButton {{
        background: {surface};
        color: {fg};
        border: 1px solid {muted};
        border-radius: 6px;
        padding: 7px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {accent};
        color: {on_accent};
        border-color: {accent};
    }}
    QPushButton:pressed {{
        background: {muted};
        color: {on_accent};
        border-color: {muted};
    }}
    QPushButton:focus {{ outline: none; border-color: {accent}; }}
    QPushButton:disabled {{
        color: {muted};
        border-color: {border};
        background: {bg};
    }}
    QPushButton[role="primary"] {{
        background: {accent}; color: {on_accent}; border-color: {accent}; font-weight: 700;
    }}
    QPushButton[role="primary"]:hover {{ background: {fg}; color: {bg}; border-color: {fg}; }}
    QPushButton[role="primary"]:disabled {{
        background: {surface}; color: {muted}; border-color: {border};
    }}

    /* Inputs */
    QLineEdit, QTextEdit, QSpinBox {{
        background: {bg};
        color: {fg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 5px 8px;
        selection-background-color: {accent};
        selection-color: {on_accent};
    }}
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{ border-color: {accent}; }}
    QLineEdit, QTextEdit {{ placeholder-text-color: {muted}; }}

    /* Spin box: explicit minimum height + 11pt font so the value reads on
       Wayland. Buttons are styled (surface bg, accent on hover) but the
       arrow images are left to Qt's active style — drawing arrows via
       Qt-stylesheet "border tricks" is unreliable across Qt versions. */
    QSpinBox {{
        font-size: 11pt;
        min-height: 26px;
    }}
    QSpinBox::up-button {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 22px;
        border-left: 1px solid {border};
        border-top-right-radius: 5px;
        background: {surface};
    }}
    QSpinBox::down-button {{
        subcontrol-origin: padding;
        subcontrol-position: bottom right;
        width: 22px;
        border-left: 1px solid {border};
        border-bottom-right-radius: 5px;
        background: {surface};
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {accent}; }}
    QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{ background: {muted}; }}
    QSpinBox::up-arrow {{
        image: url({arrow_up});
        width: 8px; height: 5px;
    }}
    QSpinBox::up-button:hover QSpinBox::up-arrow {{ image: url({arrow_up_hot}); }}
    QSpinBox::down-arrow {{
        image: url({arrow_down});
        width: 8px; height: 5px;
    }}
    QSpinBox::down-button:hover QSpinBox::down-arrow {{ image: url({arrow_down_hot}); }}

    /* Disabled inputs: dashed border + muted text + no spin arrows so it
       reads as "not editable" instead of "looks editable but ignored". */
    QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled {{
        color: {muted};
        background: {bg};
        border: 1px dashed {border};
    }}
    QSpinBox:disabled::up-button, QSpinBox:disabled::down-button {{
        width: 0; border: none; background: transparent;
    }}
    QCheckBox {{ color: {fg}; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {muted};
        border-radius: 4px;
        background: {bg};
        image: none;
    }}
    QCheckBox::indicator:hover {{ border-color: {accent}; }}
    QCheckBox::indicator:checked {{
        background: {accent};
        border-color: {accent};
        image: url({check_mark});
    }}
    QCheckBox:disabled {{ color: {muted}; }}
    QCheckBox:disabled::indicator {{ border-color: {border}; background: {bg}; image: none; }}

    QProgressBar {{
        background: {surface};
        color: {on_accent};
        border: 1px solid {border}; border-radius: 6px;
        text-align: center; min-height: 22px;
        font-weight: 700; font-size: 10pt;
    }}
    QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}

    /* QSlider — used for CPU threads and RAM cap. Volume-bar style:
       thin track, sub-page in accent, round handle floating on the
       parent background (no slider-body fill). The widget itself is
       made transparent via WA_StyledBackground=False on the instance
       so the global QWidget {{background: bg}} cascade can't paint
       over us. */
    QSlider::groove:horizontal {{
        height: 4px;
        background-color: {surface_hi};
        border-radius: 2px;
        /* Side margin so the round handle has room at the extremes
           and isn't clipped by the slider widget's edge. */
        margin: 0 9px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {accent};
        border-radius: 2px;
        margin: 0 9px;
    }}
    QSlider::add-page:horizontal {{
        background-color: {surface_hi};
        border-radius: 2px;
        margin: 0 9px;
    }}
    QSlider::handle:horizontal {{
        background-color: {fg};
        width: 16px; height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ background-color: {accent}; }}
    QSlider:disabled::sub-page:horizontal {{ background-color: {muted}; }}
    QSlider:disabled::handle:horizontal {{ background-color: {border}; }}

    QStatusBar {{ background: {surface}; color: {fg}; }}
    QStatusBar QLabel {{ color: {fg}; }}
    QMenuBar {{ background: {bg}; color: {fg}; }}
    QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {accent}; color: {on_accent}; }}
    QMenu {{ background: {surface}; color: {fg}; border: 1px solid {border}; padding: 4px; }}
    QMenu::item {{ padding: 6px 18px; }}
    QMenu::item:selected {{ background: {accent}; color: {on_accent}; }}
    QHeaderView::section {{ background: {surface}; color: {fg}; border: 0; padding: 6px; }}
    QTableWidget {{
        background: {bg}; color: {fg};
        gridline-color: {border};
        border: 1px solid {border}; border-radius: 6px;
    }}
    QRadioButton {{ background: transparent; color: {fg}; }}

    /* Scroll area + modern thin scrollbars (no ancient arrows) */
    QScrollArea {{ border: 0; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px 0 2px 6px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 6px 2px 0 2px;
    }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {border};
        border-radius: 5px;
        min-height: 24px;
        min-width: 24px;
    }}
    QScrollBar::handle:hover {{ background: {muted}; }}
    QScrollBar::handle:pressed {{ background: {accent}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0; width: 0; background: transparent; border: 0;
    }}
    QScrollBar::up-arrow, QScrollBar::down-arrow,
    QScrollBar::left-arrow, QScrollBar::right-arrow {{
        background: transparent; width: 0; height: 0;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* Model cards */
    ModelCard {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 8px;
    }}
    ModelCard:hover {{ border-color: {muted}; background: {surface_hi}; }}
    ModelCard[selected="true"] {{
        border: 2px solid {accent};
        background: {bg};
    }}
    ModelCard[selected="true"]:hover {{ border-color: {accent}; background: {bg}; }}
    ModelCard QLabel {{ color: {fg}; }}
    ModelCard[downloaded="true"] QLabel#status {{ color: {success}; font-weight: 600; }}
    ModelCard[downloaded="false"] QLabel#status {{ color: {muted}; }}

    DropLabel {{
        border: 2px dashed {border};
        border-radius: 8px;
        color: {muted};
        padding: 18px;
        font-weight: 500;
    }}
    DropLabel[hasFile="true"] {{
        border-color: {accent};
        color: {fg};
    }}

    /* Tabs at the top of the main window (File / Record) */
    QTabWidget::pane {{
        border: 1px solid {border};
        border-radius: 8px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {muted};
        padding: 8px 18px;
        border: 1px solid transparent;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 2px;
        font-weight: 600;
    }}
    QTabBar::tab:hover {{ color: {fg}; }}
    QTabBar::tab:selected {{
        background: {bg};
        color: {accent};
        border: 1px solid {border};
        border-bottom: 1px solid {bg};
    }}

    /* Segmented control: two/three pill-buttons sharing borders */
    QPushButton#SegmentBtn {{
        background: {surface};
        color: {muted};
        border: 1px solid {border};
        padding: 8px 16px;
        font-weight: 600;
        border-radius: 0;
    }}
    QPushButton#SegmentBtn[position="left"] {{
        border-top-left-radius: 6px; border-bottom-left-radius: 6px;
    }}
    QPushButton#SegmentBtn[position="right"] {{
        border-top-right-radius: 6px; border-bottom-right-radius: 6px;
        border-left: none;
    }}
    QPushButton#SegmentBtn[position="middle"] {{
        border-left: none; border-right: none;
    }}
    QPushButton#SegmentBtn:hover {{
        color: {fg};
        background: {surface_hi};
    }}
    QPushButton#SegmentBtn:checked {{
        background: {accent};
        color: {on_accent};
        border: 1px solid {accent};
    }}
    QPushButton#SegmentBtn:checked:hover {{
        background: {fg};
        color: {bg};
        border-color: {fg};
    }}

    /* Big record button with red recording state */
    QPushButton#RecordButton {{
        background: {surface};
        color: {fg};
        border: 2px solid {border};
        border-radius: 12px;
        font-size: 13pt;
        font-weight: 700;
        padding: 12px 18px;
    }}
    QPushButton#RecordButton:hover {{
        background: {surface_hi};
        border-color: {accent};
    }}
    QPushButton#RecordButton[recording="true"] {{
        background: {danger};
        color: {on_accent};
        border-color: {danger};
    }}
    QPushButton#RecordButton[recording="true"]:hover {{
        background: {fg};
        color: {bg};
        border-color: {fg};
    }}

    /* Model chooser button: full-width, button-card styled */
    QPushButton#ModelChooser {{
        background: {surface};
        color: {fg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 12px 14px;
        text-align: left;
        font-weight: 600;
    }}
    QPushButton#ModelChooser:hover {{
        background: {surface_hi};
        border-color: {accent};
    }}
    QPushButton#ModelChooser:pressed {{
        background: {bg};
        border-color: {accent};
    }}
    """


# --- Subprocess worker --------------------------------------------------

WORKER_SCRIPT = Path(__file__).resolve().parent / "whisper_transcribe_worker.py"


def _systemd_run_available() -> bool:
    return shutil.which("systemd-run") is not None


class TranscribeProcess(QObject):
    """Persistent worker subprocess.

    Spawned once and kept alive across transcriptions so the loaded
    Whisper model stays in memory between files (preload + reuse).
    Communicates via line-delimited JSON commands on stdin and
    `[STATUS] ...` lines on stderr. Result of each transcribe is
    written by the worker to a tempfile path supplied by the GUI.

    Lifecycle:
    - Worker is (re)spawned on first command, or whenever the RAM cap
      changes (the cap is wired in via `systemd-run` so it cannot be
      changed for an already-running process).
    - Cancel terminates the worker — whisper.transcribe is not
      interruptible from Python, so killing the process is the only
      way to stop it. The cached model is lost; next call respawns.
    - On normal app exit, send `quit` and let the worker exit cleanly.
    """

    progress = pyqtSignal(str)            # raw [STATUS] message
    progress_pct = pyqtSignal(int)        # tqdm percentage
    preload_done = pyqtSignal(str)        # model name once loaded
    finished_ok = pyqtSignal(dict)        # transcription result
    failed = pyqtSignal(str)              # transcription failed

    # tqdm overwrites a single line with carriage returns:
    # "  0%|          | 0/2669 [00:00<?, ?frames/s]"
    # "100%|██████████| 2669/2669 [00:37<00:00, 70.72frames/s]"
    _TQDM_RE = re.compile(r"^\s*(\d{1,3})%\|")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self._stderr_buf = ""
        self._stderr_log: list[str] = []
        self._cancelled = False
        self._active_result_path: str | None = None
        self._spawned_ram_cap_gb: int = -1   # -1 = nothing spawned yet
        self._loaded_model: str | None = None

    # --- public API -----------------------------------------------------
    def preload(self, model: str, cpu_threads: int, ram_cap_gb: int = 0) -> None:
        """Ask the worker to load `model` in the background. No-op if
        already loaded with the same cap."""
        self._ensure_running(ram_cap_gb)
        if self._loaded_model == model:
            self.preload_done.emit(model)
            return
        self._send({
            "action": "preload",
            "model": model,
            "cpu_threads": int(cpu_threads),
        })

    def transcribe(
        self,
        file_path: str,
        model: str,
        cpu_threads: int,
        language: str | None,
        ram_cap_gb: int = 0,
    ) -> None:
        self._ensure_running(ram_cap_gb)
        # Result is written to a tempfile by the worker, not piped via
        # stdout — pipes can race or be polluted by stray torch/tqdm output.
        fd, path = tempfile.mkstemp(prefix="whisper-result-", suffix=".json")
        os.close(fd)
        self._active_result_path = path
        self._cancelled = False
        self._send({
            "action": "transcribe",
            "model": model,
            "cpu_threads": int(cpu_threads),
            "file": file_path,
            "language": language or "",
            "result_path": path,
        })

    def cancel(self) -> None:
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self._cancelled = True
        self.proc.terminate()
        if not self.proc.waitForFinished(2000):
            self.proc.kill()
            self.proc.waitForFinished(2000)

    def quit_worker(self) -> None:
        """Best-effort clean shutdown. Used on app close."""
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        try:
            self._send({"action": "quit"})
            self.proc.closeWriteChannel()
            if not self.proc.waitForFinished(1500):
                self.proc.kill()
        except Exception:
            self.proc.kill()

    def is_running(self) -> bool:
        return self._active_result_path is not None

    # --- internals ------------------------------------------------------
    def _send(self, cmd: dict) -> None:
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            self.failed.emit("Worker is not running.")
            return
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self.proc.write(line)

    def _ensure_running(self, ram_cap_gb: int) -> None:
        running = self.proc.state() != QProcess.ProcessState.NotRunning
        if running and ram_cap_gb == self._spawned_ram_cap_gb:
            return
        # Different cap or not yet started — (re)spawn.
        if running:
            self.quit_worker()
        self._stderr_buf = ""
        self._stderr_log.clear()
        self._loaded_model = None
        self._active_result_path = None
        self._cancelled = False

        py = sys.executable or "python3"
        worker_args = [str(WORKER_SCRIPT)]
        if ram_cap_gb > 0 and _systemd_run_available():
            program = "systemd-run"
            args = [
                "--user", "--scope", "--quiet", "--collect",
                "-p", f"MemoryMax={ram_cap_gb}G",
                "-p", "MemorySwapMax=0",
                "--", py, *worker_args,
            ]
        else:
            program = py
            args = worker_args
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.start()
        if not self.proc.waitForStarted(5000):
            self.failed.emit(f"Failed to start worker: {self.proc.errorString()}")
            return
        self._spawned_ram_cap_gb = ram_cap_gb

    def _on_stderr(self) -> None:
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buf += data
        # Split on both \n and \r — tqdm uses \r to overwrite the same
        # progress line, so percentages would otherwise pile up in a
        # single un-terminated line forever.
        while True:
            nl = self._stderr_buf.find("\n")
            cr = self._stderr_buf.find("\r")
            if nl == -1 and cr == -1:
                break
            if nl == -1:
                idx = cr
            elif cr == -1:
                idx = nl
            else:
                idx = min(nl, cr)
            line = self._stderr_buf[:idx].strip()
            self._stderr_buf = self._stderr_buf[idx + 1:]
            if not line:
                continue
            if line.startswith("[STATUS] "):
                self._handle_status(line[len("[STATUS] "):])
                continue
            m = self._TQDM_RE.match(line)
            if m:
                self.progress_pct.emit(int(m.group(1)))
                continue
            self._stderr_log.append(line)

    def _handle_status(self, msg: str) -> None:
        # Surface to the GUI for the status bar / progress label.
        self.progress.emit(msg)
        # Track loaded-model state.
        low = msg.lower()
        if low.startswith("model ") and "loaded" in low:
            # "Model <name> loaded"
            parts = msg.split()
            if len(parts) >= 2:
                self._loaded_model = parts[1]
        if msg == "Ready":
            if self._loaded_model:
                self.preload_done.emit(self._loaded_model)
        elif msg == "Done":
            self._deliver_result()
        elif msg.startswith("ERROR:"):
            err = msg[len("ERROR:"):].strip()
            tail = "\n".join(self._stderr_log[-15:]).strip()
            self.failed.emit(f"{err}\n\n{tail}" if tail else err)
            # Cleanup any pending result file.
            if self._active_result_path:
                try:
                    Path(self._active_result_path).unlink(missing_ok=True)
                except Exception:
                    pass
                self._active_result_path = None

    def _deliver_result(self) -> None:
        path = self._active_result_path
        self._active_result_path = None
        if not path:
            return
        try:
            if not Path(path).exists():
                raise FileNotFoundError("worker did not write the result file")
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.finished_ok.emit(result)
        except Exception as e:
            tail = "\n".join(self._stderr_log[-15:]).strip()
            self.failed.emit(f"Failed to read worker result: {e}\n\n{tail}")
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def _on_error(self, _err) -> None:
        pass

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        # Drain any remaining stderr.
        self._on_stderr()
        if self._stderr_buf.strip():
            self._stderr_log.append(self._stderr_buf.strip())
            self._stderr_buf = ""
        # Mark cap state as unspawned so the next call respawns.
        self._spawned_ram_cap_gb = -1
        self._loaded_model = None

        path = self._active_result_path
        self._active_result_path = None

        if self._cancelled:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            self.failed.emit("Cancelled.")
            return

        if path is None:
            # No transcription was in flight — worker just exited (e.g.
            # user closed the app, or model-load crashed during preload).
            if exit_status == QProcess.ExitStatus.CrashExit or exit_code not in (0, 9, 137):
                tail = "\n".join(self._stderr_log[-15:]).strip()
                hint = ""
                if exit_code in (137, 9):
                    hint = "\n\nThe process was killed (likely OOM under the RAM cap)."
                self.failed.emit(
                    f"Worker exited (code {exit_code}).{hint}\n\n{tail or '(no stderr output)'}"
                )
            return

        # A transcription was in flight when the worker died.
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        tail = "\n".join(self._stderr_log[-15:]).strip()
        hint = ""
        if exit_code in (137, 9):
            hint = "\n\nThe process was killed (likely OOM under the RAM cap)."
        elif exit_status == QProcess.ExitStatus.CrashExit:
            hint = "\n\nThe worker crashed."
        self.failed.emit(
            f"Worker exited (code {exit_code}).{hint}\n\n{tail or '(no stderr output)'}"
        )


class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self) -> None:
        try:
            self.progress.emit(f"Downloading {self.model_name}…")
            import whisper
            whisper.load_model(self.model_name, device="cpu")
            self.finished_ok.emit(self.model_name)
        except Exception as e:
            self.failed.emit(str(e))


# --- Custom widgets -----------------------------------------------------

class DropLabel(QLabel):
    filesDropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("DropLabel")
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("hasFile", "false")
        self.setText("Drop audio/video files here  ·  or click Open File…")

    def setHasFile(self, has: bool) -> None:
        self.setProperty("hasFile", "true" if has else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in e.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)


class ModelCard(QFrame):
    """One model row: radio button + name, status badge, RAM estimate, disk size."""
    clicked = pyqtSignal(str)

    def __init__(self, name: str, dl_mb: int, ram_gb: float):
        super().__init__()
        self.name = name
        self.dl_mb = dl_mb
        self.ram_gb = ram_gb
        self.setProperty("selected", "false")
        self.setProperty("downloaded", "false")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.radio = QRadioButton()
        self.radio.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self.radio)

        name_lbl = QLabel(name)
        f = name_lbl.font()
        f.setBold(True)
        name_lbl.setFont(f)
        name_lbl.setMinimumWidth(90)
        layout.addWidget(name_lbl)

        self.status_lbl = QLabel()
        self.status_lbl.setObjectName("status")
        self.status_lbl.setMinimumWidth(120)
        layout.addWidget(self.status_lbl)

        self.size_lbl = QLabel()
        self.size_lbl.setProperty("role", "muted")
        layout.addWidget(self.size_lbl)

        layout.addStretch(1)

        ram_lbl = QLabel(f"~{ram_gb:.0f} GB RAM")
        ram_lbl.setProperty("role", "muted")
        layout.addWidget(ram_lbl)

        self.refresh_status()

    def mousePressEvent(self, _e):
        self.clicked.emit(self.name)

    def refresh_status(self) -> None:
        downloaded = is_model_downloaded(self.name)
        self.setProperty("downloaded", "true" if downloaded else "false")
        if downloaded:
            self.status_lbl.setText("✓ downloaded")
            self.size_lbl.setText(human_size(model_disk_size(self.name)))
        else:
            self.status_lbl.setText("not downloaded")
            self.size_lbl.setText(f"~{self.dl_mb} MB to download")
        self.style().unpolish(self)
        self.style().polish(self)

    def setSelected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        self.radio.setChecked(selected)
        self.style().unpolish(self)
        self.style().polish(self)


class ModelPicker(QWidget):
    selectionChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cards: dict[str, ModelCard] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for name, dl_mb, ram_gb in MODELS:
            card = ModelCard(name, dl_mb, ram_gb)
            card.clicked.connect(self._on_card_clicked)
            self._group.addButton(card.radio)
            layout.addWidget(card)
            self.cards[name] = card

    def _on_card_clicked(self, name: str) -> None:
        self.select(name)

    def select(self, name: str) -> None:
        if name not in self.cards:
            return
        for n, card in self.cards.items():
            card.setSelected(n == name)
        self.selectionChanged.emit(name)

    def selected(self) -> str | None:
        for n, card in self.cards.items():
            if card.property("selected") == "true":
                return n
        return None

    def refresh(self) -> None:
        for card in self.cards.values():
            card.refresh_status()


class SegmentedControl(QWidget):
    """Two- or three-segment toggle. Reads as a single visual unit, not
    as scattered radio buttons. Used for Source / Trigger choices in
    the Record panel where the radios were too inconspicuous."""

    valueChanged = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], default: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for i, (value, label) in enumerate(options):
            btn = QPushButton(label)
            btn.setObjectName("SegmentBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if i == 0:
                pos = "left"
            elif i == len(options) - 1:
                pos = "right"
            else:
                pos = "middle"
            btn.setProperty("position", pos)
            btn.clicked.connect(lambda _checked, v=value: self.set_value(v))
            self._group.addButton(btn)
            layout.addWidget(btn)
            self._buttons[value] = btn
        layout.addStretch(1)
        if default and default in self._buttons:
            self.set_value(default, emit=False)
        else:
            first = next(iter(self._buttons))
            self.set_value(first, emit=False)

    def set_value(self, value: str, emit: bool = True) -> None:
        if value not in self._buttons:
            return
        for v, btn in self._buttons.items():
            btn.setChecked(v == value)
        if emit:
            self.valueChanged.emit(value)

    def value(self) -> str:
        for v, btn in self._buttons.items():
            if btn.isChecked():
                return v
        return ""


class RecordPanel(QWidget):
    """Live recording into a temp WAV via ffmpeg + PulseAudio.

    Two modes:
    - Click to start / Click to stop  (default — toggle on click)
    - Push and hold                   (record only while button is held)

    Two sources:
    - Microphone (PulseAudio default source)
    - System audio (default sink's monitor — captures whatever is playing)

    On stop, emits `fileRecorded(path)` with the WAV path so the main
    window can drop it into the same pipeline as a picked file.
    """

    fileRecorded = pyqtSignal(str)
    statusMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._record_path: str | None = None
        self._record_started_at: float = 0.0
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(200)
        self._tick_timer.timeout.connect(self._update_duration)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Source toggle
        src_row = QHBoxLayout()
        src_lbl = QLabel("Source:")
        src_lbl.setMinimumWidth(70)
        src_row.addWidget(src_lbl)
        self.source_toggle = SegmentedControl(
            [("mic", "Microphone"), ("sys", "System audio")],
            default="mic",
        )
        src_row.addWidget(self.source_toggle)
        src_row.addStretch(1)
        layout.addLayout(src_row)

        # Trigger toggle
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Trigger:")
        mode_lbl.setMinimumWidth(70)
        mode_row.addWidget(mode_lbl)
        self.mode_toggle = SegmentedControl(
            [("click", "Click to start / stop"), ("push", "Push and hold")],
            default="click",
        )
        mode_row.addWidget(self.mode_toggle)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Big record button
        self.record_btn = QPushButton()
        self.record_btn.setObjectName("RecordButton")
        self.record_btn.setMinimumHeight(64)
        self.record_btn.setProperty("recording", "false")
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.clicked.connect(self._on_clicked)
        self.record_btn.pressed.connect(self._on_pressed)
        self.record_btn.released.connect(self._on_released)
        layout.addWidget(self.record_btn)

        self.duration_lbl = QLabel("")
        self.duration_lbl.setProperty("role", "muted")
        self.duration_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.duration_lbl)

        if not shutil.which("ffmpeg"):
            self.record_btn.setEnabled(False)
            self.record_btn.setToolTip("ffmpeg not found — recording disabled")
        elif not shutil.which("pactl"):
            self.record_btn.setToolTip(
                "pactl not found — recording will use the default Pulse source"
            )

        self.mode_toggle.valueChanged.connect(lambda _: self._refresh_button_text())
        self._refresh_button_text()

    # --- public ---------------------------------------------------------
    def is_recording(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def stop_if_recording(self) -> None:
        if self.is_recording():
            self._stop_recording()

    def _is_push_mode(self) -> bool:
        return self.mode_toggle.value() == "push"

    def _is_mic(self) -> bool:
        return self.source_toggle.value() == "mic"

    # --- handlers -------------------------------------------------------
    def _on_clicked(self) -> None:
        # `clicked` fires after `released`; in push-and-hold mode the
        # release already stopped the recording — ignore.
        if self._is_push_mode():
            return
        if self.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _on_pressed(self) -> None:
        if self._is_push_mode() and not self.is_recording():
            self._start_recording()

    def _on_released(self) -> None:
        if self._is_push_mode() and self.is_recording():
            self._stop_recording()

    # --- recording ------------------------------------------------------
    def _resolve_source(self) -> str:
        if self._is_mic():
            return "default"
        # System audio: ask pactl for the default sink and use its monitor.
        if shutil.which("pactl"):
            try:
                out = subprocess.run(
                    ["pactl", "info"], capture_output=True, text=True, timeout=2,
                ).stdout
                for line in out.splitlines():
                    if line.startswith("Default Sink:"):
                        sink = line.split(":", 1)[1].strip()
                        if sink:
                            return f"{sink}.monitor"
            except Exception:
                pass
        return "default"  # best-effort fallback

    def _start_recording(self) -> None:
        source = self._resolve_source()
        path = str(RUNTIME_DIR / f"recording-{int(time.time())}.wav")
        self._record_path = path

        self._proc = QProcess(self)
        self._proc.setProgram("ffmpeg")
        self._proc.setArguments([
            "-loglevel", "error",
            "-f", "pulse", "-i", source,
            "-ar", "16000", "-ac", "1",
            "-y", path,
        ])
        self._proc.finished.connect(self._on_proc_finished)
        self._proc.start()
        if not self._proc.waitForStarted(3000):
            err = self._proc.errorString()
            self._proc = None
            self._record_path = None
            self.statusMessage.emit(f"Recording failed to start: {err}")
            return
        self._record_started_at = time.monotonic()
        self._tick_timer.start()
        self._refresh_button_text()
        self.statusMessage.emit(
            f"Recording from {'microphone' if self._is_mic() else 'system audio'}…"
        )

    def _stop_recording(self) -> None:
        if not self._proc:
            return
        # ffmpeg quits cleanly on 'q' on stdin and writes a valid file;
        # SIGTERM also works but may truncate.
        try:
            self._proc.write(b"q\n")
            self._proc.closeWriteChannel()
        except Exception:
            self._proc.terminate()
        if not self._proc.waitForFinished(3000):
            self._proc.kill()
            self._proc.waitForFinished(1000)

    def _on_proc_finished(self, _exit_code, _exit_status) -> None:
        self._tick_timer.stop()
        path = self._record_path
        self._record_path = None
        proc = self._proc
        self._proc = None
        self._refresh_button_text()
        if not path or not Path(path).exists() or Path(path).stat().st_size < 1024:
            self.statusMessage.emit("Recording produced no audio.")
            return
        secs = time.monotonic() - self._record_started_at
        self.duration_lbl.setText(f"Recorded {secs:.1f} s → {Path(path).name}")
        self.fileRecorded.emit(path)
        self.statusMessage.emit(f"Recorded {secs:.1f} s")

    # --- ui -------------------------------------------------------------
    def _refresh_button_text(self) -> None:
        recording = self.is_recording()
        self.record_btn.setProperty("recording", "true" if recording else "false")
        if recording:
            self.record_btn.setText("⏹  Stop recording")
        elif self._is_push_mode():
            self.record_btn.setText("⏺  Push and hold to record")
        else:
            self.record_btn.setText("⏺  Click to record")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)

    def _update_duration(self) -> None:
        if not self.is_recording():
            return
        secs = time.monotonic() - self._record_started_at
        m, s = divmod(int(secs), 60)
        self.duration_lbl.setText(f"⏺  {m:02d}:{s:02d}")


class ModelPickerDialog(QDialog):
    """Modal popup that lets the user pick a model.

    Why a popup instead of an inline collapsible section: when the
    inline section opens it expands the main window, but on close the
    main window does not shrink back — Qt distributes the freed space
    to the Result text area instead. Wrapping the picker in a modal
    leaves the main window untouched.
    """

    def __init__(self, current_model: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose model")
        self.setModal(True)
        self.resize(640, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(QLabel(
            "Pick a Whisper model. Larger models are more accurate "
            "but need more RAM and time."
        ))

        self.picker = ModelPicker()
        if current_model:
            self.picker.select(current_model)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.picker)
        layout.addWidget(scroll, 1)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def selected(self) -> str | None:
        return self.picker.selected()


# --- Settings dialog ----------------------------------------------------

class SettingsDialog(QDialog):
    modelsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Models")
        self.resize(600, 440)
        self._download_worker: DownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Models cache: {WHISPER_CACHE}\n"
            "Download or delete Whisper models. Sizes shown are approximate."
        ))

        self.table = QTableWidget(len(MODELS), 4)
        self.table.setHorizontalHeaderLabels(["Model", "Status", "Size", "Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.table)

        self.status = QLabel("")
        layout.addWidget(self.status)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

        self.refresh()

    def refresh(self) -> None:
        for row, (name, dl_mb, ram_gb) in enumerate(MODELS):
            downloaded = is_model_downloaded(name)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("✓ downloaded" if downloaded else "not downloaded"))
            size_text = human_size(model_disk_size(name)) if downloaded else f"~{dl_mb} MB · ~{ram_gb:.0f} GB RAM"
            self.table.setItem(row, 2, QTableWidgetItem(size_text))

            btn = QPushButton("Delete" if downloaded else "Download")
            if downloaded:
                btn.clicked.connect(lambda _, n=name: self.delete_model(n))
            else:
                btn.clicked.connect(lambda _, n=name: self.download_model(n))
            self.table.setCellWidget(row, 3, btn)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def delete_model(self, name: str) -> None:
        path = model_cache_path(name)
        if not path.exists():
            return
        ok = QMessageBox.question(
            self, "Delete model",
            f"Delete cached model {name} ({human_size(path.stat().st_size)})?",
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
            self.status.setText(f"Deleted {name}.")
            self.modelsChanged.emit()
        except Exception as e:
            QMessageBox.critical(self, "Delete failed", str(e))
        self.refresh()

    def download_model(self, name: str) -> None:
        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, "Busy", "A download is already running.")
            return
        self.status.setText(f"Downloading {name}…")
        self._download_worker = DownloadWorker(name)
        self._download_worker.progress.connect(self.status.setText)
        self._download_worker.finished_ok.connect(self._on_download_done)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_done(self, name: str) -> None:
        self.status.setText(f"Downloaded {name}.")
        self.modelsChanged.emit()
        self.refresh()

    def _on_download_failed(self, msg: str) -> None:
        self.status.setText("Download failed.")
        QMessageBox.critical(self, "Download failed", msg)


# --- Main window --------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = load_settings()
        # Restore last window size if remembered, else use a sensible default
        # for a fresh install. Position is left to the compositor.
        saved = self.settings.get("window_size")
        if (
            isinstance(saved, list)
            and len(saved) == 2
            and all(isinstance(v, int) and 320 <= v <= 4000 for v in saved)
        ):
            self.resize(saved[0], saved[1])
        else:
            self.resize(720, 820)
        self.current_file: str | None = None
        self._file_queue: list[str] = []
        self._batch_index: int = 0
        self.last_result: dict | None = None
        self._current_model: str | None = None
        self.worker = TranscribeProcess(self)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.progress_pct.connect(self._on_worker_progress_pct)
        self.worker.finished_ok.connect(self.on_transcribe_done)
        self.worker.failed.connect(self.on_transcribe_failed)
        self.worker.preload_done.connect(self._on_preload_done)
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setInterval(200)
        self._estimate_timer.timeout.connect(self._tick_progress_estimate)
        self._estimate_total = 0.0
        self._estimate_started = 0.0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # System info bar (live) — kept bright like terminal text
        self.sys_info = QLabel()
        f = self.sys_info.font()
        f.setBold(True)
        self.sys_info.setFont(f)
        root.addWidget(self.sys_info)
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._refresh_sys_info)
        self._sys_timer.start(2000)
        self._refresh_sys_info()

        # Source tabs: File (drop / open) vs Record (live capture)
        self.source_tabs = QTabWidget()

        file_tab = QWidget()
        fbl = QVBoxLayout(file_tab)
        fbl.setContentsMargins(10, 12, 10, 10)
        self.drop = DropLabel()
        self.drop.filesDropped.connect(self.set_files)
        fbl.addWidget(self.drop)
        path_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No file selected")
        self.file_edit.setReadOnly(True)
        open_btn = QPushButton("Open Files…")
        open_btn.clicked.connect(self.pick_files)
        path_row.addWidget(self.file_edit, 1)
        path_row.addWidget(open_btn)
        fbl.addLayout(path_row)
        self.source_tabs.addTab(file_tab, "File")

        record_tab = QWidget()
        rec_layout = QVBoxLayout(record_tab)
        rec_layout.setContentsMargins(10, 12, 10, 10)
        self.record_panel = RecordPanel()
        self.record_panel.fileRecorded.connect(self._on_recording_finished)
        self.record_panel.statusMessage.connect(self.statusBar().showMessage)
        rec_layout.addWidget(self.record_panel)
        self.source_tabs.addTab(record_tab, "Record")

        root.addWidget(self.source_tabs)

        # Model — single full-width button summarising the current pick.
        # Click opens a modal popup; main window stays the same size.
        self.model_btn = QPushButton()
        self.model_btn.setObjectName("ModelChooser")
        self.model_btn.setMinimumHeight(48)
        self.model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_btn.clicked.connect(self.open_model_picker)
        root.addWidget(self.model_btn)

        self.model_warning = QLabel("")
        self.model_warning.setProperty("role", "warning")
        self.model_warning.setContentsMargins(4, 0, 4, 0)
        root.addWidget(self.model_warning)

        # Resources / language. Sliders + N/M label so a full picture
        # fits on a single line — spinboxes hid the available range.
        res_box = QGroupBox("Resources")
        form = QFormLayout(res_box)
        cpus = cpu_count()
        ram_total_int = max(1, int(total_ram_gb()))

        self.cpu_slider = QSlider(Qt.Orientation.Horizontal)
        self.cpu_slider.setRange(1, cpus)
        default_cpu = self.settings.get("cpu_threads", max(1, cpus // 2))
        self.cpu_slider.setValue(min(default_cpu, cpus))
        self.cpu_slider.setSingleStep(1)
        self.cpu_slider.setPageStep(1)
        self.cpu_slider.setMinimumWidth(180)
        self.cpu_slider.setToolTip(f"1 – {cpus} CPU threads available")
        # Force transparent background even when the global QWidget rule
        # tries to paint over us.
        self.cpu_slider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.cpu_value_lbl = QLabel()
        self.cpu_value_lbl.setMinimumWidth(80)
        self.cpu_slider.valueChanged.connect(self._refresh_resource_labels)
        cpu_row = QHBoxLayout()
        cpu_row.setContentsMargins(0, 0, 0, 0)
        cpu_row.setSpacing(10)
        cpu_row.addWidget(self.cpu_slider, 1)
        cpu_row.addWidget(self.cpu_value_lbl)
        cpu_wrap = QWidget()
        cpu_wrap.setLayout(cpu_row)
        form.addRow(QLabel("CPU threads:"), cpu_wrap)

        # Hard RAM cap. Checkbox toggles, slider sets the value.
        self.ram_cap_check = QCheckBox("Hard RAM cap")
        cap_tip = (
            "Run the transcription worker inside a systemd cgroup with a "
            "memory ceiling. If it tries to use more than the cap, the "
            "kernel kills the worker process (the GUI shows an OOM error) "
            "instead of letting it swap or eat the whole machine.\n\n"
            "Use it when running heavy models you don't want monopolising "
            "RAM. Don't use it casually — picking a cap below the model's "
            "actual need (e.g. cap=4 GB with large-v3) just causes an OOM "
            "right after model load."
        )
        self.ram_cap_check.setToolTip(cap_tip)
        if not _systemd_run_available():
            self.ram_cap_check.setEnabled(False)
            self.ram_cap_check.setToolTip(
                "systemd-run not found on PATH — hard cap unavailable."
            )
        self.ram_cap_check.setChecked(bool(self.settings.get("ram_cap_enabled", False)))

        self.ram_cap_slider = QSlider(Qt.Orientation.Horizontal)
        self.ram_cap_slider.setRange(1, ram_total_int)
        self.ram_cap_slider.setValue(
            min(self.settings.get("ram_cap_gb", max(2, ram_total_int // 2)), ram_total_int)
        )
        self.ram_cap_slider.setMinimumWidth(180)
        self.ram_cap_slider.setEnabled(self.ram_cap_check.isChecked())
        self.ram_cap_slider.setToolTip(cap_tip)
        self.ram_cap_slider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.ram_cap_value_lbl = QLabel()
        self.ram_cap_value_lbl.setMinimumWidth(80)
        self.ram_cap_check.toggled.connect(self.ram_cap_slider.setEnabled)
        self.ram_cap_check.toggled.connect(lambda _: self._refresh_resource_labels())
        self.ram_cap_slider.valueChanged.connect(self._refresh_resource_labels)
        ram_row = QHBoxLayout()
        ram_row.setContentsMargins(0, 0, 0, 0)
        ram_row.setSpacing(10)
        ram_row.addWidget(self.ram_cap_check)
        ram_row.addWidget(self.ram_cap_slider, 1)
        ram_row.addWidget(self.ram_cap_value_lbl)
        ram_widget = QWidget()
        ram_widget.setLayout(ram_row)
        form.addRow(ram_widget)

        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("auto-detect (or e.g. en, ru, de)")
        self.lang_edit.setText(self.settings.get("language", ""))
        form.addRow(QLabel("Language:"), self.lang_edit)

        self._cpu_max = cpus
        self._ram_max = ram_total_int
        self._refresh_resource_labels()

        root.addWidget(res_box)

        # Action buttons
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start transcription")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.clicked.connect(self.start_transcription)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        # Progress: phase text lives on a label above the bar so it is
        # always readable; the bar itself only shows the percentage so
        # text and chunk fill never compete for legibility.
        self.progress_phase_lbl = QLabel("")
        self.progress_phase_lbl.setProperty("role", "muted")
        self.progress_phase_lbl.setVisible(False)
        root.addWidget(self.progress_phase_lbl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Result tabs — one per transcription so a batch produces a tab
        # per file; the user can flip between them and copy/save each.
        result_box = QGroupBox("Result")
        rbl = QVBoxLayout(result_box)
        self.result_tabs = QTabWidget()
        self.result_tabs.setTabsClosable(True)
        self.result_tabs.tabCloseRequested.connect(
            lambda i: self.result_tabs.removeTab(i)
        )
        self.result_tabs.setDocumentMode(True)
        self._tab_results: list[dict] = []   # parallel to tab index
        self._tab_sources: list[str] = []    # source path per tab
        self._add_placeholder_tab()
        rbl.addWidget(self.result_tabs)
        out_btns = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to clipboard")
        self.copy_btn.clicked.connect(self.copy_result)
        self.copy_btn.setEnabled(False)
        self.save_btn = QPushButton("Save to file…")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        out_btns.addWidget(self.copy_btn)
        out_btns.addWidget(self.save_btn)
        out_btns.addStretch(1)
        rbl.addLayout(out_btns)
        root.addWidget(result_box, 1)

        # Menu
        settings_act = QAction("&Settings…", self)
        settings_act.triggered.connect(self.open_settings)
        about_act = QAction("&About", self)
        about_act.triggered.connect(self.show_about)
        m_app = self.menuBar().addMenu("&App")
        m_app.addAction(settings_act)
        m_app.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        m_app.addAction(quit_act)
        m_help = self.menuBar().addMenu("&Help")
        m_help.addAction(about_act)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        # Initial selection: prefer settings, then a downloaded model, then "base"
        last = self.settings.get("model")
        valid = {n for n, *_ in MODELS}
        candidates = ([last] if last else []) + [
            n for n, *_ in MODELS if is_model_downloaded(n)
        ] + ["base"]
        for c in candidates:
            if c and c in valid:
                self._set_current_model(c)
                break

        if initial_file:
            self.set_file(initial_file)

    # --- system info ----------------------------------------------------
    def _refresh_sys_info(self) -> None:
        total = total_ram_gb()
        avail = available_ram_gb()
        self.sys_info.setText(
            f"System · RAM {avail:.1f} GB available of {total:.1f} GB · {cpu_count()} CPU cores"
        )
        if self._current_model:
            self._update_model_warning(self._current_model)

    # --- result tabs ----------------------------------------------------
    def _add_placeholder_tab(self) -> None:
        if self.result_tabs.count() != 0:
            return
        edit = QTextEdit()
        edit.setPlaceholderText("Transcription will appear here.")
        edit.setReadOnly(True)
        self.result_tabs.addTab(edit, "Result")
        # Hide the close button on the placeholder.
        self.result_tabs.tabBar().setTabButton(0, self.result_tabs.tabBar().ButtonPosition.RightSide, None)
        self._tab_results.append({})
        self._tab_sources.append("")

    def _add_result_tab(self, source_path: str, result: dict) -> None:
        # Drop the placeholder tab once we have a real result.
        if self.result_tabs.count() == 1 and not self._tab_results[0]:
            self.result_tabs.removeTab(0)
            self._tab_results.pop(0)
            self._tab_sources.pop(0)
        edit = QTextEdit()
        edit.setReadOnly(False)
        edit.setPlainText(result.get("text", "").strip())
        title = Path(source_path).name if source_path else "Result"
        idx = self.result_tabs.addTab(edit, title)
        self.result_tabs.setCurrentIndex(idx)
        self._tab_results.append(result)
        self._tab_sources.append(source_path)
        # When a tab is closed, drop our parallel state too.
        self.result_tabs.tabCloseRequested.connect(self._on_tab_close, Qt.ConnectionType.UniqueConnection)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _on_tab_close(self, index: int) -> None:
        # Already removed by the lambda above; sync our parallel arrays.
        if 0 <= index < len(self._tab_results):
            self._tab_results.pop(index)
            self._tab_sources.pop(index)
        if self.result_tabs.count() == 0:
            self.copy_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self._add_placeholder_tab()

    def _current_result(self) -> tuple[dict, str, QTextEdit | None]:
        idx = self.result_tabs.currentIndex()
        if idx < 0:
            return ({}, "", None)
        result = self._tab_results[idx] if idx < len(self._tab_results) else {}
        source = self._tab_sources[idx] if idx < len(self._tab_sources) else ""
        widget = self.result_tabs.widget(idx)
        return (result, source, widget if isinstance(widget, QTextEdit) else None)

    # --- resources ------------------------------------------------------
    def _refresh_resource_labels(self) -> None:
        self.cpu_value_lbl.setText(f"{self.cpu_slider.value()}/{self._cpu_max} threads")
        if self.ram_cap_check.isChecked():
            self.ram_cap_value_lbl.setText(
                f"{self.ram_cap_slider.value()}/{self._ram_max} GB"
            )
        else:
            self.ram_cap_value_lbl.setText(f"off · max {self._ram_max} GB")

    def _set_current_model(self, name: str, *, preload: bool = True) -> None:
        self._current_model = name
        self._update_model_button()
        self._update_model_warning(name)
        # Kick off a background load so the next Start has the model
        # already in RAM. Skipped only when explicitly disabled (e.g.
        # initial selection during boot before the worker is wired).
        if preload and is_model_downloaded(name) and not self.worker.is_running():
            ram_cap_gb = self._current_ram_cap_gb()
            self.worker.preload(name, self.cpu_slider.value(), ram_cap_gb)

    def _current_ram_cap_gb(self) -> int:
        if self.ram_cap_check.isChecked() and _systemd_run_available():
            return self.ram_cap_slider.value()
        return 0

    def _update_model_button(self) -> None:
        name = self._current_model
        if not name:
            self.model_btn.setText("Choose a model…")
            return
        ram = next((r for n, _, r in MODELS if n == name), 0.0)
        if is_model_downloaded(name):
            loaded_marker = "  · loaded" if self.worker._loaded_model == name else ""
            status = "✓ downloaded" + loaded_marker
            tail = f"~{ram:.0f} GB RAM"
        else:
            dl = next((d for n, d, _ in MODELS if n == name), 0)
            status = "not downloaded"
            tail = f"~{dl} MB to fetch · ~{ram:.0f} GB RAM"
        self.model_btn.setText(f"  Model:  {name}    {status}    ·    {tail}")

    def _on_preload_done(self, name: str) -> None:
        if name == self._current_model:
            self.statusBar().showMessage(f"Model {name} ready")
            self._update_model_button()

    def open_model_picker(self) -> None:
        dlg = ModelPickerDialog(self._current_model, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            picked = dlg.selected()
            if picked and picked in {n for n, *_ in MODELS}:
                self._set_current_model(picked)

    def _update_model_warning(self, name: str) -> None:
        # Skip while a transcription is running — the message is only
        # actionable before pressing Start.
        if self.worker.is_running():
            return
        ram = next((r for n, _, r in MODELS if n == name), 0.0)
        avail = available_ram_gb()
        if ram > avail:
            self.model_warning.setText(
                f"⚠ {name} typically needs ~{ram:.0f} GB RAM but only {avail:.1f} GB is available."
            )
        elif ram > avail - 1:
            self.model_warning.setText(
                f"ℹ {name} needs ~{ram:.0f} GB RAM; close other apps for headroom."
            )
        else:
            self.model_warning.setText("")
        self.model_warning.setVisible(bool(self.model_warning.text()))

    # --- file selection -------------------------------------------------
    def pick_files(self) -> None:
        last_folder = self.settings.get("last_folder")
        start_dir = last_folder if last_folder and Path(last_folder).is_dir() else str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select audio or video files (Ctrl/Shift to multi-select)", start_dir,
            "Media (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.mp4 *.mkv *.webm *.mov *.avi);;All files (*)",
        )
        if paths:
            self.set_files(paths)

    def set_files(self, paths: list[str]) -> None:
        """Accept one or more files. Single → behaves as before. Many →
        queued for sequential transcription (each result is auto-saved
        as <basename>.txt next to its source)."""
        valid: list[str] = []
        unknown_asked = False
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            if p.suffix.lower() not in AUDIO_EXTS:
                if not unknown_asked:
                    ok = QMessageBox.question(
                        self, "Unknown extension(s)",
                        "Some files have unrecognised audio/video extensions. "
                        "Include them anyway?",
                    )
                    unknown_asked = True
                    if ok != QMessageBox.StandardButton.Yes:
                        continue
                # If user already said yes once, accept the rest with same ext.
            valid.append(str(p))
        if not valid:
            QMessageBox.warning(self, "No usable files", "None of the dropped paths exist.")
            return
        self._file_queue = valid
        self.current_file = valid[0]
        self.last_result = None
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        if len(valid) == 1:
            self.file_edit.setText(self.current_file)
            self.drop.setText(Path(self.current_file).name)
        else:
            head = ", ".join(Path(p).name for p in valid[:3])
            more = "" if len(valid) <= 3 else f" + {len(valid) - 3} more"
            self.file_edit.setText(f"{len(valid)} files: {head}{more}")
            self.drop.setText(f"{len(valid)} files queued")
        self.drop.setHasFile(True)
        self.settings["last_folder"] = str(Path(valid[0]).parent)
        save_settings(self.settings)

    def set_file(self, path: str) -> None:
        # Backwards-compatible single-file entry point used by the
        # recording panel.
        self.set_files([path])

    def _on_recording_finished(self, path: str) -> None:
        # Wire recording into the same pipeline as a picked file. Stay
        # on the Record tab so the user can keep recording back-to-back.
        self.set_files([path])

    # --- transcription --------------------------------------------------
    def start_transcription(self) -> None:
        if not self._file_queue:
            QMessageBox.information(self, "No file", "Pick a file first.")
            return
        model_name = self._current_model
        if not model_name:
            QMessageBox.information(self, "No model", "Pick a model first.")
            return
        cpu_threads = self.cpu_slider.value()
        language = self.lang_edit.text().strip() or None
        ram_cap_enabled = self.ram_cap_check.isChecked() and _systemd_run_available()
        ram_cap_gb = self.ram_cap_slider.value() if ram_cap_enabled else 0

        # Pre-flight RAM check (advisory, not enforced)
        ram_need = next((r for n, _, r in MODELS if n == model_name), 0.0)
        avail = available_ram_gb()
        if ram_need > avail:
            ok = QMessageBox.warning(
                self, "Likely out of memory",
                f"Model {model_name} typically needs ~{ram_need:.0f} GB RAM, "
                f"but only {avail:.1f} GB is available. Whisper will probably "
                f"fail with an allocation error. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        if ram_cap_enabled and ram_cap_gb < ram_need:
            ok = QMessageBox.warning(
                self, "Cap below model need",
                f"You set the hard RAM cap to {ram_cap_gb} GB but {model_name} "
                f"needs ~{ram_need:.0f} GB. The kernel will OOM-kill the worker. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return

        self.settings.update({
            "model": model_name,
            "cpu_threads": cpu_threads,
            "language": language or "",
            "ram_cap_enabled": ram_cap_enabled,
            "ram_cap_gb": self.ram_cap_slider.value(),
        })
        save_settings(self.settings)

        # Cache batch parameters so each step uses the same model / cap
        # even if the user fiddles with the controls mid-batch.
        self._batch_index = 0
        self._batch_params = {
            "model": model_name,
            "cpu_threads": cpu_threads,
            "language": language,
            "ram_cap_gb": ram_cap_gb,
        }

        self.set_running(True)
        self.last_result = None

        self._run_next_in_batch()

    def _run_next_in_batch(self) -> None:
        if self._batch_index >= len(self._file_queue):
            return
        self.current_file = self._file_queue[self._batch_index]
        params = self._batch_params
        total = len(self._file_queue)
        if total > 1:
            self.statusBar().showMessage(
                f"File {self._batch_index + 1}/{total}: {Path(self.current_file).name}"
            )
        self.worker.transcribe(
            self.current_file,
            params["model"],
            params["cpu_threads"],
            params["language"],
            params["ram_cap_gb"],
        )

    def _on_worker_progress(self, msg: str) -> None:
        # Phase text lives on a label above the bar, never overlaid on
        # the chunk fill (where the colours conflict).
        self.statusBar().showMessage(msg)
        self._progress_phase = msg
        prefix = self._batch_prefix()
        self.progress_phase_lbl.setText(f"{prefix}{msg}".strip())
        # Reset bar at the start of each new phase. The transcribe phase
        # additionally kicks off a time-based estimator that smoothly
        # fills 0→99 % while whisper churns on its single clip.
        self.progress.setValue(0)
        if msg.startswith("Transcribing"):
            self._start_progress_estimate()
        else:
            self._estimate_timer.stop()

    def _on_worker_progress_pct(self, pct: int) -> None:
        pct = max(0, min(100, pct))
        # Real tqdm number wins if it's ahead of the time-based estimate.
        if pct > self.progress.value():
            self.progress.setValue(pct)

    def _batch_prefix(self) -> str:
        total = len(self._file_queue)
        if total > 1:
            return f"File {min(self._batch_index + 1, total)}/{total} · "
        return ""

    def _start_progress_estimate(self) -> None:
        duration = audio_duration_seconds(self.current_file or "")
        factor = MODEL_RT_FACTOR.get(
            (self._current_model or "").replace(".en", ""), 1.0
        )
        # Threads cut wall-clock roughly with sqrt(threads) before plateau.
        threads = max(1, self.cpu_slider.value())
        speedup = max(1.0, min(threads, 6) ** 0.6)
        self._estimate_total = max(2.0, duration * factor / speedup)
        self._estimate_started = time.monotonic()
        self._estimate_timer.start()

    def _tick_progress_estimate(self) -> None:
        if not self.worker.is_running():
            self._estimate_timer.stop()
            return
        elapsed = time.monotonic() - self._estimate_started
        pct = int(min(99, 100 * elapsed / max(0.5, self._estimate_total)))
        if pct > self.progress.value():
            self.progress.setValue(pct)
            phase = getattr(self, "_progress_phase", "")
            self.progress.setFormat(f"{phase}    %p%" if phase else "%p%")

    def cancel_transcription(self) -> None:
        if self.worker.is_running():
            self.statusBar().showMessage("Cancelling…")
            self.worker.cancel()
            self.statusBar().showMessage("Cancelled")
        # Drop any remaining files from the batch — user explicitly stopped.
        self._batch_index = len(self._file_queue)
        self.set_running(False)

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress.setVisible(running)
        self.progress_phase_lbl.setVisible(running)
        if running:
            self._progress_phase = "Starting…"
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_phase_lbl.setText(f"{self._batch_prefix()}Starting…")
            self.model_warning.setVisible(False)
        else:
            self._estimate_timer.stop()
            self.progress.setValue(0)
            self.progress_phase_lbl.setText("")
            self.model_warning.setVisible(bool(self.model_warning.text()))

    def on_transcribe_done(self, result: dict) -> None:
        self.last_result = result
        self._add_result_tab(self.current_file or "", result)
        self._update_model_button()

        total = len(self._file_queue)
        # Auto-save TXT next to source file — only meaningful for batch.
        if total > 1 and self.current_file:
            try:
                src = Path(self.current_file)
                txt_path = src.with_suffix(".txt")
                txt_path.write_text(result.get("text", "").strip(), encoding="utf-8")
            except Exception as e:
                self.statusBar().showMessage(
                    f"Saved transcription failed for {Path(self.current_file).name}: {e}"
                )

        self._batch_index += 1
        if self._batch_index < total:
            self._run_next_in_batch()
            return

        if total > 1:
            self.statusBar().showMessage(f"Done — {total} files transcribed")
        else:
            self.statusBar().showMessage("Done")
        self.set_running(False)

    def on_transcribe_failed(self, msg: str) -> None:
        self.set_running(False)
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Transcription failed", msg)
        # Stop the batch on failure — likely the same error will hit
        # the next file too (model load, OOM, etc.).
        self._batch_index = len(self._file_queue)

    # --- output ---------------------------------------------------------
    def copy_result(self) -> None:
        _result, _src, edit = self._current_result()
        if edit is None:
            return
        QGuiApplication.clipboard().setText(edit.toPlainText())
        self.statusBar().showMessage("Copied to clipboard")

    def save_result(self) -> None:
        result, source, edit = self._current_result()
        if not result or edit is None:
            return
        default = Path(source or "transcript").with_suffix(".txt").name
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcription", default,
            "Plain text (*.txt);;SubRip (*.srt);;WebVTT (*.vtt);;JSON (*.json)",
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        try:
            if ext == ".srt":
                Path(path).write_text(self._to_srt(result))
            elif ext == ".vtt":
                Path(path).write_text(self._to_vtt(result))
            elif ext == ".json":
                Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                Path(path).write_text(edit.toPlainText())
            self.statusBar().showMessage(f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    @staticmethod
    def _ts(seconds: float, sep: str = ",") -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s - int(s)) * 1000):03d}"

    def _to_srt(self, result: dict) -> str:
        lines: list[str] = []
        for i, seg in enumerate(result.get("segments", []), 1):
            lines.append(str(i))
            lines.append(f"{self._ts(seg['start'])} --> {self._ts(seg['end'])}")
            lines.append(seg["text"].strip())
            lines.append("")
        return "\n".join(lines)

    def _to_vtt(self, result: dict) -> str:
        lines = ["WEBVTT", ""]
        for seg in result.get("segments", []):
            lines.append(f"{self._ts(seg['start'], '.')} --> {self._ts(seg['end'], '.')}")
            lines.append(seg["text"].strip())
            lines.append("")
        return "\n".join(lines)

    # --- misc -----------------------------------------------------------
    def open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.modelsChanged.connect(self._update_model_button)
        dlg.exec()
        self._update_model_button()

    def show_about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"{APP_NAME}\n\nSimple GUI for OpenAI Whisper.\n"
            f"Models cache: {WHISPER_CACHE}\nConfig: {CONFIG_FILE}\n"
            f"Theme: {THEME_COLORS_PATH if THEME_COLORS_PATH.exists() else 'built-in default'}",
        )

    def closeEvent(self, event) -> None:
        # Remember last window size so the next launch comes up the same way.
        self.settings["window_size"] = [self.width(), self.height()]
        save_settings(self.settings)
        # Stop a recording in flight so we don't leak ffmpeg.
        try:
            self.record_panel.stop_if_recording()
        except Exception:
            pass
        # Tell the persistent worker to exit cleanly so the cached model
        # doesn't stay resident if the user just closed the window.
        try:
            self.worker.quit_worker()
        except Exception:
            pass
        super().closeEvent(event)


def _ensure_hyprland_floating() -> None:
    """Tell Hyprland to float windows of this app, if running under Hyprland.

    Registers a transient `windowrule` via `hyprctl keyword`. The rule
    persists for the current Hyprland session — fast, scoped, and
    doesn't touch the user's hyprland.conf. Re-applied on every launch
    so the rule survives Hyprland restarts.
    """
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return
    if not shutil.which("hyprctl"):
        return
    try:
        subprocess.run(
            [
                "hyprctl", "keyword", "windowrule",
                f"float on, match:class ^({APP_ID})$",
            ],
            timeout=2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
    _ensure_hyprland_floating()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)

    theme = load_theme()
    app.setStyleSheet(build_stylesheet(theme))

    initial = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(initial_file=initial)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
