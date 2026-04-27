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
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
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
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
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


def build_stylesheet(t: dict) -> str:
    bg = t["background"]
    fg = t["foreground"]
    accent = t.get("accent") or t.get("color4", "#7aa2f7")
    sel_fg = t.get("selection_foreground", bg)
    border = t.get("color8", "#444b6a")
    surface = t.get("color0", "#32344a")
    danger = t.get("color1", "#f7768e")
    return f"""
    QMainWindow, QDialog, QWidget {{
        background: {bg};
        color: {fg};
        font-size: 10pt;
    }}
    QLabel {{ background: transparent; color: {fg}; }}
    QLabel[role="muted"] {{ color: {border}; }}
    QLabel[role="warning"] {{ color: {danger}; }}
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
    QPushButton {{
        background: {surface};
        color: {fg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ background: {accent}; color: {sel_fg}; border-color: {accent}; }}
    QPushButton:pressed {{ background: {accent}; color: {sel_fg}; }}
    QPushButton:disabled {{ color: {border}; border-color: {border}; background: {bg}; }}
    QPushButton[role="primary"] {{
        background: {accent}; color: {sel_fg}; border-color: {accent}; font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{ background: {fg}; color: {bg}; border-color: {fg}; }}
    QPushButton[role="primary"]:disabled {{ background: {surface}; color: {border}; border-color: {border}; }}
    QLineEdit, QTextEdit, QSpinBox {{
        background: {bg};
        color: {fg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 4px 6px;
        selection-background-color: {accent};
        selection-color: {sel_fg};
    }}
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{ border-color: {accent}; }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; }}
    QProgressBar {{
        background: {surface}; color: {fg};
        border: 1px solid {border}; border-radius: 6px;
        text-align: center; min-height: 18px;
    }}
    QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
    QStatusBar {{ background: {surface}; color: {fg}; }}
    QMenuBar {{ background: {bg}; color: {fg}; }}
    QMenuBar::item:selected {{ background: {accent}; color: {sel_fg}; }}
    QMenu {{ background: {surface}; color: {fg}; border: 1px solid {border}; }}
    QMenu::item:selected {{ background: {accent}; color: {sel_fg}; }}
    QHeaderView::section {{ background: {surface}; color: {fg}; border: 0; padding: 6px; }}
    QTableWidget {{ background: {bg}; color: {fg}; gridline-color: {border}; border: 1px solid {border}; border-radius: 6px; }}
    QScrollArea {{ border: 0; background: transparent; }}
    QRadioButton {{ background: transparent; color: {fg}; }}

    ModelCard {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 8px;
    }}
    ModelCard[selected="true"] {{
        border: 2px solid {accent};
        background: {bg};
    }}
    ModelCard[downloaded="true"] QLabel#status {{ color: {t.get("color2", "#9ece6a")}; }}
    ModelCard[downloaded="false"] QLabel#status {{ color: {border}; }}
    DropLabel {{
        border: 2px dashed {border};
        border-radius: 8px;
        color: {border};
        padding: 18px;
    }}
    DropLabel[hasFile="true"] {{
        border-color: {accent};
        color: {fg};
    }}
    """


# --- Worker threads -----------------------------------------------------

class TranscribeWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, model_name: str, cpu_threads: int, language: str | None):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self.language = language

    def run(self) -> None:
        try:
            self.progress.emit(f"Loading model {self.model_name}…")
            import torch
            torch.set_num_threads(max(1, self.cpu_threads))
            os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(self.cpu_threads))
            import whisper

            model = whisper.load_model(self.model_name, device="cpu")
            self.progress.emit("Transcribing… (this may take a while)")
            result = model.transcribe(
                self.file_path,
                language=self.language or None,
                fp16=False,
                verbose=False,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


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
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("DropLabel")
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("hasFile", "false")
        self.setText("Drop an audio/video file here  ·  or click Open File…")

    def setHasFile(self, has: bool) -> None:
        self.setProperty("hasFile", "true" if has else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.fileDropped.emit(url.toLocalFile())
                return


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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

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
        self.resize(880, 760)
        self.settings = load_settings()
        self.current_file: str | None = None
        self.last_result: dict | None = None
        self.worker: TranscribeWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # System info bar (live)
        self.sys_info = QLabel()
        self.sys_info.setProperty("role", "muted")
        root.addWidget(self.sys_info)
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._refresh_sys_info)
        self._sys_timer.start(2000)
        self._refresh_sys_info()

        # File group
        file_box = QGroupBox("File")
        fbl = QVBoxLayout(file_box)
        self.drop = DropLabel()
        self.drop.fileDropped.connect(self.set_file)
        fbl.addWidget(self.drop)

        path_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No file selected")
        self.file_edit.setReadOnly(True)
        open_btn = QPushButton("Open File…")
        open_btn.clicked.connect(self.pick_file)
        path_row.addWidget(self.file_edit, 1)
        path_row.addWidget(open_btn)
        fbl.addLayout(path_row)
        root.addWidget(file_box)

        # Model picker (always-expanded card list)
        model_box = QGroupBox("Model")
        mbl = QVBoxLayout(model_box)
        self.picker = ModelPicker()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.picker)
        scroll.setMinimumHeight(220)
        mbl.addWidget(scroll)
        self.model_warning = QLabel("")
        self.model_warning.setProperty("role", "warning")
        mbl.addWidget(self.model_warning)
        self.picker.selectionChanged.connect(self._on_model_changed)
        root.addWidget(model_box)

        # Resources / language
        res_box = QGroupBox("Resources")
        form = QFormLayout(res_box)
        cpus = cpu_count()
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, cpus)
        default_cpu = self.settings.get("cpu_threads", max(1, cpus // 2))
        self.cpu_spin.setValue(min(default_cpu, cpus))
        form.addRow(QLabel(f"CPU threads (1 – {cpus} available):"), self.cpu_spin)

        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("auto-detect (or e.g. en, ru, de)")
        self.lang_edit.setText(self.settings.get("language", ""))
        form.addRow(QLabel("Language:"), self.lang_edit)

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

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Result
        result_box = QGroupBox("Result")
        rbl = QVBoxLayout(result_box)
        self.result_edit = QTextEdit()
        self.result_edit.setPlaceholderText("Transcription will appear here.")
        rbl.addWidget(self.result_edit)
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
        candidates = (
            [last] if last else []
        ) + [n for n, *_ in MODELS if is_model_downloaded(n)] + ["base"]
        for c in candidates:
            if c and c in self.picker.cards:
                self.picker.select(c)
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
        # also re-check selected-model warning (available RAM moves)
        sel = self.picker.selected() if hasattr(self, "picker") else None
        if sel:
            self._update_model_warning(sel)

    def _on_model_changed(self, name: str) -> None:
        self._update_model_warning(name)

    def _update_model_warning(self, name: str) -> None:
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

    # --- file selection -------------------------------------------------
    def pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio or video file", str(Path.home()),
            "Media (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.mp4 *.mkv *.webm *.mov *.avi);;All files (*)",
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Not found", f"File does not exist:\n{path}")
            return
        if p.suffix.lower() not in AUDIO_EXTS:
            ok = QMessageBox.question(
                self, "Unknown extension",
                f"{p.suffix} is not a recognized audio/video extension. Continue?",
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        self.current_file = str(p)
        self.file_edit.setText(self.current_file)
        self.drop.setText(p.name)
        self.drop.setHasFile(True)

    # --- transcription --------------------------------------------------
    def start_transcription(self) -> None:
        if not self.current_file:
            QMessageBox.information(self, "No file", "Pick a file first.")
            return
        model_name = self.picker.selected()
        if not model_name:
            QMessageBox.information(self, "No model", "Pick a model first.")
            return
        cpu_threads = self.cpu_spin.value()
        language = self.lang_edit.text().strip() or None

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

        self.settings.update({
            "model": model_name,
            "cpu_threads": cpu_threads,
            "language": language or "",
        })
        save_settings(self.settings)

        self.set_running(True)
        self.result_edit.clear()
        self.last_result = None
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.worker = TranscribeWorker(self.current_file, model_name, cpu_threads, language)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished_ok.connect(self.on_transcribe_done)
        self.worker.failed.connect(self.on_transcribe_failed)
        self.worker.start()

    def cancel_transcription(self) -> None:
        if self.worker and self.worker.isRunning():
            # whisper.transcribe is not interruptible; we terminate the QThread.
            self.worker.terminate()
            self.worker.wait(2000)
            self.statusBar().showMessage("Cancelled")
        self.set_running(False)

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress.setVisible(running)

    def on_transcribe_done(self, result: dict) -> None:
        self.last_result = result
        self.result_edit.setPlainText(result.get("text", "").strip())
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.statusBar().showMessage("Done")
        self.set_running(False)
        self.picker.refresh()  # in case the model just got downloaded mid-run

    def on_transcribe_failed(self, msg: str) -> None:
        self.set_running(False)
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Transcription failed", msg)

    # --- output ---------------------------------------------------------
    def copy_result(self) -> None:
        QGuiApplication.clipboard().setText(self.result_edit.toPlainText())
        self.statusBar().showMessage("Copied to clipboard")

    def save_result(self) -> None:
        if not self.last_result:
            return
        default = Path(self.current_file or "transcript").with_suffix(".txt").name
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcription", default,
            "Plain text (*.txt);;SubRip (*.srt);;WebVTT (*.vtt);;JSON (*.json)",
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        try:
            if ext == ".srt":
                Path(path).write_text(self._to_srt(self.last_result))
            elif ext == ".vtt":
                Path(path).write_text(self._to_vtt(self.last_result))
            elif ext == ".json":
                Path(path).write_text(json.dumps(self.last_result, indent=2, ensure_ascii=False))
            else:
                Path(path).write_text(self.result_edit.toPlainText())
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
        dlg.modelsChanged.connect(self.picker.refresh)
        dlg.exec()
        self.picker.refresh()

    def show_about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"{APP_NAME}\n\nSimple GUI for OpenAI Whisper.\n"
            f"Models cache: {WHISPER_CACHE}\nConfig: {CONFIG_FILE}\n"
            f"Theme: {THEME_COLORS_PATH if THEME_COLORS_PATH.exists() else 'built-in default'}",
        )


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
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
