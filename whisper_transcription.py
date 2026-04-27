#!/usr/bin/env python3
"""Whisper Transcription for Arch — simple PyQt6 GUI for OpenAI Whisper.

Features:
- Drag & drop or file picker for audio/video
- Model selector (tiny, base, small, medium, large-v3, turbo)
- CPU thread cap and RAM cap (RLIMIT_AS)
- Progress while transcribing (runs in a worker thread)
- Save result to .txt / .srt / .vtt or copy to clipboard
- Settings dialog: download/delete cached models
"""
from __future__ import annotations

import json
import multiprocessing
import os
import resource
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import QAction, QGuiApplication, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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

# (model_id, approx download MB, approx peak RAM)
MODELS: list[tuple[str, int, str]] = [
    ("tiny", 75, "~1 GB RAM"),
    ("tiny.en", 75, "~1 GB RAM"),
    ("base", 142, "~1 GB RAM"),
    ("base.en", 142, "~1 GB RAM"),
    ("small", 466, "~2 GB RAM"),
    ("small.en", 466, "~2 GB RAM"),
    ("medium", 1500, "~5 GB RAM"),
    ("medium.en", 1500, "~5 GB RAM"),
    ("large-v3", 2900, "~10 GB RAM"),
    ("turbo", 1500, "~6 GB RAM"),
]

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".aiff", ".mp4", ".mkv", ".webm", ".mov",
    ".avi", ".mpeg", ".mpg", ".3gp", ".ts",
}


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
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


# --- Worker threads -----------------------------------------------------

class TranscribeWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        file_path: str,
        model_name: str,
        cpu_threads: int,
        ram_limit_gb: int,
        language: str | None,
    ):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self.ram_limit_gb = ram_limit_gb
        self.language = language

    def run(self) -> None:
        try:
            if self.ram_limit_gb > 0:
                bytes_limit = self.ram_limit_gb * 1024**3
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
                except (ValueError, OSError) as e:
                    self.progress.emit(f"RAM limit not applied: {e}")

            self.progress.emit(f"Loading model {self.model_name}…")
            import torch
            torch.set_num_threads(max(1, self.cpu_threads))
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


# --- Settings dialog ----------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Models")
        self.resize(560, 420)
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
        for row, (name, dl_mb, ram) in enumerate(MODELS):
            downloaded = is_model_downloaded(name)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            status_text = "✓ downloaded" if downloaded else "not downloaded"
            self.table.setItem(row, 1, QTableWidgetItem(status_text))
            if downloaded:
                size_text = human_size(model_disk_size(name))
            else:
                size_text = f"~{dl_mb} MB · {ram}"
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
        self.refresh()

    def _on_download_failed(self, msg: str) -> None:
        self.status.setText("Download failed.")
        QMessageBox.critical(self, "Download failed", msg)


# --- Drop area ----------------------------------------------------------

class DropLabel(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drop an audio/video file here\nor click Open File…")
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px;"
            " padding: 24px; color: #888; }"
        )

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.fileDropped.emit(url.toLocalFile())
                return


# --- Main window --------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(820, 640)
        self.settings = load_settings()
        self.current_file: str | None = None
        self.last_result: dict | None = None
        self.worker: TranscribeWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # File row
        file_row = QHBoxLayout()
        self.drop = DropLabel()
        self.drop.fileDropped.connect(self.set_file)
        file_row.addWidget(self.drop, 1)
        root.addLayout(file_row)

        path_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No file selected")
        self.file_edit.setReadOnly(True)
        open_btn = QPushButton("Open File…")
        open_btn.clicked.connect(self.pick_file)
        path_row.addWidget(self.file_edit, 1)
        path_row.addWidget(open_btn)
        root.addLayout(path_row)

        # Options
        opts = QGroupBox("Options")
        opts_layout = QFormLayout(opts)

        self.model_combo = QComboBox()
        for name, dl_mb, ram in MODELS:
            label = f"{name}   ({ram}, ~{dl_mb} MB)"
            self.model_combo.addItem(label, userData=name)
        last_model = self.settings.get("model", "base")
        idx = self.model_combo.findData(last_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        opts_layout.addRow("Model:", self.model_combo)

        cpus = max(1, multiprocessing.cpu_count())
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, cpus)
        self.cpu_spin.setValue(self.settings.get("cpu_threads", max(1, cpus // 2)))
        opts_layout.addRow(f"CPU threads (1–{cpus}):", self.cpu_spin)

        self.ram_spin = QSpinBox()
        self.ram_spin.setRange(0, 256)
        self.ram_spin.setSuffix(" GB")
        self.ram_spin.setSpecialValueText("Unlimited")
        self.ram_spin.setValue(self.settings.get("ram_gb", 0))
        opts_layout.addRow("RAM cap (0 = off):", self.ram_spin)

        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("auto-detect (or e.g. en, ru, de)")
        self.lang_edit.setText(self.settings.get("language", ""))
        opts_layout.addRow("Language:", self.lang_edit)

        root.addWidget(opts)

        # Buttons
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start transcription")
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
        root.addWidget(QLabel("Result:"))
        self.result_edit = QTextEdit()
        self.result_edit.setPlaceholderText("Transcription will appear here.")
        root.addWidget(self.result_edit, 1)

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
        root.addLayout(out_btns)

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

        if initial_file:
            self.set_file(initial_file)

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

    # --- transcription --------------------------------------------------
    def start_transcription(self) -> None:
        if not self.current_file:
            QMessageBox.information(self, "No file", "Pick a file first.")
            return
        model_name = self.model_combo.currentData()
        cpu_threads = self.cpu_spin.value()
        ram_gb = self.ram_spin.value()
        language = self.lang_edit.text().strip() or None

        self.settings.update({
            "model": model_name,
            "cpu_threads": cpu_threads,
            "ram_gb": ram_gb,
            "language": language or "",
        })
        save_settings(self.settings)

        self.set_running(True)
        self.result_edit.clear()
        self.last_result = None
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.worker = TranscribeWorker(
            self.current_file, model_name, cpu_threads, ram_gb, language,
        )
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

    def on_transcribe_failed(self, msg: str) -> None:
        self.set_running(False)
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Transcription failed", msg)

    # --- output ---------------------------------------------------------
    def copy_result(self) -> None:
        text = self.result_edit.toPlainText()
        QGuiApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied to clipboard")

    def save_result(self) -> None:
        if not self.last_result:
            return
        default = Path(self.current_file or "transcript").with_suffix(".txt").name
        path, selected = QFileDialog.getSaveFileName(
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
        lines = []
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
        dlg.exec()

    def show_about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"{APP_NAME}\n\nSimple GUI for OpenAI Whisper.\n"
            f"Models cache: {WHISPER_CACHE}\nConfig: {CONFIG_FILE}",
        )


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)

    initial = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(initial_file=initial)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
