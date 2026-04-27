# Whisper Transcription for Arch

Simple PyQt6 GUI around [OpenAI Whisper](https://github.com/openai/whisper) for
transcribing audio and video on Arch Linux (Wayland / Hyprland friendly).
Integrates with [Walker](https://github.com/abenz1267/walker) and any other
launcher that reads `.desktop` files.

## Features

- Drag and drop a file, or pick one with a file dialog
- Model picker as a card list — every model shows its download status,
  disk size, and approximate RAM cost; the currently selected one is
  highlighted with the theme accent color
- Live system bar — total / available RAM and CPU count; pre-flight warning
  if the chosen model is unlikely to fit in available RAM
- CPU thread cap with the upper bound matching `nproc`
- Optional **hard RAM cap** via `systemd-run --user --scope -p MemoryMax=NG`
  (kernel OOM-kills the worker if exceeded — clean and instant)
- Transcription runs in an isolated subprocess, so **Cancel kills the
  worker process** instead of trying to interrupt PyTorch from the GUI
  thread (the previous QThread-based approach hung the app)
- Theme palette pulled from Omarchy
  (`~/.config/omarchy/current/theme/colors.toml`), with a Tokyo Night
  fallback for non-Omarchy systems
- Live progress while transcribing (cancellable)
- Save result as **TXT / SRT / VTT / JSON** or **Copy to clipboard**
- Settings dialog: download or delete cached models, see disk usage

## Install

### Dependencies (Arch Linux)

```bash
sudo pacman -S --needed python python-pyqt6 ffmpeg
pip install --user openai-whisper        # pulls torch
```

If `pip` refuses on system Python, use a venv:

```bash
python -m venv ~/.venvs/whisper
source ~/.venvs/whisper/bin/activate
pip install -r requirements.txt
```

…then edit the launcher in `~/.local/bin/whisper-transcription` to use that
Python.

### Register in Walker / desktop

```bash
./install.sh
```

This drops a `.desktop` entry into `~/.local/share/applications/` and a
launcher script into `~/.local/bin/`. Open Walker and type **Whisper
Transcription**.

To remove:

```bash
./uninstall.sh
```

## Usage

- Launch from Walker, or from a terminal:
  `whisper-transcription [path/to/file]`
- File managers can also "Open with…" → Whisper Transcription, since the
  desktop entry advertises common audio/video MIME types.

## Where things live

| Path | Purpose |
| --- | --- |
| `~/.cache/whisper/` | Whisper model `.pt` files (managed in Settings) |
| `~/.config/whisper-transcription-arch/settings.json` | Last-used options |
| `~/.local/share/applications/whisper-transcription-arch.desktop` | Launcher entry |
| `~/.local/bin/whisper-transcription` | Wrapper script |

## Notes

- Transcription runs in `whisper_transcribe_worker.py` as a child process.
  Cancel sends `SIGTERM` (then `SIGKILL` after 2 s) to that process — the
  GUI never has to interrupt PyTorch in-process, which previously hung the
  app for tens of seconds and could leave a zombie.
- The hard RAM cap uses `systemd-run --user --scope -p MemoryMax=NG
  -p MemorySwapMax=0`. On overrun the worker is `SIGKILL`-ed by the kernel
  (exit 137); the GUI surfaces this with a clear OOM hint.
- The earlier `RLIMIT_AS`-based cap was dropped because PyTorch reserves
  much more virtual address space than its actual RSS, so any reasonable
  value killed model loading with `Cannot allocate memory`.
- The app forces `fp16=False` because Whisper on CPU uses fp32 anyway.

## License

MIT — see [LICENSE](LICENSE).
