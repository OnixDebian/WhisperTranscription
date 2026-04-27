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
- Theme palette pulled from Omarchy
  (`~/.config/omarchy/current/theme/colors.toml`), with a Tokyo Night
  fallback for non-Omarchy systems
- Live progress while transcribing (runs in a worker thread; cancellable)
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

- Cancelling a running job uses `QThread.terminate()` because
  `whisper.transcribe` is not interruptible. The worker is fully owned by the
  app, so this is safe in this context but avoid spamming it.
- RAM is **not** hard-capped by the app: an earlier version used `RLIMIT_AS`,
  but PyTorch reserves much more virtual address space than its actual RSS,
  so any reasonable cap killed loading even small models with
  `Cannot allocate memory`. The picker now shows live available RAM and
  warns before starting if the chosen model is unlikely to fit. If you need
  a real hard cap, wrap the launcher in
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`.
- The app forces `fp16=False` because Whisper on CPU uses fp32 anyway.

## License

MIT — see [LICENSE](LICENSE).
