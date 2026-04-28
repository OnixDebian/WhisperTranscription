#!/usr/bin/env python3
"""Whisper transcription worker — long-running command loop.

Runs as a child process of the GUI. Keeps the loaded model in memory
between transcriptions so consecutive files do not pay the model-load
cost over and over (which was 5–30 s each on CPU for the larger
models).

Protocol:
  stdin  : one JSON object per line, each is a command. Commands:
             {"action": "preload", "model": "...", "cpu_threads": N}
                 — load the model and answer with `[STATUS] Ready`.
             {"action": "transcribe", "model": "...", "cpu_threads": N,
              "file": "/path", "language": "en", "result_path": "/path"}
                 — load the model if it differs from the cached one,
                   transcribe, write JSON to result_path, answer with
                   `[STATUS] Done`.
             {"action": "quit"}
                 — exit cleanly.
  stderr : human progress lines, prefixed `[STATUS] `, plus tqdm output.
  stdout : not used. Stays redirected to stderr at startup so any stray
           print from torch / whisper cannot pollute the channel.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def emit(msg: str) -> None:
    sys.stderr.write(f"[STATUS] {msg}\n")
    sys.stderr.flush()


def main() -> int:
    # Belt & braces: redirect stdout to stderr so any stray print from
    # torch / whisper / tqdm cannot mix into the parent's stdout pipe.
    sys.stdout = sys.stderr

    model = None
    current_model_name: str | None = None
    current_cpu_threads: int = 0

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            emit(f"ERROR: malformed command — {e}")
            continue

        action = cmd.get("action")
        if action == "quit":
            emit("Bye")
            return 0

        if action not in ("preload", "transcribe"):
            emit(f"ERROR: unknown action {action!r}")
            continue

        try:
            cpu = max(1, int(cmd.get("cpu_threads", 1)))
            if cpu != current_cpu_threads:
                os.environ["OMP_NUM_THREADS"] = str(cpu)
                os.environ["MKL_NUM_THREADS"] = str(cpu)
                os.environ["OPENBLAS_NUM_THREADS"] = str(cpu)
                current_cpu_threads = cpu

            target_model = cmd.get("model")
            if not target_model:
                emit("ERROR: missing model")
                continue

            if target_model != current_model_name:
                emit(f"Loading model {target_model}…")
                import torch
                torch.set_num_threads(cpu)
                import whisper
                model = whisper.load_model(target_model, device="cpu")
                current_model_name = target_model
                emit(f"Model {target_model} loaded")

            if action == "preload":
                emit("Ready")
                continue

            # action == "transcribe"
            file_path = cmd.get("file")
            result_path = cmd.get("result_path")
            if not file_path or not result_path:
                emit("ERROR: transcribe needs file and result_path")
                continue

            emit("Transcribing… (this may take a while)")
            result = model.transcribe(
                file_path,
                language=cmd.get("language") or None,
                fp16=False,
                verbose=False,
            )
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            emit("Done")
        except Exception as e:
            emit(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emit(f"ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
