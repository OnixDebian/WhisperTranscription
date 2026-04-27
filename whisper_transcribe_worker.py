#!/usr/bin/env python3
"""Whisper transcription worker.

Spawned by the main GUI as an isolated subprocess so that:
- Cancel can simply kill this process — the GUI never blocks on torch.
- A hard RAM cap can be applied by wrapping us in
  `systemd-run --user --scope -p MemoryMax=NG -p MemorySwapMax=0 -- python3 ...`.

Protocol:
  stdin  : single JSON object — {file, model, cpu_threads, language?, result_path}
  stderr : human progress, one per line, prefixed `[STATUS] `
  result : written as JSON to `result_path` (NOT stdout — too easy for
           torch / whisper / tqdm to pollute stdout in a child process).
  exit 0 : success, result_path is readable
  exit 1 : failure, stderr contains traceback
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
    raw = sys.stdin.read()
    if not raw.strip():
        emit("ERROR: empty config on stdin")
        return 2
    cfg = json.loads(raw)

    result_path = cfg.get("result_path")
    if not result_path:
        emit("ERROR: missing result_path in config")
        return 2

    # Belt & braces: redirect stdout to stderr so any stray print from
    # torch / whisper / tqdm cannot mix into the parent's stdout pipe.
    sys.stdout = sys.stderr

    cpu = max(1, int(cfg.get("cpu_threads", 1)))
    os.environ["OMP_NUM_THREADS"] = str(cpu)
    os.environ["MKL_NUM_THREADS"] = str(cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu)

    emit(f"Loading model {cfg['model']}…")
    import torch
    torch.set_num_threads(cpu)
    import whisper

    model = whisper.load_model(cfg["model"], device="cpu")

    emit("Transcribing… (this may take a while)")
    result = model.transcribe(
        cfg["file"],
        language=cfg.get("language") or None,
        fp16=False,
        verbose=False,
    )

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    emit("Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emit(f"ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
