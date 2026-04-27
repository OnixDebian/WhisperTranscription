#!/usr/bin/env python3
"""Whisper transcription worker.

Spawned by the main GUI as an isolated subprocess so that:
- Cancel can simply kill this process — the GUI never blocks on torch.
- A hard RAM cap can be applied by wrapping us in
  `systemd-run --user --scope -p MemoryMax=NG -p MemorySwapMax=0 -- python3 ...`.

Protocol:
  stdin  : single JSON object — {file, model, cpu_threads, language?}
  stderr : human progress, one per line, prefixed `[STATUS] `
  stdout : final transcription result as a single JSON document
  exit 0 : success, stdout contains JSON
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

    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()
    emit("Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emit(f"ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
