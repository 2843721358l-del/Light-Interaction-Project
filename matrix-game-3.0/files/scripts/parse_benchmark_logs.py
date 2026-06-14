#!/usr/bin/env python3
"""Parse Light Interaction Matrix-Game-3.0 benchmark logs into CSV rows."""

import re
import sys
from pathlib import Path


def search(pattern: str, text: str):
    match = re.search(pattern, text)
    return match.group(1) if match else "NA"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: parse_benchmark_logs.py LOG_DIR", file=sys.stderr)
        return 2

    log_dir = Path(sys.argv[1])
    print("preset,dit_core_time_s,video_generation_time_s,peak_vram_allocated_gb,peak_vram_reserved_gb")
    for log_path in sorted(log_dir.glob("*.log")):
        text = log_path.read_text(errors="replace")
        preset = search(r"Preset\s*:\s*([A-Za-z0-9_-]+)", text)
        if preset == "NA":
            preset = log_path.stem
        dit = search(r"DiT Core Time:\s*([0-9.]+)\s*s", text)
        gen = search(r"Video Generation Time\s*:\s*([0-9.]+)\s*s", text)
        alloc = search(r"Peak VRAM Allocated\s*:\s*([0-9.]+)\s*GB", text)
        reserved = search(r"Peak VRAM Reserved\s*:\s*([0-9.]+)\s*GB", text)
        print(f"{preset},{dit},{gen},{alloc},{reserved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
