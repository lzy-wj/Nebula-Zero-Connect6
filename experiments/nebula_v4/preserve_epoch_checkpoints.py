"""Preserve epoch checkpoints that would otherwise be overwritten by last.pth."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--idle-polls", type=int, default=120)
    arguments = parser.parse_args()
    root = Path(arguments.root).expanduser().resolve()
    idle = 0
    while idle < arguments.idle_polls:
        changed = False
        active = False
        for name in ("dense64", "moe64"):
            directory = root / name
            history_path = directory / "history.json"
            last_path = directory / "last.pth"
            if not history_path.exists() or not last_path.exists():
                active = True
                continue
            try:
                epochs = len(json.loads(history_path.read_text())["epochs"])
            except (OSError, KeyError, json.JSONDecodeError):
                active = True
                continue
            target = directory / f"epoch-{epochs}.pth"
            if epochs and not target.exists():
                shutil.copy2(last_path, target)
                print(f"preserved {target}", flush=True)
                changed = True
            if epochs < 3:
                active = True
        idle = 0 if changed or active else idle + 1
        time.sleep(arguments.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
