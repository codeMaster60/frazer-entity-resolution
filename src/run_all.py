"""Run the whole pipeline end to end.

    python -m src.run_all              # everything, from the raw TSVs
    python -m src.run_all --from train # skip the stages already done

Each stage is also runnable on its own; see the module docstrings.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    ("parquet", [["-m", "src.to_parquet"]]),
    ("prepare", [["-m", "src.prepare", "--split", "train"],
                 ["-m", "src.prepare", "--split", "test"]]),
    ("block", [["-m", "src.blocking", "--split", "train"],
               ["-m", "src.blocking", "--split", "test"]]),
    ("block_eval", [["-m", "src.block_eval"]]),
    ("train", [["-m", "src.train"]]),
    ("predict", [["-m", "src.predict"]]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None,
                    choices=[name for name, _ in STAGES])
    args = ap.parse_args()

    started = args.start is None
    for name, commands in STAGES:
        if name == args.start:
            started = True
        if not started:
            print(f"== skip {name}")
            continue
        for cmd in commands:
            print(f"== {name}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, *cmd])
            if r.returncode:
                print(f"== {name} FAILED ({r.returncode})")
                return r.returncode
            print(f"== {name} ok in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
