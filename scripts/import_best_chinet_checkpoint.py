#!/usr/bin/env python3
"""Import the local best ChiNet checkpoint into this repo's ignored weights dir."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCE = Path(
    "/home/aellis/ChiNet/checkpoints_sad2elf/batch2/"
    "SAD2ELF_20251104_124933/best_epoch=0114.ckpt"
)
DEST = Path(__file__).resolve().parents[1] / "weights" / "best_chinet_epoch0114.ckpt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SOURCE.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {SOURCE}")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() or DEST.is_symlink():
        if not args.overwrite:
            print(f"Already exists: {DEST}")
            return
        DEST.unlink()

    if args.mode == "symlink":
        DEST.symlink_to(SOURCE)
    else:
        shutil.copy2(SOURCE, DEST)

    print(f"{args.mode} created: {DEST} -> {SOURCE}")


if __name__ == "__main__":
    main()
