"""Checkpoint metadata and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = {
    "name": "elfnet_sad2elf.ckpt",
    "epoch": 114,
    "global_step": 499905,
    "validation_metric": "weighted_smooth_l1",
    "validation_score": 0.0088327322,
    "hparams": {
        "base": 24,
        "depth": 5,
        "blocks_per_stage": 1,
        "sym_every_stage": True,
        "lr": 3e-4,
        "high_value_weight": 5.0,
    },
}


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve a checkpoint path from an explicit path, env var, or package weights dir."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())

    env_path = os.environ.get("ELFNET_CHECKPOINT")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    name = str(DEFAULT_CHECKPOINT["name"])
    candidates.extend([
        Path.cwd() / "weights" / name,
        Path(__file__).resolve().parents[2] / "weights" / name,
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "ELFNet checkpoint not found. Pass --checkpoint, set ELFNET_CHECKPOINT, "
        "or place the file at weights/elfnet_sad2elf.ckpt.\n"
        f"Checked:\n{checked}"
    )


def load_model(path: str | Path | None = None, map_location: str | None = "cpu") -> Any:
    """Load the ELFNet model from a checkpoint."""
    from .model import Sad2ElfLitModule

    checkpoint = resolve_checkpoint(path)
    return Sad2ElfLitModule.load_from_checkpoint(str(checkpoint), map_location=map_location)
