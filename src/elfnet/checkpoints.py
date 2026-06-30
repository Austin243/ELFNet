"""Checkpoint resolution and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = {
    "name": "elfnet.ckpt",
}


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve a checkpoint path from an explicit path, env var, or local weights dir."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())

    env_path = os.environ.get("ELFNET_CHECKPOINT")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    name = str(DEFAULT_CHECKPOINT["name"])
    candidates.extend(
        [
            Path.cwd() / "weights" / name,
            Path(__file__).resolve().parents[2] / "weights" / name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Checkpoint not found. Pass a checkpoint path, set ELFNET_CHECKPOINT, "
        "or use the default weights/elfnet.ckpt file.\n"
        f"Checked:\n{checked}"
    )


def load_model(path: str | Path | None = None, map_location: str | None = "cpu") -> Any:
    """Load the full-grid ELFNet predictor from a checkpoint."""
    from .model import ELFPredictor

    checkpoint = resolve_checkpoint(path)
    return ELFPredictor.load_from_checkpoint(str(checkpoint), map_location=map_location)
