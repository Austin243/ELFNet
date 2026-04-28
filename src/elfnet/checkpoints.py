"""Checkpoint metadata and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = {
    "name": "elfnet_sad2elf.ckpt",
    "source": "/home/aellis/ChiNet/epoch1000.ckpt",
    "epoch": 999,
    "global_step": 175000,
    "model_class": "ELFPredictor",
    "architecture": "ResidualUNet3D",
    "validation_metric": "val/l_vox_raw",
    "validation_score": 0.0984048,
    "hparams": {
        "lambda_vox": 1.0,
        "lambda_grad": 0.2,
        "lambda_hist": 0.05,
        "hist_bins": 30,
        "hist_sigma": 0.02,
        "delta": 0.1,
        "lr": 6e-4,
        "aux_weight": 0.3,
        "gamma_w": 2.0,
    },
    "production_training_hparams": {
        "lambda_vox": 1.0,
        "lambda_grad": 0.2,
        "lambda_cdf": 0.05,
        "cdf_bins": 64,
        "cdf_sigma": 0.02,
        "cdf_tail_start": 0.60,
        "cdf_tail_weight": 2.0,
        "cdf_max_voxels": 200000,
        "delta": 0.1,
        "lr": 6e-4,
        "aux_weight": 0.3,
        "gamma_w": 2.0,
    },
}


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve a checkpoint path from an explicit path, env var, or weights dir."""
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
        "ELFNet checkpoint not found. Pass a checkpoint path, set ELFNET_CHECKPOINT, "
        "or place the file at weights/elfnet_sad2elf.ckpt.\n"
        f"Checked:\n{checked}"
    )


def load_model(path: str | Path | None = None, map_location: str | None = "cpu") -> Any:
    """Load the full-grid ELFNet predictor from a checkpoint."""
    from .model import ELFPredictor

    checkpoint = resolve_checkpoint(path)
    return ELFPredictor.load_from_checkpoint(str(checkpoint), map_location=map_location)
