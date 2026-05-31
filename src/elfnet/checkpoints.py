"""Checkpoint resolution and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = {
    "name": "elfnet.ckpt",
    "bundled": True,
    "sha256": "66dac5953e2b93cb0629b708c77cd444b20a40daa586514ab4187b6f2c995c34",
    "source_run": "pressure_flatresnet_c32_b16_k5_kendall_fixed_order/ELF_20260430_123836",
    "epoch": 59,
    "global_step": 72780,
    "best_val_loss": -9.523093223571777,
    "training_structures": 326009,
    "production_training_hparams": {
        "arch": "flat_resnet",
        "base": 32,
        "depth": 4,
        "flat_blocks": 16,
        "flat_kernel": 5,
        "flat_attention_every": 4,
        "in_ch": 1,
        "use_checkpoint": False,
        "loss_mode": "kendall",
        "lambda_vox": 1.0,
        "lambda_grad": 1.0,
        "lambda_cdf": 1.0,
        "lambda_hist": 1.0,
        "cdf_bins": 64,
        "cdf_sigma": 0.02,
        "cdf_tail_start": 0.60,
        "cdf_tail_weight": 2.0,
        "cdf_max_voxels": 20000,
        "hist_bins": 64,
        "hist_sigma": 0.02,
        "delta": 0.1,
        "lr": 1e-4,
        "aux_weight": 0.0,
        "gamma_w": 2.0,
        "interstitial_weight": 0.0,
        "peak_all_weight": 0.0,
        "peak_weight": 0.0,
        "peak_atomic_weight": 0.0,
        "peak_margin": 0.07,
        "peak_min_value": 0.0,
        "peak_location_tau": 0.75,
        "peak_location_margin": 0.07,
        "peak_location_temperature": 0.05,
        "peak_topk_frac": 0.002,
        "peak_topk_min": 32,
        "peak_topk_max": 512,
        "peak_location_loss_weight": 1.0,
        "peak_value_loss_weight": 1.0,
        "false_peak_loss_weight": 1.0,
        "kendall_log_var_min": -5.0,
        "kendall_log_var_max": 5.0,
        "weight_decay": 1e-4,
    },
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
