#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train or fine-tune the full-grid ELFNet ``ELFPredictor`` model.

This trainer matches the old ChiNet model family: each sample is a complete
paired SAD/ELF grid, and the collate function periodically tiles mixed-size
samples to the largest shape in the batch. By default, samples are bucketed by
exact grid shape so that tiling is normally a no-op. The trainer does not use
symmetry operations or patch sampling.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
from typing import Sequence

import torch

from .data import make_loaders
from .model import ELFPredictor


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _world_info() -> tuple[int, int, int]:
    world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    rank = _env_int("RANK", _env_int("SLURM_PROCID", 0))
    nodes = _env_int("SLURM_JOB_NUM_NODES", _env_int("SLURM_NNODES", 1))
    return max(world_size, 1), max(rank, 0), max(nodes, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="Directory with paired *_sad.npy and *_elf.npy")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per process")
    parser.add_argument("--batching", choices=("shape", "random"), default="shape",
                        help="Batch same-size grids together for less padding, or use fully random batches")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", "--val-fraction", dest="val_frac", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--output-dir", "--checkpoint-root", dest="output_dir", type=Path, default=_repo_root() / "checkpoints_sad2elf" / "full_grid")
    parser.add_argument("--resume-from-ckpt", type=Path, default=None)
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable activation checkpointing during training")
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lambda-vox", type=float, default=1.0)
    parser.add_argument("--lambda-grad", type=float, default=0.20)
    parser.add_argument("--lambda-cdf", type=float, default=0.05)
    parser.add_argument("--cdf-bins", type=int, default=64)
    parser.add_argument("--cdf-sigma", type=float, default=0.02)
    parser.add_argument("--cdf-tail-start", type=float, default=0.60)
    parser.add_argument("--cdf-tail-weight", type=float, default=2.0)
    parser.add_argument("--cdf-max-voxels", type=int, default=200_000)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--gamma-w", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-metric", type=str, default="l_vox_raw", help="Validation metric suffix under val/ used for best checkpointing")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Checkpoint-era loss aliases. New runs should use the CDF names above.
    parser.add_argument("--lambda-hist", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hist-bins", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hist-sigma", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import lightning as L
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    except Exception as exc:
        raise SystemExit(
            "Training requires Lightning. Install with `python -m pip install -e '.[train]'`."
        ) from exc

    args = build_parser().parse_args(argv)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader = make_loaders(
        args.data_dir,
        batch=args.batch,
        num_workers=args.num_workers,
        seed=args.seed,
        val_frac=args.val_frac,
        batching=args.batching,
    )

    if args.lambda_hist is not None:
        args.lambda_cdf = args.lambda_hist
    if args.hist_bins is not None:
        args.cdf_bins = args.hist_bins
    if args.hist_sigma is not None:
        args.cdf_sigma = args.hist_sigma

    model = ELFPredictor(
        lr=args.lr,
        lambda_vox=args.lambda_vox,
        lambda_grad=args.lambda_grad,
        lambda_cdf=args.lambda_cdf,
        cdf_bins=args.cdf_bins,
        cdf_sigma=args.cdf_sigma,
        cdf_tail_start=args.cdf_tail_start,
        cdf_tail_weight=args.cdf_tail_weight,
        cdf_max_voxels=args.cdf_max_voxels,
        delta=args.delta,
        aux_weight=args.aux_weight,
        gamma_w=args.gamma_w,
        weight_decay=args.weight_decay,
        base=args.base,
        depth=args.depth,
        use_checkpoint=args.use_checkpoint,
    )

    ckpt_dir = args.output_dir.expanduser().resolve() / f"ELF_{_timestamp()}"
    class _SetBatchSamplerEpoch(L.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            loader = getattr(trainer, "train_dataloader", None)
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(int(trainer.current_epoch))

    callbacks = [
        _SetBatchSamplerEpoch(),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:04d}",
            save_top_k=-1,
            every_n_epochs=25,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_{epoch:04d}",
            monitor=f"val/{args.val_metric}",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    world_size, _, num_nodes = _world_info()
    strategy = "ddp" if world_size > 1 else "auto"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "gpu":
        devices = _env_int("SLURM_NTASKS_PER_NODE", 0) or torch.cuda.device_count() or 1
    else:
        devices = 1
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum,
        gradient_clip_algorithm="norm",
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        log_every_n_steps=50,
        use_distributed_sampler=False,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume_from_ckpt,
    )
    print(f"Training complete. Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
