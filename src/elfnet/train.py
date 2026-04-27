#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ELFNet on periodic SAD-to-ELF patches with DDP-safe jitter.

Key points:
- Explicit DistributedSampler built from env (WORLD_SIZE/RANK or SLURM_*).
- Epoch-wise jitter and sampler seeds advanced every epoch under DDP.
- One process per GPU: devices=1 (your SLURM job already launches 1 task/GPU).
- Keeps your fixed-weights validation metric for stable checkpoint selection.

This training entry point is compatible with the architecture used by the
best ChiNet checkpoint, but writes new runs to this repository by default.
"""

from __future__ import annotations
import argparse, datetime as _dt, os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .data import ElfSadPatchDataset, collate_patches
from .model import Sad2ElfLitModule

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
    _LIGHTNING_IMPORT_ERROR = None
except Exception as exc:
    L = None
    ModelCheckpoint = None
    LearningRateMonitor = None
    _LIGHTNING_IMPORT_ERROR = exc

    class Callback:
        pass

# --------------------------- helpers --------------------------- #

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ckpt_root(batch: int, output_dir: Optional[Path] = None) -> Path:
    root = (output_dir or (Path.cwd() / "runs" / "checkpoints")) / f"batch{batch}"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _world_info() -> Tuple[int, int, int]:
    """Return (world_size, global_rank, num_nodes) from common envs (PyTorch/SLURM)."""
    ws = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    rk = _env_int("RANK", _env_int("SLURM_PROCID", 0))
    nn = _env_int("SLURM_NNODES", 1)
    return max(ws, 1), max(rk, 0), max(nn, 1)

def _discover_stems(root: Path) -> List[str]:
    """Find stems that have *_sad.npy, *_elf.npy, *_sym.npy all present."""
    sad = {p.stem[:-4] for p in root.glob("*_sad.npy")}
    elf = {p.stem[:-4] for p in root.glob("*_elf.npy")}
    sym = {p.stem[:-4] for p in root.glob("*_sym.npy")}
    stems = sorted(sad & elf & sym)
    if not stems:
        raise FileNotFoundError(f"No matching *_sad.npy, *_elf.npy, *_sym.npy triplets in {root}")
    return stems

# -------------------- adapter LightningModule ------------------ #

class Sad2ElfLitFromELFLoader(Sad2ElfLitModule):
    """
    Adapter layer that keeps the original training/validation logging semantics
    (e.g. sync'ing metrics across ranks and exposing a deterministic checkpoint
    metric) while reusing the updated base module.
    """
    def __init__(self, *args, val_metric: str = "loss/vox", **kwargs):
        super().__init__(*args, **kwargs)
        self.val_metric = str(val_metric)

    def training_step(self, batch, batch_idx):
        X, seitz, mask, origins, shapes, stems = batch
        x_sad = X[:, self.hparams.sad_idx:self.hparams.sad_idx+1]
        y_elf = X[:, self.hparams.elf_idx:self.hparams.elf_idx+1]
        y_pred = self.forward(x_sad, seitz, mask, shapes)
        loss, logs = self._loss(y_pred, y_elf)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=X.size(0))
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True, batch_size=X.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        X, seitz, mask, origins, shapes, stems = batch
        x_sad = X[:, self.hparams.sad_idx:self.hparams.sad_idx+1]
        y_elf = X[:, self.hparams.elf_idx:self.hparams.elf_idx+1]
        with torch.no_grad():
            y_pred = self.forward(x_sad, seitz, mask, shapes)
        loss, logs = self._loss(y_pred, y_elf)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=X.size(0),
            sync_dist=True,
        )
        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()},
            on_epoch=True,
            batch_size=X.size(0),
            sync_dist=True,
        )

        val_fixed = logs.get(self.val_metric, loss)
        self.log("val/loss_fixed", val_fixed, on_epoch=True, batch_size=X.size(0), sync_dist=True)

# ---------------------------- epoch-jitter callback ---------------------------- #

class PatchJitterCallback(Callback):
    """
    Advance both the sampler seed and the dataset's epoch-wise origin jitter every epoch.
    Works for single-GPU and DDP.
    """
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        dl = trainer.train_dataloader
        if dl is None:
            return
        # 1) Advance DistributedSampler RNG (ensures new shuffling per epoch)
        smp = getattr(dl, "sampler", None)
        if hasattr(smp, "set_epoch"):
            try:
                smp.set_epoch(trainer.current_epoch)
            except Exception:
                pass
        # 2) Advance dataset jitter offset
        ds = getattr(dl, "dataset", None)
        base = getattr(ds, "dataset", ds)  # unwrap Subset
        if hasattr(base, "set_epoch"):
            try:
                base.set_epoch(trainer.current_epoch)
            except Exception:
                pass

# ---------------------------- CLI ----------------------------- #

def _get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("data_dir", type=Path, help="Folder with *_sad.npy, *_elf.npy, *_sym.npy")

    # Data / loader knobs
    p.add_argument("--patch-size", type=int, default=32, help="Patch edge length")
    p.add_argument("--stride", type=int, default=None, help="Training stride (default: patch-size)")
    p.add_argument("--val-fraction", type=float, default=0.10, help="Fraction of structures reserved for validation")
    p.add_argument("--no-jitter", action="store_true", help="Disable epoch-wise origin jitter")
    p.add_argument("--num-workers", type=int, default=None)

    # Optimization
    p.add_argument("--batch", type=int, default=2, help="Batch size per GPU")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--accum", type=int, default=1, help="Gradient-accumulation steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Model knobs (forwarded to Sad2ElfLitModule)
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--blocks-per-stage", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-sym-every-stage", dest="sym_every_stage", action="store_false",
                   help="Disable symmetry pooling after each stage")
    p.set_defaults(sym_every_stage=True)
    p.add_argument("--sad-idx", type=int, default=0)
    p.add_argument("--elf-idx", type=int, default=1)

    # High-value weighting (forwarded to Sad2ElfLitModule)
    p.add_argument("--high-value-margin", type=float, default=0.25,
                   help="Margin below per-sample max ELF treated as high-value")
    p.add_argument("--high-value-weight", type=float, default=5.0,
                   help="Importance weight applied to high-value voxels")
    p.add_argument("--val-metric", type=str, default="loss/vox",
                   help="Metric key from Sad2ElfLitModule logs to use for checkpoint ranking")

    # Trainer / cluster knobs
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["16-mixed", "bf16-mixed", "32"])
    p.add_argument("--compile", type=str, default="none",
                   choices=["none", "default", "reduce-overhead"])
    p.add_argument("--output-dir", type=Path, default=Path("runs/checkpoints"),
                   help="Directory for Lightning checkpoints")
    p.add_argument("--devices", type=int, default=1,
                   help="Devices per node for Lightning. Use one process per GPU under SLURM/DDP.")
    return p.parse_args()

# ---------------------------- data builders ----------------------------- #

def _build_loaders(root: Path,
                   patch_size: int,
                   stride: Optional[int],
                   batch_size: int,
                   val_fraction: float,
                   num_workers: Optional[int],
                   jitter_train: bool) -> Tuple[DataLoader, DataLoader]:
    """Create *separate* train/val datasets (jitter on train only) and DDP-safe loaders."""
    stems = _discover_stems(root)
    # Shuffle stems deterministically for a fair split
    g = torch.Generator().manual_seed(1337)
    idx = torch.randperm(len(stems), generator=g).tolist()
    stems = [stems[i] for i in idx]

    n_val = max(1, int(round(val_fraction * len(stems))))
    val_stems = set(stems[-n_val:])
    train_stems = set(stems[:-n_val])

    ds_train = ElfSadPatchDataset(
        root=root, patch_size=patch_size, stride=stride,
        jitter=jitter_train, allowed_stems=train_stems
    )
    ds_val = ElfSadPatchDataset(
        root=root, patch_size=patch_size, stride=stride,
        jitter=False, allowed_stems=val_stems
    )

    # Worker defaults
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    persistent_workers = (num_workers or 0) > 0
    pin_memory = torch.cuda.is_available()

    # Distributed or single-GPU samplers
    world_size, rank, _ = _world_info()
    is_dist = world_size > 1
    if is_dist:
        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(ds_val,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(ds_train)
        val_sampler   = SequentialSampler(ds_val)

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, sampler=train_sampler, shuffle=False,
        num_workers=num_workers, collate_fn=collate_patches,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, sampler=val_sampler, shuffle=False,
        num_workers=num_workers, collate_fn=collate_patches,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False
    )
    return train_loader, val_loader

# ---------------------------- main ----------------------------- #

def main() -> None:
    args = _get_args()
    if L is None:
        raise RuntimeError(
            "Training requires the 'lightning' package. Install ELFNet with "
            "`python -m pip install -e '.[train,symmetry]'` in an environment with "
            "compatible torch/torchvision/lightning builds."
        ) from _LIGHTNING_IMPORT_ERROR

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    print(f"--- Starting training with arguments: {args} ---")

    # Data loaders (built with explicit DistributedSampler so epochs are de-duplicated)
    train_ld, val_ld = _build_loaders(
        root=args.data_dir,
        patch_size=args.patch_size,
        stride=(args.patch_size if args.stride is None else args.stride),
        batch_size=args.batch,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        jitter_train=(not args.no_jitter),
    )

    # Model
    model = Sad2ElfLitFromELFLoader(
        sad_idx=args.sad_idx,
        elf_idx=args.elf_idx,
        base=args.base,
        depth=args.depth,
        blocks_per_stage=args.blocks_per_stage,
        dropout=args.dropout,
        sym_every_stage=args.sym_every_stage,
        lr=args.lr,
        max_epochs=args.epochs,
        high_value_margin=args.high_value_margin,
        high_value_weight=args.high_value_weight,
        val_metric=args.val_metric,
    )

    if args.compile != "none" and torch.cuda.is_available():
        mode = args.compile if args.compile != "default" else None
        try:
            model = torch.compile(model, mode=mode)  # PyTorch 2.x
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); continuing without compilation.")

    # Checkpoints & logging
    ckpt_dir = _ckpt_root(args.batch, args.output_dir) / f"SAD2ELF_{_timestamp()}"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:04d}",
            save_top_k=-1,
            every_n_epochs=25,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(  # "best" by stable validation metric
            dirpath=ckpt_dir,
            filename="best_{epoch:04d}",
            monitor="val/loss_fixed",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        PatchJitterCallback(),  # advance sampler+jitter every epoch
    ]

    # ---- Trainer configuration (DDP-safe) ----
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    world_size, _, _ = _world_info()
    use_ddp = world_size > 1
    strategy = "ddp" if world_size > 1 else "auto"

    env_devices = int(os.environ.get("SLURM_NTASKS_PER_NODE", "0"))
    if use_ddp and env_devices > 0:
        devices = env_devices
    else:
        devices = max(int(args.devices), 1)

    env_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "0"))
    if env_nodes <= 0:
        env_nodes = 1

    trainer = L.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        num_nodes=env_nodes,
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum,
        gradient_clip_algorithm="norm",
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_progress_bar=True,
        # We already pass explicit samplers; let them be.
        # (Lightning won't replace a provided sampler.)
    )

    print("Launching SAD-to-ELF training ...")
    trainer.fit(model, train_ld, val_ld)
    print("Training complete.")

if __name__ == "__main__":
    main()
