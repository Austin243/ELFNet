#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAD-to-ELF periodic, symmetry-aware 3D model.

- Periodic UNet3D backbone (circular padding everywhere).
- Symmetry pooling via patch-local Seitz {R|t'} using batched grid_sample:
    zeta_in = S^-1 R^-1 S zeta_out - S^-1 R^-1 t' (mod 1),
    then rho_in = 2 zeta_in - 1
  where S = diag(ps/Nx, ps/Ny, ps/Nz) rescales from patch coords to unit-cell fractions.
- Loss = Smooth L1 (Huber) voxel loss with extra weight on high-value voxels.

The packaged pretrained checkpoint is documented in ``MODEL_CARD.md``.
"""

from __future__ import annotations
import argparse
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
    _LightningModuleBase = L.LightningModule
except Exception:
    L = None

    class _LightningModuleBase(nn.Module):
        """Small inference-only fallback when Lightning is unavailable."""

        def save_hyperparameters(self) -> None:
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                self.hparams = SimpleNamespace()
                return
            values = {
                k: v for k, v in frame.f_back.f_locals.items()
                if k != "self" and not k.startswith("_")
            }
            self.hparams = SimpleNamespace(**values)

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=map_location,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
            hparams = dict(checkpoint.get("hyper_parameters", {}))
            hparams.update(kwargs)
            model = cls(**hparams)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            return model

from .data import make_patch_loaders


# -------------------------- utilities --------------------------

def _group_norm(ch: int) -> nn.GroupNorm:
    # 32 groups unless channels are small
    g = min(32, max(1, ch // 4))
    return nn.GroupNorm(g, ch)

def periodic_pad3d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply equal circular padding on all sides."""
    if pad == 0:
        return x
    # PyTorch pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
    return F.pad(x, (pad, pad, pad, pad, pad, pad), mode="circular")

class PeriodicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, bias=True, groups=1):
        super().__init__()
        assert k % 2 == 1, "Use odd kernels for symmetric padding"
        self.pad = k // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=stride, padding=0,
                              bias=bias, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = periodic_pad3d(x, self.pad)
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = PeriodicConv3d(ch, ch, 3)
        self.gn1 = _group_norm(ch)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = PeriodicConv3d(ch, ch, 3)
        self.gn2 = _group_norm(ch)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        return self.act2(h + x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv_s2 = PeriodicConv3d(in_ch, out_ch, k=3, stride=2)
        self.gn = _group_norm(out_ch)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(*[ResBlock(out_ch, dropout) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn(self.conv_s2(x)))
        return self.blocks(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, blocks: int = 1, dropout: float = 0.0):
        super().__init__()
        # upsample + periodic conv to keep periodic behavior
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv_up = PeriodicConv3d(in_ch, out_ch, 3)
        self.blocks = nn.Sequential(*[ResBlock(out_ch + skip_ch, dropout), *[ResBlock(out_ch + skip_ch, dropout) for _ in range(blocks - 1)]])
        self.merge = PeriodicConv3d(out_ch + skip_ch, out_ch, 3)
        self.gn = _group_norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.blocks(x)
        x = self.act(self.gn(self.merge(x)))
        return x


# -------------------------- symmetry pooling --------------------------

def _wrap01(z: torch.Tensor) -> torch.Tensor:
    # Wrap to [0,1) in a differentiable way
    return z - torch.floor(z)

@dataclass
class SymmAvgConfig:
    align_corners: bool = True  # keeps integer-grid correspondences exact
    mode: str = "bilinear"      # 'bilinear' == trilinear for 5D
    eps: float = 1e-8

class SymmAvg3D(nn.Module):
    """
    Average features across per-sample symmetry ops in the patch frame.

    Inputs:
      f: (B, C, D, H, W)
      seitz: (B, R, 4, 4)  with integer R and fractional t' for the patch frame
      mask:  (B, R) bool
      orig_shape: (B, 3) longs with (NX, NY, NZ)

    Returns:
      f_sym: (B, C, D, H, W)
    """
    def __init__(self, cfg: Optional[SymmAvgConfig] = None):
        super().__init__()
        self.cfg = cfg or SymmAvgConfig()

    @staticmethod
    def _make_base_grid(D: int, H: int, W: int, device, dtype, align_corners: bool) -> torch.Tensor:
        # Build rho_out in [-1, 1] and corresponding zeta_out = (rho+1)/2 in [0,1]
        if align_corners:
            xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
            zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        else:
            # Not used by default, but keep for completeness
            xs = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=device, dtype=dtype)
            ys = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=device, dtype=dtype)
            zs = torch.linspace(-1 + 1/D, 1 - 1/D, D, device=device, dtype=dtype)
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")  # (D,H,W)
        rho = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D,H,W,3) -> x,y,z order
        zeta = 0.5 * (rho + 1.0)
        return rho, zeta  # each (D,H,W,3)

    def forward(self, f: torch.Tensor, seitz: torch.Tensor, mask: torch.Tensor, orig_shape: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = f.shape
        device, dtype = f.device, f.dtype
        R = seitz.shape[1]
        assert seitz.shape[-2:] == (4, 4), "seitz must be (B,R,4,4)"
        assert mask.shape == (B, R)

        # Extract R and t' in float
        #Rmat = seitz[:, :, :3, :3].to(dtype)    # (B,R,3,3)
        Rmat = seitz[:, :, :3, :3].to(dtype)    # (B,R,3,3)
        #tvec = seitz[:, :, :3, 3].to(dtype)     # (B,R,3)
        tvec = seitz[:, :, :3, 3].to(dtype)     # (B,R,3)

        # S = diag(ps/Nx, ps/Ny, ps/Nz) per sample
        NXNYNZ = orig_shape.to(dtype)  # (B,3)
        ps = torch.tensor([W, H, D], dtype=dtype, device=device).flip(0)  # careful with order later if needed
        # grid_sample expects x=W, y=H, z=D; we map S with ordering (x,y,z)
        S_diag = torch.stack([
            (W / NXNYNZ[:, 0]),  # ps/Nx
            (H / NXNYNZ[:, 1]),  # ps/Ny
            (D / NXNYNZ[:, 2]),  # ps/Nz
        ], dim=-1)  # (B,3)
        S_inv_diag = 1.0 / S_diag  # (B,3)

        # Precompute base grids
        rho_out, zeta_out = self._make_base_grid(D, H, W, device, dtype, self.cfg.align_corners)  # (D,H,W,3)
        zeta_out = zeta_out.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W,3)

        # Build per-(B,R) linear map: Q = S^{-1} R^{-1} S ; d = S^{-1} R^{-1} t'
        # broadcasting over (B,R)
        S = torch.diag_embed(S_diag)            # (B,3,3)
        S_inv = torch.diag_embed(S_inv_diag)    # (B,3,3)
        S = S[:, None]                          # (B,1,3,3)
        S_inv = S_inv[:, None]                  # (B,1,3,3)

#        R_inv = torch.linalg.inv(Rmat)          # (B,R,3,3)
#        Q = S_inv @ R_inv @ S                   # (B,R,3,3)
#        d = torch.matmul(S_inv @ R_inv, tvec.unsqueeze(-1)).squeeze(-1)  # (B,R,3)

        mask_exp = mask[..., None, None]
        I3 = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3).expand(B, R, -1, -1)
        R_eff = torch.where(mask_exp, Rmat, I3)
        t_eff = torch.where(mask[..., None], tvec, torch.zeros_like(tvec))

        R_inv = R_eff.transpose(-1, -2)
        Q = S_inv @ R_inv @ S
        d = torch.matmul(S_inv @ R_inv, t_eff.unsqueeze(-1)).squeeze(-1)

        # Expand and compute zeta_in = wrap(Q zeta_out - d)
        Qe = Q.view(B * R, 1, 1, 1, 1, 3, 3)
        de = d.view(B * R, 1, 1, 1, 1, 3)
        zout = zeta_out.expand(B * R, 1, D, H, W, 3)  # (B*R,1,D,H,W,3)
        zout_vec = zout.unsqueeze(-1)                 # (B*R,1,D,H,W,3,1)
        zq = torch.matmul(Qe, zout_vec).squeeze(-1)   # (B*R,1,D,H,W,3)
        zin = _wrap01(zq - de)                        # (B*R,1,D,H,W,3)
        grid = 2.0 * zin - 1.0                        # rho_in in [-1,1]

        # Sample and average
        f_rep = f.repeat_interleave(R, dim=0)         # (B*R,C,D,H,W)
        grid = grid.view(B * R, D, H, W, 3)
        f_g = F.grid_sample(
            f_rep, grid,
            mode=self.cfg.mode,
            padding_mode="zeros",  # all points are in [-1,1] after wrap
            align_corners=self.cfg.align_corners,
        )                           # (B*R,C,D,H,W)
        f_g = f_g.view(B, R, C, D, H, W)

        w = mask.to(f_g.dtype)      # (B,R)
        denom = w.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)  # (B,1)
        w = (w / denom).view(B, R, 1, 1, 1, 1)
        f_avg = (w * f_g).sum(dim=1)  # (B,C,D,H,W)
        return f_avg


# -------------------------- UNet backbone --------------------------

class UNet3DPeriodic(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16, depth: int = 4, blocks_per_stage: int = 1, dropout: float = 0.0,
                 sym_every_stage: bool = True, min_patch_size: int = 32):
        super().__init__()
        assert depth >= 2
        self.sym_every_stage = sym_every_stage
        self.sym = SymmAvg3D()
        self.min_patch_size = int(min_patch_size)

        chs = [base * (2 ** i) for i in range(depth)]
        self.stem = nn.Sequential(
            PeriodicConv3d(in_ch, chs[0], 3), _group_norm(chs[0]), nn.GELU(),
            *[ResBlock(chs[0], dropout) for _ in range(blocks_per_stage)]
        )
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(chs[i], chs[i + 1], blocks=blocks_per_stage, dropout=dropout))

        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(Up(chs[i + 1], chs[i], chs[i], blocks=blocks_per_stage, dropout=dropout))

        self.head = nn.Sequential(
            PeriodicConv3d(chs[0], chs[0], 3), _group_norm(chs[0]), nn.GELU(),
            PeriodicConv3d(chs[0], 1, 1),
        )
        self._min_spatial = 2 ** len(self.downs)

    def forward(self, x: torch.Tensor, seitz: torch.Tensor, mask: torch.Tensor, orig_shape: torch.Tensor) -> torch.Tensor:
        self._validate_spatial_dims(x)
        # Optional symmetry pooling at input
        x = self.sym(x, seitz, mask, orig_shape)

        # Encoder
        xs = []
        x = self.stem(x)
        if self.sym_every_stage:
            x = self.sym(x, seitz, mask, orig_shape)
        xs.append(x)
        for d in self.downs:
            x = d(x)
            if self.sym_every_stage:
                x = self.sym(x, seitz, mask, orig_shape)
            xs.append(x)

        # Decoder with skips
        for up in self.ups:
            skip = xs.pop(-2)  # last encoder activation (excluding current x)
            x = up(x, skip)
            if self.sym_every_stage:
                x = self.sym(x, seitz, mask, orig_shape)

        y = self.head(x)
        y = torch.clamp(y, 0.0, 1.0)
        return y

    def _validate_spatial_dims(self, x: torch.Tensor) -> None:
        if x.dim() < 5:
            raise ValueError("UNet3DPeriodic expects inputs shaped (B,C,D,H,W)")
        d, h, w = (int(x.size(-3)), int(x.size(-2)), int(x.size(-1)))
        if min(d, h, w) < self.min_patch_size:
            raise ValueError(
                f"input spatial dims {(d, h, w)} must each be >= {self.min_patch_size}"
            )
        if any(dim % self._min_spatial != 0 for dim in (d, h, w)):
            raise ValueError(
                f"input spatial dims {(d, h, w)} must be divisible by {self._min_spatial} for depth={len(self.downs)+1}"
            )


# -------------------------- Lightning module --------------------------

class Sad2ElfLitModule(_LightningModuleBase):
    def __init__(
        self,
        sad_idx: int = 0,
        elf_idx: int = 1,
        base: int = 16,
        depth: int = 4,
        blocks_per_stage: int = 1,
        dropout: float = 0.0,
        sym_every_stage: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 500,
        high_value_margin: float = 0.25,
        high_value_weight: float = 5.0,
        min_patch_size: int = 32,
        val_metric: str = "loss/vox",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = UNet3DPeriodic(
            in_ch=1,
            base=base,
            depth=depth,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
            sym_every_stage=sym_every_stage,
            min_patch_size=min_patch_size,
        )

        self.vox_loss = nn.SmoothL1Loss(reduction="none")
        self.high_value_margin = float(high_value_margin)
        self.high_value_weight = float(high_value_weight)
        self.weight_eps = 1e-6

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs_cfg = max_epochs
        self.min_patch_size = int(min_patch_size)

    def forward(self, x_sad: torch.Tensor, seitz: torch.Tensor, mask: torch.Tensor, orig_shape: torch.Tensor) -> torch.Tensor:
        return self.net(x_sad, seitz, mask, orig_shape)

    def _split_xy(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert X.dim() == 5
        sad = X[:, self.hparams.sad_idx:self.hparams.sad_idx + 1]
        elf = X[:, self.hparams.elf_idx:self.hparams.elf_idx + 1]
        return sad, elf

    def _importance_weights(self, target: torch.Tensor):
        B = target.shape[0]
        flat = target.view(B, -1)
        max_vals = flat.max(dim=1).values.view(B, 1, 1, 1, 1)
        threshold = (max_vals - self.high_value_margin).clamp_min(0.0)
        high_mask = target >= threshold
        high_weight = target.new_tensor(self.high_value_weight)
        weights = torch.ones_like(target)
        weights = torch.where(high_mask, high_weight, weights)
        return weights, high_mask

    def _loss(self, pred: torch.Tensor, target: torch.Tensor):
        weights, high_mask = self._importance_weights(target)
        loss_map = self.vox_loss(pred, target)
        weighted = (loss_map * weights).view(loss_map.size(0), -1).sum(dim=1)
        denom = weights.view(weights.size(0), -1).sum(dim=1).clamp_min(self.weight_eps)
        loss = (weighted / denom).mean()
        high_frac = high_mask.float().mean()
        logs = {"loss/vox": loss, "stats/high_frac": high_frac}
        return loss, logs

    def training_step(self, batch, batch_idx):
        X, seitz, mask, origins, shapes, stems = batch
        x_sad, y_elf = self._split_xy(X)
        y_pred = self.forward(x_sad, seitz, mask, shapes)
        loss, logs = self._loss(y_pred, y_elf)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=X.size(0))
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True, batch_size=X.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        X, seitz, mask, origins, shapes, stems = batch
        x_sad, y_elf = self._split_xy(X)
        with torch.no_grad():
            y_pred = self.forward(x_sad, seitz, mask, shapes)
        loss, logs = self._loss(y_pred, y_elf)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=X.size(0))
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, on_epoch=True, batch_size=X.size(0))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs_cfg)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def parse_args():
    p = argparse.ArgumentParser(description="Train SAD-to-ELF symmetry-aware periodic UNet on patches")
    p.add_argument("root", type=str, help="Dataset root directory with *_sad.npy, *_elf.npy, *_sym.npy")
    p.add_argument("--patch-size", type=int, default=32, help="Patch edge length (must match loader)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--channels", type=str, default="sad,elf", help="Loader channels; include both to supply target")
    p.add_argument("--sad-idx", type=int, default=0, help="Index of SAD channel in X")
    p.add_argument("--elf-idx", type=int, default=1, help="Index of ELF channel in X")
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--blocks-per-stage", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-sym-every-stage", dest="sym_every_stage", action="store_false",
                   help="Disable symmetry pooling after each stage")
    p.set_defaults(sym_every_stage=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--high-value-margin", type=float, default=0.25, help="Margin below max ELF treated as high-value")
    p.add_argument("--high-value-weight", type=float, default=5.0, help="Importance weight for high-value voxels")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32", choices=["16-mixed", "32", "bf16-mixed"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()

def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    chans = tuple([c.strip() for c in args.channels.split(",") if c.strip()])
    if "sad" not in chans or "elf" not in chans:
        raise ValueError("--channels must include both 'sad' and 'elf' so the model gets target data")

    train_loader, val_loader = make_patch_loaders(
        root=args.root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        channels=chans,
        num_workers=args.num_workers
    )

    model = Sad2ElfLitModule(
        sad_idx=args.sad_idx,
        elf_idx=args.elf_idx,
        base=args.base,
        depth=args.depth,
        blocks_per_stage=args.blocks_per_stage,
        dropout=args.dropout,
        sym_every_stage=args.sym_every_stage,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        high_value_margin=args.high_value_margin,
        high_value_weight=args.high_value_weight,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
