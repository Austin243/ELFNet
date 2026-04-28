#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-grid SAD-to-ELF model.

``ELFPredictor`` predicts a full ELF grid from a full superposed atomic
density (SAD) grid in one forward pass. It does not take symmetry operations
and it does not run patch inference.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    import lightning as L

    _LightningModuleBase = L.LightningModule
except Exception:
    L = None

    class _LightningModuleBase(nn.Module):
        """Small inference-only fallback when Lightning is unavailable."""

        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            if args and isinstance(args[0], dict):
                self.hparams = SimpleNamespace(**args[0])
                return
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                self.hparams = SimpleNamespace()
                return
            values = {
                key: value
                for key, value in frame.f_back.f_locals.items()
                if key != "self" and not key.startswith("_")
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


def conv3(ci: int, co: int, k: int = 3, s: int = 1) -> nn.Conv3d:
    """Circular-padding 3D convolution used throughout the old model."""
    p = (k - 1) // 2
    return nn.Conv3d(
        ci,
        co,
        k,
        s,
        padding=p,
        padding_mode="circular",
        bias=False,
    )


class SEBlock(nn.Module):
    """Squeeze-excitation block for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _, _ = x.shape
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1, 1, 1)
        return x * y.expand_as(x)


class _ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.max = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class _SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=pad, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True)[0]
        att = self.sig(self.conv(torch.cat([avg, mx], 1)))
        return x * att


class CBAM3D(nn.Module):
    """Channel plus spatial attention at the bottleneck."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 3):
        super().__init__()
        self.ca = _ChannelAttention3D(channels, reduction)
        self.sa = _SpatialAttention3D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class ResBlock(nn.Module):
    """Residual block with circular convolutions, GroupNorm, GELU, and SE."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        group_count = min(groups, channels)
        self.use_checkpoint = bool(use_checkpoint)
        self.f = nn.Sequential(
            conv3(channels, channels),
            nn.GroupNorm(group_count, channels),
            nn.GELU(),
            conv3(channels, channels),
            nn.GroupNorm(group_count, channels),
            SEBlock(channels),
        )

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            y = activation_checkpoint(self._f, x, use_reentrant=False)
        else:
            y = self._f(x)
        return F.gelu(x + y)


class Down(nn.Module):
    def __init__(self, ci: int, co: int, use_checkpoint: bool = False):
        super().__init__()
        self.pre = conv3(ci, co, s=2)
        self.res = ResBlock(co, use_checkpoint=use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.pre(x))


class Up(nn.Module):
    def __init__(
        self,
        ci: int,
        cskip: int,
        co: int,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(ci, co, 2, stride=2, bias=False)
        self.res = ResBlock(co + cskip, use_checkpoint=use_checkpoint)
        self.post = nn.Conv3d(co + cskip, co, 1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        target = [min(xs, ss) for xs, ss in zip(x.shape[2:], skip.shape[2:])]
        crop_x = [(size - tgt) // 2 for size, tgt in zip(x.shape[2:], target)]
        crop_s = [(size - tgt) // 2 for size, tgt in zip(skip.shape[2:], target)]
        x = x[
            :,
            :,
            crop_x[0] : crop_x[0] + target[0],
            crop_x[1] : crop_x[1] + target[1],
            crop_x[2] : crop_x[2] + target[2],
        ]
        skip = skip[
            :,
            :,
            crop_s[0] : crop_s[0] + target[0],
            crop_s[1] : crop_s[1] + target[1],
            crop_s[2] : crop_s[2] + target[2],
        ]
        return self.post(self.res(torch.cat([x, skip], 1)))


class ResidualUNet3D(nn.Module):
    """3D residual U-Net with CBAM bottleneck and auxiliary decoder heads."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 16,
        depth: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        widths = [base * 2**i for i in range(depth)]
        self.stem = nn.Sequential(
            conv3(in_ch, base),
            ResBlock(base, use_checkpoint=use_checkpoint),
        )
        self.enc = nn.ModuleList(
            [
                Down(widths[i - 1] if i else base, width, use_checkpoint)
                for i, width in enumerate(widths)
            ]
        )
        bot_ch = widths[-1] * 2
        self.bot = nn.Sequential(
            conv3(widths[-1], bot_ch),
            ResBlock(bot_ch, use_checkpoint=use_checkpoint),
            ResBlock(bot_ch, use_checkpoint=use_checkpoint),
            CBAM3D(bot_ch),
        )
        skip_ch = ([base] + widths[:-1])[::-1]
        dec_in_ch = [bot_ch] + skip_ch[:-1]
        self.dec = nn.ModuleList(
            [
                Up(ci, cs, cs, use_checkpoint=use_checkpoint)
                for ci, cs in zip(dec_in_ch, skip_ch)
            ]
        )
        self.head = nn.Sequential(nn.Conv3d(skip_ch[-1], 1, 1), nn.Sigmoid())
        self.aux_heads = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(c, 1, 1), nn.Sigmoid()) for c in skip_ch[:-1]]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for block in self.enc:
            x = block(x)
            skips.append(x)

        x = self.bot(skips.pop())
        aux_preds = []
        for idx, (block, skip) in enumerate(zip(self.dec, reversed(skips))):
            x = block(x, skip)
            if idx < len(self.aux_heads):
                aux_preds.append(self.aux_heads[idx](x))

        main_pred = self.head(x)
        return main_pred, aux_preds


class ELFPredictor(_LightningModuleBase):
    """Full-grid ELF predictor.

    The constructor accepts both the public ASCII hyperparameter names and the
    legacy checkpoint keys, including ``"lambda1"``, ``"lambdag"``,
    ``"lambda_hist"``, and the original Greek-key variants. ``eta_hist`` is
    kept as the state-dict parameter name for checkpoint compatibility, but it
    now weights the soft CDF distribution loss.
    """

    def __init__(
        self,
        lambda_vox: float = 1.0,
        lambda_grad: float = 0.20,
        lambda_cdf: float | None = None,
        lambda_hist: float | None = None,
        cdf_bins: int = 64,
        cdf_sigma: float = 0.02,
        cdf_tail_start: float = 0.60,
        cdf_tail_weight: float = 2.0,
        cdf_max_voxels: int = 200_000,
        hist_bins: int | None = None,
        hist_sigma: float | None = None,
        delta: float = 0.1,
        lr: float = 6e-4,
        aux_weight: float = 0.3,
        gamma_w: float = 2.0,
        weight_decay: float = 1e-4,
        base: int = 16,
        depth: int = 4,
        in_ch: int = 1,
        use_checkpoint: bool = False,
        **legacy_hparams: Any,
    ):
        super().__init__()
        lambda_vox = float(
            legacy_hparams.pop("lambda1", legacy_hparams.pop("\u03bb1", lambda_vox))
        )
        lambda_grad = float(
            legacy_hparams.pop("lambdag", legacy_hparams.pop("\u03bbg", lambda_grad))
        )
        legacy_lambda_hist = legacy_hparams.pop(
            "lambda_hist",
            legacy_hparams.pop("\u03bb_hist", None),
        )
        legacy_lambda_cdf = legacy_hparams.pop("lambda_cdf", None)
        if lambda_cdf is None:
            if legacy_lambda_cdf is not None:
                lambda_cdf = float(legacy_lambda_cdf)
            elif lambda_hist is not None:
                lambda_cdf = float(lambda_hist)
            elif legacy_lambda_hist is not None:
                lambda_cdf = float(legacy_lambda_hist)
            else:
                lambda_cdf = 0.05
        if hist_bins is not None:
            cdf_bins = int(hist_bins)
        if hist_sigma is not None:
            cdf_sigma = float(hist_sigma)
        hparams = {
            "lambda_vox": lambda_vox,
            "lambda_grad": lambda_grad,
            "lambda_cdf": float(lambda_cdf),
            "lambda_hist": float(lambda_cdf),
            "cdf_bins": int(cdf_bins),
            "cdf_sigma": float(cdf_sigma),
            "cdf_tail_start": float(cdf_tail_start),
            "cdf_tail_weight": float(cdf_tail_weight),
            "cdf_max_voxels": int(cdf_max_voxels),
            "hist_bins": int(cdf_bins),
            "hist_sigma": float(cdf_sigma),
            "delta": float(delta),
            "lr": float(lr),
            "aux_weight": float(aux_weight),
            "gamma_w": float(gamma_w),
            "weight_decay": float(weight_decay),
            "base": int(base),
            "depth": int(depth),
            "in_ch": int(in_ch),
            "use_checkpoint": bool(use_checkpoint),
        }
        hparams.update(legacy_hparams)
        try:
            self.save_hyperparameters(hparams)
        except TypeError:
            self.hparams = SimpleNamespace(**hparams)

        self.net = ResidualUNet3D(
            in_ch=int(in_ch),
            base=int(base),
            depth=int(depth),
            use_checkpoint=bool(use_checkpoint),
        )
        self.eta_vox = nn.Parameter(torch.tensor(math.log(1.0 / lambda_vox)))
        self.eta_grad = nn.Parameter(torch.tensor(math.log(1.0 / lambda_grad)))
        self.eta_hist = nn.Parameter(torch.tensor(math.log(1.0 / float(lambda_cdf))))
        torch.backends.cuda.matmul.allow_tf32 = True

    @staticmethod
    def _to_multiple_of_16(dim: int) -> int:
        return (int(dim) + 15) // 16 * 16

    def _pad16(self, volume: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        d, h, w = volume.shape[-3:]
        dm, hm, wm = map(self._to_multiple_of_16, (d, h, w))
        pad = (0, wm - w, 0, hm - h, 0, dm - d)
        return F.pad(volume, pad, mode="circular"), (int(d), int(h), int(w))

    @staticmethod
    def _gradient(volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dz = F.pad(volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        dy = F.pad(volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0))
        dx = F.pad(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))
        return dz, dy, dx

    def _soft_cdf_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Approximate 1D Wasserstein distance between ELF value distributions.

        The CDF at each threshold is smoothed with a sigmoid kernel so gradients
        flow to voxel values. High-ELF thresholds are upweighted because missed
        localization peaks are usually more important than small low-ELF shifts.
        """
        bins = int(self.hparams.cdf_bins)
        sigma = max(float(self.hparams.cdf_sigma), 1e-4)
        max_voxels = int(self.hparams.cdf_max_voxels)
        pred_flat = pred.reshape(-1).float()
        target_flat = target.reshape(-1).float()
        if max_voxels > 0 and pred_flat.numel() > max_voxels:
            idx = torch.linspace(
                0,
                pred_flat.numel() - 1,
                max_voxels,
                device=pred_flat.device,
            ).long()
            pred_flat = pred_flat.index_select(0, idx)
            target_flat = target_flat.index_select(0, idx)

        thresholds = torch.linspace(
            0.0,
            1.0,
            bins + 2,
            device=pred_flat.device,
            dtype=pred_flat.dtype,
        )[1:-1]
        pred_cdf = torch.sigmoid((thresholds[None, :] - pred_flat[:, None]) / sigma).mean(0)
        target_cdf = torch.sigmoid((thresholds[None, :] - target_flat[:, None]) / sigma).mean(0)

        tail_start = float(self.hparams.cdf_tail_start)
        tail_weight = float(self.hparams.cdf_tail_weight)
        if tail_weight != 1.0 and tail_start < 1.0:
            tail = ((thresholds - tail_start) / max(1.0 - tail_start, 1e-6)).clamp(0.0, 1.0)
            weights = 1.0 + (tail_weight - 1.0) * tail
        else:
            weights = torch.ones_like(thresholds)
        return (torch.abs(pred_cdf - target_cdf) * weights).sum() / weights.sum().clamp_min(1e-12)

    def forward(self, sad: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        sad_padded, orig_shape = self._pad16(sad)
        main_padded, aux_padded = self.net(sad_padded)
        d, h, w = orig_shape
        main = main_padded[..., :d, :h, :w]
        aux = [pred[..., :d, :h, :w] for pred in aux_padded]
        return main, aux

    def predict_elf(self, sad: torch.Tensor) -> torch.Tensor:
        """Return only the main ELF prediction for a full SAD grid."""
        pred, _ = self(sad)
        return pred

    def _loss_and_logs(self, sad: torch.Tensor, elf: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main, aux = self(sad)
        with torch.no_grad():
            denom = sad.max().clamp_min(1e-12)
            weights = (1.0 - sad / denom).clamp(min=0)
            weights = weights.pow(float(self.hparams.gamma_w))
            weights = weights / weights.mean().clamp_min(1e-12)

        diff_main_weighted = F.smooth_l1_loss(
            main,
            elf,
            beta=float(self.hparams.delta),
            reduction="none",
        )
        l_vox_main = (diff_main_weighted * weights).sum() / weights.sum().clamp_min(1e-12)
        l_vox_main_raw = F.smooth_l1_loss(main, elf, beta=float(self.hparams.delta))

        l_vox_aux = torch.zeros((), dtype=main.dtype, device=main.device)
        l_vox_aux_raw = torch.zeros((), dtype=main.dtype, device=main.device)
        for pred in aux:
            pred_up = F.interpolate(
                pred,
                size=elf.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            diff_aux = F.smooth_l1_loss(
                pred_up,
                elf,
                beta=float(self.hparams.delta),
                reduction="none",
            )
            l_vox_aux = l_vox_aux + (diff_aux * weights).sum() / weights.sum().clamp_min(1e-12)
            l_vox_aux_raw = l_vox_aux_raw + F.smooth_l1_loss(
                pred_up,
                elf,
                beta=float(self.hparams.delta),
            )
        l_vox = l_vox_main + float(self.hparams.aux_weight) * l_vox_aux / max(len(aux), 1)
        l_vox_raw = l_vox_main_raw + float(self.hparams.aux_weight) * l_vox_aux_raw / max(len(aux), 1)

        gp = self._gradient(main + 1e-6)
        gt = self._gradient(elf + 1e-6)
        wz = weights * weights.roll(-1, dims=2)
        wy = weights * weights.roll(-1, dims=3)
        wx = weights * weights.roll(-1, dims=4)
        l_grad = (
            (torch.abs(gp[0] - gt[0]) * wz).sum() / wz.sum().clamp_min(1e-12)
            + (torch.abs(gp[1] - gt[1]) * wy).sum() / wy.sum().clamp_min(1e-12)
            + (torch.abs(gp[2] - gt[2]) * wx).sum() / wx.sum().clamp_min(1e-12)
        ) / 3.0
        l_grad_raw = sum(F.l1_loss(a, b) for a, b in zip(gp, gt)) / 3.0

        l_cdf = self._soft_cdf_loss(main, elf)
        weighted_vox = l_vox * torch.exp(-self.eta_vox) + self.eta_vox
        weighted_grad = l_grad * torch.exp(-self.eta_grad) + self.eta_grad
        weighted_cdf = l_cdf * torch.exp(-self.eta_hist) + self.eta_hist
        loss = weighted_vox + weighted_grad + weighted_cdf
        logs = {
            "l_vox_raw": l_vox_raw,
            "l_grad_raw": l_grad_raw,
            "l_cdf_raw": l_cdf,
        }
        return loss, logs

    def _loss(self, sad: torch.Tensor, elf: torch.Tensor) -> torch.Tensor:
        loss, _ = self._loss_and_logs(sad, elf)
        return loss

    def _raw_losses(self, sad: torch.Tensor, elf: torch.Tensor) -> dict[str, torch.Tensor]:
        _, logs = self._loss_and_logs(sad, elf)
        return logs

    def training_step(self, batch, _):
        sad, elf = batch
        loss, logs = self._loss_and_logs(sad, elf)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        for key, value in logs.items():
            self.log(f"train/{key}", value, on_epoch=True, sync_dist=True)
        self.log("train/eta_vox", self.eta_vox.detach(), on_epoch=True, sync_dist=True)
        self.log("train/eta_grad", self.eta_grad.detach(), on_epoch=True, sync_dist=True)
        self.log("train/eta_cdf", self.eta_hist.detach(), on_epoch=True, sync_dist=True)
        self.log("train/eff_weight_vox", torch.exp(-self.eta_vox).detach(), on_epoch=True, sync_dist=True)
        self.log("train/eff_weight_grad", torch.exp(-self.eta_grad).detach(), on_epoch=True, sync_dist=True)
        self.log("train/eff_weight_cdf", torch.exp(-self.eta_hist).detach(), on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        sad, elf = batch
        loss, logs = self._loss_and_logs(sad, elf)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        for key, value in logs.items():
            self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)
        self.log("val/eta_vox", self.eta_vox.detach(), on_epoch=True, sync_dist=True)
        self.log("val/eta_grad", self.eta_grad.detach(), on_epoch=True, sync_dist=True)
        self.log("val/eta_cdf", self.eta_hist.detach(), on_epoch=True, sync_dist=True)
        self.log("val/eff_weight_vox", torch.exp(-self.eta_vox).detach(), on_epoch=True, sync_dist=True)
        self.log("val/eff_weight_grad", torch.exp(-self.eta_grad).detach(), on_epoch=True, sync_dist=True)
        self.log("val/eff_weight_cdf", torch.exp(-self.eta_hist).detach(), on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            betas=(0.9, 0.999),
            eps=1e-4,
            weight_decay=float(self.hparams.weight_decay),
        )
        try:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None) or 1000
        except RuntimeError:
            total_steps = 1000
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(self.hparams.lr),
            pct_start=0.3,
            total_steps=total_steps,
            cycle_momentum=False,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


def build_model(**kwargs) -> ELFPredictor:
    """Convenience factory retained for external scripts."""
    return ELFPredictor(**kwargs)


# Backward-compatible aliases for older imports from the previous public repo.
Sad2ElfLitModule = ELFPredictor
UNet3DPeriodic = ResidualUNet3D
