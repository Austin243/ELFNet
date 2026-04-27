#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ELF+SAD Patch DataLoader (periodic, symmetry-aware) with epoch-wise origin jitter
and optional *overlapping stride* to prevent patch-phase striping.

- Produces periodic 3D patches (SAD, ELF) with per-patch Seitz {R|t'} in the patch frame.
- Jitter: every epoch, a random offset in [0, stride)^3 is added to all patch origins.
- Works in DDP and single-GPU:
    - In DDP we use a DistributedSampler subclass that calls dataset.set_epoch(epoch)
      when Lightning advances the sampler epoch.
    - In single-GPU we use a RandomSampler subclass that increments epoch on each __iter__.

Public API (unchanged for callers):
    make_patch_loaders(root, patch_size=32, batch_size=64, ...)

License: MIT
"""

from __future__ import annotations

import os
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler


# ----------------------------- utilities -----------------------------

def _is_dist_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Sample:
    stem: str
    sad_path: Path
    elf_path: Path
    sym_path: Path
    shape: Tuple[int, int, int]  # (NX, NY, NZ)


def _discover_samples(root: Path) -> List[Sample]:
    """
    Scan a directory for triplets: *_sad.npy, *_elf.npy, *_sym.npy sharing the same stem.
    """
    root = Path(root)
    sad_files = {p.stem[:-4]: p for p in root.glob("*_sad.npy")}
    elf_files = {p.stem[:-4]: p for p in root.glob("*_elf.npy")}
    sym_files = {p.stem[:-4]: p for p in root.glob("*_sym.npy")}
    stems = sorted(set(sad_files) & set(elf_files) & set(sym_files))
    samples: List[Sample] = []
    for stem in stems:
        sad_p = sad_files[stem]
        elf_p = elf_files[stem]
        sym_p = sym_files[stem]
        arr = np.load(elf_p, mmap_mode="r")  # cheap header read
        shape = tuple(int(x) for x in arr.shape)
        samples.append(Sample(stem=stem, sad_path=sad_p, elf_path=elf_p, sym_path=sym_p, shape=shape))  # type: ignore[arg-type]
    return samples


def _periodic_patch(arr: np.ndarray, ix: int, iy: int, iz: int, ps: int) -> np.ndarray:
    """Extract a ps x ps x ps patch with wrap-around."""
    xs = (ix + np.arange(ps))
    ys = (iy + np.arange(ps))
    zs = (iz + np.arange(ps))
    out = np.take(arr, xs, axis=0, mode="wrap")
    out = np.take(out, ys, axis=1, mode="wrap")
    out = np.take(out, zs, axis=2, mode="wrap")
    return out


def _sym_patchify(R_int: np.ndarray, t_frac: np.ndarray, origin_frac: np.ndarray) -> np.ndarray:
    """
    Map global Seitz {R|t} (fractional) to patch-local Seitz {R|t'} with
       t' = (R o + t - o) mod 1
    for a patch whose origin in global fractional coords is o.
    """
    Ro = np.einsum("nij,j->ni", R_int.astype(np.float64), origin_frac.astype(np.float64))  # (N_ops, 3)
    t_prime = (Ro + t_frac.astype(np.float64) - origin_frac.astype(np.float64)) % 1.0
    t_prime = t_prime.astype(np.float32)
    N = R_int.shape[0]
    seitz = np.zeros((N, 4, 4), dtype=np.float32)
    seitz[:, :3, :3] = R_int.astype(np.float32)
    seitz[:, :3, 3] = t_prime
    seitz[:, 3, 3] = 1.0
    return seitz


# ----------------------------- dataset -----------------------------

class ElfSadPatchDataset(Dataset):
    """
    Yields dicts with:
      - x            : (C, ps, ps, ps) float32    (channels in order from `channels`, default SAD, ELF)
      - sym_patch    : (R,4,4) float32            local {R|t'}
      - sym_global   : (R,4,4) float32            original {R|t}
      - sym_mask     : (R,)   bool                all True (caller may pad across batch)
      - origin_frac  : (3,)   float32             patch origin in global fractional coords
      - origin_idx   : (3,)   long                (ix, iy, iz) voxel starts
      - orig_shape   : (3,)   long                (NX, NY, NZ)
      - stem         : str

    Key additions vs your previous version:
      - `stride` controls start spacing (default == patch_size).
      - `jitter` makes every epoch use a new random offset in [0, stride)^3.
        Call `set_epoch(epoch)` to change the offset (handled automatically by the samplers below).
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 32,
        stride: Optional[int] = None,
        channels: Sequence[str] = ("sad", "elf"),
        cache_syms: bool = True,
        mmap_mode: str = "r",
        jitter: bool = True,
        seed: int = 12345,
        allowed_stems: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.ps = int(patch_size)
        assert self.ps > 0, "patch_size must be positive"
        self.stride = int(self.ps if stride is None else stride)
        assert 1 <= self.stride <= self.ps, f"stride ({self.stride}) must be in [1, patch_size]"
        self.channels = tuple(channels)
        assert set(self.channels) <= {"sad", "elf"}, "channels must be subset of {'sad','elf'}"
        self.jitter = bool(jitter)
        self._seed = int(seed)
        self._epoch = 0
        self._offset = (0, 0, 0)  # (ox, oy, oz) in [0, stride)

        # Discover & optionally filter by stem
        all_samples = _discover_samples(self.root)
        if allowed_stems is not None:
            allowed = set(allowed_stems)
            self.samples = [s for s in all_samples if s.stem in allowed]
        else:
            self.samples = all_samples
        if not self.samples:
            raise FileNotFoundError(f"No matching *_sad.npy, *_elf.npy, *_sym.npy triplets found in {root}")

        # Precompute per-sample patch grid counts (based on stride)
        self._per_sample_counts: List[Tuple[int, int, int, int]] = []  # (npx, npy, npz, total)
        for s in self.samples:
            NX, NY, NZ = s.shape
            npx = _ceil_div(NX, self.stride)
            npy = _ceil_div(NY, self.stride)
            npz = _ceil_div(NZ, self.stride)
            self._per_sample_counts.append((npx, npy, npz, npx * npy * npz))

        # Prefix sums for O(log n) global-index -> (sample,i,j,k)
        self._cum_counts: List[int] = [0]
        for (_, _, _, tot) in self._per_sample_counts:
            self._cum_counts.append(self._cum_counts[-1] + tot)
        self._total = self._cum_counts[-1]

        self._cache_syms = bool(cache_syms)
        self._sym_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # si -> (R_int, t_frac, seitz_global)

    # epoch-wise jitter API (samplers/trainer should call this once per epoch)
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if not self.jitter:
            self._offset = (0, 0, 0)
            return
        rng = np.random.RandomState(self._seed + self._epoch)
        ox = int(rng.randint(0, self.stride))
        oy = int(rng.randint(0, self.stride))
        oz = int(rng.randint(0, self.stride))
        self._offset = (ox, oy, oz)

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[int, int, int, int]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        si = bisect.bisect_right(self._cum_counts, idx) - 1
        base = self._cum_counts[si]
        local = idx - base
        npx, npy, npz, _ = self._per_sample_counts[si]
        i = local // (npy * npz)
        rem = local % (npy * npz)
        j = rem // npz
        k = rem % npz
        return si, i, j, k

    def _load_arrays(self, s: Sample) -> Tuple[np.ndarray, np.ndarray]:
        sad = np.load(s.sad_path, mmap_mode="r").astype(np.float32, copy=False)
        elf = np.load(s.elf_path, mmap_mode="r").astype(np.float32, copy=False)
        return sad, elf

    def _load_syms(self, si: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache_syms and si in self._sym_cache:
            return self._sym_cache[si]
        sym = np.load(self.samples[si].sym_path, mmap_mode="r").astype(np.float32, copy=False)
        R_int = sym[:, :3, :3].astype(np.int8)
        t_frac = sym[:, :3, 3].astype(np.float32)
        seitz_global = np.zeros_like(sym, dtype=np.float32)
        seitz_global[:, :3, :3] = R_int.astype(np.float32)
        seitz_global[:, :3, 3] = t_frac
        seitz_global[:, 3, 3] = 1.0
        if self._cache_syms:
            self._sym_cache[si] = (R_int, t_frac, seitz_global)
        return R_int, t_frac, seitz_global

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        si, I, J, K = self._locate(idx)
        s = self.samples[si]
        NX, NY, NZ = s.shape

        # Start indices with stride and epoch offset
        ox, oy, oz = self._offset
        ix = (I * self.stride + ox) % NX
        iy = (J * self.stride + oy) % NY
        iz = (K * self.stride + oz) % NZ

        sad_arr, elf_arr = self._load_arrays(s)
        sad_patch = _periodic_patch(sad_arr, ix, iy, iz, self.ps)
        elf_patch = _periodic_patch(elf_arr, ix, iy, iz, self.ps)

        chans: List[np.ndarray] = []
        if "sad" in self.channels:
            chans.append(sad_patch[None, ...])
        if "elf" in self.channels:
            chans.append(elf_patch[None, ...])
        x = np.concatenate(chans, axis=0).astype(np.float32)  # (C, ps, ps, ps)

        R_int, t_frac, seitz_global = self._load_syms(si)
        origin_frac = np.array([ix / NX, iy / NY, iz / NZ], dtype=np.float32)
        seitz_patch = _sym_patchify(R_int, t_frac, origin_frac)

        out = {
            "x": torch.from_numpy(x),
            "sym_patch": torch.from_numpy(seitz_patch),
            "sym_global": torch.from_numpy(seitz_global),
            "sym_mask": torch.ones((seitz_patch.shape[0],), dtype=torch.bool),
            "origin_frac": torch.from_numpy(origin_frac),
            "origin_idx": torch.tensor([ix, iy, iz], dtype=torch.long),
            "orig_shape": torch.tensor([NX, NY, NZ], dtype=torch.long),
            "stem": s.stem,
        }
        return out


# ----------------------------- collate -----------------------------

def collate_patches(batch: List[Dict[str, torch.Tensor | str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Pad per-item symmetry lists to the max in batch and stack tensors.

    Returns:
      X           : (B, C, ps, ps, ps) float32
      sym_batch   : (B, R_max, 4, 4) float32
      sym_mask    : (B, R_max) bool
      origin_frac : (B, 3) float32
      orig_shape  : (B, 3) long
      stems       : list[str]
    """
    B = len(batch)
    X = torch.stack([b["x"] for b in batch], dim=0)  # type: ignore[arg-type]
    origins = torch.stack([b["origin_frac"] for b in batch], dim=0)  # type: ignore[arg-type]
    shapes = torch.stack([b["orig_shape"] for b in batch], dim=0)  # type: ignore[arg-type]
    stems = [str(b["stem"]) for b in batch]

    n_ops = [int(b["sym_patch"].shape[0]) for b in batch]  # type: ignore[arg-type]
    R_max = max(n_ops)
    sym_pad = torch.zeros((B, R_max, 4, 4), dtype=torch.float32)
    mask = torch.zeros((B, R_max), dtype=torch.bool)
    for i, b in enumerate(batch):
        ops = b["sym_patch"]  # type: ignore[assignment]
        m = ops.shape[0]  # type: ignore[attr-defined]
        sym_pad[i, :m] = ops  # type: ignore[index]
        mask[i, :m] = True
    return X, sym_pad, mask, origins, shapes, stems


# ----------------------------- samplers with jitter -----------------------------

class _RandomSamplerWithEpoch(RandomSampler):
    """RandomSampler that bumps an internal epoch counter and calls dataset.set_epoch(epoch) each __iter__."""
    def __init__(self, data_source: Dataset, *, replacement: bool = False, num_samples: Optional[int] = None):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self._epoch = 0
        self._base = data_source

    def __iter__(self):
        if hasattr(self._base, "set_epoch"):
            # Call before shuffling for this epoch
            try:
                self._base.set_epoch(self._epoch)
            except Exception:
                pass
        self._epoch += 1
        return super().__iter__()


class _DistributedSamplerWithJitter(DistributedSampler):
    """DistributedSampler that forwards set_epoch(epoch) to the underlying dataset (for jitter)."""
    def __init__(self, dataset: Dataset, **kwargs):
        # Always shuffle for training
        super().__init__(dataset, shuffle=True, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        base = getattr(self, "dataset", None)
        if hasattr(base, "set_epoch"):
            try:
                base.set_epoch(epoch)
            except Exception:
                pass


# ----------------------------- loaders (train / val) -----------------------------

def make_patch_loaders(
    root: str | Path,
    patch_size: int = 32,
    stride: Optional[int] = None,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    channels: Sequence[str] = ("sad", "elf"),
    num_workers: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    jitter: bool = True,
    split_seed: int = 1337,
    shuffle_stems: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders over periodic patches **with epoch-wise jitter**.

    Notes:
      - Validation dataset uses the same stride but **jitter=False** so metrics are stable.
      - Stems are optionally shuffled with a fixed seed before splitting to avoid biased histograms.
    """
    root = Path(root)
    all_samples = _discover_samples(root)
    all_stems = [s.stem for s in all_samples]

    # Shuffle stems for a fair split
    stems = all_stems[:]
    if shuffle_stems:
        rng = np.random.RandomState(split_seed)
        rng.shuffle(stems)

    num_structs = len(stems)
    n_val = max(1, int(round(val_fraction * num_structs)))
    val_stem_set = set(stems[-n_val:])
    train_stem_set = set(stems[:-n_val])

    # Instantiate two datasets so we can disable jitter for validation
    ds_train = ElfSadPatchDataset(
        root=root, patch_size=patch_size, stride=stride, channels=channels,
        jitter=jitter, allowed_stems=train_stem_set,
    )
    ds_val = ElfSadPatchDataset(
        root=root, patch_size=patch_size, stride=stride, channels=channels,
        jitter=False, allowed_stems=val_stem_set,
    )

    # Samplers
    if _is_dist_initialized():
        train_sampler: Sampler[int] = _DistributedSamplerWithJitter(ds_train, drop_last=False)
        val_sampler: Sampler[int] = DistributedSampler(ds_val, shuffle=False, drop_last=False)
    else:
        train_sampler = _RandomSamplerWithEpoch(ds_train)
        val_sampler = SequentialSampler(ds_val)

    # Worker defaults
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if persistent_workers is None:
        persistent_workers = (num_workers or 0) > 0

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,                   # sampler handles shuffling / epoch change
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader


# ----------------------------- CLI sanity check -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ELF+SAD periodic patch DataLoader (with jitter & stride)")
    parser.add_argument("root", type=str, help="Directory with *_sad.npy, *_elf.npy, *_sym.npy triplets")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch edge length")
    parser.add_argument("--stride", type=int, default=None, help="Training stride (defaults to patch-size)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of structures for validation")
    parser.add_argument("--channels", type=str, default="sad,elf", help="Comma-separated subset of channels (sad,elf)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: SLURM_CPUS_PER_TASK or 4)")
    parser.add_argument("--no-jitter", action="store_true", help="Disable epoch-wise origin jitter (for debugging)")
    args = parser.parse_args()

    chans = tuple([c.strip() for c in args.channels.split(",") if c.strip()])
    train_loader, val_loader = make_patch_loaders(
        root=args.root,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        channels=chans,  # type: ignore[arg-type]
        num_workers=args.num_workers,
        jitter=not args.no_jitter,
    )

    # Print one batch from each split
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        try:
            batch = next(iter(loader))
        except StopIteration:
            print(f"[{split_name}] loader is empty.")
            continue
        X, sym, mask, origins, shapes, stems = batch
        print(f"[{split_name}] X: {tuple(X.shape)}  sym: {tuple(sym.shape)}  mask: {tuple(mask.shape)}")
        print(f"  stems[:3]={stems[:3]}  origin(frac)[0]={origins[0].tolist()}  shape[0]={shapes[0].tolist()}")
        break
