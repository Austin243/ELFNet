#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-grid SAD/ELF data loaders for the verified ELFNet checkpoint family.

The ``epoch1000.ckpt`` model was trained on paired ``*_sad.npy`` and
``*_elf.npy`` grids. Each dataset item is an entire periodic unit-cell grid,
not a patch. Production loaders bucket samples by exact grid shape, so the
periodic tiling collate path is normally a no-op inside each batch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class Sample:
    stem: str
    sad_path: Path
    elf_path: Path
    shape: Tuple[int, int, int]


class ShapeBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield same-shape batches, with deterministic rank sharding for DDP."""

    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        shuffle: bool,
        seed: int,
        world_size: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.world_size = max(int(world_size), 1)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.by_shape: Dict[Tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            self.by_shape[sample.shape].append(idx)
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} outside world_size {self.world_size}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batches(self) -> list[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        batches: list[list[int]] = []
        for shape in sorted(self.by_shape):
            indices = list(self.by_shape[shape])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        if self.world_size > 1 and batches:
            remainder = len(batches) % self.world_size
            if remainder:
                pad_count = self.world_size - remainder
                batches.extend([list(batches[i % len(batches)]) for i in range(pad_count)])
            batches = batches[self.rank :: self.world_size]
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.by_shape.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += int(math.ceil(len(indices) / self.batch_size))
        return int(math.ceil(total / self.world_size))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _world_info() -> tuple[int, int]:
    world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    rank = _env_int("RANK", _env_int("SLURM_PROCID", 0))
    return max(world_size, 1), max(rank, 0)


def _discover_samples(root: Path) -> List[Sample]:
    """Find paired ``*_sad.npy`` and ``*_elf.npy`` files."""
    root = Path(root)
    manifest = root / "manifest.tsv"
    if manifest.exists():
        samples: List[Sample] = []
        with manifest.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            try:
                status_idx = header.index("status")
                stem_idx = header.index("stem")
                shape_idx = header.index("shape")
            except ValueError:
                samples = []
            else:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(status_idx, stem_idx, shape_idx):
                        continue
                    if parts[status_idx] != "wrote":
                        continue
                    dims = tuple(int(dim) for dim in parts[shape_idx].split("x"))
                    if len(dims) != 3:
                        continue
                    stem = parts[stem_idx]
                    samples.append(
                        Sample(
                            stem=stem,
                            sad_path=root / f"{stem}_sad.npy",
                            elf_path=root / f"{stem}_elf.npy",
                            shape=dims,  # type: ignore[arg-type]
                        )
                    )
        if samples:
            return samples

    sad_files = {p.stem[:-4]: p for p in root.glob("*_sad.npy")}
    elf_files = {p.stem[:-4]: p for p in root.glob("*_elf.npy")}
    stems = sorted(set(sad_files) & set(elf_files))
    samples: List[Sample] = []
    for stem in stems:
        sad_path = sad_files[stem]
        elf_path = elf_files[stem]
        arr = np.load(elf_path, mmap_mode="r")
        shape = tuple(int(x) for x in arr.shape)
        samples.append(
            Sample(
                stem=stem,
                sad_path=sad_path,
                elf_path=elf_path,
                shape=shape,  # type: ignore[arg-type]
            )
        )
    return samples


def _tile_to_shape(tensor: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """Periodically tile ``tensor`` shaped ``(1,D,H,W)`` to ``shape``."""
    _, d, h, w = tensor.shape
    rep_d = int(math.ceil(shape[0] / d))
    rep_h = int(math.ceil(shape[1] / h))
    rep_w = int(math.ceil(shape[2] / w))
    tiled = tensor.repeat(1, rep_d, rep_h, rep_w)
    return tiled[:, : shape[0], : shape[1], : shape[2]].contiguous()


class SadElfDataset(Dataset):
    """Dataset yielding full paired SAD and ELF grids.

    ``__getitem__`` returns ``(sad, elf)`` tensors, each shaped ``(1,D,H,W)``.
    SAD and ELF shapes must already match for each structure.
    """

    def __init__(
        self,
        data_dir: str | Path,
        allowed_stems: Optional[Iterable[str]] = None,
        samples: Optional[Sequence[Sample]] = None,
        mmap_mode: str | None = "r",
    ) -> None:
        super().__init__()
        self.root = Path(data_dir).expanduser()
        samples = list(samples) if samples is not None else _discover_samples(self.root)
        if allowed_stems is not None:
            allowed = set(allowed_stems)
            samples = [sample for sample in samples if sample.stem in allowed]
        if not samples:
            raise FileNotFoundError(f"No paired *_sad.npy and *_elf.npy files in {self.root}")
        self.samples = samples
        self.mmap_mode = mmap_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sad = np.load(sample.sad_path, mmap_mode=self.mmap_mode).astype(np.float32)
        elf = np.load(sample.elf_path, mmap_mode=self.mmap_mode).astype(np.float32)
        if sad.shape != elf.shape:
            raise ValueError(
                f"SAD and ELF grids differ for {sample.stem}: {sad.shape} vs {elf.shape}"
            )
        return torch.from_numpy(np.asarray(sad))[None], torch.from_numpy(np.asarray(elf))[None]


def collate_full_grids(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate full grids by periodically tiling each sample to batch max shape."""
    shapes = [sad.shape[-3:] for sad, _ in batch]
    target = tuple(max(shape[axis] for shape in shapes) for axis in range(3))
    sad_batch = torch.stack([_tile_to_shape(sad, target) for sad, _ in batch])
    elf_batch = torch.stack([_tile_to_shape(elf, target) for _, elf in batch])
    return sad_batch, elf_batch


def split_stems(
    root: str | Path,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return deterministic train/validation stem lists."""
    samples = _discover_samples(Path(root).expanduser())
    stems = [sample.stem for sample in samples]
    if not stems:
        raise FileNotFoundError(f"No paired *_sad.npy and *_elf.npy files in {root}")
    rng = np.random.RandomState(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(float(val_frac) * len(stems))))
    return stems[:-n_val] or stems, stems[-n_val:]


def split_samples(
    samples: Sequence[Sample],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Return deterministic train/validation sample lists without rediscovery."""
    samples = list(samples)
    if not samples:
        raise ValueError("No samples to split")
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    n_val = max(1, int(round(float(val_frac) * len(samples))))
    return samples[:-n_val] or samples, samples[-n_val:]


def make_loaders(
    data_dir: str | Path,
    batch: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int = 42,
    val_frac: float = 0.1,
    val_fraction: float | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    batching: str = "shape",
) -> tuple[DataLoader, DataLoader]:
    """Build full-grid train/validation loaders for paired SAD/ELF arrays."""
    if batch is None:
        batch = 1 if batch_size is None else int(batch_size)
    if val_fraction is not None:
        val_frac = val_fraction
    samples = _discover_samples(Path(data_dir).expanduser())
    train_samples, val_samples = split_samples(samples, val_frac=val_frac, seed=seed)
    train_ds = SadElfDataset(data_dir, samples=train_samples)
    val_ds = SadElfDataset(data_dir, samples=val_samples)

    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    world_size, rank = _world_info()
    batching = str(batching).lower()
    if batching not in {"shape", "random"}:
        raise ValueError(f"batching must be 'shape' or 'random', got {batching!r}")

    if batching == "shape":
        common = dict(
            collate_fn=collate_full_grids,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ShapeBucketBatchSampler(
                train_ds.samples,
                batch_size=batch,
                shuffle=True,
                seed=seed,
                world_size=world_size,
                rank=rank,
            ),
            **common,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=ShapeBucketBatchSampler(
                val_ds.samples,
                batch_size=batch,
                shuffle=False,
                seed=seed,
                world_size=world_size,
                rank=rank,
            ),
            **common,
        )
        return train_loader, val_loader

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, seed=seed, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    common = dict(
        batch_size=batch,
        collate_fn=collate_full_grids,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


# Compatibility aliases for older imports; these now return full-grid samples.
make_full_grid_loaders = make_loaders
ElfSadFullDataset = SadElfDataset
