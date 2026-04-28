# ELFNet Dataset Notes

The bundled `epoch1000.ckpt` model uses paired full-grid SAD/ELF arrays.

Required files for training or fine-tuning this model family:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Optional files that may exist in older dataset releases:

```text
<stem>_sym.npy
```

The current full-grid `ELFPredictor` ignores `_sym.npy` files. They are kept in
some local and release datasets because later SAD2ELF experiments used symmetry
operations, but they are not part of the verified `epoch1000.ckpt` inference or
training path.

## Full-Grid Convention

Each `*_sad.npy` and `*_elf.npy` pair must have identical shape for a given
structure. The loader yields complete unit-cell grids shaped:

```text
sad: (1, D, H, W)
elf: (1, D, H, W)
```

There is no patch extraction. Production training buckets samples by exact
grid shape, so the periodic tiling collate path is normally a no-op inside
each batch.

## Historical Checkpoint Data

The bundled `epoch1000.ckpt` checkpoint was traced to the old ChiNet full-grid
training path:

```text
/home/aellis/ChiNet/data/ELFCAR/elf_processed
```

That processed directory is no longer present locally, so the exact sample
stems cannot be recovered from the checkpoint alone. Surviving evidence points
to a July 19, 2025 batch-32 Lightning run saved under:

```text
/home/aellis/ChiNet/checkpoints_elf/batch32/ELF_20250719_202621
```

The historical run used paired `*_sad.npy` and `*_elf.npy` full grids, a
deterministic `90/10` split with seed `42`, periodic full-grid tiling in the
collate function, and no symmetry or patch loader. The current production
loader additionally buckets by exact shape before batching. The preprocessing
trail converted VASP
`ELFCAR_*.vasp` outputs into NumPy arrays and rebuilt SAD inputs from neutral
atomic densities using the old ChiNet convention.

The closest surviving local dataset with the inferred scale is:

```text
/home/aellis/TestNet/datasets/data
```

It contains `50,000` matched triplets, but it should be treated as a likely
sibling or copy rather than proven byte-identical training data for
`epoch1000.ckpt`.

## SAD Definition

The SAD input is a project-defined superposed atomic density:

1. neutral spherical atomic density tables are loaded from packaged `.pkl`
   files;
2. each atom contributes a periodic neutral density centered at its fractional
   position;
3. the grid is normalized to the configured total valence electron count;
4. the density is scaled into the convention used by the old ChiNet training
   scripts.

This SAD is not VASP's internal `ICHARG=2` density.

## ELF Definition

The target ELF grid is parsed from VASP ELFCAR volumetric output and stored as a
float32 NumPy array. Values are expected to be ELF-like and typically lie in the
range `[0, 1]`.

## Published Metadata

This repository includes lightweight metadata for the A/AB sweep dataset:

```text
dataset/elfnet_aab_sweep_manifest.csv.gz
dataset/elfnet_aab_sweep_stats.json
release/dataset-v1/
```

Those files describe a larger triplet dataset that includes SAD, ELF, and
symmetry arrays. For this full-grid checkpoint, use the SAD and ELF arrays and
ignore symmetry arrays.

Dataset summary from the metadata:

```text
Complete triplets: 77,279
NumPy files: 231,837
Dataset size: about 21 GB unpacked
Elemental A examples: 2,158
Binary AB examples: 75,121
Unique grid shapes: 38
```

## Training Example

```bash
elfnet-train /path/to/paired_arrays \
  --epochs 100 \
  --batch 32 \
  --batching shape \
  --val-frac 0.05 \
  --lambda-cdf 0.05
```

The trainer expects all required `*_sad.npy` and `*_elf.npy` files to be in the
same directory.
