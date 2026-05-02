# ELFNet

ELFNet predicts electron localization function (ELF) grids from superposed
atomic density (SAD) grids for periodic crystal structures.

The repository includes inference and training code, a POSCAR example, packaged
neutral-density tables, a default checkpoint, and Git LFS dataset archives.

## Install

```bash
git clone git@github.com:Austin243/ELFNet.git
cd ELFNet
python -m pip install -e .
```

For training or fine-tuning:

```bash
python -m pip install -e ".[train]"
```

## Quick Inference

Inputs are POSCAR files named `POSCAR_*`. By default, inference uses
`weights/elfnet.ckpt`.

```bash
elfnet-predict \
  examples/poscars \
  runs/example_outputs
```

You can pass another checkpoint path explicitly or set `ELFNET_CHECKPOINT`.

The inference pipeline:

1. parses each `POSCAR_*`;
2. estimates a VASP-like grid from the lattice, `ENCUT=680 eV`, and `PREC=Accurate`;
3. builds the project SAD grid from packaged neutral-density tables;
4. runs one full-grid forward pass through `ELFPredictor`;
5. writes `ELFCAR_<id>.vasp`.

This model does not consume crystallographic symmetry operations and does not
tile inputs into patches.

## Model Architecture

`ELFPredictor` wraps `FlatResNet3D`.

Key details:

- input: one SAD channel, shape `(B, 1, D, H, W)`
- output: one ELF channel
- base channels: `32`
- residual blocks: `16`
- kernel size: `5`
- padding: circular 3D convolutions
- residual blocks: Conv3D, GroupNorm, GELU, Conv3D, GroupNorm, squeeze-excitation
- attention: 3D CBAM every four residual blocks
- output head: sigmoid-bounded ELF prediction
- grid handling: full-grid same-resolution inference

The default configuration has about `4.23M` parameters.

## Datasets

Large datasets are stored as Git LFS archive assets under `release/`. See
`DATA_RELEASES.md` for download and extraction commands.

- `dataset-v1`: 77,279 A/AB sweep SAD/ELF/symmetry triplets.
- `pressure-triplets-326k-v1`: 326,009 SAD/ELF/symmetry triplets for training.
- `dft-reference-elfs-75k-v1`: 75,000 selected DFT reference ELFCAR files.

## Training And Fine-Tuning Data

Training uses paired NumPy arrays:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Each pair must have identical full-grid shape. The loader yields complete
unit-cell grids, not patches. Shape-bucketed training groups samples by exact
grid shape.

If a dataset also contains `<stem>_sym.npy`, those files are ignored by this
model family.

Fine-tuning example:

```bash
elfnet-train /path/to/paired_sad_elf_arrays \
  --epochs 100 \
  --batch 32 \
  --batching shape \
  --val-frac 0.05 \
  --lr 6e-4 \
  --lambda-cdf 0.05 \
  --cdf-bins 64 \
  --cdf-tail-start 0.6 \
  --cdf-tail-weight 2.0
```

## Loss

Default training settings use `lambda_cdf=0.05`, `cdf_bins=64`,
`cdf_sigma=0.02`, `cdf_tail_start=0.60`, `cdf_tail_weight=2.0`, and
`cdf_max_voxels=200000`.

The objective combines voxel loss, periodic gradient loss, sorted-value CDF
distribution loss, an adaptive peak objective, and learned Kendall uncertainty
weights.

## Repository Layout

```text
src/elfnet/model.py       ELFPredictor and model backbones
src/elfnet/inference.py   POSCAR-to-ELFCAR full-grid inference
src/elfnet/data.py        full-grid paired SAD/ELF loaders
src/elfnet/train.py       Lightning trainer for training/fine-tuning
configs/default.yaml      architecture/training defaults
examples/poscars/         small POSCAR inference example
```

See `MODEL_CARD.md` for model details and limitations. See `DATASET.md` for
the expected external training-data format.
