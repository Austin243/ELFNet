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

## Bundled Model

The bundled checkpoint is a full-grid `ELFPredictor` with a `FlatResNet3D`
backbone:

- checkpoint: `weights/elfnet.ckpt`
- source run: `pressure_flatresnet_c32_b16_k5_kendall_fixed_order/ELF_20260430_123836`
- checkpoint epoch: `59`
- global step: `72780`
- best monitored `val/loss`: `-9.523093223571777`
- SHA256: `66dac5953e2b93cb0629b708c77cd444b20a40daa586514ab4187b6f2c995c34`
- architecture: base `32`, `16` flat residual blocks, kernel size `5`,
  CBAM attention every `4` blocks
- parameter count: `4,232,989`
- training set: `326,009` full-grid pressure SAD/ELF triplets
- training objective: Kendall-weighted voxel, periodic-gradient, sorted-CDF,
  and adaptive-peak losses

The inference pipeline:

1. parses each `POSCAR_*`;
2. estimates a VASP-like grid from the lattice, `ENCUT=680 eV`, and `PREC=Accurate`;
3. builds the project SAD grid from packaged neutral-density tables;
4. runs one full-grid forward pass through `ELFPredictor`;
5. writes `ELFCAR_<id>.vasp`.

## Datasets

Large datasets are stored as Git LFS archive assets under `release/`. See
`DATA_RELEASES.md` for download and extraction commands.

- `pressure-triplets-326k-v1`: 326,009 SAD/ELF triplets for training.
- `dft-reference-elfs-75k-v1`: 75,000 DFT reference ELFCAR files.

## Training And Fine-Tuning Data

Training uses paired NumPy arrays:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Each pair must have identical full-grid shape. Shape-bucketed training groups
samples by exact grid shape.

Fine-tuning example:

```bash
elfnet-train /path/to/paired_sad_elf_arrays \
  --epochs 60 \
  --batch 16 \
  --accum 2 \
  --batching shape \
  --val-frac 0.05 \
  --lr 1e-4 \
  --arch flat_resnet \
  --base 32 \
  --flat-blocks 16 \
  --flat-kernel 5 \
  --flat-attention-every 4 \
  --loss-mode kendall \
  --lambda-grad 1.0 \
  --lambda-cdf 1.0 \
  --cdf-max-voxels 20000 \
  --val-metric loss
```

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
