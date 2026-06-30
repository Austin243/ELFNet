# ELFNet

ELFNet predicts electron localization function (ELF) grids from superposed
atomic density (SAD) grids for periodic crystal structures.

The repository includes inference and training code, a POSCAR example, and
packaged neutral-density tables.

All ELFNet data is available on Zenodo: <https://zenodo.org/records/20481629>.

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

Inputs are POSCAR files named `POSCAR_*`. By default, inference looks for
`weights/elfnet.ckpt`.

```bash
elfnet-predict \
  examples/poscars \
  runs/example_outputs
```

You can pass another checkpoint path explicitly or set `ELFNET_CHECKPOINT`.

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

See `MODEL_CARD.md` for model behavior and limitations.
