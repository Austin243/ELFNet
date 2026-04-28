# ELFNet

ELFNet predicts electron localization function (ELF) grids from superposed
atomic density (SAD) grids for periodic crystal structures.

The current repository is production code only: it does not bundle a model
checkpoint or dataset. Training data and checkpoints should be supplied
externally.

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

Inputs are POSCAR files named `POSCAR_*`. A checkpoint path is required unless
`ELFNET_CHECKPOINT` is set or you have placed a local checkpoint at
`weights/elfnet.ckpt`.

```bash
elfnet-predict \
  /path/to/checkpoint.ckpt \
  examples/poscars \
  runs/example_outputs
```

The inference pipeline:

1. parses each `POSCAR_*`;
2. estimates a VASP-like grid from the lattice, `ENCUT=680 eV`, and `PREC=Accurate`;
3. builds the project SAD grid from packaged neutral-density tables;
4. runs one full-grid forward pass through `ELFPredictor`;
5. writes `ELFCAR_<id>.vasp`.

This model does not consume crystallographic symmetry operations and does not
tile inputs into patches.

## Model Architecture

`ELFPredictor` wraps `ResidualUNet3D`.

Key details:

- input: one SAD channel, shape `(B, 1, D, H, W)`
- output: one ELF channel plus auxiliary decoder predictions
- base channels: `16`
- depth: `4`
- padding: circular 3D convolutions
- residual blocks: Conv3D, GroupNorm, GELU, Conv3D, GroupNorm, squeeze-excitation
- bottleneck: two residual blocks plus 3D CBAM attention
- decoder: transposed-convolution upsampling with skip connections
- output head: sigmoid-bounded ELF prediction
- grid handling: pad to multiples of 16, run full-grid model, crop back

The default production configuration has about `10.86M` parameters.


## Large Data Releases

Large datasets are distributed as Git LFS archive assets under `release/`.
The current layout is documented in `DATA_RELEASES.md`:

- `dataset-v1`: 77,279 A/AB sweep SAD/ELF/symmetry triplets.
- `pressure-triplets-326k-v1`: 326,009 SAD/ELF/symmetry triplets for training.
- `dft-reference-elfs-75k-v1`: 75,000 selected DFT reference ELFCAR files for the ChiNet epoch1000 best-75k analysis.

## Training And Fine-Tuning Data

Training uses paired NumPy arrays:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Each pair must have identical full-grid shape. The loader yields complete
unit-cell grids, not patches. Production training buckets samples by exact
grid shape, so periodic tiling is normally a no-op inside each batch.

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

Current production training defaults use `lambda_cdf=0.05`, `cdf_bins=64`,
`cdf_sigma=0.02`, `cdf_tail_start=0.60`, `cdf_tail_weight=2.0`, and
`cdf_max_voxels=200000`.

The objective combines weighted voxel loss, auxiliary decoder loss, weighted
gradient loss, a soft tail-weighted CDF distribution loss, and learned dynamic
weights for the three main terms.

## Repository Layout

```text
src/elfnet/model.py       ELFPredictor and ResidualUNet3D
src/elfnet/inference.py   POSCAR-to-ELFCAR full-grid inference
src/elfnet/data.py        full-grid paired SAD/ELF loaders
src/elfnet/train.py       Lightning trainer for training/fine-tuning
configs/default.yaml      production architecture/training defaults
examples/poscars/         small POSCAR inference example
```

See `MODEL_CARD.md` for model details and limitations. See `DATASET.md` for
the expected external training-data format.
