# ELFNet

ELFNet predicts electron localization function (ELF) grids from superposed
atomic density (SAD) grids for periodic crystal structures.

This repository is now centered on the verified ChiNet full-grid model:

```text
local source checkpoint: /home/aellis/ChiNet/epoch1000.ckpt
repo checkpoint:         weights/elfnet_sad2elf.ckpt
model class:             ELFPredictor
architecture:            ResidualUNet3D
inference mode:          full SAD grid -> full ELF grid
symmetry input:          none
patch inference:         none
```

The bundled checkpoint exactly reproduces the flat prediction files currently
stored in `/home/aellis/ChiNet/outputs` when run through the old ChiNet
inference convention.

## Install

```bash
git lfs install
git lfs pull
python -m pip install -e .
```

For training or fine-tuning:

```bash
python -m pip install -e ".[train]"
```

## Quick Inference

Inputs are POSCAR files named `POSCAR_*`.

```bash
elfnet-predict \
  weights/elfnet_sad2elf.ckpt \
  examples/poscars \
  runs/example_outputs
```

You can also omit the checkpoint if `weights/elfnet_sad2elf.ckpt` exists or
`ELFNET_CHECKPOINT` is set:

```bash
elfnet-predict examples/poscars runs/example_outputs
```

The inference pipeline:

1. parses each `POSCAR_*`;
2. estimates a VASP-like grid from the lattice, `ENCUT=680 eV`, and `PREC=Accurate`;
3. builds the project SAD grid from packaged neutral-density tables;
4. runs one full-grid forward pass through `ELFPredictor`;
5. writes `ELFCAR_<id>.vasp`.

This model does not detect or consume crystallographic symmetry operations.
It does not tile the input into patches or blend overlapping patch predictions.

## Verified Output Provenance

The model used for the current flat files in:

```text
/home/aellis/ChiNet/outputs
```

is:

```text
/home/aellis/ChiNet/epoch1000.ckpt
```

Verification performed on 2026-04-27:

```text
rerun checkpoint: /home/aellis/ChiNet/epoch1000.ckpt
inputs:           /home/aellis/ChiNet/inputs/POSCAR_*
scratch outputs:  /home/aellis/SAD2ELFNet/provenance_chinet_outputs/epoch1000_outputs
comparison:       every ELFCAR_*.vasp matched ChiNet/outputs byte-for-byte
```

The later epoch-114 SAD2ELF checkpoint is a different symmetry-aware model
family and does not explain these output files.

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

The checkpoint has about `10.86M` parameters.

## Checkpoint Metadata

```text
file:        weights/elfnet_sad2elf.ckpt
sha256:      103cd271e736b215fb81cc8232f84b00f8243e9f0872a5704e5e5fdd452e6f51
source:      /home/aellis/ChiNet/epoch1000.ckpt
epoch:       999
global_step: 175000
format:      PyTorch Lightning checkpoint
```

Bundled legacy checkpoint hyperparameters:

```yaml
lambda_vox: 1.0
lambda_grad: 0.2
lambda_hist: 0.05
hist_bins: 30
hist_sigma: 0.02
delta: 0.1
lr: 0.0006
aux_weight: 0.3
gamma_w: 2.0
```

The legacy bundled checkpoint may store the distribution term with histogram
keys. The loader accepts those names and maps them onto the current CDF term
for checkpoint compatibility.

Current production training defaults use `lambda_cdf=0.05`, `cdf_bins=64`,
`cdf_sigma=0.02`, `cdf_tail_start=0.60`, `cdf_tail_weight=2.0`, and
`cdf_max_voxels=200000`.

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

## Repository Layout

```text
src/elfnet/model.py       ELFPredictor and ResidualUNet3D
src/elfnet/inference.py   POSCAR-to-ELFCAR full-grid inference
src/elfnet/data.py        full-grid paired SAD/ELF loaders
src/elfnet/train.py       Lightning trainer for fine-tuning
weights/                  Git LFS checkpoint
examples/poscars/         small POSCAR inference example
dataset/                  dataset metadata and manifest
```

## Limitations

- Predictions are model estimates, not substitutes for converged DFT when
  high-accuracy electronic structure is required.
- Direct comparison to VASP ELFCAR files requires both fields on a common grid.
- The SAD input is a project-defined neutral-density superposition, not VASP's
  internal `ICHARG=2` charge density.
- The visually verified ChiNet outputs come from this older full-grid model;
  they should not be cited as outputs of the later symmetry-aware epoch-114
  SAD2ELF checkpoint.
