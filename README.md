# ELFNet

ELFNet predicts electron localization function (ELF) grids from superposed
atomic density (SAD) grids for periodic crystal structures.

The package includes:

- a pretrained symmetry-aware 3D U-Net checkpoint
- POSCAR-to-ELFCAR inference utilities
- periodic patch data loaders for training and fine-tuning
- bundled neutral-density tables for SAD construction
- a published A/AB sweep dataset release for training and evaluation

## Install

```bash
git clone git@github.com:Austin243/ELFNet.git
cd ELFNet
git lfs install
git lfs pull
python -m pip install -e ".[symmetry]"
```

`pymatgen` is optional but recommended for symmetry detection. Without it,
inference can still run with `--identity-symmetry`.

Training additionally needs Lightning:

```bash
python -m pip install -e ".[train,symmetry]"
```

## Predict ELF From POSCAR Files

Inputs must be files named `POSCAR_*`. Outputs are written as `ELFCAR_*.vasp`.

```bash
elfnet-predict \
  weights/elfnet_sad2elf.ckpt \
  examples/poscars \
  runs/example_outputs \
  --device auto \
  --batch-size 8
```

The inference pipeline builds a SAD grid from the POSCAR, estimates a
VASP-style FFT grid from `ENCUT`, applies symmetry-aware patch inference, and
stitches predictions into an ELFCAR-like volumetric file.

## Fine-Tune Or Train

Training data should be stored as matched triplets sharing a stem:

```text
<stem>_sad.npy
<stem>_elf.npy
<stem>_sym.npy
```

Run:

```bash
elfnet-train /path/to/triplets --epochs 150 --batch 2
```

The dataset used with the included checkpoint is available as split GitHub
release assets. See `DATASET.md` for provenance, statistics, and download
instructions.

The default training configuration matches the included pretrained model:
`base=24`, `depth=5`, `blocks_per_stage=1`, `sym_every_stage=True`,
`lr=3e-4`, and `high_value_weight=5.0`.

## Layout

```text
src/elfnet/model.py       periodic symmetry-aware U-Net
src/elfnet/data.py        SAD/ELF patch datasets and loaders
src/elfnet/inference.py   POSCAR-to-ELFCAR prediction pipeline
src/elfnet/checkpoints.py checkpoint metadata and loading helpers
configs/default.yaml      default model and training configuration
dataset/                  lightweight dataset manifest and statistics
weights/                  pretrained checkpoint tracked with Git LFS
examples/poscars/         small POSCAR example for smoke testing
```

See `MODEL_CARD.md` for model details, intended use, and limitations. See
`DATASET.md` for dataset download and provenance.
