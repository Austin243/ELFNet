# ELFNet

ELFNet is a clean release directory for the best-performing ChiNet SAD-to-ELF
model found in the local model inventory.

The released architecture is a periodic, symmetry-aware 3D U-Net trained on
SAD/ELF patches with Seitz symmetry pooling. The canonical checkpoint is:

```text
best_chinet_epoch0114.ckpt
source: /home/aellis/ChiNet/checkpoints_sad2elf/batch2/SAD2ELF_20251104_124933/best_epoch=0114.ckpt
val/loss_fixed: 0.0088327322
epoch: 114
global_step: 499905
```

The checkpoint is 334 MB and is tracked with Git LFS rather than normal git.
After cloning, install Git LFS and pull the model file:

```bash
git lfs install
git lfs pull
```

## Install

```bash
cd ~/github/ELFNet
python -m pip install -e ".[symmetry]"
```

`pymatgen` is optional but recommended. Without it, inference can run with
identity symmetry by passing `--identity-symmetry`. Training additionally needs
Lightning:

```bash
python -m pip install -e ".[train,symmetry]"
```

## Predict ELF from POSCAR files

```bash
elfnet-predict \
  weights/best_chinet_epoch0114.ckpt \
  examples/poscars \
  runs/example_outputs \
  --device auto \
  --batch-size 8
```

Inputs must be files named `POSCAR_*`. Outputs are written as `ELFCAR_*.vasp`.
The package includes the neutral-density tables needed to construct the SAD
input grids.

## Train or fine-tune

Training data must be triplets sharing a stem:

```text
<stem>_sad.npy
<stem>_elf.npy
<stem>_sym.npy
```

Run:

```bash
elfnet-train /path/to/triplets --epochs 150 --batch 2
```

The default architecture and loss settings match the best ChiNet checkpoint:
`base=24`, `depth=5`, `blocks_per_stage=1`, `sym_every_stage=True`,
`lr=3e-4`, and `high_value_weight=5.0`.

## Repository layout

```text
src/elfnet/model.py       periodic symmetry-aware U-Net and Lightning module
src/elfnet/data.py        SAD/ELF patch dataset and DDP-safe loaders
src/elfnet/inference.py   POSCAR-to-ELFCAR prediction pipeline
src/elfnet/checkpoints.py checkpoint metadata and loading helpers
configs/best_chinet.yaml  canonical checkpoint configuration
examples/poscars/         small POSCAR example for smoke testing
```

See `MODEL_CARD.md` for provenance, expected inputs, and current limitations.
