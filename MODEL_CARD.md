# ELFNet Model Card

## Model Details

ELFNet is a periodic, symmetry-aware 3D U-Net for predicting electron
localization function (ELF) fields from superposed atomic density (SAD) fields.

- Input: one SAD channel shaped `(B, 1, D, H, W)`
- Output: one ELF channel clipped to `[0, 1]`
- Architecture: periodic 3D U-Net with patch-local Seitz symmetry averaging
- Training mode: periodic patch training with origin jitter
- Loss: weighted Smooth L1 with additional emphasis on high-ELF voxels

## Included Checkpoint

- File: `weights/elfnet_sad2elf.ckpt`
- Epoch: `114`
- Global step: `499905`
- Validation metric: weighted Smooth L1
- Validation score: `0.0088327322`
- Checkpoint format: PyTorch Lightning

Checkpoint hyperparameters:

```yaml
sad_idx: 0
elf_idx: 1
base: 24
depth: 5
blocks_per_stage: 1
dropout: 0.0
sym_every_stage: true
lr: 0.0003
weight_decay: 0.0001
max_epochs: 501
high_value_margin: 0.25
high_value_weight: 5.0
min_patch_size: 32
val_metric: loss/vox
```

## Intended Use

- Fast ELF prediction from POSCAR structures
- Screening workflows where approximate ELF fields are useful before running
  more expensive electronic-structure calculations
- Visualization and analysis of periodic materials
- Fine-tuning on compatible SAD/ELF/symmetry triplet datasets

## Input Assumptions

The bundled inference pipeline constructs SAD grids from neutral-density tables
included in the package. For best results, use the same SAD construction for
training, fine-tuning, and inference.

## Limitations

- ELFNet predictions are model estimates, not replacements for converged DFT
  calculations when high-accuracy electronic structure is required.
- Direct quantitative comparison to VASP ELFCAR files requires putting both
  fields on a common grid.
- The SAD representation used here is a project-defined neutral-density input,
  not VASP's internal `ICHARG=2` charge density.
- Git LFS is required to retrieve the included checkpoint after cloning.
