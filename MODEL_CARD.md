# ELFNet Best ChiNet Checkpoint

## Model

- Architecture: periodic 3D U-Net with patch-local Seitz symmetry averaging.
- Task: map superposed atomic density (SAD) grids to electron localization
  function (ELF) grids.
- Input: one SAD channel, shaped `(B, 1, D, H, W)`.
- Output: one ELF channel clamped to `[0, 1]`.
- Training mode: periodic patches with origin jitter and high-ELF voxel weighting.

## Best Checkpoint

- Name for release: `best_chinet_epoch0114.ckpt`
- Local source path:
  `/home/aellis/ChiNet/checkpoints_sad2elf/batch2/SAD2ELF_20251104_124933/best_epoch=0114.ckpt`
- Epoch: `114`
- Global step: `499905`
- Monitor: `val/loss_fixed`
- Best score: `0.0088327322`
- Lightning version recorded in checkpoint: `2.5.5`

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

## Provenance

This repository is centered on the best ChiNet checkpoint. The compatible model
and data-loader lineage is from `TestNet/experiments/scripts2`, because the
inventory found that current `ChiNet/scripts` does not match this checkpoint
family cleanly.

## Intended Use

- Rapid ELF prediction from POSCAR structures for screening and visualization.
- Research use where the SAD construction, neutral-density tables, and symmetry
  handling match the assumptions in this repository.
- Fine-tuning or retraining on triplet datasets containing `*_sad.npy`,
  `*_elf.npy`, and `*_sym.npy` files.

## Limitations

- Quantitative comparison to VASP ELFCAR references requires putting model and
  DFT grids on a common grid.
- The bundled SAD construction is a project-defined neutral-density
  representation, not VASP's internal `ICHARG=2` charge density.
- The best checkpoint is large enough that it should be published through Git
  LFS, a GitHub Release asset, or a model-hosting service rather than normal
  git history.
