# ELFNet Model Card

## Model Details

This repository contains the full-grid SAD-to-ELF model code.

- Model class: `ELFPredictor`
- Backbone: `FlatResNet3D`
- Architecture: base `32`, `16` flat residual blocks, kernel size `5`,
  CBAM attention every `4` blocks
- Input: one SAD channel shaped `(B, 1, D, H, W)`
- Output: one sigmoid-bounded ELF channel
- Inference mode: one full-grid forward pass
- Symmetry input: none
- Patch inference: none
- Default parameter count: `4,232,989`

The default checkpoint is `weights/elfnet.ckpt`. You can pass another
checkpoint path at inference time or set `ELFNET_CHECKPOINT`.

Bundled checkpoint provenance:

```text
source run: pressure_flatresnet_c32_b16_k5_kendall_fixed_order/ELF_20260430_123836
checkpoint epoch: 59
global step: 72780
best val/loss: -9.523093223571777
training structures: 326009
checkpoint SHA256: 66dac5953e2b93cb0629b708c77cd444b20a40daa586514ab4187b6f2c995c34
```

## Architecture

`FlatResNet3D` uses:

- circular-padding 3D convolutions;
- residual blocks with GroupNorm, GELU, and squeeze-excitation;
- same-resolution residual blocks;
- 3D CBAM attention after every four residual blocks;
- a sigmoid final head for the main ELF prediction.

`ELFPredictor.forward` runs the backbone directly on the full grid.

## Training Objective

The training objective combines:

- voxel L1 loss;
- periodic gradient loss;
- sorted-value CDF distribution loss;
- adaptive peak objective;
- learned Kendall uncertainty weights for the composite objective.

The voxel and gradient terms use an interstitial emphasis map derived from the
SAD input:

```text
w = (1 - sad / sad.max()) ** gamma_w
w = w / mean(w)
```

Default loss settings:

```text
lambda_vox = 1.0
lambda_grad = 1.0
lambda_cdf = 1.0
loss_mode = kendall
cdf_bins = 64
cdf_sigma = 0.02
cdf_tail_start = 0.60
cdf_tail_weight = 2.0
cdf_max_voxels = 20000
delta = 0.1
aux_weight = 0.0
gamma_w = 2.0
peak_margin = 0.07
peak_min_value = 0.0
peak_location_tau = 0.75
peak_location_margin = 0.07
peak_location_temperature = 0.05
peak_topk_frac = 0.002
peak_topk_min = 32
peak_topk_max = 512
peak_location_loss_weight = 1.0
peak_value_loss_weight = 1.0
false_peak_loss_weight = 1.0
kendall_log_var_clamp = [-5.0, 5.0]
```

## Intended Use

- Fast approximate ELF generation from POSCAR structures
- Visual screening of bonding/interstitial ELF structure
- Training or fine-tuning on compatible full-grid paired SAD/ELF data

## Input Assumptions

The inference pipeline builds SAD grids from packaged neutral-density tables and
normalizes them to the total configured valence electron count. This SAD is a
project-defined neutral-density superposition, not a raw VASP `ICHARG=2`
charge density.

Training/fine-tuning data should be paired full-grid arrays:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Any `<stem>_sym.npy` files are ignored by this model family.

## Limitations

- Predictions are model estimates, not DFT calculations.
- Quantitative comparison to VASP ELFCAR references requires a common grid.
- Dataset archives are available through Git LFS.
