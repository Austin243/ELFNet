# ELFNet Model Card

## Model Details

This repository contains the production full-grid SAD-to-ELF model code.

- Model class: `ELFPredictor`
- Backbone: `ResidualUNet3D`
- Input: one SAD channel shaped `(B, 1, D, H, W)`
- Output: one sigmoid-bounded ELF channel plus auxiliary decoder predictions
- Inference mode: one full-grid forward pass
- Symmetry input: none
- Patch inference: none
- Default parameter count: about `10.86M`

No model checkpoint is bundled in this repository. Supply a checkpoint path at
inference time or set `ELFNET_CHECKPOINT`.

## Architecture

`ResidualUNet3D` uses:

- circular-padding 3D convolutions;
- residual blocks with GroupNorm, GELU, and squeeze-excitation;
- four downsampling stages with stride-2 circular convolutions;
- a bottleneck with two residual blocks and 3D CBAM attention;
- transposed-convolution decoder stages with skip connections;
- auxiliary decoder heads for deep supervision;
- a sigmoid final head for the main ELF prediction.

`ELFPredictor.forward` pads inputs to multiples of 16 with circular padding,
runs the backbone, and crops predictions back to the original grid shape.

## Training Objective

The production training objective combines:

- weighted SmoothL1/Huber voxel loss;
- auxiliary decoder voxel loss;
- weighted gradient matching loss;
- soft tail-weighted CDF distribution loss;
- learned dynamic weights for voxel, gradient, and CDF terms.

The voxel and gradient terms use an interstitial emphasis map derived from the
SAD input:

```text
w = (1 - sad / sad.max()) ** gamma_w
w = w / mean(w)
```

Default production loss settings:

```text
lambda_vox = 1.0
lambda_grad = 0.2
lambda_cdf = 0.05
cdf_bins = 64
cdf_sigma = 0.02
cdf_tail_start = 0.60
cdf_tail_weight = 2.0
cdf_max_voxels = 200000
delta = 0.1
aux_weight = 0.3
gamma_w = 2.0
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
- No checkpoint or dataset is included with the current repository state.
