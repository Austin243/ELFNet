# ELFNet Model Card

## Model Details

This release contains the verified full-grid ChiNet SAD-to-ELF model.

- Model class: `ELFPredictor`
- Backbone: `ResidualUNet3D`
- Input: one SAD channel shaped `(B, 1, D, H, W)`
- Output: one sigmoid-bounded ELF channel plus auxiliary decoder predictions
- Inference mode: one full-grid forward pass
- Symmetry input: none
- Patch inference: none
- Parameter count: about `10.86M`

This is not the later epoch-114 symmetry-aware SAD2ELF checkpoint. The current
checkpoint is the model that exactly reproduces the flat `ChiNet/outputs`
prediction files.

## Included Checkpoint

- File: `weights/elfnet_sad2elf.ckpt`
- Source artifact: `/home/aellis/ChiNet/epoch1000.ckpt`
- SHA256: `103cd271e736b215fb81cc8232f84b00f8243e9f0872a5704e5e5fdd452e6f51`
- Epoch: `999`
- Global step: `175000`
- Checkpoint format: PyTorch Lightning
- Associated TensorBoard run: `/home/aellis/ChiNet/lightning_logs/version_1196221`

Stored legacy checkpoint hyperparameters:

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

Current production training defaults replace the legacy histogram term with a
soft tail-weighted CDF term using `lambda_cdf=0.05`, `cdf_bins=64`,
`cdf_sigma=0.02`, `cdf_tail_start=0.60`, `cdf_tail_weight=2.0`, and
`cdf_max_voxels=200000`.

Recorded validation summary for the associated run:

```text
best raw voxel validation loss: about 0.0984048
last raw voxel validation loss: about 0.099468
best raw gradient validation loss: about 0.0435017
best raw distribution validation loss: about 0.0220433 under the legacy
histogram metric
```

These metrics are not directly comparable to the later SAD2ELF
`val/loss_fixed` metrics, because the architectures and loss definitions differ.

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

## Intended Use

- Fast approximate ELF generation from POSCAR structures
- Visual screening of bonding/interstitial ELF structure
- Reproducing and studying the existing flat `ChiNet/outputs` prediction set
- Fine-tuning on compatible full-grid paired SAD/ELF data

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
- Quantitative comparison to VASP ELFCAR references requires resampling to a
  common grid when grid shapes differ.
- The bundled checkpoint is visually/provenance-verified against local ChiNet
  outputs, but it is not the lowest local validation-loss checkpoint found in
  the broader workspace.
- The later epoch-114 symmetry-aware model must be evaluated separately.
