# Architecture

ELFNet uses `ELFPredictor`, a full-grid SAD-to-ELF model.

## Top-Level Flow

```text
SAD grid (B, 1, D, H, W)
  -> circular pad to multiples of 16
  -> ResidualUNet3D
  -> crop back to original D, H, W
  -> main ELF prediction (B, 1, D, H, W)
```

The model returns both the main prediction and auxiliary decoder predictions:

```python
main, aux = model(sad)
```

For inference, use:

```python
elf = model.predict_elf(sad)
```

## ResidualUNet3D

Default production configuration:

```text
input channels: 1
base channels: 16
depth: 4
bottleneck channels: 256
```

The backbone contains:

1. A circular-padding convolution stem.
2. Four downsampling blocks.
3. A CBAM attention bottleneck.
4. Four transposed-convolution upsampling blocks with skip connections.
5. A sigmoid main head.
6. Auxiliary sigmoid heads on decoder stages.

## Residual Blocks

Each residual block contains:

```text
circular Conv3d
GroupNorm
GELU
circular Conv3d
GroupNorm
squeeze-excitation
residual add
GELU
```

Circular padding is the only explicit periodic treatment in this architecture.

## Bottleneck Attention

The bottleneck uses a 3D CBAM block:

```text
channel attention: global average/max pooling -> 1x1 Conv3d MLP -> sigmoid
spatial attention: channel average/max maps -> Conv3d -> sigmoid
```

## Output Bounds

The final head uses `Sigmoid`, so the main predicted ELF channel is bounded in
`[0, 1]` before writing.

## Loss Used During Training

The production training objective combines:

```text
weighted SmoothL1 voxel loss
auxiliary decoder voxel loss
weighted gradient loss
soft tail-weighted CDF distribution loss
learned dynamic uncertainty weights
```

The SAD-derived interstitial weight map is:

```text
w = (1 - sad / sad.max()) ** gamma_w
w = w / mean(w)
```

Current production training defaults:

```text
gamma_w = 2.0
delta = 0.1
aux_weight = 0.3
cdf_bins = 64
cdf_sigma = 0.02
cdf_tail_start = 0.60
cdf_tail_weight = 2.0
cdf_max_voxels = 200000
```

## What The Model Does Not Do

This model does not use:

```text
crystallographic symmetry operations
Seitz matrices
patch-local symmetry transforms
patch extraction
overlap blending
```
