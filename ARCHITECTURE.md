# Architecture

ELFNet uses `ELFPredictor`, a full-grid SAD-to-ELF model.

## Top-Level Flow

```text
SAD grid (B, 1, D, H, W)
  -> FlatResNet3D
  -> main ELF prediction (B, 1, D, H, W)
```

The model returns the main prediction and an empty auxiliary list for this
backbone:

```python
main, aux = model(sad)
```

For inference, use:

```python
elf = model.predict_elf(sad)
```

## FlatResNet3D

Default production configuration:

```text
input channels: 1
base channels: 32
residual blocks: 16
kernel size: 5
attention every: 4 blocks
```

The backbone contains:

1. A circular-padding convolution stem.
2. Same-resolution residual blocks.
3. Periodic CBAM attention blocks.
4. A residual post block.
5. A sigmoid main head.

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

## Attention

The flat backbone inserts 3D CBAM blocks through the residual body:

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
periodic gradient loss
sorted-value CDF distribution loss
adaptive peak objective
learned Kendall uncertainty weights
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
aux_weight = 0.0
cdf_bins = 64
cdf_sigma = 0.02
cdf_tail_start = 0.60
cdf_tail_weight = 2.0
cdf_max_voxels = 20000
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
