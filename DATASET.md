# ELFNet Dataset Notes

All ELFNet data is available on Zenodo: <https://zenodo.org/records/20481629>.
Custom training data should use the same paired full-grid SAD/ELF NumPy
convention.

Required files:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Optional files:

```text
<stem>_sym.npy
```

The current full-grid `ELFPredictor` ignores `_sym.npy` files.

## Full-Grid Convention

Each `*_sad.npy` and `*_elf.npy` pair must have identical shape for a given
structure. The loader yields complete unit-cell grids shaped:

```text
sad: (1, D, H, W)
elf: (1, D, H, W)
```

There is no patch extraction. Shape-bucketed training groups samples by exact
grid shape.

## SAD Definition

The SAD input is a project-defined superposed atomic density:

1. neutral spherical atomic density tables are loaded from packaged `.pkl`
   files;
2. each atom contributes a periodic neutral density centered at its fractional
   position;
3. the grid is normalized to the configured total valence electron count;
4. the density is scaled into the ELFNet training convention.

This SAD is not VASP's internal `ICHARG=2` density.

## ELF Definition

The target ELF grid is parsed from VASP ELFCAR volumetric output and stored as a
float32 NumPy array. Values are expected to be ELF-like and typically lie in the
range `[0, 1]`.

## Training Example

```bash
elfnet-train /path/to/paired_arrays \
  --epochs 60 \
  --batch 16 \
  --accum 2 \
  --batching shape \
  --val-frac 0.05 \
  --lr 1e-4 \
  --arch flat_resnet \
  --base 32 \
  --flat-blocks 16 \
  --flat-kernel 5 \
  --flat-attention-every 4 \
  --loss-mode kendall \
  --lambda-grad 1.0 \
  --lambda-cdf 1.0 \
  --cdf-max-voxels 20000 \
  --val-metric loss
```

The trainer expects all required `*_sad.npy` and `*_elf.npy` files to be in the
same directory.
