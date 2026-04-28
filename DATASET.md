# ELFNet Dataset Notes

This repository does not include raw training arrays in regular Git history.
Large immutable datasets are stored as Git LFS archive assets under `release/`;
see `DATA_RELEASES.md` for the current 326,009-triplet training set, the earlier
A/AB sweep dataset, and the separate 75,000-structure DFT reference ELFCAR set.
For custom training, use an external dataset of paired full-grid SAD/ELF NumPy
arrays.

Required files:

```text
<stem>_sad.npy
<stem>_elf.npy
```

Optional files that may exist in older local datasets:

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

There is no patch extraction. Production training buckets samples by exact
grid shape, so the periodic tiling collate path is normally a no-op inside
each batch.

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
  --epochs 100 \
  --batch 32 \
  --batching shape \
  --val-frac 0.05 \
  --lambda-cdf 0.05
```

The trainer expects all required `*_sad.npy` and `*_elf.npy` files to be in the
same directory.
