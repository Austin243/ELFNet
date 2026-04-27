# ELFNet A/AB Sweep Dataset

This repository publishes the dataset used with the included ELFNet checkpoint
as GitHub release assets. The git repository only stores lightweight metadata;
the full NumPy triplets are distributed as split compressed archives so normal
clones stay small.

## Contents

The dataset contains matched triplets:

```text
<stem>_sad.npy
<stem>_elf.npy
<stem>_sym.npy
```

- `*_sad.npy`: superposed atomic density input grid
- `*_elf.npy`: VASP ELFCAR-derived electron localization function target grid
- `*_sym.npy`: crystallographic symmetry operations for the structure

Dataset summary:

```text
Complete triplets: 77,279
NumPy files: 231,837
Dataset size: about 21 GB unpacked
Elemental A examples: 2,158
Binary AB examples: 75,121
Unique grid shapes: 38
Symmetry operation range: 4 to 192
```

## Provenance

The data were generated from a controlled A/AB prototype sweep over metallic
elements, binary pairings, crystal prototypes, and lattice-parameter tags.

Elemental A prototypes:

```text
SC, BCC, FCC, HCP, DC, Trigonal
```

Binary AB prototypes:

```text
CsCl, HgS, NaCl, NiAs, PbO, ZB, ZW
```

The lattice-parameter sweep samples compressed and expanded fixed-cell
environments. Smaller lattice parameters create higher-density, pressure-like
environments; larger lattice parameters create lower-density environments.

Target ELF fields were computed from static PBE PAW VASP calculations with ELF
output enabled (`LELF = .TRUE.`). The processed triplets contain derived arrays
only; raw VASP POTCAR files are not included.

## Metadata

This repository includes lightweight metadata:

```text
dataset/elfnet_aab_sweep_manifest.csv.gz
dataset/elfnet_aab_sweep_stats.json
```

The manifest has one row per triplet and includes the stem, split, formula,
prototype, lattice tag, grid shape, voxel count, symmetry-operation count, and
file sizes.

## Download

The full dataset is attached to the `dataset-v1` GitHub release as split archive
parts. Download all parts plus checksums:

```bash
gh release download dataset-v1 \
  --repo Austin243/ELFNet \
  --pattern 'elfnet-aab-sweep-v1.tar.zst.part-*' \
  --pattern 'SHA256SUMS*'
```

Verify the downloaded parts:

```bash
sha256sum -c SHA256SUMS
```

Reassemble and verify the combined archive:

```bash
cat elfnet-aab-sweep-v1.tar.zst.part-* > elfnet-aab-sweep-v1.tar.zst
sha256sum -c SHA256SUMS.full
```

Extract:

```bash
mkdir -p data
tar --use-compress-program=unzstd -xf elfnet-aab-sweep-v1.tar.zst -C data
```

After extraction, the triplets will be under:

```text
data/artificial_data/
```

## Training

Use the extracted triplet directory with the training command:

```bash
elfnet-train data/artificial_data --epochs 150 --batch 2
```

The pretrained checkpoint in this repository was selected from the same
triplet format.
