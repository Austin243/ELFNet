# ELFNet Data Releases

ELFNet keeps raw large datasets out of regular Git history. The code repository
tracks source, configuration, small examples, documentation, metadata, and Git
LFS pointers for immutable compressed dataset archives.

This follows GitHub guidance for repository health: regular Git files over 100 MiB
are rejected and repositories are recommended to stay small. The large archive
parts are therefore stored through Git LFS, not as normal Git blobs.

## Published LFS Datasets

| Release tag | Contents | Local staging directory |
|---|---:|---|
| `dataset-v1` | 77,279 A/AB sweep SAD/ELF/symmetry triplets, 231,837 `.npy` files | `release/dataset-v1/` |
| `pressure-triplets-326k-v1` | 326,009 pressure SAD/ELF/symmetry triplets, 978,027 `.npy` files | `release/pressure-triplets-326k-v1/` |
| `dft-reference-elfs-75k-v1` | 75,000 selected DFT reference `ELFCAR` files used by the ChiNet epoch1000 best-75k analysis | `release/dft-reference-elfs-75k-v1/` |

`pressure-triplets-326k-v1` is generated from
`/home/aellis/SAD2ELFNet/datasets/pressure_triplets`. The source structures and
pressure dataset provenance live in the local SAD2ELFNet and mp-aloe extension
workspaces; this repository publishes the training arrays and manifests.

## Download With Git LFS

Install Git LFS before cloning, or run `git lfs pull` after cloning:

```bash
git lfs install
git clone git@github.com:Austin243/ELFNet.git
cd ELFNet
git lfs pull --include="release/**"
```

To fetch only one dataset:

```bash
git lfs pull --include="release/pressure-triplets-326k-v1/**"
```

## Extract Single-Archive Datasets

`dataset-v1` and `dft-reference-elfs-75k-v1` use one split archive each.
Download all parts and metadata for the dataset, verify the part checksums,
then reassemble the archive:

```bash
cd release/dft-reference-elfs-75k-v1
sha256sum -c SHA256SUMS
cat dft-reference-elfs-75k-v1.tar.zst.part-* > dft-reference-elfs-75k-v1.tar.zst
sha256sum -c SHA256SUMS.full
tar --use-compress-program=unzstd -xf dft-reference-elfs-75k-v1.tar.zst
```

Use the same pattern for `release/dataset-v1/elfnet-aab-sweep-v1.tar.zst.part-*`.

## Extract Pressure 326k Shards

`pressure-triplets-326k-v1` is split into 32 shard archives. Shard 006 is
itself split into three parts to keep every LFS object below 2 GiB.

```bash
cd release/pressure-triplets-326k-v1
sha256sum -c SHA256SUMS
mkdir -p pressure_triplets
for shard in $(seq -f "%03g" 0 31); do
  cat pressure-triplets-326k-v1-shard${shard}.tar.zst.part-* > pressure-triplets-326k-v1-shard${shard}.tar.zst
done
sha256sum -c SHA256SUMS.full
for archive in pressure-triplets-326k-v1-shard*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$archive" -C pressure_triplets
done
gzip -dk pressure-triplets-326k-v1-manifest.tsv.gz
```

The triplet archives extract to paired NumPy arrays named:

```text
<stem>_sad.npy
<stem>_elf.npy
<stem>_sym.npy
```

Current ELFNet training consumes `_sad.npy` and `_elf.npy`; `_sym.npy` is kept
for provenance and compatibility with older SAD2ELF experiments.

The 75k DFT reference archive extracts selected reference `ELFCAR` files under
paths matching the source Yuanhui structure tree. Use its manifest to map each
archive path back to the selected-set rank, prototype, substituted elements, and
ELF maxima values.
