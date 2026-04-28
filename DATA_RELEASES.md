# ELFNet Data Releases

ELFNet stores large datasets as compressed archives tracked with Git LFS. Regular
Git history contains code, small examples, documentation, checksums, and LFS
pointers.

## Datasets

| Dataset | Contents | Path |
|---|---:|---|
| `dataset-v1` | 77,279 A/AB sweep SAD/ELF/symmetry triplets, 231,837 `.npy` files | `release/dataset-v1/` |
| `pressure-triplets-326k-v1` | 326,009 pressure SAD/ELF/symmetry triplets, 978,027 `.npy` files | `release/pressure-triplets-326k-v1/` |
| `dft-reference-elfs-75k-v1` | 75,000 selected DFT reference `ELFCAR` files | `release/dft-reference-elfs-75k-v1/` |

## Download With Git LFS

Install Git LFS before cloning, or run `git lfs pull` after cloning.

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
Verify the part checksums, reassemble the archive, then extract it:

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
