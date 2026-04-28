# ELFNet Pressure Triplets 326k v1

This release stages the full pressure-triplet dataset generated under `/home/aellis/SAD2ELFNet/datasets/pressure_triplets`. The archive contains paired full-grid SAD and ELF NumPy arrays plus the historical symmetry array for each structure. Current ELFNet training uses the `_sad.npy` and `_elf.npy` files and ignores `_sym.npy`.

- Manifest rows: 326,009
- Complete triplets: 326,009
- NumPy files: 978,027
- Unique grid shapes: 77

The dataset is packaged as 32 shard archives. Every Git LFS object is split
below 2 GiB; shard 006 has three parts and all other shards currently have one
part.

After `git lfs pull --include="release/pressure-triplets-326k-v1/**"`, verify
the part checksums, reassemble each shard, verify whole-shard checksums, and
extract:

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
