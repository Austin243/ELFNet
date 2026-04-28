# ELFNet Pressure Triplets 326k v1

Full-grid pressure triplets containing SAD, ELF, and symmetry arrays.

- Manifest rows: 326,009
- NumPy files: 978,027
- Unique grid shapes: 77
- Shards: 32

Fetch with `git lfs pull --include="release/pressure-triplets-326k-v1/**"`.
Verify with `SHA256SUMS` and `SHA256SUMS.full`.

```bash
cd release/pressure-triplets-326k-v1
mkdir -p pressure_triplets
for shard in $(seq -f "%03g" 0 31); do
  cat pressure-triplets-326k-v1-shard${shard}.tar.zst.part-* > pressure-triplets-326k-v1-shard${shard}.tar.zst
done
for archive in pressure-triplets-326k-v1-shard*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$archive" -C pressure_triplets
done
```
