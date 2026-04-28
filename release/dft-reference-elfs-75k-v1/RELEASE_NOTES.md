# ELFNet 75k DFT Reference ELFCAR Set v1

This release stages the selected 75,000 DFT reference ELFCAR files used for the ChiNet epoch1000 best-75k ELF maxima analysis. The selected structures are the lowest-error 75,000 rows from `reports/chinet_epoch1000_best75000_structure_manifest.tsv`, ranked by `abs(predicted_global_max - reference_global_max)`.

- Reference ELFCAR files: 75,000
- Missing ELFCAR files: 0
- Uncompressed ELFCAR bytes: 37,008,424,188 bytes, 34.47 GiB
- Source root: `/home/aellis/Databases/New_structures/Yuanhui_structures`
- Companion manifest: `dft-reference-elfs-75k-v1-manifest.tsv`

The archive is split below 2 GiB per Git LFS object. After
`git lfs pull --include="release/dft-reference-elfs-75k-v1/**"`, verify
checksums, reassemble the archive with
`cat dft-reference-elfs-75k-v1.tar.zst.part-* > dft-reference-elfs-75k-v1.tar.zst`,
then extract with
`tar --use-compress-program=unzstd -xf dft-reference-elfs-75k-v1.tar.zst`.
