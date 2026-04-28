# ELFNet A/AB Sweep Dataset v1

This release contains the A/AB sweep dataset metadata and split compressed
archives.

The archive contains triplets:

```text
<stem>_sad.npy
<stem>_elf.npy
<stem>_sym.npy
```

For the current bundled checkpoint, `weights/elfnet_sad2elf.ckpt`
(`ELFPredictor` / `epoch1000.ckpt`), only the SAD and ELF arrays are used.
The symmetry arrays are retained for historical compatibility with later
SAD2ELF experiments, but they are ignored by the full-grid model.

Summary:

```text
complete triplets: 77,279
unpacked size: about 21 GB
elemental A examples: 2,158
binary AB examples: 75,121
unique grid shapes: 38
```

The archive is split below 2 GiB per Git LFS object. After
`git lfs pull --include="release/dataset-v1/**"`, use `SHA256SUMS` and
`SHA256SUMS.full` to verify the downloaded archive parts.
