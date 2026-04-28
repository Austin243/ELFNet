# Provenance

This file records the local audit that changed the repository from the later
epoch-114 symmetry-aware SAD2ELF checkpoint to the verified full-grid ChiNet
model.

Audit date: 2026-04-27.

## Verified Model

The model that generated the current flat predictions in:

```text
/home/aellis/ChiNet/outputs
```

is:

```text
/home/aellis/ChiNet/epoch1000.ckpt
```

The checkpoint has:

```text
epoch: 999
global_step: 175000
sha256: 103cd271e736b215fb81cc8232f84b00f8243e9f0872a5704e5e5fdd452e6f51
model class: ELFPredictor
backbone: ResidualUNet3D
```

The repository copy is:

```text
weights/elfnet_sad2elf.ckpt
```

## Training Data Trace

The exact processed training directory used for `epoch1000.ckpt` is no longer
present in the local `ChiNet` tree. The strongest surviving evidence points to
the old full-grid ChiNet ELFCAR dataset:

```text
/home/aellis/ChiNet/data/ELFCAR/elf_processed
```

This is a historical local path, not a packaged dataset in this repository.

The checkpoint itself records the original Lightning checkpoint callback path:

```text
/home/aellis/ChiNet/checkpoints_elf/batch32/ELF_20250719_202621/epochepoch=0999.ckpt
```

That ties the saved `epoch1000.ckpt` artifact to a batch-32 full-grid run that
started on 2025-07-19. The checkpoint loop state records:

```text
epoch: 999
global_step: 175000
train batches per epoch/rank: 175
validation batches per epoch/rank: 20
```

The matching launch script present at the time was:

```text
/home/aellis/ChiNet/2submit.sh
```

with the relevant settings:

```text
DATA_DIR=/home/aellis/ChiNet/data/ELFCAR/elf_processed
CKPT_PATH=/home/aellis/ChiNet/epoch600.ckpt
--batch 32
--epochs 1001
--num-workers $SLURM_CPUS_PER_TASK
```

The corresponding training loader lineage is:

```text
/home/aellis/ChiNet/scripts/working_versions/elf_data_loaderv5.py
```

That loader consumed only paired full-grid arrays:

```text
<stem>_sad.npy
<stem>_elf.npy
```

It used a deterministic random `90/10` train/validation split with seed `42`,
periodically tiled variable-size full grids to the largest shape in each batch,
and used `DistributedSampler` when DDP was active. It did not load
`<stem>_sym.npy` files and did not extract patches.

The preprocessing trail for the disappeared `elf_processed` directory points
to VASP ELFCAR calculations collected from Materials Project style structures:

```text
/home/aellis/ChiNet/data/utility_scripts/2.py
  -> pulled MP structures and wrote VASP inputs with LELF = .TRUE.

/home/aellis/ChiNet/data/conv6.py
  -> copied converged ELFCAR files into
     /home/aellis/ChiNet/data/ELFCAR/6th_matproj_elfs

/home/aellis/ChiNet/scripts/npy_matproj2.py
  -> converted ELFCAR files into *_elf.npy targets and rebuilt *_sad.npy inputs
     using the old ChiNet neutral-density SAD convention
```

Because `/home/aellis/ChiNet/data/ELFCAR/elf_processed` has been deleted, the
exact stem list and sample count for `epoch1000.ckpt` cannot be proven from the
surviving files alone. The loop counters are consistent with a roughly
50k-structure full-grid dataset under the observed batch/DDP configuration.
The closest surviving local dataset with that scale is:

```text
/home/aellis/TestNet/datasets/data
```

which has exactly `50,000` matched `*_sad.npy`, `*_elf.npy`, and `*_sym.npy`
triplets and the matching `structures_list.txt`. Treat this as a likely sibling
or copy of the old processed data, not as proven byte-identical training data.

## Reproduction Test

The old full-grid ChiNet inference path was reproduced using:

```text
checkpoint: /home/aellis/ChiNet/epoch1000.ckpt
inputs:     /home/aellis/ChiNet/inputs/POSCAR_*
outputs:    /home/aellis/SAD2ELFNet/provenance_chinet_outputs/epoch1000_outputs
```

Every reproduced file matched the existing flat output file byte-for-byte:

```text
/home/aellis/ChiNet/outputs/ELFCAR_*.vasp
```

This proves the current flat output set was produced by `epoch1000.ckpt`.

## Why Epoch 114 Was Removed From The Public Default

The later checkpoint:

```text
/home/aellis/ChiNet/checkpoints_sad2elf/batch2/SAD2ELF_20251104_124933/best_epoch=0114.ckpt
```

is a different model family. It uses a symmetry-aware SAD2ELF architecture and
different inference code. It was created on 2025-11-11, while the flat
`ChiNet/outputs` predictions were written on 2025-08-27. It cannot be the
source of those files.

Epoch 114 may still be useful as a separate research artifact, but it should
not be used as the provenance for the current ChiNet output examples.

## Full-Grid Inference Path

The verified path is:

```text
POSCAR_* -> full SAD grid -> ELFPredictor -> full ELF grid -> ELFCAR_*.vasp
```

It does not use:

```text
symmetry tensors
Seitz operations
patch extraction
patch-local coordinate transforms
overlap blending
```

## Output Set

The reproduced output files are:

```text
ELFCAR_BaH6_Imm2_200_Mn_Ni.vasp
ELFCAR_BaH6_Imm2_200_Nb_Bi.vasp
ELFCAR_Ca8_sublat.vasp
ELFCAR_NdH9_F-43m_200_Au_Ga.vasp
ELFCAR_NdH9_F-43m_200_Fe_Pb.vasp
ELFCAR_ZrH6_P21c_200.vasp
ELFCAR_ZrH6_P21c_200_Ca.vasp
ELFCAR_ZrH6_P21c_200_Ca_Ba.vasp
ELFCAR_ZrH6_P21c_200_Ca_Ce.vasp
ELFCAR_ZrH6_P21c_200_Ta_Bi.vasp
```

All matched with `max_abs = 0` against the current local `ChiNet/outputs`
versions.
