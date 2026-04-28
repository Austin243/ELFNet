# Weights

The bundled checkpoint is tracked with Git LFS:

```text
weights/elfnet_sad2elf.ckpt
```

Current checkpoint identity:

```text
source:      /home/aellis/ChiNet/epoch1000.ckpt
model:       ELFPredictor / ResidualUNet3D
epoch:       999
global_step: 175000
sha256:      103cd271e736b215fb81cc8232f84b00f8243e9f0872a5704e5e5fdd452e6f51
```

The current package code is the production full-grid CDF-loss training code.
This checkpoint is retained for inference/provenance until a new CDF-trained
checkpoint is produced.

After cloning:

```bash
git lfs install
git lfs pull
```

If the checkpoint file is only a small text pointer, Git LFS has not downloaded
the model weights yet.
