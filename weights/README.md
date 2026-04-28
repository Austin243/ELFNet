# Weights

No checkpoint is bundled in this repository.

For inference, pass an explicit checkpoint path:

```bash
elfnet-predict /path/to/checkpoint.ckpt examples/poscars runs/example_outputs
```

You can also set `ELFNET_CHECKPOINT` or place a local checkpoint at:

```text
weights/elfnet.ckpt
```

Checkpoint files remain Git-LFS eligible through `.gitattributes`, but no
checkpoint object is tracked in the current repository state.
