# Weights

No checkpoint is tracked in this repository.

For inference, pass an explicit checkpoint path:

```bash
elfnet-predict /path/to/checkpoint.ckpt examples/poscars runs/example_outputs
```

You can also set `ELFNET_CHECKPOINT` or place a local checkpoint at:

```text
weights/elfnet.ckpt
```

Checkpoint files are Git-LFS eligible through `.gitattributes`.
