# Weights

The default inference checkpoint is:

```text
weights/elfnet.ckpt
```

Checkpoint details:

```text
source run: pressure_flatresnet_c32_b16_k5_kendall_fixed_order/ELF_20260430_123836
checkpoint epoch: 59
global step: 72780
best val/loss: -9.523093223571777
SHA256: 66dac5953e2b93cb0629b708c77cd444b20a40daa586514ab4187b6f2c995c34
```

You can still pass another checkpoint path explicitly or set `ELFNET_CHECKPOINT`.
