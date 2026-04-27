import torch

from elfnet.model import Sad2ElfLitModule


def test_small_model_forward_shape():
    model = Sad2ElfLitModule(
        base=4,
        depth=2,
        blocks_per_stage=1,
        min_patch_size=8,
        sym_every_stage=True,
    )
    x = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    seitz = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4)
    mask = torch.ones((1, 1), dtype=torch.bool)
    shape = torch.tensor([[8, 8, 8]], dtype=torch.long)

    y = model(x, seitz, mask, shape)

    assert tuple(y.shape) == (1, 1, 8, 8, 8)
    assert torch.isfinite(y).all()
