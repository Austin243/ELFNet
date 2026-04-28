import torch
import numpy as np

from elfnet.data import make_loaders
from elfnet.model import ELFPredictor, ResidualUNet3D


def test_residual_unet_forward_shape():
    net = ResidualUNet3D(base=4, depth=2)
    x = torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)

    with torch.no_grad():
        main, aux = net(x)

    assert tuple(main.shape) == (1, 1, 16, 16, 16)
    assert len(aux) == 1
    assert torch.isfinite(main).all()


def test_elf_predictor_forward_shape():
    model = ELFPredictor()
    x = torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)

    with torch.no_grad():
        main, aux = model(x)

    assert tuple(main.shape) == (1, 1, 16, 16, 16)
    assert len(aux) == 3
    assert torch.isfinite(main).all()


def test_cdf_loss_is_finite_and_differentiable():
    model = ELFPredictor(base=4, depth=2, cdf_bins=16, cdf_max_voxels=4096)
    sad = torch.rand((1, 1, 17, 18, 19), dtype=torch.float32)
    elf = torch.rand((1, 1, 17, 18, 19), dtype=torch.float32)

    loss, logs = model._loss_and_logs(sad, elf)
    loss.backward()

    assert torch.isfinite(loss)
    assert "l_cdf_raw" in logs
    assert torch.isfinite(logs["l_cdf_raw"])
    assert model.net.head[0].weight.grad is not None


def test_shape_bucket_loader_uses_manifest_shapes(tmp_path):
    shapes = [(4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4), (6, 4, 4)]
    rows = ["status\tstem\tcalc_dir\telfcar\tshape\tspecies\tcounts\tsym_ops\tmessage\n"]
    for idx, shape in enumerate(shapes):
        stem = f"s{idx}"
        arr = np.full(shape, idx / 10.0, dtype=np.float32)
        np.save(tmp_path / f"{stem}_sad.npy", arr)
        np.save(tmp_path / f"{stem}_elf.npy", arr)
        rows.append(
            f"wrote\t{stem}\t.\t.\t{'x'.join(str(dim) for dim in shape)}\tH\t1\t\t\n"
        )
    (tmp_path / "manifest.tsv").write_text("".join(rows), encoding="utf-8")

    train, _ = make_loaders(tmp_path, batch=2, num_workers=0, val_frac=0.2, batching="shape")

    for sad, elf in train:
        assert tuple(elf.shape) == tuple(sad.shape)
        assert tuple(sad.shape[-3:]) in {(4, 4, 4), (6, 4, 4)}
        if tuple(sad.shape[-3:]) == (6, 4, 4):
            assert sad.shape[0] == 1
