"""Deep CWFS through the common interface (skipped without torch / checkpoints / data)."""
import glob
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from speckle_opd_retrieval import Setup, get_method, load_image
from speckle_opd_retrieval.methods.deep_cwfs.optical_flow_network_builders import _sea_raft_args, guess_model_name
from conftest import CKPTS, DEVICE, ORIGINAL_REPO, SAMPLES, speckle


def _random_checkpoint(tmp_path, name):
    if name.startswith("sea_raft"):
        from speckle_opd_retrieval.methods.deep_cwfs.sea_raft import RAFT
        net = RAFT(_sea_raft_args(name))
        net.fnet.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3); net.cnet.conv1 = torch.nn.Conv2d(2, 64, 7, 2, 3)
    else:
        from torchvision.models.optical_flow import raft_large, raft_small
        net = raft_large() if name == "raft_large" else raft_small()
        ch = 64 if name == "raft_large" else 32
        net.feature_encoder.convnormrelu[0] = torch.nn.Conv2d(1, ch, 7, 2, 3)
        net.context_encoder.convnormrelu[0] = torch.nn.Conv2d(1, ch, 7, 2, 3)
    p = tmp_path / f"test_{name}.pth"
    torch.save({"model": net.state_dict()}, p)
    return str(p)


def _ckpt(name):
    c = sorted(glob.glob(os.path.join(CKPTS, f"*_{name}_*.pth")))
    return c[-1] if c else None


def test_guess_model_name():
    assert guess_model_name("x/ellipsoids_sea_raft_medium_2026.pth") == "sea_raft_medium"
    assert guess_model_name("ellipsoids_raft_small_2026.pth") == "raft_small"


@pytest.mark.parametrize("name", ["raft_small", "sea_raft_medium"])
def test_shapes_and_interface_with_random_weights(tmp_path, name):
    m = get_method("deep_cwfs", Setup(3.45e-6, 2.48e-3), checkpoint=_random_checkpoint(tmp_path, name), device=DEVICE,
                   patch_size=256, overlap=64, batch_size=2, iters=2)
    ref = speckle((300, 420))
    r = m.retrieve(ref, np.roll(ref, 1, axis=1))
    assert r.opd.shape == (300, 420) and r.displacement.shape == (2, 300, 420) and np.isfinite(r.opd).all()
    assert r.method == "deep_cwfs" and r.params["model"] == name


def test_legacy_sign_flips_displacement(tmp_path):
    ck = _random_checkpoint(tmp_path, "raft_small")
    ref = speckle((200, 200)); spec = np.roll(ref, 1, axis=0)
    a = get_method("deep_cwfs", Setup(1, 1), checkpoint=ck, device=DEVICE, patch_size=200, overlap=0, iters=2).retrieve(ref, spec)
    b = get_method("deep_cwfs", Setup(1, 1), checkpoint=ck, device=DEVICE, patch_size=200, overlap=0, iters=2, legacy_sign=True).retrieve(ref, spec)
    assert np.allclose(a.displacement, -b.displacement) and np.allclose(a.opd, -b.opd)


@pytest.mark.parametrize("name", ["raft_small", "raft_large", "sea_raft_medium"])
def test_legacy_sign_reproduces_original_inference(name):
    """Same OPD as the original qpi-deep-cwfs-infer script (single-patch branch) on real data."""
    repo = ORIGINAL_REPO
    ck = _ckpt(name)
    if not (ck and os.path.isdir(repo) and os.path.isdir(SAMPLES)):
        pytest.skip("checkpoints, archived repo or sample data not found")
    pytest.importorskip("huggingface_hub")   # imported by the original utils
    sys.path.insert(0, repo)
    import torchvision
    from qpi_deep_cwfs import hRAFT, hSea_RAFT, utils
    I0 = load_image(os.path.join(SAMPLES, "reference", "0.tiff")); I1 = load_image(os.path.join(SAMPLES, "specimen", "0.tiff"))
    px, z = 3.45e-6, 2.5e-3
    model, transforms, settings = utils.get_model_and_transforms(name, DEVICE, checkpoint=ck); model.eval()
    pre = hSea_RAFT.preprocess if "sea_raft" in name else hRAFT.preprocess
    with torch.no_grad():
        a, b = pre([torch.tensor(I0).float() / I0.max()], [torch.tensor(I1).float() / I0.max()], transforms)
        a, b = a.to(DEVICE), b.to(DEVICE)
        flows = (model(a, b, **settings)["final"] if "sea_raft" in name else model(a, b)[-1]).cpu()
        f0, f1 = torchvision.transforms.functional.resize(flows, size=list(I0.shape))[0]
        legacy = -(px**2) / z * utils.integrate_flow_field_torch(torch.stack([f1, f0], dim=2)).double().numpy()
    res = get_method("deep_cwfs", Setup(px, z), checkpoint=ck, device=DEVICE, patch_size=I0.shape[0], legacy_sign=True).retrieve(I0, I1)
    legacy = legacy - legacy.mean()
    np.testing.assert_allclose(res.opd, legacy, atol=1e-3 * np.abs(legacy).max())


def test_sign_agrees_with_dic():
    """Default (non-legacy) Deep CWFS and DIC should give positively correlated OPD on a real pair.
    If this fails, the default sign in deep_cwfs_opd_retriever.py must be flipped."""
    ck = _ckpt("raft_large") or _ckpt("raft_small")
    if not (ck and os.path.isdir(SAMPLES)):
        pytest.skip("checkpoint or sample data not found")
    I0 = load_image(os.path.join(SAMPLES, "reference", "0.tiff")); I1 = load_image(os.path.join(SAMPLES, "specimen", "0.tiff"))
    s = Setup(3.45e-6, 2.48e-3)
    deep = get_method("deep_cwfs", s, checkpoint=ck, device=DEVICE, patch_size=I0.shape[0]).retrieve(I0, I1).opd
    dic = get_method("dic", s, saturation=float(I0.max())).retrieve(I0, I1).opd
    sl = (slice(40, -40), slice(40, -40))
    c = np.corrcoef(deep[sl].ravel(), dic[sl].ravel())[0, 1]
    print(f"corr(deep, dic) = {c:.3f}")
    assert c > 0.3, f"Deep CWFS and DIC disagree (corr={c:.3f}): check the flow sign"
