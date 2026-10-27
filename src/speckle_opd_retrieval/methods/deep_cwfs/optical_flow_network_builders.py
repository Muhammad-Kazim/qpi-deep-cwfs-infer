"""Optical-flow networks used by Deep CWFS.

Each network is adapted to single-channel (grey-scale) speckle images and
loaded from a CWFS-trained checkpoint. :func:`build_flow_model` returns a
:class:`FlowModel` that maps a batch of normalised reference/specimen
patches ``(B, H, W)`` to a flow field ``(B, 2, H, W)`` at the same size.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

log = logging.getLogger(__name__)

#: Model names that can be built, in the order used to guess the name from a
#: checkpoint file name (``sea_raft_*`` must come before ``raft_*``).
MODEL_NAMES = ("sea_raft_medium", "sea_raft_small", "raft_large", "raft_small")

#: Network input size (H, W) each family was trained with.
INPUT_SIZE = {
    "raft": (520, 960),
    # 540 rather than 520 so that H is divisible by 8 after SEA-RAFT's padding.
    "sea_raft": (540, 960),
}


def guess_model_name(checkpoint: str) -> str:
    """Infer the architecture from a checkpoint file name such as
    ``ellipsoids_sea_raft_medium_20260219_082610_9_5027.pth``."""
    base = os.path.basename(checkpoint).lower()
    for name in MODEL_NAMES:
        if name in base:
            return name
    raise ValueError(
        f"Cannot infer the model from checkpoint name '{base}'. "
        f"Pass model=... explicitly (one of {MODEL_NAMES})."
    )


def _load_state_dict(checkpoint: str, device) -> dict:
    try:
        weights = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception:  # older checkpoints may pickle extra objects
        weights = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(weights, dict) and "model" in weights:
        weights = weights["model"]
    return weights


def _to_network_input(ref: torch.Tensor, spec: torch.Tensor, size: Tuple[int, int]):
    """Resize ``(B, H, W)`` patches to the network size and map [0, 1] -> [-1, 1]
    (identical to the original ``preprocess`` + ``OpticalFlowTransform*``)."""
    ref = TF.resize(ref, size=list(size), antialias=True).unsqueeze(1)
    spec = TF.resize(spec, size=list(size), antialias=True).unsqueeze(1)
    ref = TF.normalize(ref, mean=[0.5], std=[0.5]).contiguous()
    spec = TF.normalize(spec, mean=[0.5], std=[0.5]).contiguous()
    return ref, spec


@dataclass
class FlowModel:
    name: str
    network: torch.nn.Module
    input_size: Tuple[int, int]
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: torch.device

    @torch.no_grad()
    def __call__(self, ref: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """``ref``, ``spec``: ``(B, H, W)`` float tensors, normalised by the
        reference maximum. Returns flow ``(B, 2, H, W)`` on the CPU, resized back
        to the patch size. As in the original pipeline the flow *values* are
        not rescaled: the networks were trained to output displacement in
        patch pixels on the resized grid."""
        h, w = ref.shape[-2:]
        a, b = _to_network_input(ref, spec, self.input_size)
        flow = self.forward(a.to(self.device), b.to(self.device)).detach().cpu()
        return TF.resize(flow, size=[h, w], antialias=True)


def _build_raft(name: str, checkpoint: Optional[str], device, iters: Optional[int]) -> FlowModel:
    from torchvision.models.optical_flow import raft_large, raft_small

    net = raft_large(progress=False) if name == "raft_large" else raft_small(progress=False)
    ch = 64 if name == "raft_large" else 32
    if checkpoint is None:
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights

        warnings.warn("No checkpoint given: using generic RAFT weights (not CWFS-trained).")
        w = (Raft_Large_Weights if name == "raft_large" else Raft_Small_Weights).DEFAULT.get_state_dict()
        for k in ("feature_encoder.convnormrelu.0.weight", "context_encoder.convnormrelu.0.weight"):
            w[k] = w[k].mean(dim=1, keepdim=True)
    else:
        w = _load_state_dict(checkpoint, device)
    net.feature_encoder.convnormrelu[0] = torch.nn.Conv2d(1, ch, kernel_size=7, stride=2, padding=3)
    net.context_encoder.convnormrelu[0] = torch.nn.Conv2d(1, ch, kernel_size=7, stride=2, padding=3)
    net.load_state_dict(w)
    net = net.to(device).eval()

    n_iter = 12 if iters is None else int(iters)

    def forward(a, b):
        return net(a, b, num_flow_updates=n_iter)[-1]

    return FlowModel(name, net, INPUT_SIZE["raft"], forward, torch.device(device))


def _sea_raft_args(name: str) -> argparse.Namespace:
    if name not in ("sea_raft_medium", "sea_raft_small"):
        raise ValueError(f"SEA-RAFT variant '{name}' not available")
    # Same settings as the original utils.get_model_and_transforms; medium
    # and small share the architecture there.
    return argparse.Namespace(
        name=name,
        gma=False,
        restore_ckpt=None,
        image_size=list(INPUT_SIZE["sea_raft"]),
        use_var=True,
        var_min=0,
        var_max=10,
        initial_dim=64,
        radius=4,
        num_heads=4,
        hidden_dim=128,
        context_dim=128,
        dropout=0.0,
        alternate_corr=False,
        mixed_precision=False,
        num_blocks=2,
        pretrain="resnet34",
        block_dims=[64, 128, 256],
        dim=128,
        iters=4,
        init_weight=False,  # never download ImageNet weights; checkpoint overrides them anyway
    )


def _build_sea_raft(name: str, checkpoint: Optional[str], device, iters: Optional[int]) -> FlowModel:
    from .sea_raft import RAFT as SeaRAFT

    if checkpoint is None:
        raise ValueError("SEA-RAFT needs a CWFS-trained checkpoint (checkpoint=...).")
    args = _sea_raft_args(name)
    net = SeaRAFT(args)
    w = _load_state_dict(checkpoint, device)
    ch = w["fnet.conv1.weight"].shape[0]
    net.fnet.conv1 = torch.nn.Conv2d(1, ch, kernel_size=7, stride=2, padding=3)
    net.cnet.conv1 = torch.nn.Conv2d(2, ch, kernel_size=7, stride=2, padding=3)
    net.load_state_dict(w, strict=True)
    net = net.to(device).eval()

    n_iter = args.iters if iters is None else int(iters)

    def forward(a, b):
        return net(a, b, iters=n_iter, test_mode=True)["final"]

    return FlowModel(name, net, INPUT_SIZE["sea_raft"], forward, torch.device(device))


def build_flow_model(
    model: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    iters: Optional[int] = None,
) -> FlowModel:
    """Build an optical-flow network for Deep CWFS.

    Args:
        model: one of :data:`MODEL_NAMES`; guessed from ``checkpoint`` if omitted.
        checkpoint: path to a CWFS-trained ``.pth`` file.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``...
        iters: number of recurrent refinement iterations (default: 12 for
            RAFT, 4 for SEA-RAFT, as in the original scripts).
    """
    if model is None:
        if checkpoint is None:
            raise ValueError("Give a checkpoint and/or a model name.")
        model = guess_model_name(checkpoint)
    model = model.lower()
    if checkpoint is not None and not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    log.info("Loading %s from %s", model, checkpoint)
    if model.startswith("sea_raft"):
        return _build_sea_raft(model, checkpoint, device, iters)
    if model in ("raft_large", "raft_small"):
        return _build_raft(model, checkpoint, device, iters)
    raise NotImplementedError(f"Model '{model}' is not supported (use one of {MODEL_NAMES}).")
