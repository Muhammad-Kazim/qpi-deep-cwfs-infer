"""Deep CWFS as an :class:`OPDRetriever`: CWFS-trained optical-flow network → displacement → OPD.

Ported from ``archive/speckle_qpi_package_classical_and_deep_cwfs_to_port`` (which reproduces the
original ``qpi-deep-cwfs-infer`` inference) with two changes: the shared interface / result type of this
package, and integration via ``core`` (``cg`` = NumPy port of the original solver, default; ``dct`` faster).

Sign
----
The network's channel 0 is the row (y) gradient and channel 1 the column (x) gradient. The networks
were trained on labels equal to the NEGATIVE of the displacement d in the package convention
specimen(x + d) = reference(x) (SynthEllips: DIC correlates -0.7 with the labels, the flow +0.9; GPU run
2026-10-27). So ``displacement = -flow``; it agrees with DIC (corr +0.98), and the core's textbook
OPD = +px²/z·∫d = -px²/z·∫flow is exactly the original script's output. There is no sign option:
see ``notebooks/opd_sign_convention_explained.ipynb``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core import MethodOutput, OPDRetriever, Setup
from .patch_tiling_and_stitching import pad_to, stitch, tile_grid

log = logging.getLogger(__name__)


class DeepCWFS(OPDRetriever):
    """Deep coded wavefront sensing.

    Args:
        setup: pixel size and mask–sensor distance.
        checkpoint: CWFS-trained ``.pth`` file. Default: the longer-trained SEA-RAFT medium
            (``ellipsoids_sea_raft_medium_20260219_082610_9_5027.pth``), looked up in
            ``$SPECKLE_OPD_CHECKPOINTS``, ``$SPECKLE_OPD_DATA/deep_cwfs_model_checkpoints`` or ``../data/``.
        model: ``raft_large`` | ``raft_small`` | ``sea_raft_medium`` | ``sea_raft_small``; guessed from the
            checkpoint name if omitted. A model without a checkpoint loads that model's newest checkpoint.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.
        patch_size: square tile size [px] (networks were trained on ~500 px crops); smaller
            images are mirror-padded up to it.
        overlap: tile overlap [px].
        blend: ``"mean"`` (original) or ``"feather"`` (linear ramps, fewer seams).
        batch_size: tiles per forward pass.
        iters: refinement iterations (default 12 for RAFT, 4 for SEA-RAFT, as originally).
        integrator: ``"cg"`` (default; original solver) or ``"dct"``.
    """

    name = "deep_cwfs"
    default_integrator = "cg"

    def __init__(self, setup: Setup, checkpoint: Optional[str] = None, model: Optional[str] = None,
                 device: str = "cpu", patch_size: int = 500, overlap: int = 150, blend: str = "mean",
                 batch_size: int = 1, iters: Optional[int] = None, integrator: Optional[str] = None):
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError("Deep CWFS needs PyTorch: pip install -e '.[deep]'") from e
        from .optical_flow_network_builders import build_flow_model, resolve_checkpoint

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; use device='cpu'.")
        if not 0 <= int(overlap) < int(patch_size):
            raise ValueError("need 0 <= overlap < patch_size")
        checkpoint = resolve_checkpoint(model, checkpoint)          # recorded in params for provenance
        self.flow_model = build_flow_model(model, checkpoint, device=device, iters=iters)
        super().__init__(setup, integrator=integrator, checkpoint=checkpoint, model=self.flow_model.name,
                         device=str(device), patch_size=int(patch_size), overlap=int(overlap), blend=blend,
                         batch_size=max(1, int(batch_size)), iters=iters)

    def estimate_flow(self, reference: np.ndarray, specimen: np.ndarray) -> np.ndarray:
        """Raw network output (2, H, W) for the whole image, stitched from tiles. Each tile is
        normalised by the maximum of its reference tile, as in the original script."""
        import torch

        p = self.params
        P = p["patch_size"]
        H, W = reference.shape
        ref = pad_to(reference.astype(np.float32), P, P)
        spec = pad_to(specimen.astype(np.float32), P, P)
        corners = tile_grid(ref.shape, P, p["overlap"])
        log.info("deep_cwfs (%s): %d tile(s) of %d px", self.flow_model.name, len(corners), P)
        flows = []
        for i in range(0, len(corners), p["batch_size"]):
            chunk = corners[i:i + p["batch_size"]]
            r = np.stack([ref[a:a + P, b:b + P] for a, b in chunk])
            s = np.stack([spec[a:a + P, b:b + P] for a, b in chunk])
            scale = r.reshape(len(chunk), -1).max(axis=1)[:, None, None]
            scale[scale == 0] = 1.0
            out = self.flow_model(torch.from_numpy(r / scale), torch.from_numpy(s / scale))
            flows.extend(out.double().numpy())
        flow = stitch(flows, corners, ref.shape, p["overlap"], p["blend"])
        return flow[:, :H, :W]

    def _estimate(self, reference, specimen) -> MethodOutput:
        flow = self.estimate_flow(reference, specimen)          # channels (y, x)
        # the networks' labels are -d (module docstring)
        return MethodOutput(displacement=-flow, info={"model": self.flow_model.name})
