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

import torch

from ...core import MethodOutput, OPDRetriever, Setup
from .optical_flow_network_builders import build_flow_model, resolve_checkpoint
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
        device: ``None`` (CUDA if available, else CPU), ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.
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
                 device: Optional[str] = None, patch_size: int = 500, overlap: int = 150, blend: str = "mean",
                 batch_size: int = 1, iters: Optional[int] = None, integrator: Optional[str] = None):
        """Resolve the checkpoint and load the network onto the device (see the class docstring for arguments)."""
        if not 0 <= int(overlap) < int(patch_size):
            raise ValueError("need 0 <= overlap < patch_size")
        checkpoint = resolve_checkpoint(model, checkpoint)  # recorded in params for provenance
        super().__init__(setup, integrator=integrator, device=device, checkpoint=checkpoint, model=model,
                         patch_size=int(patch_size), overlap=int(overlap), blend=blend,
                         batch_size=max(1, int(batch_size)), iters=iters)
        self.flow_model = build_flow_model(model, checkpoint, device=self.device, iters=iters)
        self.params["model"] = self.flow_model.name

    def estimate_flow(self, reference: torch.Tensor, specimen: torch.Tensor) -> torch.Tensor:
        """Raw network output (2, H, W) for the whole image, stitched from tiles (on ``self.device``).
        Each tile is normalised by the maximum of its reference tile, as in the original script."""
        p = self.params
        P = p["patch_size"]
        H, W = reference.shape

        # the networks run in float32, as in the original script
        ref = pad_to(reference.to(self.device).float(), P, P)
        spec = pad_to(specimen.to(self.device).float(), P, P)
        corners = tile_grid(tuple(ref.shape), P, p["overlap"])
        log.info("deep_cwfs (%s): %d tile(s) of %d px", self.flow_model.name, len(corners), P)

        flows = []
        for i in range(0, len(corners), p["batch_size"]):
            batch = corners[i:i + p["batch_size"]]
            ref_tiles = torch.stack([ref[r:r + P, c:c + P] for r, c in batch])
            spec_tiles = torch.stack([spec[r:r + P, c:c + P] for r, c in batch])

            # normalise both tiles by the maximum of the reference tile
            scale = ref_tiles.reshape(len(batch), -1).amax(dim=1)[:, None, None]
            scale[scale == 0] = 1.0
            flow_tiles = self.flow_model(ref_tiles / scale, spec_tiles / scale)
            flows.extend(flow_tiles.to(self.dtype))

        flow = stitch(flows, corners, tuple(ref.shape), p["overlap"], p["blend"])
        return flow[:, :H, :W]  # remove the padding

    def _estimate(self, reference, specimen) -> MethodOutput:
        """Network flow -> displacement in the package convention."""
        flow = self.estimate_flow(reference, specimen)  # channels (y, x)
        # the networks' labels are -d (module docstring)
        return MethodOutput(displacement=-flow, info={"model": self.flow_model.name})
