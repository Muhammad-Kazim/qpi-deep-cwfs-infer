"""Run several OPD retrieval methods on the same speckle pair and plot the results side by side.

    python examples/example_compare_methods_on_one_speckle_pair.py REF.tiff SPEC.tiff \
        --methods dic deep_cwfs        # Deep CWFS: default checkpoint unless --checkpoint is given

Every method is used through the same interface: get_method(name, setup, **params).retrieve(ref, spec).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from speckle_opd_retrieval import Setup, get_method, load_image

ap = argparse.ArgumentParser()
ap.add_argument("reference")
ap.add_argument("specimen")
ap.add_argument("--methods", nargs="+", default=["dic"])
ap.add_argument("--pixel-size", type=float, default=3.45e-6)
ap.add_argument("--distance", type=float, default=2.48e-3)
ap.add_argument("--checkpoint", default=None, help="deep_cwfs checkpoint (default: longer-trained SEA-RAFT medium)")
ap.add_argument("--device", default=None, help="cpu, cuda, ... (default: CUDA if available, else CPU)")
ap.add_argument("--saturation", type=float, default=65520)
ap.add_argument("--out", default=None, help="folder to save results (optional)")
a = ap.parse_args()

setup = Setup(a.pixel_size, a.distance)
ref, spec = load_image(a.reference), load_image(a.specimen)
params = {"dic": dict(saturation=a.saturation, device=a.device),
          "deep_cwfs": dict(checkpoint=a.checkpoint, device=a.device)}

results = {}
for name in a.methods:
    res = get_method(name, setup, **params.get(name, {})).retrieve(ref, spec)
    results[name] = res
    print(f"{name:10s} runtime {res.info['runtime_s']:.1f} s, OPD range {res.opd.min()*1e6:.2f} … {res.opd.max()*1e6:.2f} µm")
    if a.out:
        res.save(a.out, f"{name}", formats=["tiff", "png"])

fig, axs = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
for ax, (name, res) in zip(axs[0], results.items()):
    im = ax.imshow(res.opd * 1e6, cmap="gray")
    fig.colorbar(im, ax=ax, label="OPD [µm]", shrink=0.85)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()
