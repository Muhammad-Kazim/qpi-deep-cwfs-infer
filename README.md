# speckle_opd_retrieval

Retrieve the **optical path difference (OPD)**, its gradient and the speckle displacement from a
reference / specimen speckle image pair (coded wavefront sensing), with interchangeable methods
behind **one interface**.

| Method | Name | Status | Needs |
|---|---|---|---|
| Digital image correlation (amplitude-robust ZNCC) | `dic` | ✅ tested on synthetic data | numpy, scipy |
| Deep CWFS (CWFS-trained RAFT / SEA-RAFT) | `deep_cwfs` | ✅ ported; ⚠ not yet run with torch, sign check pending | `[deep]` = torch, torchvision |
| Classical CWFS (ADMM) | — | ⏳ to port from `archive/` | torch |

## Install
```bash
pip install -e ".[test]"          # DIC only
pip install -e ".[deep,test]"     # + Deep CWFS (install the CUDA build of torch first for a GPU)
pytest -v
```

## The common interface
```python
from speckle_opd_retrieval import Setup, get_method, available_methods, load_image

setup = Setup(pixel_size=3.45e-6, distance=2.48e-3)           # metres
ref, spec = load_image("ref.tiff"), load_image("specimen.tiff")

available_methods()        # {'dic': '…', 'deep_cwfs': '…'}  (does not import torch)

dic  = get_method("dic", setup, window=32, step=8, search=7, saturation=65520)
deep = get_method("deep_cwfs", setup,
                  checkpoint="../data/deep_cwfs_model_checkpoints/ellipsoids_raft_large_20260216_124504_4_5012.pth",
                  device="cuda")

for method in (dic, deep):
    res = method.retrieve(ref, spec)            # → RetrievalResult
    res.opd             # (H, W) OPD [m], mean removed
    res.opd_gradient    # (2, H, W) dOPD/dy, dOPD/dx
    res.displacement    # (2, H, W) speckle displacement [px]
    res.params, res.info["runtime_s"]
    res.save("results", f"cell01_{method.name}", formats=["tiff", "png"], unit="um")

results = dic.retrieve_many(ref, [spec1, spec2, spec3])   # time series, same settings / loaded model
```

**Conventions, identical for every method:**
- displacement `d`: `specimen(x + d) = reference(x)`
- fields ordered **(y, x)**
- `OPD = pixel_size²/distance · ∫d`
- `opd_gradient = pixel_size/distance · d`

**How each method plugs in:**
- **The base class** `OPDRetriever` handles input checks, integration (`integrator="dct"` or `"cg"`), physical scaling, timing and the result type.
- **A method only implements** `_estimate(reference, specimen) → MethodOutput(displacement, [phi], [amplitude], info)`.

## Command line
```bash
speckle-opd-retrieval methods
speckle-opd-retrieval run -m dic -m deep_cwfs -r ref.tiff -s "frames/*.tiff" \
    --pixel-size 3.45e-6 --distance 2.48e-3 \
    --param saturation=65520 --param checkpoint=../data/deep_cwfs_model_checkpoints/<file>.pth -o results
speckle-opd-retrieval run -c config.json      # keys as the flags, plus "params": {...}
```
Method parameters are passed with `--param name=value`. With several methods, each method receives only the parameters it accepts. A parameter that no selected method accepts is an error.

## Method notes
**DIC** (`methods/dic/`):
1. Local intensity normalisation (σ = 6 px) removes background and transmission changes.
2. Saturated pixels are masked.
3. Exact weighted ZNCC is computed at every integer shift within ±`search`.
4. Sub-pixel peak by a parabolic fit; outliers by the median test.
5. The grid result is bilinearly upsampled to full resolution, then integrated (DCT).

Accuracy on synthetic speckle: 0.03–0.04 px RMS. With amplitude change only, the bias stays below 0.04 px.

**Deep CWFS** (`methods/deep_cwfs/`):
- Tiles of `patch_size` (500) with `overlap` (150). The last tile moves inward so all tiles hold real data, any image shape works, and small images get one mirror-pad. Blending: `mean` or `feather`.
- Each tile is normalised by its reference maximum, then resized to the network input (RAFT 520×960, SEA-RAFT 540×960).
- Integration: CG by default, matching the original.
- **Sign:** the original script gives OPD = −px²/z·∫flow, so beads of higher RI come out negative. Here the default is `displacement = +flow`, matching DIC and the physical sign; `legacy_sign=True` reproduces the original maps.
  ⚠ Confirm with `tests/test_deep_cwfs_opd_retriever.py::test_sign_agrees_with_dic`. If it fails, flip the default in `deep_cwfs_opd_retriever.py`.

## Folder guide
| Folder | Contents |
|---|---|
| `src/speckle_opd_retrieval/` | the package; `core/` shared code, `methods/<name>/` one folder per method |
| `tests/` | automated tests (`pytest -v`) |
| `examples/` | runnable scripts |
| `gpu_server/` | Deep CWFS tests + inference on the GPU server; results come back via git in `results/deep_cwfs_gpu_server/` |
| `results/` | outputs (only `deep_cwfs_gpu_server/` is tracked by git) |
| `archive/` | original reference code (classical CWFS still to port), read-only, delete once the port is complete |

Every folder has a README listing its files.
