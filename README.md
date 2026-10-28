# speckle_opd_retrieval

Retrieve the **optical path difference (OPD)**, its gradient and the speckle displacement from a
reference / specimen speckle image pair (coded wavefront sensing), with interchangeable methods
behind **one interface**.

| Method | Name | Status |
|---|---|---|
| Digital image correlation (amplitude-robust ZNCC) | `dic` | ✅ tested on synthetic data; torch port identical to the validated NumPy version (≤1e-10) |
| Deep CWFS (CWFS-trained SEA-RAFT; default checkpoint `ellipsoids_sea_raft_medium_20260219_082610_9_5027.pth`, RAFT on request) | `deep_cwfs` | ✅ ported; reproduces the original script on GPU; sign agrees with DIC |
| Classical CWFS (ADMM) | — | ⏳ to port from `archive/` |

Everything (methods, Poisson integration, tiling) runs in **PyTorch, on the CPU or a GPU**: every method takes
`device=` (`None` = CUDA if available, else CPU; or `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`). Computation is float64
(float32 on MPS); inputs may be NumPy arrays or tensors, results are NumPy arrays on the CPU.

The displacement is integrated into OPD by one of four integrators; every method takes `integrator=`:

| `integrator=` | Border | Notes |
|---|---|---|
| `"poisson_neumann"` (**default**) | no assumption on the values | least-squares Poisson (DCT), fast, keeps tilts |
| `"poisson_zero_boundary"` | OPD = 0 on the border | conjugate gradients; the original Deep CWFS solver (slow; distorts objects touching the border) |
| `"fourier_mirror"` | gradients mirror-padded by N/2, signs kept | FFT; loses tilts |
| `"fourier_asdi"` | anti-symmetric derivative extension | FFT; most accurate for point-sampled gradients |

`available_integrators()` (or `speckle-opd-retrieval methods`) lists them; `integrate(displacement, name)` integrates
any field. Compared, with how to remove piston/tilt before an RMSE, in
[`notebooks/integration_boundary_conditions_explained.ipynb`](notebooks/integration_boundary_conditions_explained.ipynb).

## Install
`pip install -e` installs into the *active* environment and does not create one, so create and activate
an environment first. Run everything from the project root.

```bash
conda create -n speckle_qpi python=3.10 -y && conda activate speckle_qpi
#    without conda: python3 -m venv .venv && source .venv/bin/activate
```

### Using the methods (no development tools needed)
```bash
# torch first, the build for your hardware: GPU .../whl/cu121 (cuXXX matching nvidia-smi), or CPU .../whl/cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .                                          # or: pip install -r requirements.txt
```

### Development (tests, reproduction check, notebooks)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a cuXXX build
pip install -e ".[dev]"                                   # or: pip install -r requirements-dev.txt
pytest -v                                                 # SPECKLE_OPD_DEVICE=cuda pytest -v: tests on the GPU
```
`dev` adds pytest, huggingface_hub (imported by the original qpi-deep-cwfs-infer code in the reproduction test),
jupyterlab, ipykernel, nbformat and nbclient.

### GPU server (development env, all methods on CUDA)
```bash
conda env create -f environment_gpu_server.yml        # CUDA torch from conda-forge + this package + dev tools
conda activate speckle_qpi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
#    without conda: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#                   (cuXXX to match nvidia-smi), then pip install -r requirements-dev.txt
```
Data, running the GPU checks and copying results back: `gpu_server/README.md`.

## The common interface
```python
from speckle_opd_retrieval import Setup, get_method, available_methods, load_image

setup = Setup(pixel_size=3.45e-6, distance=2.48e-3)           # metres
ref, spec = load_image("ref.tiff"), load_image("specimen.tiff")

available_methods()        # {'dic': '…', 'deep_cwfs': '…'}

dic  = get_method("dic", setup, window=32, step=8, search=7, saturation=65520)   # device=None: CUDA if available
deep = get_method("deep_cwfs", setup, device="cuda")   # default: longer-trained SEA-RAFT medium
# other model: get_method("deep_cwfs", setup, model="raft_large")  or  checkpoint="path/to/file.pth"

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
- `OPD = pixel_size²/distance · ∫d`, the textbook OPD = ∫(n − n_medium) dz: an object of **higher RI** than the
  medium (phase lag, converging field) gives **positive** OPD, lower RI (lead, diverging) negative. Speckles move
  toward a converging region. Explained with BPM and speckle simulations in
  [`notebooks/opd_sign_convention_explained.ipynb`](notebooks/opd_sign_convention_explained.ipynb); no method has a sign option
- `opd_gradient = pixel_size/distance · d`

**How each method plugs in:**
- **The base class** `OPDRetriever` handles input checks, integration (`integrator=`, default `"poisson_neumann"`), physical scaling, timing and the result type.
- **A method only implements** `_estimate(reference, specimen) → MethodOutput(displacement, [phi], [amplitude], info)`.

## Command line
```bash
speckle-opd-retrieval methods                 # lists methods and integrators
speckle-opd-retrieval run -m dic -m deep_cwfs -r ref.tiff -s "frames/*.tiff" \
    --pixel-size 3.45e-6 --distance 2.48e-3 \
    --param saturation=65520 -o results            # Deep CWFS: default checkpoint, or --param model=raft_large
speckle-opd-retrieval run -c config.json      # keys as the flags, plus "params": {...}
```
Method parameters are passed with `--param name=value`. With several methods, each method receives only the parameters it accepts. A parameter that no selected method accepts is an error.

## Method notes
**DIC** (`methods/dic/`):
1. Local intensity normalisation (σ = 6 px) removes background and transmission changes.
2. Saturated pixels are masked.
3. Exact weighted ZNCC is computed at every integer shift within ±`search`.
4. Sub-pixel peak by a parabolic fit; outliers by the median test.
5. The grid result is bilinearly upsampled to full resolution, then integrated (default `poisson_neumann`).

Accuracy on synthetic speckle: 0.03–0.04 px RMS. With amplitude change only, the bias stays below 0.04 px.

**Deep CWFS** (`methods/deep_cwfs/`):

> **Input size: the networks expect 500 × 500 px.** They were trained on 500 × 500 px inputs, and the flow values are
> not rescaled after the network's internal resize (as in the original script). A patch of another size is magnified
> differently and gives wrong displacement magnitudes (256 px patches: about 1.8× too large). Therefore:
> - keep `patch_size` at 500 (default); larger images are tiled into 500 px patches;
> - smaller images, and patches smaller than 500 px, are reflect-padded up to 500 × 500 (`pad_size=500`, default),
>   and the result is cropped back. `pad_size=None` switches the padding off;
> - do not use patches larger than 500 px.

- Tiles of `patch_size` (500) with `overlap` (150). The last tile moves inward so all tiles hold real data, any image shape works, and small images get one mirror-pad. Blending: `mean` or `feather`.
- Each tile is normalised by its reference maximum, then resized to the network input (RAFT 520×960, SEA-RAFT 540×960).
- Integration: default `poisson_neumann` like every method; `integrator="poisson_zero_boundary"` is the original
  script's solver and reproduces its OPD exactly (`test_reproduces_original_inference`).
- **Sign:** the networks were trained on labels equal to −d, so `displacement = −flow` (fixed). It agrees with DIC
  (corr +0.98), and the textbook OPD = px²/z·∫d = −px²/z·∫flow is exactly the original script's output.

## Folder guide
| Folder / file | Contents |
|---|---|
| `pyproject.toml` | package metadata and dependencies (torch, torchvision, …; extra `dev` for development) |
| `environment_gpu_server.yml` | development conda env for the GPU server (CUDA torch from conda-forge + this package + dev tools) |
| `requirements.txt` | runtime only (use the methods): torch, torchvision, this package |
| `requirements-dev.txt` | development: runtime + `[dev]` (pytest, huggingface_hub, jupyterlab, ipykernel, nbformat, nbclient) |
| `src/speckle_opd_retrieval/` | the package; `core/` shared code, `methods/<name>/` one folder per method |
| `tests/` | automated tests (`pytest -v`) |
| `examples/` | runnable examples (method comparison script; stitching and integrators notebook) |
| `notebooks/` | explanatory notebooks (OPD sign convention; integrators and boundary conditions) |
| `gpu_server/` | Deep CWFS tests + inference on the GPU server; results are copied back with rsync into `results/deep_cwfs_gpu_server/` |
| `results/` | outputs, git-ignored |
| `archive/` | original reference code (classical CWFS still to port), read-only, delete once the port is complete |

Every folder has a README listing its files.
