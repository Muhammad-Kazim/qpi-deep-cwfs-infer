# speckle_opd_retrieval: project context summary
*Hand-over summary, 27 Oct 2026. Import this into a new Claude project to continue the work.*

## Goal
One Python package that retrieves **optical path difference (OPD)**, its gradient and (optionally) amplitude from a
**reference / specimen speckle image pair** (coded wavefront sensing, CWFS), with several interchangeable methods:
- DIC (digital image correlation)
- Classical CWFS (ADMM)
- Deep CWFS (optical-flow networks: RAFT, SEA-RAFT)

It should be usable from the terminal and from Python. Owner: Syed Muhammad Kazim (Uni Siegen).

## Where things are
Root: `~/sciebo - Kazim, Syed Muhammad (...)/Lab/Claude+Projects/`
```
data/                                  shared, READ-ONLY raw measurements (one folder per dataset), see data/README.md
speckle_opd_retrieval/                 this project
  src/speckle_opd_retrieval/
    core/                              shared: Setup/RetrievalResult, OPDRetriever interface + lazy method registry,
                                       integration (poisson_neumann default, poisson_zero_boundary, fourier_mirror, fourier_asdi), device helpers, image I/O
    methods/dic/                       DIC retriever + ZNCC core
    methods/deep_cwfs/                 Deep CWFS retriever, network builders, tiling, vendored SEA-RAFT
    command_line_interface.py          `speckle-opd-retrieval methods | run`
  tests/, examples/                    every folder has a README listing its files
  archive/                             original code (classical CWFS still to port), read-only, delete when done
tvam_qpi_epfl/                         separate application project (resin RI study), uses this package
```
The folder conventions are in `Claude+Projects/README.md` and `CLAUDE.md`:
- snake_case project names
- descriptive file names
- a README in every folder with one line per file
- `data/` and `archive/` are read-only

## Common interface (v0.1.0, 27 Oct 2026)
```python
setup = Setup(pixel_size=3.45e-6, distance=2.48e-3)
m = get_method("dic" | "deep_cwfs", setup, device=None, **params)   # device None: CUDA if available, else CPU
res = m.retrieve(reference, specimen)                       # or m.retrieve_many(reference, [spec, ...])
res.opd [m] (H,W) · res.opd_gradient (2,H,W) · res.displacement [px] (2,H,W) · res.amplitude · res.params · res.info · res.save(...)
```
- **Everything is PyTorch** (since 28 Oct 2026): methods, integration and tiling run on the retriever's `device`, in
  float64 (float32 on MPS). Inputs: NumPy or tensors; `RetrievalResult`: NumPy on the CPU. The torch DIC and integrators
  reproduce the validated NumPy versions to ≤1e-10 (frozen copy in `tests/reference_numpy_implementations_validated.py`).
- **The base class** `OPDRetriever` does input checks, device placement, integration (`integrator=`, one default for all methods: `"poisson_neumann"`; others listed by `available_integrators()`; since 28 Oct 2026, before that Deep CWFS defaulted to the original zero-boundary CG solver), scaling, timing and the result.
- **A method implements** `_estimate(ref, spec) -> MethodOutput(displacement, phi=None, amplitude=None, info)`.
- **To add a method:** create `methods/<name>/` and add one line to `METHODS`.

## Status
| Component | State |
|---|---|
| Core interface, registry, integration, I/O, CLI | **Done, tested**, torch (CPU/GPU) |
| DIC | **Done, tested**, torch (CPU/GPU), identical to the validated NumPy version. The grid result is bilinearly upsampled to full resolution before integration. |
| Deep CWFS | **Done.** Reproduces the original script (GPU run 27 Oct 2026); sign agrees with DIC (corr +0.98). |
| Classical CWFS | To port into `methods/classical_cwfs/`, returning `phi` and `amplitude` via `MethodOutput`. |

## Physics and conventions (keep consistent)
- **Scaling:**
  - OPD [m] = `pixel_size² / distance` × ∫ d·dl, with the displacement d in pixels integrated over pixel units
  - OPD gradient = `pixel_size / distance` × d
- **Camera:** Basler acA1440-220um, 3.45 µm pixels, 12-bit stored as 16-bit, so the maximum is 65520 = saturated.
- **Distance:** the user's DIC/OPD request used **z = 2.48 mm**, but the original `configs/args.json` has `DIST_PM_IM = 2.5e-3`. Confirm which is right.
- **Displacement convention:** `specimen(x + d) = reference(x)`, the same as `cws.m`. Vector fields are ordered **(y, x)** = (rows, columns).
- **OPD sign convention (decided 2026-10-27): textbook** OPD = ∫(n − n_m) dz, so higher RI / phase lag / converging →
  **positive**. Package: OPD = **+**px²/z·∫d for every method, applied once in core; no sign options. BPM and speckle
  simulations in `notebooks/opd_sign_convention_explained.ipynb`; physics test
  `tests/test_opd_sign_convention_physical_speckle_simulation.py`. (The user's earlier "higher RI negative" habit equals
  phase/k₀ in the exp(i(ωt − kz)) convention.)
- **Deep CWFS sign:** the SynthEllips `gt_gradients` are the network's training labels; DIC displacement ≈ −labels, so
  `displacement = −flow` (agrees with DIC, corr +0.98). The textbook OPD is then −px²/z·∫(flow1, flow0), exactly the
  original script. Network channel 0 = row gradient, 1 = column.
  - Open: an earlier note said that in the original Deep CWFS output beads of higher RI come out negative (downstream TVAM
    scripts take `abs()`). The simulations say the original sign gives higher RI positive for direct mask→sensor
    propagation. Check on a real sample of known RI.
  - Classical CWFS (to port): original OPD = **+**px²/z·φ; check that its φ follows specimen(x + d) = reference(x).
- **SEA-RAFT input-scaling quirk:** the preprocessing maps images to [−1, 1], and SEA-RAFT's own `forward` then applies `2·(x/255) − 1` again, so the network sees about −1 ± 0.01. It was kept as is because the checkpoints were trained that way. Check before changing.
- **Deep CWFS flow values:** these are **not** rescaled after resizing back to the patch size, which is how the networks were trained.
- **Bugs in the original Deep CWFS tiling, fixed in the archived `speckle_qpi`:**
  - `grad_stitch_n` assumes a square n×n grid.
  - Edge patches are reflect-padded, and can end up one pixel short, which then stretches them.
  - The small-image branch fails when only one dimension is smaller than the patch.
  - The `speckle_qpi` fix: the last tile moves inward, tiles stitch at any shape, small images get one mirror-pad, and there's an optional "feather" blend.
- **Other changes in the archived SEA-RAFT copy:** relative imports; no Hugging Face hub mixin; no ImageNet ResNet download at build time (`init_weight=False`), because the checkpoint overwrites those weights anyway.
- **GMFlow/GMA branches:** the original `utils.py` references code that isn't in the repo, so these were dropped.

## DIC method (implemented)
1. Local intensity normalisation of each image: `(I − G_σ*I) / sqrt(G_σ*(I − G_σ*I)²)`, σ = 6 px. The speckle grain is about 2 px HWHM. This removes background and amplitude (transmission) changes larger than a few grains.
2. Saturated pixels (≥ 65520) and a 1 px margin get zero weight.
3. Exact weighted ZNCC at every integer lag within ±`search` px, using integral-image box sums, so the score is invariant to offset and gain at every shift.
4. Separable parabolic sub-pixel peak. Border peaks are rejected. Outliers are replaced by the normalised median test (Westerweel & Scarano).

Defaults: window 32 px, step 8 px, search ±7 px.

**Accuracy on synthetic speckle:**
- with motion: 0.03–0.04 px RMS
- with amplitude change only (blobs, stripes, gradient, offset): bias ≤ 0.04 px

An earlier FFT-correlation version normalised only per window. It had 0.25 px worst-case amplitude bias and 8.5% outliers on real data (3.5% with the new method).

**Integration:** neighbouring samples are averaged to gradients between grid nodes, multiplied by the grid step, then integrated (default: Neumann least squares, `poisson_neumann`). Avoid large ramps: a small constant displacement bias (0.1 px over 1000 px) becomes about 0.5 µm of OPD tilt. Objects cut by the frame are kept by `poisson_neumann` and `fourier_asdi` but distorted by `poisson_zero_boundary` (synthetic test, `notebooks/integration_boundary_conditions_explained.ipynb`). For point-sampled gradients `fourier_asdi` is the most accurate (≤5e-4 vs ~2 % for `poisson_neumann`, a half-pixel shift); evaluate with piston and tilt removed, or on the gradients.

## Test and development data (in `Claude+Projects/data/`)
- `2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth/`: 10 simulated pairs with ground-truth gradients and amplitude. **Use for method validation and the sign test.**
- `2026-04-13_cwfs_speckle_pairs_10_samples/`: real sample pairs from the original repo (500×500 and 250×250).
- `deep_cwfs_model_checkpoints/`:
  - `ellipsoids_raft_small_20260211_…pth` (12 MB)
  - `ellipsoids_raft_large_20260216_…pth` (63 MB)
  - two `ellipsoids_sea_raft_medium_2026021x_…pth` (236 MB each)
- `2026-10-02_red_led_vial_cwfs_first_test/`: the first lab CWFS test.
- The lab bead datasets (2026-10-xx) also make good real-world tests. See `data/README.md`.

## Environment and working setup
- **Laptop:** Linux, CPU only; files sync through Sciebo.
- **GPU host:** a university server is available. The user does **not** want Claude to have SSH access to it.
- **Claude Code** runs in the VS Code extension with a sandbox (bubblewrap/socat):
  - writes only inside the opened folder, plus the conda, pip and torch caches
  - network only to PyPI, PyTorch, Anaconda and GitHub
  - `~/.ssh` denied, and `ssh`/`scp`/`sftp`/`rsync` denied
  - GPU runs are done by the user manually.
- **Conda env:** `speckle_qpi` (Python 3.10). The archived `setup_and_demo.sh` creates it, installs CPU torch, and runs pytest and the synthetic demo.

## Next steps
1. Port classical and deep CWFS from the archive into `src/speckle_opd_retrieval/`, behind one interface:
   `(reference, specimen) → RetrievalResult(opd [m], opd_gradient (y,x), displacement [px], amplitude?, info)`.
   Everything in torch (decided 28 Oct 2026: torch is a required dependency so every method can run on the GPU).
2. Add regression tests against the originals: `cws_torch.cws` must give the same φ, and the original inference script the same OPD on the 2026-04-13 samples with the real checkpoints.
3. Resolve the classical-vs-deep sign on synthellips ground truth.
4. Add a CLI (`run` with multiple methods and specimen globs, plus JSON config with the legacy key mapping).
5. Benchmark DIC vs classical vs deep on synthellips (RMSE vs ground truth) and on real bead data.
6. Delete `archive/` once everything is ported and tested.

## Deep CWFS input size (28 Oct 2026)
The networks expect 500 × 500 px inputs: flow values are not rescaled after the internal resize, so patches of another
size give wrong magnitudes (GPU example run: 256 px patches ≈ 1.8× too large). For now every patch smaller than 500 px is
reflect-padded to 500 × 500 (`pad_size=500`, default) and the flow cropped back; `patch_size` stays 500. A proper fix
(rescaling the flow by patch size / training size) is deferred until the exact training crop size is confirmed.

