# gpu_server/
Deep CWFS checks that run on the GPU server. Code goes laptop → GitHub → server; results come back
server → laptop with rsync (`results/` is git-ignored, so nothing large goes to GitHub).

| File | Purpose |
|---|---|
| `plot_deep_cwfs_gpu_run_summary.py` | `summary.png` for a run folder (called by the runner; rerun by hand with `python gpu_server/plot_deep_cwfs_gpu_run_summary.py results/deep_cwfs_gpu_server/<run>`) |
| `run_deep_cwfs_gpu_tests_and_inference.py` | runs the Deep CWFS pytest file on the GPU (incl. reproduction of the original script and the sign check against DIC), then infers every checkpoint on both test sets and writes `results/deep_cwfs_gpu_server/<date>_<host>/` |

## On the server
```bash
# 1. code (later updates: git pull)
git clone https://github.com/Muhammad-Kazim/qpi-deep-cwfs-infer.git speckle_opd_retrieval
cd speckle_opd_retrieval && git checkout speckle-opd-retrieval        # `origin/main` = original code, used by the tests

# 2. environment (once; `pip install -e` does not create an env, it installs into the active one)
conda env create -f environment_gpu_server.yml                        # conda-forge, CUDA build of torch matching the driver
conda activate speckle_qpi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```
Without conda (any env with Python >= 3.9, e.g. `python -m venv .venv && source .venv/bin/activate`):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # first; cuXXX to match nvidia-smi
pip install -r requirements.txt
```
If the conda torch needs a newer driver than the server has, pin the CUDA version: add `- cuda-version=12.4`
(≤ the "CUDA Version" shown by `nvidia-smi`) to `environment_gpu_server.yml`.

```bash
# 3. data: one folder holding these three (copied from the laptop's Claude+Projects/data/)
#    deep_cwfs_model_checkpoints/  2026-04-13_cwfs_speckle_pairs_10_samples/
#    2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth/
# 4. run (one call: all checkpoints on both test sets)
python gpu_server/run_deep_cwfs_gpu_tests_and_inference.py --data /path/to/data --device cuda
```
Options: `--no-maps` (skip the OPD maps, ~60 MB per run), `--checkpoint-glob '*raft_large*'`, `--limit N`
(first N pairs per set), `--skip-pytest`. The original code is extracted from `origin/main` into `.cache/` (git-ignored).

## Laptop side (project root)
```bash
# data to the server (once; the path has spaces, so cd into it)
cd ../data && rsync -av deep_cwfs_model_checkpoints 2026-04-13_cwfs_speckle_pairs_10_samples \
  2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth <server>:/path/to/data/ && cd -
# results back
rsync -av <server>:<path>/speckle_opd_retrieval/results/deep_cwfs_gpu_server/ results/deep_cwfs_gpu_server/
```

## Output (`results/deep_cwfs_gpu_server/<date>_<host>/`)
| File | Contents |
|---|---|
| `summary.png` | heat maps of corr(deep, DIC) and corr(displacement, gt labels) per checkpoint and pair; OPD maps (DIC, best checkpoint, integrated gt labels / legacy sign) with colour bars |
| `summary.md` | per checkpoint and test set: mean/min corr(deep, DIC) OPD, corr of the deep displacement with the SynthEllips ground-truth gradients, time per pair |
| `metrics.csv` | the same per pair; `*_dispI_gtJ` = corr(displacement channel I (0 = y, 1 = x), gt gradient `N_J`) for deep and DIC |
| `pytest_deep_cwfs_gpu.log`, `pytest_deep_cwfs_junit.xml` | pytest output (prints `corr(deep, dic)`) |
| `environment.json` | commit, torch/CUDA, GPU, checkpoints |
| `opd_maps/*.npz` | OPD [m] per pair (keys = pair index), deep per checkpoint and DIC (skipped with `--no-maps`) |
