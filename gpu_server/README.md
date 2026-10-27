# gpu_server/
Deep CWFS checks that run on the GPU server. Results are committed there, pushed, and pulled back here.

| File | Purpose |
|---|---|
| `run_deep_cwfs_gpu_tests_and_inference.py` | runs the Deep CWFS pytest file on the GPU (incl. reproduction of the original script and the sign check against DIC), then infers every checkpoint on both test sets and writes `results/deep_cwfs_gpu_server/<date>_<host>/` |

## On the server
```bash
git clone https://github.com/Muhammad-Kazim/qpi-deep-cwfs-infer.git speckle_opd_retrieval
cd speckle_opd_retrieval && git checkout speckle-opd-retrieval        # `origin/main` = original code, used by the tests
# env: install the CUDA build of torch + torchvision first, then
pip install -e ".[deep,test]" huggingface_hub                         # huggingface_hub: imported by the original code

# data folder (copy from the shared ../data/): deep_cwfs_model_checkpoints/,
# 2026-04-13_cwfs_speckle_pairs_10_samples/, 2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth/
python gpu_server/run_deep_cwfs_gpu_tests_and_inference.py --data /path/to/data --device cuda

git add results/deep_cwfs_gpu_server && git commit -m "Deep CWFS GPU run" && git push
```
Options: `--save-maps` (all OPD maps as `.npz`, ~60 MB), `--checkpoint-glob '*raft_large*'`, `--limit N`
(first N pairs per set), `--skip-pytest`. The original code is extracted from `origin/main` into `.cache/` (git-ignored).

## Output (`results/deep_cwfs_gpu_server/<date>_<host>/`)
| File | Contents |
|---|---|
| `summary.md` | per checkpoint and test set: mean/min corr(deep, DIC) OPD, corr of the deep displacement with the SynthEllips ground-truth gradients, time per pair |
| `metrics.csv` | the same per pair; `*_dispI_gtJ` = corr(displacement channel I (0 = y, 1 = x), gt gradient `N_J`) for deep and DIC |
| `pytest_deep_cwfs_gpu.log`, `pytest_deep_cwfs_junit.xml` | pytest output (prints `corr(deep, dic)`) |
| `environment.json` | commit, torch/CUDA, GPU, checkpoints |
| `opd_maps/*.npz` | with `--save-maps`: OPD [m] per pair (keys = pair index), deep per checkpoint and DIC |
