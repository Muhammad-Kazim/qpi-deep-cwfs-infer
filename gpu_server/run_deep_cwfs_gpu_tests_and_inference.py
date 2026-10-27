"""Deep CWFS checks on the GPU server; writes a results folder that is copied back with rsync (not git).

Run from the project root of a clone of this repository (branch ``speckle-opd-retrieval``):

    python gpu_server/run_deep_cwfs_gpu_tests_and_inference.py --data /path/to/data --device cuda

Steps
-----
1. ``pytest tests/test_deep_cwfs_opd_retriever.py`` on the GPU, including
   ``test_reproduces_original_inference`` (against the original code, extracted from
   ``origin/main``) and ``test_sign_agrees_with_dic`` (prints ``corr(deep, dic)``).
2. Inference with every checkpoint on every pair of the two test sets:
   - corr(deep OPD, DIC OPD) for each pair (should be > 0);
   - SynthEllips only: correlation of the Deep CWFS displacement channels (y, x) with the
     ground-truth gradient channels (``N_0``, ``N_1``), which checks sign and channel order
     independently of DIC (DIC is reported the same way for comparison).
3. ``summary.md``, ``summary.png`` (heat maps + OPD maps with colour bars, see
   ``plot_deep_cwfs_gpu_run_summary.py``), ``metrics.csv``, ``environment.json`` and (unless ``--no-maps``) the OPD maps as
   compressed float32 ``.npz`` into ``results/deep_cwfs_gpu_server/<date>_<host>/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import io
import json
import os
import platform
import socket
import subprocess
import sys
import tarfile
import time

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = "2026-04-13_cwfs_speckle_pairs_10_samples"
SYNTH = "2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth"
CROP = (slice(40, -40), slice(40, -40))      # border excluded from correlations, as in the tests


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()


def extract_original_repo(dest: str, ref: str = "origin/main") -> str:
    """Write the original qpi-deep-cwfs-infer code (``qpi_deep_cwfs/``) from ``ref`` into ``dest``."""
    if not os.path.isdir(os.path.join(dest, "qpi_deep_cwfs")):
        tar = subprocess.run(["git", "archive", ref, "qpi_deep_cwfs"], cwd=PROJECT, capture_output=True, check=True).stdout
        os.makedirs(dest, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(dest)
    return dest


def run_pytest(out_dir: str, env: dict) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/test_deep_cwfs_opd_retriever.py", "-v", "-rs", "-s",
           "-p", "no:cacheprovider", f"--junitxml={os.path.join(out_dir, 'pytest_deep_cwfs_junit.xml')}"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True)
    log = p.stdout + p.stderr + f"\n[exit code {p.returncode}, {time.perf_counter() - t0:.0f} s]\n"
    with open(os.path.join(out_dir, "pytest_deep_cwfs_gpu.log"), "w") as f:
        f.write(log)
    print(log)
    return p.returncode


def pairs(folder: str):
    for ref in sorted(glob.glob(os.path.join(folder, "reference", "*.tiff")), key=lambda p: int(os.path.basename(p)[:-5])):
        yield os.path.basename(ref)[:-5], ref, os.path.join(folder, "specimen", os.path.basename(ref))


def corr(a, b) -> float:
    return float(np.corrcoef(a[CROP].ravel(), b[CROP].ravel())[0, 1])


def channel_corr(disp, gt) -> list:
    """2×2 matrix: corr(displacement[i], gt[j]), i, j ∈ (0 = y / N_0, 1 = x / N_1)."""
    return [[corr(disp[i], gt[j]) for j in range(2)] for i in range(2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("SPECKLE_OPD_DATA", os.path.join(PROJECT, "..", "data")),
                    help="folder holding deep_cwfs_model_checkpoints/ and the two test sets")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--original-ref", default="origin/main", help="git ref holding the original qpi_deep_cwfs code")
    ap.add_argument("--no-maps", action="store_true", help="do not save the OPD maps (.npz, ~60 MB per run)")
    ap.add_argument("--checkpoint-glob", default="*.pth", help="restrict checkpoints, e.g. '*raft_small*'")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs of each set (quick check)")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    import torch
    from speckle_opd_retrieval import Setup, get_method, load_image

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    data = os.path.abspath(args.data)
    ckpts = sorted(glob.glob(os.path.join(data, "deep_cwfs_model_checkpoints", args.checkpoint_glob)))
    if not ckpts:
        sys.exit(f"no checkpoints in {data}/deep_cwfs_model_checkpoints")

    run = f"{datetime.datetime.now():%Y-%m-%d_%H%M}_{socket.gethostname()}"
    out = os.path.join(PROJECT, "results", "deep_cwfs_gpu_server", run)
    os.makedirs(out, exist_ok=True)
    original = extract_original_repo(os.path.join(PROJECT, ".cache", "original_qpi_deep_cwfs_infer_main"), args.original_ref)

    env_info = {
        "run": run, "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "original_ref": f"{args.original_ref} = {git('rev-parse', args.original_ref)}",
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoints": [os.path.basename(c) for c in ckpts], "data": data,
    }
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    pytest_code = None
    if not args.skip_pytest:
        env = dict(os.environ, SPECKLE_OPD_DATA=data, SPECKLE_OPD_DEVICE=args.device, SPECKLE_OPD_ORIGINAL_REPO=original)
        pytest_code = run_pytest(out, env)

    sets = {"samples": os.path.join(data, SAMPLES), "synthellips": os.path.join(data, SYNTH)}
    synth_cfg = json.load(open(os.path.join(sets["synthellips"], "data_creation_cwfs_params.json")))
    gscale, gbias = synth_cfg["data_saving_params"]["GRAD_DATA_SCALE"], synth_cfg["data_saving_params"]["GRAD_DATA_BIAS"]
    synth_dist = {r["img_id"]: float(r["dist_pm_im"]) * 1e-6 for r in csv.DictReader(open(os.path.join(sets["synthellips"], "dataset_details.csv")))}

    def setup_for(set_name, idx):
        if set_name == "synthellips":
            return Setup(synth_cfg["system_params"]["DIGITAL_PX_SIZE"], synth_dist[idx])
        return Setup(3.45e-6, 2.48e-3)

    # DIC reference (CPU), once per pair
    loaded, dic = {}, {}
    for set_name, folder in sets.items():
        for idx, rp, sp in list(pairs(folder))[:args.limit]:
            I0, I1 = load_image(rp), load_image(sp)
            gt = None
            if set_name == "synthellips":
                gt = np.stack([load_image(os.path.join(folder, "gt_gradients", f"{idx}_{c}.tiff")) / gscale - gbias for c in (0, 1)])
            loaded[set_name, idx] = (I0, I1, gt)
            dic[set_name, idx] = get_method("dic", setup_for(set_name, idx), saturation=float(I0.max())).retrieve(I0, I1)
            print(f"dic {set_name}/{idx} done")

    rows, maps = [], {}
    for ck in ckpts:
        ck_name = os.path.basename(ck)[:-4]
        for (set_name, idx), (I0, I1, gt) in loaded.items():
            m = get_method("deep_cwfs", setup_for(set_name, idx), checkpoint=ck, device=args.device, patch_size=500, overlap=150)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = m.retrieve(I0, I1)
            dt = time.perf_counter() - t0
            d = dic[set_name, idx]
            row = {"checkpoint": ck_name, "set": set_name, "pair": idx, "shape": "x".join(map(str, I0.shape)),
                   "seconds": round(dt, 2), "corr_deep_dic_opd": round(corr(r.opd, d.opd), 4)}
            if gt is not None:
                cd, cg = channel_corr(r.displacement, gt), channel_corr(d.displacement, gt)
                row.update({f"deep_disp{i}_gt{j}": round(cd[i][j], 4) for i in range(2) for j in range(2)})
                row.update({f"dic_disp{i}_gt{j}": round(cg[i][j], 4) for i in range(2) for j in range(2)})
            rows.append(row)
            maps.setdefault((ck_name, set_name), {})[idx] = r.opd.astype(np.float32)
            print(row)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    if not args.no_maps:
        os.makedirs(os.path.join(out, "opd_maps"), exist_ok=True)
        for (ck_name, set_name), m in maps.items():
            np.savez_compressed(os.path.join(out, "opd_maps", f"deep_cwfs_{ck_name}_{set_name}_opd_m.npz"), **m)
        for set_name in sets:
            np.savez_compressed(os.path.join(out, "opd_maps", f"dic_{set_name}_opd_m.npz"),
                                **{i: dic[s, i].opd.astype(np.float32) for (s, i) in dic if s == set_name})

    lines = [f"# Deep CWFS GPU run {run}", "",
             f"commit `{env_info['git_commit'][:10]}`{' (dirty)' if env_info['git_dirty'] else ''}, torch {env_info['torch']}, "
             f"CUDA {env_info['cuda']}, {env_info['gpu']}", "",
             f"pytest exit code: {pytest_code}" + ("" if pytest_code is None else "  (see `pytest_deep_cwfs_gpu.log`)"), "",
             "| checkpoint | set | mean corr(deep, DIC) OPD | min | mean corr(deep y, gt N_0) | mean corr(deep x, gt N_1) | mean s/pair |",
             "|---|---|---|---|---|---|---|"]
    for ck in ckpts:
        for set_name in sets:
            sel = [r for r in rows if r["checkpoint"] == os.path.basename(ck)[:-4] and r["set"] == set_name]
            c = [r["corr_deep_dic_opd"] for r in sel]
            gy = f"{np.mean([r['deep_disp0_gt0'] for r in sel]):.3f}" if set_name == "synthellips" else "–"
            gx = f"{np.mean([r['deep_disp1_gt1'] for r in sel]):.3f}" if set_name == "synthellips" else "–"
            lines.append(f"| {os.path.basename(ck)[:-4]} | {set_name} | {np.mean(c):.3f} | {np.min(c):.3f} | {gy} | {gx} | "
                         f"{np.mean([r['seconds'] for r in sel]):.2f} |")
    synth = [r for r in rows if r["set"] == "synthellips"]
    if synth:
        lines += ["", f"DIC vs gt (synthellips): mean corr(y, N_0) = {np.mean([r['dic_disp0_gt0'] for r in synth]):.3f}, "
                      f"mean corr(x, N_1) = {np.mean([r['dic_disp1_gt1'] for r in synth]):.3f}"]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    from plot_deep_cwfs_gpu_run_summary import plot_run     # same folder as this script
    print(f"figure: {plot_run(out, data)}")
    print(f"\nresults in {out}\ncopy to the laptop's results/deep_cwfs_gpu_server/ with rsync (results/ is git-ignored)")


if __name__ == "__main__":
    main()
