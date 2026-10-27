"""Summary figure (``summary.png``) for one Deep CWFS GPU run folder.

    python gpu_server/plot_deep_cwfs_gpu_run_summary.py results/deep_cwfs_gpu_server/<run> [--data ../data]

Panels: corr(deep, DIC) OPD per checkpoint and pair; corr of the displacement with the SynthEllips
ground-truth labels (mean of the y and x channels) for every checkpoint and DIC; OPD maps of one real
and one synthetic pair (DIC, best checkpoint, and for SynthEllips the integrated ground-truth labels),
each with its own colour bar. Called automatically at the end of
``run_deep_cwfs_gpu_tests_and_inference.py``; the OPD maps need the run's ``opd_maps/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = "2025-12-03_synthellips_v0_synthetic_cwfs_pairs_with_ground_truth"
CMAP = "RdBu_r"                     # diverging, neutral at 0: signed correlations and signed OPD


def short(ck: str) -> str:
    """ellipsoids_sea_raft_medium_20260219_082610_9_5027 -> sea_raft_medium 0219"""
    m = re.match(r"(?:ellipsoids_)?(.+?)_\d{4}(\d{4})_", ck)
    return f"{m.group(1)} {m.group(2)}" if m else ck


def heatmap(ax, fig, values, rows, cols, title, cbar_label):
    im = ax.imshow(values, cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else "black")
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.01)


def opd_panel(ax, fig, opd, title):
    um = opd * 1e6
    v = np.percentile(np.abs(um), 99.5) or 1.0
    im = ax.imshow(um, cmap=CMAP, vmin=-v, vmax=v)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="OPD [µm]", fraction=0.046, pad=0.02)


def gt_label_opd(data: str, idx: str):
    """OPD from the integrated ground-truth labels (labels = -d, so OPD = -px²/z ∫labels), same scaling as the run."""
    from speckle_opd_retrieval import Setup, integrate, load_image

    folder = os.path.join(data, SYNTH)
    cfg = json.load(open(os.path.join(folder, "data_creation_cwfs_params.json")))
    dist = {r["img_id"]: float(r["dist_pm_im"]) * 1e-6 for r in csv.DictReader(open(os.path.join(folder, "dataset_details.csv")))}
    s, b = cfg["data_saving_params"]["GRAD_DATA_SCALE"], cfg["data_saving_params"]["GRAD_DATA_BIAS"]
    gt = np.stack([load_image(os.path.join(folder, "gt_gradients", f"{idx}_{c}.tiff")) / s - b for c in (0, 1)])
    opd = -Setup(cfg["system_params"]["DIGITAL_PX_SIZE"], dist[idx]).opd_scale * integrate(gt, "cg")
    return opd - opd.mean()


def plot_run(run_dir: str, data: str | None = None, real_pair: str = "0", synth_pair: str = "0") -> str:
    rows = list(csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))))
    env = json.load(open(os.path.join(run_dir, "environment.json")))
    ckpts = list(dict.fromkeys(r["checkpoint"] for r in rows))
    sets = list(dict.fromkeys(r["set"] for r in rows))
    pairs = {s: list(dict.fromkeys(r["pair"] for r in rows if r["set"] == s)) for s in sets}
    get = {(r["checkpoint"], r["set"], r["pair"]): r for r in rows}

    cols = [(s, p) for s in sets for p in pairs[s]]
    c_dic = np.array([[float(get[c, s, p]["corr_deep_dic_opd"]) for s, p in cols] for c in ckpts])
    synth = [p for s, p in cols if s == "synthellips"]
    c_gt = None
    if synth:
        c_gt = np.array([[(float(get[c, "synthellips", p]["deep_disp0_gt0"]) + float(get[c, "synthellips", p]["deep_disp1_gt1"])) / 2
                          for p in synth] for c in ckpts]
                        + [[(float(get[ckpts[0], "synthellips", p]["dic_disp0_gt0"]) + float(get[ckpts[0], "synthellips", p]["dic_disp1_gt1"])) / 2
                            for p in synth]])
        best = ckpts[int(np.argmax(c_gt[:-1].mean(axis=1)))]
    else:
        best = ckpts[0]

    maps_dir = os.path.join(run_dir, "opd_maps")
    have_maps = os.path.isdir(maps_dir)
    fig = plt.figure(figsize=(16, 18 if have_maps else 7), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[7, 11]) if have_maps else (fig, None)
    gs = top.add_gridspec(2, 1)

    log = open(os.path.join(run_dir, "pytest_deep_cwfs_gpu.log")).read() if os.path.exists(os.path.join(run_dir, "pytest_deep_cwfs_gpu.log")) else ""
    tally = re.findall(r"=+ (.*?) in [\d.]+s", log)
    corr_test = re.search(r"corr\(deep, dic\) = (-?[\d.]+)", log)
    fig.suptitle(f"Deep CWFS GPU run {env['run']}  ·  {env.get('gpu')}  ·  commit {env['git_commit'][:7]}\n"
                 f"pytest: {tally[-1] if tally else 'not run'}"
                 + (f"  ·  test_sign_agrees_with_dic: corr(deep, dic) = {corr_test.group(1)}" if corr_test else ""), fontsize=11)

    ax = top.add_subplot(gs[0])
    heatmap(ax, top, c_dic, [short(c) for c in ckpts], [f"{s[:5]} {p}" for s, p in cols],
            "corr(Deep CWFS OPD, DIC OPD)  (> 0 = same sign; border of 40 px excluded)", "Pearson r")
    if c_gt is not None:
        ax = top.add_subplot(gs[1])
        heatmap(ax, top, c_gt, [short(c) for c in ckpts] + ["DIC"], [f"synth {p}" for p in synth],
                "SynthEllips: corr(displacement, ground-truth labels), mean of y and x channels", "Pearson r")

    if have_maps:
        axs = bottom.subplots(2, 3)
        for row, (set_name, idx) in enumerate([("samples", real_pair), ("synthellips", synth_pair)]):
            fd, fb = (os.path.join(maps_dir, f) for f in (f"dic_{set_name}_opd_m.npz", f"deep_cwfs_{best}_{set_name}_opd_m.npz"))
            if not (os.path.exists(fd) and os.path.exists(fb)):
                for a in axs[row]:
                    a.set_axis_off()
                continue
            dic, deep = np.load(fd)[idx], np.load(fb)[idx]
            r = get[best, set_name, idx]["corr_deep_dic_opd"]
            opd_panel(axs[row, 0], bottom, dic, f"{set_name} {idx}: DIC")
            opd_panel(axs[row, 1], bottom, deep, f"{set_name} {idx}: Deep CWFS {short(best)}\ncorr with DIC = {r}")
            ax = axs[row, 2]
            if set_name == "synthellips" and data and os.path.isdir(os.path.join(data, SYNTH)):
                opd_panel(ax, bottom, gt_label_opd(data, idx), f"{set_name} {idx}: from ground-truth labels\n(−px²/z ∫labels, labels = −d)")
            else:
                opd_panel(ax, bottom, deep - dic, f"{set_name} {idx}: Deep CWFS − DIC")

    out = os.path.join(run_dir, "summary.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=os.environ.get("SPECKLE_OPD_DATA", os.path.join(PROJECT, "..", "data")))
    ap.add_argument("--real-pair", default="0")
    ap.add_argument("--synth-pair", default="0")
    a = ap.parse_args()
    print(plot_run(a.run_dir, a.data, a.real_pair, a.synth_pair))


if __name__ == "__main__":
    main()
