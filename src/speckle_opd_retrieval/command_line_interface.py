"""Command line: ``speckle-opd-retrieval methods`` and ``speckle-opd-retrieval run``.

Examples::

    speckle-opd-retrieval methods
    speckle-opd-retrieval run -m dic -r ref.tiff -s "frames/*.tiff" --pixel-size 3.45e-6 --distance 2.48e-3 \
        --param saturation=65520 -o results
    speckle-opd-retrieval run -m deep_cwfs -r ref.tiff -s spec.tiff --pixel-size 3.45e-6 --distance 2.48e-3
        # default checkpoint (longer-trained SEA-RAFT medium); --param model=raft_large or --param checkpoint=... for others
    speckle-opd-retrieval run -c config.json        # same keys as the flags; flags override the file
    speckle-opd-retrieval run ... --device cpu      # default: CUDA if available, else CPU

Method parameters are passed generically as ``--param name=value`` (repeatable); values are parsed
as JSON where possible (numbers, true/false, null, lists), otherwise kept as strings. With several
methods, each receives only the parameters its constructor accepts; a parameter no method accepts is an error.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import logging
import os
import sys

from . import __version__
from .core import Setup, available_methods, crop, get_method_class, load_image


def _parse_param(s: str):
    """``"name=value"`` -> ``(name, value)``; the value is parsed as JSON if possible, else kept as a string."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param expects name=value, got '{s}'")
    k, v = s.split("=", 1)
    try:
        return k.strip(), json.loads(v)
    except json.JSONDecodeError:
        return k.strip(), v


def _parser() -> argparse.ArgumentParser:
    """Argument parser with the ``methods`` and ``run`` sub-commands."""
    p = argparse.ArgumentParser(prog="speckle-opd-retrieval", description="OPD retrieval from speckle (CWFS) image pairs.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("methods", help="list available methods")
    r = sub.add_parser("run", help="retrieve OPD for one reference and one or more specimen images")
    S = argparse.SUPPRESS
    r.add_argument("-c", "--config", help="JSON file with any of the keys below (and 'params': {...})")
    r.add_argument("-m", "--method", action="append", default=S, help="method name; repeat for several")
    r.add_argument("-r", "--reference", default=S)
    r.add_argument("-s", "--specimen", nargs="+", default=S, help="file(s) or glob(s)")
    r.add_argument("--pixel-size", dest="pixel_size", type=float, default=S, help="[m]")
    r.add_argument("--distance", type=float, default=S, help="mask–sensor distance [m]")
    r.add_argument("--crop", type=int, nargs=4, metavar=("ROW0", "COL0", "ROW1", "COL1"), default=S)
    r.add_argument("--param", dest="params", action="append", type=_parse_param, default=S,
                   help="method parameter name=value (repeatable)")
    r.add_argument("--device", default=S, help="cpu, cuda, cuda:1, mps (default: CUDA if available, else CPU)")
    r.add_argument("-o", "--out", default=S, help="output folder (default ./results)")
    r.add_argument("--unit", choices=["m", "mm", "um", "nm"], default=S, help="OPD unit in files (default um)")
    r.add_argument("--formats", nargs="+", choices=["tiff", "npy", "png"], default=S, help="default: tiff png")
    r.add_argument("--save-gradient", dest="save_gradient", action="store_true", default=S)
    r.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _run(args) -> int:
    """``run``: merge defaults, config file and flags, then retrieve and save OPD for every specimen and method."""
    cfg = {"method": ["dic"], "out": "results", "unit": "um", "formats": ["tiff", "png"], "save_gradient": False,
           "crop": None, "params": {}, "device": None}
    if args.config:
        with open(args.config) as fh:
            file_cfg = json.load(fh)
        base = os.path.dirname(os.path.abspath(args.config))
        for k in ("reference", "out"):
            if isinstance(file_cfg.get(k), str) and not os.path.isabs(file_cfg[k]):
                file_cfg[k] = os.path.join(base, file_cfg[k])
        if isinstance(file_cfg.get("specimen"), str):
            file_cfg["specimen"] = [file_cfg["specimen"]]
        if "specimen" in file_cfg:
            file_cfg["specimen"] = [s if os.path.isabs(s) else os.path.join(base, s) for s in file_cfg["specimen"]]
        if isinstance(file_cfg.get("method"), str):
            file_cfg["method"] = [file_cfg["method"]]
        cfg.update(file_cfg)
    cli = {k: v for k, v in vars(args).items() if k not in ("cmd", "config", "verbose")}
    if "params" in cli:
        cfg["params"] = {**cfg["params"], **dict(cli.pop("params"))}
    cfg.update(cli)
    for req in ("reference", "specimen", "pixel_size", "distance"):
        if cfg.get(req) in (None, []):
            raise SystemExit(f"missing required setting: {req}")
    setup = Setup(float(cfg["pixel_size"]), float(cfg["distance"]))
    ref = crop(load_image(cfg["reference"]), cfg["crop"])
    specimens = []
    for s in cfg["specimen"]:
        specimens += sorted(glob.glob(s)) or [s]
    classes = {m: get_method_class(m) for m in cfg["method"]}
    accepted = {m: set(inspect.signature(c.__init__).parameters) - {"self", "setup"} for m, c in classes.items()}
    unknown = set(cfg["params"]) - set().union(*accepted.values())
    if unknown:
        raise SystemExit(f"parameter(s) {sorted(unknown)} not accepted by {cfg['method']}")
    params = dict(cfg["params"])
    if cfg["device"]:
        params["device"] = cfg["device"]
    for mname, cls in classes.items():
        method = cls(setup, **{k: v for k, v in params.items() if k in accepted[mname]})
        for path in specimens:
            res = method.retrieve(ref, crop(load_image(path), cfg["crop"]))
            stem = os.path.splitext(os.path.basename(path))[0]
            files = res.save(cfg["out"], f"{stem}_{method.name}", formats=cfg["formats"], unit=cfg["unit"],
                             save_gradient=cfg["save_gradient"])
            print(f"[{method.name}] {os.path.basename(path)}: {res.info['runtime_s']:.1f} s -> {files[0]}")
    return 0


def main(argv=None) -> int:
    """Entry point of ``speckle-opd-retrieval``; returns the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(getattr(args, "verbose", 0), 2),
                        format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "methods":
        for k, d in available_methods().items():
            print(f"{k:12s} {d}")
        return 0
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
