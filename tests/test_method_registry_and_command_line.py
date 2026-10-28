"""Registry (lazy) and the command line."""
import json

import numpy as np
import pytest

from speckle_opd_retrieval import OPDRetriever, Setup, available_methods, get_method, register, retrieve
from speckle_opd_retrieval.command_line_interface import main
from conftest import DEVICE, bump, speckle, warp


def test_listing_methods():
    assert {"dic", "deep_cwfs"} <= set(available_methods())


def test_get_method_and_one_call_retrieve():
    I0 = speckle((160, 200)); spec = warp(I0, bump((160, 200), amp=20))
    m = get_method("DIC", Setup(1.0, 1.0), search=4)
    assert isinstance(m, OPDRetriever) and m.name == "dic"
    r = retrieve(I0, spec, "dic", Setup(1.0, 1.0), search=4)
    assert r.opd.shape == I0.shape
    with pytest.raises(KeyError):
        get_method("nope", Setup(1.0, 1.0))


def test_register_external_method():
    register("dic_alias", "speckle_opd_retrieval.methods.dic:DIC", "alias for tests")
    assert "dic_alias" in available_methods()


def test_cli_run_dic_with_config_and_params(tmp_path, capsys):
    tifffile = pytest.importorskip("tifffile")
    I0 = speckle((160, 200)) * 1000 + 10
    tifffile.imwrite(tmp_path / "ref.tiff", I0.astype(np.float32))
    for i in range(2):
        tifffile.imwrite(tmp_path / f"spec_{i}.tiff", warp(I0, bump((160, 200), amp=10 * (i + 1))).astype(np.float32))
    (tmp_path / "cfg.json").write_text(json.dumps({"method": "dic", "reference": "ref.tiff", "specimen": "spec_*.tiff",
                                                   "pixel_size": 3.45e-6, "distance": 2.48e-3, "out": "out", "device": DEVICE,
                                                   "params": {"search": 4}}))
    assert main(["run", "-c", str(tmp_path / "cfg.json"), "--formats", "npy", "--param", "window=24"]) == 0
    for i in range(2):
        assert np.load(tmp_path / "out" / f"spec_{i}_dic_opd.npy").shape == (160, 200)
    meta = json.load(open(tmp_path / "out" / "spec_0_dic_meta.json"))
    assert meta["params"]["window"] == 24 and meta["params"]["search"] == 4
    assert meta["params"]["device"].startswith(DEVICE.split(":")[0])
    assert main(["methods"]) == 0 and "deep_cwfs" in capsys.readouterr().out


def test_cli_rejects_unknown_parameter(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    I0 = speckle((120, 120)).astype(np.float32)
    tifffile.imwrite(tmp_path / "r.tiff", I0); tifffile.imwrite(tmp_path / "s.tiff", I0)
    with pytest.raises(SystemExit):
        main(["run", "-r", str(tmp_path / "r.tiff"), "-s", str(tmp_path / "s.tiff"), "--pixel-size", "1e-6",
              "--distance", "1e-3", "--param", "not_a_param=1", "-o", str(tmp_path / "o")])
