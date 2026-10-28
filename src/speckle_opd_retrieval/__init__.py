"""speckle_opd_retrieval: optical path difference (OPD) retrieval from speckle (CWFS) image pairs.

All methods share one interface::

    from speckle_opd_retrieval import Setup, get_method, load_image

    setup = Setup(pixel_size=3.45e-6, distance=2.48e-3)
    ref, spec = load_image("ref.tiff"), load_image("specimen.tiff")

    dic  = get_method("dic", setup, window=32, step=8, saturation=65520)
    deep = get_method("deep_cwfs", setup)    # default checkpoint: longer-trained SEA-RAFT medium
    for m in (dic, deep):
        res = m.retrieve(ref, spec)          # RetrievalResult
        res.opd, res.opd_gradient, res.displacement, res.info

Everything runs in PyTorch: ``device=None`` (default) uses CUDA if available, else the CPU; pass
``device="cpu"``, ``"cuda:1"``, … to choose. Results are NumPy arrays on the CPU.
``available_methods()`` lists the methods; each lives in ``methods/<name>/``.
"""
from .core import (MethodOutput, OPDRetriever, RetrievalResult, Setup, available_methods, crop, get_method,
                   get_method_class, integrate, integrate_cg, integrate_dct, load_image, register,
                   resolve_device, retrieve, save_result)

__version__ = "0.1.0"

__all__ = ["Setup", "RetrievalResult", "OPDRetriever", "MethodOutput", "available_methods", "get_method",
           "get_method_class", "register", "retrieve", "integrate", "integrate_dct", "integrate_cg",
           "load_image", "crop", "save_result", "resolve_device"]
