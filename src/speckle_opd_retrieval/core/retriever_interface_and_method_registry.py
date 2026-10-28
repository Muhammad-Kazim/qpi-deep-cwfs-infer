"""The common interface every OPD retrieval method implements, and the method registry.

Adding a method
---------------
1. Create ``methods/<name>/`` with a subclass of :class:`OPDRetriever` that implements
   :meth:`OPDRetriever._estimate` (return at least a full-resolution displacement field).
2. Add one line to ``METHODS`` below (name, "module:Class", one-line description).
Everything else (input checks, integration, scaling, result type, CLI) comes for free.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .optical_setup_and_retrieval_result import RetrievalResult, Setup
from .poisson_integration_gradient_to_opd import DEFAULT_INTEGRATOR, INTEGRATORS, integrate
from .torch_device_and_tensor_helpers import as_tensor, resolve_device, to_numpy, working_dtype

# OPD sign (textbook): OPD = ∫(n - n_medium) dz, positive for an object of higher RI (phase lag,
# converging field). Speckles move toward a converging region, i.e. along +∇OPD, so with
# specimen(x + d) = reference(x): OPD = +px²/z · ∫d. Explained in notebooks/opd_sign_convention_explained.ipynb,
# checked in tests/test_opd_sign_convention_physical_speckle_simulation.py.


@dataclass
class MethodOutput:
    """What a method's ``_estimate`` returns to the base class (tensors on the method's device).

    displacement: (2, H, W) in px, convention specimen(x + d) = reference(x), order (y, x).
    phi: optional (H, W) integrated displacement [px²]; if given, the base class uses it
         instead of integrating ``displacement`` (for methods that solve for phase directly).
    amplitude: optional (H, W).
    info: method-specific extras.
    """

    displacement: torch.Tensor
    phi: Optional[torch.Tensor] = None
    amplitude: Optional[torch.Tensor] = None
    info: Dict[str, Any] = field(default_factory=dict)


class OPDRetriever:
    """Base class: ``retrieve(reference, specimen) -> RetrievalResult``.

    Subclasses set ``name`` and implement ``_estimate``.
    Constructor keyword arguments are stored in ``self.params`` (provenance),
    ``integrator`` (a name from :func:`available_integrators`; default ``"poisson_neumann"``) chooses how the
    displacement is integrated and ``device`` where everything
    runs (``None``: CUDA if available, else CPU). ``_estimate`` receives the images as tensors on
    ``self.device`` in ``self.dtype`` (float64; float32 on MPS); the result is NumPy on the CPU.
    """

    name: str = ""

    def __init__(self, setup: Setup, integrator: Optional[str] = None, device: Optional[str] = None, **params):
        """Store the setup, integrator, device and method parameters (``params``, kept for provenance)."""
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a speckle_opd_retrieval.Setup")
        self.setup = setup
        self.integrator = integrator or DEFAULT_INTEGRATOR
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"unknown integrator '{self.integrator}'; available: {sorted(INTEGRATORS)}")
        self.device = resolve_device(device)
        self.dtype = working_dtype(self.device)
        self.params = dict(params, integrator=self.integrator, device=str(self.device))

    # -- to implement -------------------------------------------------------------
    def _estimate(self, reference: torch.Tensor, specimen: torch.Tensor) -> MethodOutput:  # pragma: no cover
        """Method-specific part: two (H, W) tensors on ``self.device`` -> :class:`MethodOutput`."""
        raise NotImplementedError

    # -- common behaviour ---------------------------------------------------------
    def _as_pair(self, reference, specimen) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check the two images (2-D, same shape, finite) and move them to ``self.device``."""
        ref = as_tensor(reference, self.device, self.dtype)
        spec = as_tensor(specimen, self.device, self.dtype)
        if ref.ndim != 2 or ref.shape != spec.shape:
            raise ValueError(f"reference {tuple(ref.shape)} and specimen {tuple(spec.shape)} must be 2-D and equal in shape")
        if not bool(torch.isfinite(ref).all() & torch.isfinite(spec).all()):
            raise ValueError("images contain NaN or inf")
        return ref, spec

    def _synchronize(self):
        """Wait for queued GPU work, so that ``runtime_s`` measures the real computation time."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def retrieve(self, reference, specimen) -> RetrievalResult:
        """``reference``, ``specimen``: 2-D NumPy arrays or tensors (any device; moved to ``self.device``)."""
        ref, spec = self._as_pair(reference, specimen)
        self._synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._estimate(ref, spec)
            disp = as_tensor(out.displacement, self.device, self.dtype)
            if tuple(disp.shape) != (2,) + tuple(ref.shape):
                raise RuntimeError(f"{self.name}: displacement must have shape (2, H, W), got {tuple(disp.shape)}")
            if out.phi is None:
                phi = integrate(disp, self.integrator)
            else:
                phi = as_tensor(out.phi, self.device, self.dtype)
                phi = phi - phi.mean()
        self._synchronize()
        # everything returned to the user is NumPy on the CPU
        info = {k: to_numpy(v) for k, v in out.info.items()}
        info["runtime_s"] = time.perf_counter() - t0
        return RetrievalResult(
            opd=self.setup.opd_scale * to_numpy(phi),
            opd_gradient=self.setup.gradient_scale * to_numpy(disp),
            displacement=to_numpy(disp),
            amplitude=to_numpy(out.amplitude),
            method=self.name,
            params=dict(self.params),
            setup=self.setup,
            info=info,
        )

    def retrieve_many(self, reference, specimens: Iterable) -> List[RetrievalResult]:
        """Same reference, several specimen frames (e.g. a time series). The model / settings are reused."""
        return [self.retrieve(reference, s) for s in specimens]

    __call__ = retrieve

    def __repr__(self):
        """``DIC(window=32, ..., device='cuda')``."""
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({p})"


# -- registry ------------------------------------------------------------------------
# name -> ("module:Class", one-line description). Modules are imported only when used.
METHODS: Dict[str, Tuple[str, str]] = {
    "dic": ("speckle_opd_retrieval.methods.dic:DIC",
            "Digital image correlation (ZNCC), robust to amplitude changes"),
    "deep_cwfs": ("speckle_opd_retrieval.methods.deep_cwfs:DeepCWFS",
                  "Deep CWFS: CWFS-trained RAFT / SEA-RAFT optical-flow networks"),
}


def register(name: str, target: str, description: str = "") -> None:
    """Register an extra method at runtime, e.g. ``register("my", "mypkg.mod:MyRetriever")``."""
    METHODS[name] = (target, description)


def available_methods() -> Dict[str, str]:
    """``{name: description}`` for all registered methods (does not import them)."""
    return {k: v[1] for k, v in METHODS.items()}


def get_method_class(name: str):
    """The retriever class registered under ``name`` (imports its module on first use)."""
    key = name.lower().replace("-", "_")
    if key not in METHODS:
        raise KeyError(f"unknown method '{name}'; available: {sorted(METHODS)}")
    mod, cls = METHODS[key][0].split(":")
    return getattr(importlib.import_module(mod), cls)


def get_method(name: str, setup: Setup, **params) -> OPDRetriever:
    """Instantiate a method by name: ``get_method("dic", setup, window=32)``."""
    return get_method_class(name)(setup, **params)


def retrieve(reference, specimen, method: str, setup: Setup, **params) -> RetrievalResult:
    """One-call convenience: ``retrieve(ref, spec, "dic", setup)``."""
    return get_method(method, setup, **params).retrieve(reference, specimen)
