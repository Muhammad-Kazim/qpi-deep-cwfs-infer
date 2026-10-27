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

import numpy as np

from .optical_setup_and_retrieval_result import RetrievalResult, Setup
from .poisson_integration_gradient_to_opd import integrate


@dataclass
class MethodOutput:
    """What a method's ``_estimate`` returns to the base class.

    displacement: (2, H, W) in px, convention specimen(x + d) = reference(x), order (y, x).
    phi: optional (H, W) integrated displacement [px²]; if given, the base class uses it
         instead of integrating ``displacement`` (for methods that solve for phase directly).
    amplitude: optional (H, W).
    info: method-specific extras.
    """

    displacement: np.ndarray
    phi: Optional[np.ndarray] = None
    amplitude: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)


class OPDRetriever:
    """Base class: ``retrieve(reference, specimen) -> RetrievalResult``.

    Subclasses set ``name`` and ``default_integrator`` and implement ``_estimate``.
    Constructor keyword arguments are stored in ``self.params`` (provenance) and
    ``integrator`` ("dct" or "cg") chooses the Poisson solver.
    """

    name: str = ""
    default_integrator: str = "dct"

    def __init__(self, setup: Setup, integrator: Optional[str] = None, **params):
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a speckle_opd_retrieval.Setup")
        self.setup = setup
        self.integrator = integrator or self.default_integrator
        self.params = dict(params, integrator=self.integrator)

    # -- to implement -------------------------------------------------------------
    def _estimate(self, reference: np.ndarray, specimen: np.ndarray) -> MethodOutput:  # pragma: no cover
        raise NotImplementedError

    # -- common behaviour ---------------------------------------------------------
    @staticmethod
    def _as_pair(reference, specimen) -> Tuple[np.ndarray, np.ndarray]:
        ref = np.asarray(reference, dtype=np.float64)
        spec = np.asarray(specimen, dtype=np.float64)
        if ref.ndim != 2 or ref.shape != spec.shape:
            raise ValueError(f"reference {ref.shape} and specimen {spec.shape} must be 2-D and equal in shape")
        if not (np.isfinite(ref).all() and np.isfinite(spec).all()):
            raise ValueError("images contain NaN or inf")
        return ref, spec

    def retrieve(self, reference, specimen) -> RetrievalResult:
        ref, spec = self._as_pair(reference, specimen)
        t0 = time.perf_counter()
        out = self._estimate(ref, spec)
        disp = np.asarray(out.displacement, dtype=np.float64)
        if disp.shape != (2,) + ref.shape:
            raise RuntimeError(f"{self.name}: displacement must have shape (2, H, W), got {disp.shape}")
        phi = integrate(disp, self.integrator) if out.phi is None else np.asarray(out.phi, float) - np.mean(out.phi)
        info = dict(out.info, runtime_s=time.perf_counter() - t0)
        return RetrievalResult(
            opd=phi * self.setup.opd_scale,
            opd_gradient=disp * self.setup.gradient_scale,
            displacement=disp,
            amplitude=out.amplitude,
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
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({p})"


# -- registry ------------------------------------------------------------------------
# name -> ("module:Class", one-line description). Modules are imported only when used,
# so listing methods never imports torch.
METHODS: Dict[str, Tuple[str, str]] = {
    "dic": ("speckle_opd_retrieval.methods.dic:DIC",
            "Digital image correlation, robust to amplitude changes (numpy/scipy)"),
    "deep_cwfs": ("speckle_opd_retrieval.methods.deep_cwfs:DeepCWFS",
                  "Deep CWFS: CWFS-trained RAFT / SEA-RAFT optical-flow networks (needs torch)"),
}


def register(name: str, target: str, description: str = "") -> None:
    """Register an extra method at runtime, e.g. ``register("my", "mypkg.mod:MyRetriever")``."""
    METHODS[name] = (target, description)


def available_methods() -> Dict[str, str]:
    """``{name: description}`` for all registered methods (does not import them)."""
    return {k: v[1] for k, v in METHODS.items()}


def get_method_class(name: str):
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
