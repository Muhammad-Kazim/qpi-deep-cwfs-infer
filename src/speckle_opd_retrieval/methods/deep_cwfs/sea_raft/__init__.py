"""Vendored SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT, BSD-3-Clause).

Changes from upstream: relative imports, the Hugging Face Hub mixin removed,
and ImageNet backbone initialisation made opt-in (``args.init_weight``) so that
building a model to load a CWFS checkpoint never downloads anything.
"""
from .raft import RAFT

__all__ = ["RAFT"]
