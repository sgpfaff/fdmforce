"""Layer-1 generation engines.

* ``grf``  — 3B local Gaussian-random-field patch (fast, on-demand).
* ``eigenmode`` — 3A global eigenmode construction (validation/reference). [TODO]
"""
from .grf import LocalGRFPatch

__all__ = ["LocalGRFPatch"]
