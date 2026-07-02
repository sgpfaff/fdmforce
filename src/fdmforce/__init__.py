"""fdmforce — fast Fuzzy Dark Matter potentials and forces.

Layer 1 (generation): construct a physically-correct frozen-background FDM halo
field or its statistics (eigenmode / local-GRF engines).
Layer 2 (fast evaluation): compact surrogate for the hot path (POD / stochastic
state-space) — TBD after the generation benchmark.

See ``docs/method_selection.md`` for the design rationale.
"""
from __future__ import annotations

from . import constants
from .backgrounds import FDMBackground

__version__ = "0.0.1"
__all__ = ["FDMBackground", "constants", "__version__"]
