"""Layer-2 fast evaluators (surrogates fit to Layer-1 generation).

* ``stochastic`` — B: random-Fourier + per-mode complex-OU state-space force
  field.  Mesh-free, O(M) to advance the latent state, hook-native.
"""
from .stochastic import StochasticForceField

__all__ = ["StochasticForceField"]
