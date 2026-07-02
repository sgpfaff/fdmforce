"""tambora adapter.

Exposes an :class:`~fdmforce.halo.FDMHalo` as a native tambora
``ExternalConservativeForce`` — ``acc(pos, t)`` / ``potential(pos, t)`` in tambora
internal units (kpc, Msun, Gyr).  The granular realization is a deterministic
function of ``t`` (per-mode phase rotation via ``surrogate.state_at(t)``), so the
force satisfies the position-and-time-only contract of an external force.

This does **not** route the fluctuating field through galpy (no
``MultipoleExpansionPotential`` re-solve per step): the surrogate is evaluated
directly, which is the whole point of the fast path.
"""
from __future__ import annotations


def make_tambora_force(halo, granular=True, mean=True):
    """Build a tambora ``ExternalConservativeForce`` for ``halo``.

    Parameters
    ----------
    halo : fdmforce.halo.FDMHalo
    granular : bool
        Include the fluctuating granule surrogate.
    mean : bool
        Include the smooth mean field (soliton + NFW).  Set False if you add the
        mean field yourself (e.g. via a galpy potential + ``ExternalGalpyPotential``).
    """
    from tambora.dynamics.forces.external_force.ExternalConservativeForce import (
        ExternalConservativeForce,
    )

    class FDMForce(ExternalConservativeForce):
        """FDM mean + granular force as a tambora external conservative force."""

        def __init__(self, halo, granular, mean):
            self._halo = halo
            self._granular = granular
            self._mean = mean

        def acc(self, pos, t):
            F = 0.0
            if self._mean:
                F = F + self._halo.mean_force(pos)
            if self._granular:
                F = F + self._halo.fluct_force(pos, t)
            return F

        def potential(self, pos, t):
            P = 0.0
            if self._mean:
                P = P + self._halo.mean_potential(pos)
            if self._granular:
                P = P + self._halo.fluct_potential(pos, t)
            return P

    return FDMForce(halo, granular, mean)
