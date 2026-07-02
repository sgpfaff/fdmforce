"""galpy adapter.

Wraps an :class:`~fdmforce.halo.FDMHalo` as a galpy ``Potential`` (mean field +
granular fluctuations).  fdmforce works in physical internal units (kpc, Gyr,
kpc/Gyr); galpy works in natural units set by ``(ro, vo)``, in cylindrical
coordinates, and (for the non-axisymmetric granule field) needs ``_phitorque``.
All of that conversion lives here.

Note: this is a *custom* galpy potential, so it is not on tambora's supported-
potential whitelist — tambora users should use :func:`make_tambora_force`
instead.  This adapter is for direct galpy use (orbits, action-angle, plotting).
"""
from __future__ import annotations

import numpy as np

from ..constants import KMS_TO_KPCGYR


def make_galpy_potential(halo, granular=True, mean=True, ro=8.0, vo=220.0):
    """Build a galpy ``Potential`` wrapping ``halo`` (physical units ro,vo)."""
    from galpy.potential import Potential

    class FDMGalpyPotential(Potential):
        def __init__(self, halo, granular, mean, ro, vo):
            Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
            self._halo = halo
            self._granular = granular
            self._mean = mean
            self._ro = ro
            self._vo = vo
            self._vo_kpcgyr = vo * KMS_TO_KPCGYR
            self._D = self._vo_kpcgyr**2                 # potential natural unit (phys)
            self._T0 = (ro / vo) / KMS_TO_KPCGYR         # galpy time unit in Gyr
            self.isNonAxi = bool(granular)               # granule field breaks axisymmetry
            self.hasC = False

        # --- coordinate/unit helpers ---------------------------------------
        def _phys_pos(self, R, z, phi):
            R = np.atleast_1d(np.asarray(R, float))
            z = np.atleast_1d(np.asarray(z, float))
            phi = np.atleast_1d(np.asarray(phi, float))
            R, z, phi = np.broadcast_arrays(R, z, phi)
            x = R * np.cos(phi) * self._ro
            y = R * np.sin(phi) * self._ro
            zk = z * self._ro
            return np.stack([x, y, zk], axis=-1).reshape(-1, 3), phi.ravel(), R.ravel()

        def _phys(self, pos, t):
            """(force_cartesian [kpc/Gyr^2], potential [(kpc/Gyr)^2]) from the halo."""
            tg = t * self._T0
            F = np.zeros_like(pos)
            P = np.zeros(pos.shape[0])
            if self._mean:
                F = F + self._halo.mean_force(pos)
                P = P + self._halo.mean_potential(pos)
            if self._granular:
                F = F + self._halo.fluct_force(pos, tg)
                P = P + self._halo.fluct_potential(pos, tg)
            return F, P

        @staticmethod
        def _scalarize(a, R):
            return a[0] if np.ndim(R) == 0 else a

        # --- galpy interface (natural units) --------------------------------
        def _evaluate(self, R, z, phi=0.0, t=0.0):
            pos, _, _ = self._phys_pos(R, z, phi)
            _, P = self._phys(pos, t)
            return self._scalarize(P / self._D, R)

        def _Rforce(self, R, z, phi=0.0, t=0.0):
            pos, ph, _ = self._phys_pos(R, z, phi)
            F, _ = self._phys(pos, t)
            FR = F[:, 0] * np.cos(ph) + F[:, 1] * np.sin(ph)     # kpc/Gyr^2
            return self._scalarize(FR * self._ro / self._D, R)

        def _zforce(self, R, z, phi=0.0, t=0.0):
            pos, _, _ = self._phys_pos(R, z, phi)
            F, _ = self._phys(pos, t)
            return self._scalarize(F[:, 2] * self._ro / self._D, R)

        def _phitorque(self, R, z, phi=0.0, t=0.0):
            pos, ph, Rn = self._phys_pos(R, z, phi)
            F, _ = self._phys(pos, t)
            Fphi = -F[:, 0] * np.sin(ph) + F[:, 1] * np.cos(ph)  # kpc/Gyr^2
            # phitorque = -dPhi/dphi = R * Fphi  (natural)
            return self._scalarize(Rn * Fphi * self._ro / self._D, R)

    return FDMGalpyPotential(halo, granular, mean, ro, vo)
