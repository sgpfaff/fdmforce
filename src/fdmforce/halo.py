"""High-level FDM halo: mean field (soliton+NFW) + fluctuating granule surrogate.

`FDMHalo` bundles the smooth mean-field background with the fast stochastic force
surrogate and exposes ``force``/``potential`` as functions of position and time in
**tambora/internal units** (kpc, Msun, Gyr; accelerations kpc/Gyr^2, potentials
(kpc/Gyr)^2).  The granule surrogate is local, calibrated at a reference radius
``r_fluct`` (default r_s), so it is valid for a localized system (stream, dwarf,
cluster) orbiting near that radius; whole-halo position-dependent stitching is on
the roadmap.

The adapters in :mod:`fdmforce.adapters` wrap this object for galpy and tambora.
"""
from __future__ import annotations

import numpy as np

from .backgrounds import FDMBackground
from .constants import G_INTERNAL, KMS_TO_KPCGYR
from .engines import LocalGRFPatch
from .surrogate import StochasticForceField


class FDMHalo:
    def __init__(self, m22, M_halo, r_fluct=None, n_modes=2048, seed=0,
                 calibrate=True, background=None, **bg_kwargs):
        self.bg = background if background is not None else \
            FDMBackground(m22=m22, M_halo=M_halo, **bg_kwargs)
        self.m22 = self.bg.m22
        self.r_fluct = float(r_fluct if r_fluct is not None else self.bg.r_s)

        rho = float(self.bg.density(self.r_fluct))
        sigma = float(self.bg.sigma(self.r_fluct))
        L_coh = float(self.bg.scale_height(self.r_fluct))
        self.surrogate = StochasticForceField(
            m22=self.m22, rho_mean=rho, sigma_kms=sigma,
            coherence_scale=L_coh, n_modes=n_modes, seed=seed,
        )
        self.lambda_db = self.bg.lambda_db(self.r_fluct)
        self.valid_local = L_coh / self.lambda_db >= 8.0
        if calibrate:
            self.calibrate()

    # --- calibration against a Layer-1 (3B) ground-truth patch ---------------
    def calibrate(self, N=None, seed=1):
        rho = self.surrogate.rho_mean
        sigma = self.surrogate.sigma_kms
        L = 2.0 * np.pi / self.surrogate.k_min          # L_coh
        if N is None:
            N = int(np.clip(2 * np.ceil(L / (self.lambda_db / 4) / 2), 48, 128))
        patch = LocalGRFPatch(m22=self.m22, rho_mean=rho, sigma_kms=sigma,
                              L=L, N=N, seed=seed)
        _, F = patch.potential_force(0.0)
        target_var = float(np.mean(np.sum(F**2, axis=0)))
        self.surrogate.calibrate_amplitude(target_var)
        return self

    # --- mean field (internal units) -----------------------------------------
    def _r(self, pos):
        pos = np.atleast_2d(np.asarray(pos, float))
        return pos, np.linalg.norm(pos, axis=1)

    def mean_potential(self, pos):
        pos, r = self._r(pos)
        return self.bg.potential(r) * KMS_TO_KPCGYR**2          # (kpc/Gyr)^2

    def mean_force(self, pos):
        pos, r = self._r(pos)
        g = G_INTERNAL * self.bg.enclosed_mass(r) / r**2         # kpc/Gyr^2 (inward mag)
        return -(g / r)[:, None] * pos                            # (N,3)

    # --- fluctuating granule field (internal units) --------------------------
    def fluct_force(self, pos, t=0.0):
        b = self.surrogate.state_at(t)
        return self.surrogate.force(pos, b=b)

    def fluct_potential(self, pos, t=0.0):
        b = self.surrogate.state_at(t)
        return self.surrogate.potential(pos, b=b)

    # --- total ----------------------------------------------------------------
    def force(self, pos, t=0.0, granular=True):
        F = self.mean_force(pos)
        if granular:
            F = F + self.fluct_force(pos, t)
        return F

    def potential(self, pos, t=0.0, granular=True):
        P = self.mean_potential(pos)
        if granular:
            P = P + self.fluct_potential(pos, t)
        return P

    # --- adapters -------------------------------------------------------------
    def as_tambora_force(self, granular=True, mean=True):
        """Return a tambora ``ExternalConservativeForce`` for this halo.

        See :func:`fdmforce.adapters.tambora.make_tambora_force`.
        """
        from .adapters.tambora import make_tambora_force
        return make_tambora_force(self, granular=granular, mean=mean)

    def as_galpy_potential(self, granular=True, mean=True, ro=8.0, vo=220.0):
        """Return a galpy ``Potential`` (list) for this halo.

        See :func:`fdmforce.adapters.galpy.make_galpy_potential`.
        """
        from .adapters.galpy import make_galpy_potential
        return make_galpy_potential(self, granular=granular, mean=mean, ro=ro, vo=vo)

    def summary(self):
        s = dict(self.bg.summary())
        s.update(r_fluct_kpc=self.r_fluct, n_modes=self.surrogate.M,
                 L_coh_over_lambda=2 * np.pi / self.surrogate.k_min / self.lambda_db,
                 valid_local=self.valid_local, c_backend=self.surrogate.use_c)
        return s
