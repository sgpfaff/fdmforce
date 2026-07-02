"""3A — Global eigenmode construction of an FDM halo (Yavetz-Li-Hui 2022).

In the fixed spherical background Phi0(r) the wavefunction is expanded in energy
eigenstates,

    psi(x,t) = sum_{n l m} a_{nlm} R_{nl}(r) Y_{lm}(theta,phi) e^{-i eps_{nl} t / b},

with b = hbar/m.  Writing the specific-energy operator (H/m) and u = r R:

    -(b^2/2) u'' + [Phi0(r) + (b^2/2) l(l+1)/r^2] u = eps u,     u(0)=u(rmax)=0.

Each l gives a 1D symmetric-tridiagonal eigenproblem.  Mode occupations
<|a_{nlm}|^2> are set from the isotropic distribution function f(E) (Eddington
inversion of the target profile), so the ensemble density reproduces rho(r).
Random Gaussian a's give one halo realization.

This engine is the reference/validation oracle: globally coherent, spherically
self-consistent, but the number of modes grows steeply with m (see design doc),
so it is used on affordable halos to cross-check 3B and to expose the beat-
frequency spectrum that informs the Layer-2 surrogate choice.

Internal units (kpc, Msun, Gyr; velocities kpc/Gyr).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from ..constants import KMS_TO_KPCGYR, hbar_over_m


class EigenmodeHalo:
    def __init__(self, background, l_max=20, n_r=3000, r_max=None,
                 eps_floor=None, seed=None):
        self.bg = background
        self.m22 = background.m22
        self.b = hbar_over_m(self.m22, internal=True)  # kpc^2/Gyr
        self.l_max = int(l_max)
        self.n_r = int(n_r)
        self.r_max = float(r_max if r_max is not None else 3.0 * background.r_vir)
        self.eps_floor = eps_floor  # None -> auto (1.05*min Phi0)

        # uniform radial grid (u(0)=0 boundary at r=0)
        self.r = np.linspace(self.r_max / self.n_r, self.r_max, self.n_r)
        self.dr = self.r[1] - self.r[0]
        # background potential in internal velocity^2 units
        self.Phi0 = background.potential(self.r) * KMS_TO_KPCGYR**2

        self._solve_modes()
        self._build_df()
        self._populate(seed)

    # --- radial eigenproblem per l ------------------------------------------
    def _solve_modes(self):
        kin = 0.5 * self.b**2 / self.dr**2
        offd = -kin * np.ones(self.n_r - 1)
        # energy window of occupied states: from an energy floor (avoid the
        # deepest, under-resolved soliton-core states the thermal DF barely
        # weights) up to the bound edge eps<0.  eps_floor set a few core
        # binding-depths below the envelope so the halo states are all included.
        eps_floor = self.eps_floor if getattr(self, "eps_floor", None) else \
            1.05 * float(self.Phi0.min())
        modes = []  # each: (l, eps, u(r) normalized)
        for l in range(self.l_max + 1):
            centrifugal = 0.5 * self.b**2 * l * (l + 1) / self.r**2
            diag = 2.0 * kin + self.Phi0 + centrifugal
            # all bound states in the window (value selection)
            vals, vecs = eigh_tridiagonal(
                diag, offd, select="v", select_range=(eps_floor, -1e-9)
            )
            if vals.size == 0:
                continue  # no bound states at this l (centrifugal barrier)
            for j, eps in enumerate(vals):
                u = vecs[:, j]
                u = u / np.sqrt(np.trapezoid(u**2, self.r))
                modes.append((l, float(eps), u))
        self.modes = modes
        self.l_arr = np.array([mm[0] for mm in modes])
        self.eps_arr = np.array([mm[1] for mm in modes])
        self.u_mat = np.array([mm[2] for mm in modes])  # (Nmodes, n_r)
        # R_nl(r) = u/r
        self.R_mat = self.u_mat / self.r[None, :]

    # --- Eddington DF f(E), E = -eps ----------------------------------------
    def _build_df(self):
        r = self.r
        Psi = -self.Phi0                      # relative potential (>0), increasing inward
        rho = self.bg.density(r) * 1.0        # Msun/kpc^3
        # sort by increasing Psi (i.e. decreasing r)
        order = np.argsort(Psi)
        Psi_s, rho_s = Psi[order], rho[order]
        # keep strictly increasing Psi
        keep = np.concatenate([[True], np.diff(Psi_s) > 0])
        Psi_s, rho_s = Psi_s[keep], rho_s[keep]
        drho_dPsi = np.gradient(rho_s, Psi_s)

        # f(E) = 1/(sqrt(8) pi^2) d/dE  int_0^E drho/dPsi / sqrt(E-Psi) dPsi
        # substitute u=sqrt(E-Psi): integral = int_0^sqrt(E) 2 drho/dPsi(E-u^2) du
        E_grid = np.linspace(Psi_s[1], Psi_s[-1], 400)
        dgrad = np.interp(E_grid, Psi_s, drho_dPsi)  # for boundary term (~0)
        I = np.zeros_like(E_grid)
        for i, E in enumerate(E_grid):
            uu = np.linspace(0.0, np.sqrt(E), 300)
            Psi_u = E - uu**2
            g = np.interp(Psi_u, Psi_s, drho_dPsi)
            I[i] = 2.0 * np.trapezoid(g, uu)
        f = np.gradient(I, E_grid) / (np.sqrt(8.0) * np.pi**2)
        f = np.clip(f, 0.0, None)
        self._E_grid = E_grid
        self._f_grid = f

    def f_of_eps(self, eps):
        """Isotropic DF evaluated at total specific energy eps (<0)."""
        return np.interp(-eps, self._E_grid, self._f_grid, left=0.0, right=0.0)

    # --- mode population -----------------------------------------------------
    def _populate(self, seed):
        # <|a_nl|^2> proportional to f(eps_nl); each (l) carries (2l+1) m-states.
        w = self.f_of_eps(self.eps_arr)
        w = np.clip(w, 0.0, None)
        self.mode_weight = w  # per (n,l), before m multiplicity
        # target-mass normalization: match total mass within r_max
        # rho_recon(r) = sum_nl (2l+1)/(4pi) <|a|^2> |R_nl|^2
        rho_shape = ((2 * self.l_arr + 1) / (4 * np.pi))[:, None] * self.R_mat**2
        rho_recon_unit = (w[:, None] * rho_shape).sum(0)
        M_unit = np.trapezoid(4 * np.pi * self.r**2 * rho_recon_unit, self.r)
        M_target = self.bg.enclosed_mass(self.r_max)
        self.amp2_scale = M_target / M_unit
        self.rho_recon = self.amp2_scale * rho_recon_unit

        # random Gaussian realization of a_{nlm}
        rng = np.random.default_rng(seed)
        self._a = {}
        for i, (l, eps, _) in enumerate(self.modes):
            var = self.amp2_scale * w[i]
            ms = np.arange(-l, l + 1)
            a = (rng.standard_normal(len(ms)) + 1j * rng.standard_normal(len(ms)))
            a *= np.sqrt(var / 2.0)
            self._a[i] = (ms, a)
        self.omega = self.eps_arr / self.b  # 1/Gyr, mode frequencies

    # --- diagnostics ---------------------------------------------------------
    def n_modes(self):
        return len(self.modes), int(np.sum(2 * self.l_arr + 1))

    def beat_frequencies(self, r0=None):
        """Pairwise beat frequencies |eps_i-eps_j|/b and their weights.

        If ``r0`` is given, weight each pair by its density overlap at that
        radius, (2l_i+1)(2l_j+1)|R_i(r0)|^2|R_j(r0)|^2 * occupations — i.e. the
        beats that actually drive fluctuations *there* (the granule coherence),
        rather than global core-envelope cross terms.
        """
        occ = self.mode_weight * self.amp2_scale
        if r0 is not None:
            i0 = np.argmin(np.abs(self.r - r0))
            occ = occ * (2 * self.l_arr + 1) * self.R_mat[:, i0] ** 2
        dom = np.abs(self.eps_arr[:, None] - self.eps_arr[None, :]) / self.b
        weight = occ[:, None] * occ[None, :]
        iu = np.triu_indices(len(occ), k=1)
        return dom[iu], weight[iu]

    def density_profile_recon(self):
        """Ensemble-mean reconstructed rho(r) vs target (validation)."""
        return self.r, self.rho_recon, self.bg.density(self.r)
