"""Spherical FDM background: soliton core + NFW envelope.

This is the fixed mean-field the granular fluctuations live in ("frozen
background").  It provides the radial density, enclosed mass, potential, and
isotropic velocity dispersion that both generation engines (3A eigenmode,
3B local GRF) consume.

Physics notes / conventions
---------------------------
* Soliton (Schive et al. 2014b empirical fit):
      rho_sol(r) = rho_c / (1 + 0.091 (r/r_c)^2)^8,
      rho_c = 1.9e7 * m22^-2 * (r_c/kpc)^-4   [Msun/kpc^3]
  so the soliton is a one-parameter family in r_c at fixed m22.  Its mass is
      M_c(r_c) = 1.9e7 * I_sol * m22^-2 / (r_c/kpc)   [Msun],
  with I_sol = 4*pi*int_0^inf x^2 (1+0.091 x^2)^-8 dx (computed once, below).
* Envelope: NFW, spliced onto the soliton at r_t = SOLITON_MATCH * r_c with
  density continuity (rho_s fixed so rho_NFW(r_t) = rho_sol(r_t)).
* Core--halo relation: default sets M_core = core_fraction * M_halo.  The exact
  Schive+2014 coefficient is contested across the literature; pass r_c, M_core,
  or core_fraction explicitly to override.  (TODO: calibrated schive2014 option.)

All quantities are in tambora/internal units: kpc, Msun, Gyr; velocities are
reported in km/s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, interpolate, optimize

from ..constants import G_KPC_KMS, de_broglie_length, coherence_time_gyr

# --- soliton profile shape ---------------------------------------------------
_SOL_A = 0.091  # shape coefficient in the Schive fit
_SOL_EXP = 8

# dimensionless profile mass integrals  m(<X) = 4 pi int_0^X x^2 (1+a x^2)^-8 dx
def _sol_mass_integral(X):
    return 4.0 * np.pi * integrate.quad(
        lambda x: x**2 / (1.0 + _SOL_A * x**2) ** _SOL_EXP, 0.0, X
    )[0]

_I_SOL = _sol_mass_integral(np.inf)  # total (all radii)
_J_SOL = _sol_mass_integral(1.0)     # within one core radius r_c

#: mean matter density * 200 for virial radius (z=0, h=0.7): 200 * rho_crit
_RHO_200 = 200.0 * 135.98  # Msun/kpc^3


def soliton_rho_c(m22: float, r_c: float) -> float:
    """Central soliton density [Msun/kpc^3] for axion mass m22 and core radius r_c [kpc]."""
    return 1.9e7 * m22**-2 * r_c**-4


def soliton_mass_within_rc(m22: float, r_c: float) -> float:
    """Soliton mass within r_c [Msun] (the Schive M_c definition)."""
    return soliton_rho_c(m22, r_c) * r_c**3 * _J_SOL


def soliton_total_mass(m22: float, r_c: float) -> float:
    """Total (all-radii) soliton mass [Msun] from the profile integral."""
    return soliton_rho_c(m22, r_c) * r_c**3 * _I_SOL


def core_mass_schive(m22: float, M_halo: float) -> float:
    """Schive+2014b core--halo relation (z=0): M_c(within r_c) [Msun].

    M_c = 1.25e9 * m22^-1 * (M_halo / 1e12 Msun)^(1/3).
    """
    return 1.25e9 * m22**-1 * (M_halo / 1e12) ** (1.0 / 3.0)


def core_radius_from_core_mass(m22: float, M_c: float) -> float:
    """r_c [kpc] from the within-r_c core mass M_c [Msun]:  M_c = 1.9e7 J m22^-2 / r_c."""
    return 1.9e7 * _J_SOL * m22**-2 / M_c


def _mass_concentration(M_halo: float) -> float:
    """Rough c(M) relation (Dutton & Maccio-like), z=0.  Override via ``c``."""
    return 10.0 * (M_halo / 1e12) ** -0.1


@dataclass
class FDMBackground:
    """Spherical soliton+NFW background for an FDM halo.

    Parameters
    ----------
    m22 : float
        Axion mass in 1e-22 eV.
    M_halo : float
        Halo (virial) mass in Msun.
    r_c, M_core : float, optional
        Soliton size controls (precedence r_c > M_core).  ``M_core`` is the
        within-r_c core mass.  Default: Schive+2014b core--halo relation.
    c : float, optional
        NFW concentration; default from a c(M) relation.  The NFW envelope is
        normalized to carry the full halo mass M_halo within r_vir.
    n_grid : int
        Radial grid points (log-spaced) for tabulating M, Phi, sigma.
    """

    m22: float
    M_halo: float
    r_c: float | None = None
    M_core: float | None = None
    c: float | None = None
    n_grid: int = 4096

    # derived (filled in __post_init__)
    rho_c: float = field(init=False)
    M_core_total: float = field(init=False)
    r_vir: float = field(init=False)
    r_s: float = field(init=False)
    rho_s: float = field(init=False)
    r_t: float = field(init=False)

    def __post_init__(self):
        # --- soliton size (Schive core--halo by default) ---------------------
        if self.r_c is None:
            if self.M_core is None:
                self.M_core = core_mass_schive(self.m22, self.M_halo)
            self.r_c = core_radius_from_core_mass(self.m22, self.M_core)
        else:
            self.M_core = soliton_mass_within_rc(self.m22, self.r_c)
        self.rho_c = soliton_rho_c(self.m22, self.r_c)
        self.M_core_total = soliton_total_mass(self.m22, self.r_c)

        # --- virial radius and NFW normalized to carry M_halo ----------------
        self.r_vir = (3.0 * self.M_halo / (4.0 * np.pi * _RHO_200)) ** (1.0 / 3.0)
        if self.c is None:
            self.c = _mass_concentration(self.M_halo)
        self.r_s = self.r_vir / self.c
        mu_c = np.log(1.0 + self.c) - self.c / (1.0 + self.c)
        self.rho_s = self.M_halo / (4.0 * np.pi * self.r_s**3 * mu_c)

        # --- splice at the density crossover (soliton inside, NFW outside) ---
        self.r_t = self._find_crossover()

        self._build_tables()

    def _find_crossover(self) -> float:
        """Radius where the (steeply falling) soliton drops below the NFW envelope."""
        f = lambda r: self._rho_soliton(r) - self._rho_nfw(r)
        lo, hi = 0.3 * self.r_c, 30.0 * self.r_c
        if f(lo) <= 0:  # soliton never dominates (unusual params) -> tiny core region
            return lo
        if f(hi) > 0:   # still dominant far out -> extend search
            hi = self.r_vir
            if f(hi) > 0:
                return hi
        return optimize.brentq(f, lo, hi)

    # --- profile pieces ------------------------------------------------------
    def _rho_soliton(self, r):
        return self.rho_c / (1.0 + _SOL_A * (r / self.r_c) ** 2) ** _SOL_EXP

    def _nfw_shape(self, r):
        x = r / self.r_s
        return 1.0 / (x * (1.0 + x) ** 2)

    def _rho_nfw(self, r):
        return self.rho_s * self._nfw_shape(r)

    def density(self, r):
        """Total mean density [Msun/kpc^3] at radius r [kpc]."""
        r = np.asarray(r, dtype=float)
        return np.where(r < self.r_t, self._rho_soliton(r), self._rho_nfw(r))

    # --- tabulated integrals -------------------------------------------------
    def _build_tables(self):
        r = np.logspace(np.log10(self.r_c * 1e-3), np.log10(self.r_vir * 5.0), self.n_grid)
        rho = self.density(r)

        # enclosed mass M(<r) = int_0^r 4 pi r'^2 rho dr'
        integrand = 4.0 * np.pi * r**2 * rho
        M_enc = integrate.cumulative_trapezoid(integrand, r, initial=0.0)

        # gravitational field g(r) = G M(<r)/r^2   [ (km/s)^2 / kpc ]
        g = G_KPC_KMS * M_enc / r**2

        # potential Phi(r) = -G M(<r_max)/r_max - int_r^{r_max} g dr'
        # integrate g outward, then Phi = Phi_edge - (cumulative from r to edge)
        cum_g = integrate.cumulative_trapezoid(g, r, initial=0.0)
        Phi_edge = -G_KPC_KMS * M_enc[-1] / r[-1]
        Phi = Phi_edge - (cum_g[-1] - cum_g)

        # isotropic Jeans: rho sigma^2 = int_r^{r_max} rho g dr'
        rho_g = rho * g
        cum_rg = integrate.cumulative_trapezoid(rho_g, r, initial=0.0)
        rho_sig2 = cum_rg[-1] - cum_rg
        sigma2 = np.clip(rho_sig2 / rho, 0.0, None)  # (km/s)^2

        self._r_tab = r
        self._M_tab = M_enc
        self._Phi_tab = Phi
        self._sigma_tab = np.sqrt(sigma2)

        # interpolators (log-r where sensible)
        logr = np.log10(r)
        self._M_interp = interpolate.interp1d(logr, M_enc, bounds_error=False,
                                              fill_value=(0.0, M_enc[-1]))
        self._Phi_interp = interpolate.interp1d(logr, Phi, bounds_error=False,
                                                fill_value=(Phi[0], 0.0))
        self._sigma_interp = interpolate.interp1d(logr, self._sigma_tab,
                                                  bounds_error=False,
                                                  fill_value=(self._sigma_tab[0],
                                                              self._sigma_tab[-1]))

        # local density scale height H = rho/|drho/dr| = r/|dln rho/dln r|
        dln = np.gradient(np.log(rho), np.log(r))
        H = r / np.maximum(np.abs(dln), 1e-3)
        self._H_tab = H
        self._H_interp = interpolate.interp1d(logr, H, bounds_error=False,
                                              fill_value=(H[0], H[-1]))

    # --- public radial accessors --------------------------------------------
    def enclosed_mass(self, r):
        """M(<r) [Msun]."""
        return self._M_interp(np.log10(np.asarray(r, float)))

    def potential(self, r):
        """Mean-field potential Phi(r) [(km/s)^2], Phi(inf)->0."""
        return self._Phi_interp(np.log10(np.asarray(r, float)))

    def sigma(self, r):
        """Isotropic 1D velocity dispersion [km/s] at radius r [kpc]."""
        return self._sigma_interp(np.log10(np.asarray(r, float)))

    def v_circ(self, r):
        """Circular velocity sqrt(G M(<r)/r) [km/s]."""
        r = np.asarray(r, float)
        return np.sqrt(G_KPC_KMS * self.enclosed_mass(r) / r)

    def scale_height(self, r):
        """Local density scale height H = rho/|drho/dr| [kpc].

        Physically motivated infrared coherence scale L_coh for the granule
        field (region over which the homogeneous/Maxwellian approximation holds).
        """
        return self._H_interp(np.log10(np.asarray(r, float)))

    # --- de Broglie diagnostics ---------------------------------------------
    def lambda_db(self, r):
        """Local reduced de Broglie length [kpc], using sigma(r)."""
        return de_broglie_length(self.m22, self.sigma(r))

    def coherence_time(self, r):
        """Local granule coherence time [Gyr], using sigma(r)."""
        return coherence_time_gyr(self.m22, self.sigma(r))

    def summary(self) -> dict:
        return {
            "m22": self.m22,
            "M_halo": self.M_halo,
            "M_core_within_rc": self.M_core,
            "M_core_total": self.M_core_total,
            "r_c_kpc": self.r_c,
            "rho_c": self.rho_c,
            "r_vir_kpc": self.r_vir,
            "c": self.c,
            "r_s_kpc": self.r_s,
            "r_t_kpc": self.r_t,
            "sigma_peak_kms": float(np.max(self._sigma_tab)),
            "lambda_db_at_rs_kpc": float(self.lambda_db(self.r_s)),
            "coh_time_at_rs_Gyr": float(self.coherence_time(self.r_s)),
        }
