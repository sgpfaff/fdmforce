"""3B — Local Gaussian-random-field (Widrow-Kaiser) FDM granule patch.

In a patch small compared to the halo, the FDM wavefunction is well approximated
by a homogeneous Gaussian random field: a superposition of plane waves whose
momenta sample the *local* velocity distribution.  For an isotropic Maxwellian
of dispersion sigma, the momentum-space power spectrum is

    P(k) proportional to exp(-k^2 / (2 k_sig^2)),   k_sig = sigma / (hbar/m),

with velocity v = (hbar/m) k.  Each mode evolves freely, psi_k(t) ~ e^{-i w_k t}
with w_k = (hbar/m) k^2 / 2, so granules move and decohere on ~hbar/(m sigma^2).

The patch is periodic; the fluctuating potential is obtained by an in-patch FFT
Poisson solve.  This is the fast, on-demand generator (cost tied to the patch,
not the whole halo) and the source of the granule/force statistics used to pick
the Layer-2 surrogate.

All quantities internal units (kpc, Msun, Gyr); velocities in km/s on input.
"""
from __future__ import annotations

import numpy as np

from ..constants import G_INTERNAL, KMS_TO_KPCGYR, hbar_over_m


class LocalGRFPatch:
    """A periodic cubic patch of the FDM granule field at one halo radius.

    Parameters
    ----------
    m22 : float
        Axion mass in 1e-22 eV.
    rho_mean : float
        Local mean density [Msun/kpc^3] the granules average to.
    sigma_kms : float
        Local 1D velocity dispersion [km/s] setting the granule scale.
    L : float
        Patch side length [kpc].  Should span several granules.
    N : int
        Grid points per side.
    seed : int, optional
        RNG seed for a reproducible realization.
    """

    def __init__(self, m22, rho_mean, sigma_kms, L, N=64, seed=None):
        self.m22 = float(m22)
        self.rho_mean = float(rho_mean)
        self.sigma_kms = float(sigma_kms)
        self.L = float(L)
        self.N = int(N)
        self.hbar_m = hbar_over_m(self.m22, internal=True)  # kpc^2/Gyr
        # sigma in internal velocity units (kpc/Gyr) -> k_sig in 1/kpc
        sigma_int = self.sigma_kms * KMS_TO_KPCGYR
        self.k_sig = sigma_int / self.hbar_m  # 1/kpc

        # real-space grid and k-grid (periodic)
        self.dx = self.L / self.N
        k1 = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)  # 1/kpc
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, k1, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.kmag = np.sqrt(self.k2)

        # momentum-space amplitude sqrt(P(k)); free-particle frequency w_k
        Pk = np.exp(-self.k2 / (2.0 * self.k_sig**2))
        self.sqrtP = np.sqrt(Pk)
        self.omega = 0.5 * self.hbar_m * self.k2  # 1/Gyr

        # draw a fixed complex-Gaussian realization of the mode coefficients
        rng = np.random.default_rng(seed)
        g = (rng.standard_normal(self.kx.shape)
             + 1j * rng.standard_normal(self.kx.shape)) / np.sqrt(2.0)
        self._coeff = g * self.sqrtP

        # normalization so <|psi|^2> = rho_mean (density = m|psi|^2 absorbed here:
        # we treat |psi|^2 directly as mass density)
        psi0 = self._psi_realspace(self._coeff)
        norm = np.sqrt(self.rho_mean / np.mean(np.abs(psi0) ** 2))
        self._coeff *= norm

    # --- field reconstruction -------------------------------------------------
    def _psi_realspace(self, coeff):
        # inverse FFT; scale by N^3 so amplitude is grid-independent
        return np.fft.ifftn(coeff) * self.N**3

    def psi(self, t=0.0):
        """Complex wavefunction on the patch grid at time t [Gyr]."""
        coeff_t = self._coeff * np.exp(-1j * self.omega * t)
        return self._psi_realspace(coeff_t)

    def density(self, t=0.0):
        """Mass density |psi|^2 [Msun/kpc^3] on the grid at time t."""
        return np.abs(self.psi(t)) ** 2

    def delta_rho(self, t=0.0):
        rho = self.density(t)
        return rho - np.mean(rho)

    # --- in-patch Poisson -> potential and force ------------------------------
    def potential_force(self, t=0.0):
        """Return (delta_Phi [(kpc/Gyr)^2], F [kpc/Gyr^2, shape (3,N,N,N)]).

        Solves grad^2 delta_Phi = 4 pi G delta_rho on the periodic patch and
        F = -grad(delta_Phi), both via FFT.
        """
        drho_k = np.fft.fftn(self.delta_rho(t))
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_k = -4.0 * np.pi * G_INTERNAL * drho_k / self.k2
        phi_k[0, 0, 0] = 0.0  # remove k=0 (mean) mode
        phi = np.real(np.fft.ifftn(phi_k))
        # F = -grad Phi ; F_k = -i k phi_k
        Fx = np.real(np.fft.ifftn(-1j * self.kx * phi_k))
        Fy = np.real(np.fft.ifftn(-1j * self.ky * phi_k))
        Fz = np.real(np.fft.ifftn(-1j * self.kz * phi_k))
        return phi, np.stack([Fx, Fy, Fz])

    # --- diagnostics ----------------------------------------------------------
    def granule_size(self, t=0.0):
        """FWHM-like correlation length of delta_rho along an axis [kpc].

        Defined as the lag where the (normalized) 1D density autocorrelation
        first drops to 1/2.
        """
        drho = self.delta_rho(t)
        # autocorrelation along x, averaged over the other two axes, via FFT
        f = np.fft.fftn(drho)
        ac = np.real(np.fft.ifftn(np.abs(f) ** 2))
        ac = ac / ac[0, 0, 0]
        line = ac[:, 0, 0]
        lags = np.arange(self.N) * self.dx
        half = np.argmax(line < 0.5)
        if half == 0:
            return np.nan
        # linear interpolation to the 0.5 crossing
        x0, x1 = lags[half - 1], lags[half]
        y0, y1 = line[half - 1], line[half]
        return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)

    def lambda_db(self):
        """Predicted reduced de Broglie length hbar/(m sigma) [kpc]."""
        return self.hbar_m / (self.sigma_kms * KMS_TO_KPCGYR)

    def coherence_time_pred(self):
        """Predicted granule coherence time hbar/(m sigma^2) [Gyr]."""
        sigma_int = self.sigma_kms * KMS_TO_KPCGYR
        return self.hbar_m / sigma_int**2
