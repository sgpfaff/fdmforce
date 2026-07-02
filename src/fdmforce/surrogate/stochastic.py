"""Surrogate B — random-Fourier + complex-OU stochastic force field.

Physics
-------
In the local-GRF picture the wavefunction is psi = sum_q c_q e^{i(q.x - w_q t)},
w_q = (hbar/m) q^2/2.  A density Fourier mode is

    delta_rho_k(t) = sum_q c*_q c_{q+k} e^{-i[(hbar/m)(k^2/2 + q.k)] t},

so for fixed k it is a **narrowband** complex process: centre frequency
w_c(k) = (hbar/m) k^2/2 and bandwidth gamma(k) = (hbar/m) |k| k_sig (the spread
of q.k over the momentum distribution, k_sig = sigma/(hbar/m)).  The force mode is

    F_k(t) = -i k delta_Phi_k = i (4 pi G) (k / k^2) delta_rho_k(t).

We therefore represent the force field as M random Fourier modes,

    F(x,t) = Re sum_{j=1..M} a_j b_j(t) e^{i k_j . x},   a_j = C * i * (4 pi G) k_j / |k_j|^2,

with k_j drawn from the density power spectrum P_drho(k) ~ exp(-k^2/(4 k_sig^2))
and each latent scalar b_j(t) an exact complex Ornstein-Uhlenbeck process

    db_j = (-i w_c,j - gamma_j) b_j dt + noise,   <|b_j|^2> = 1 (stationary),

advanced by the exact discrete update for any dt.  The single overall amplitude
C is calibrated so <|F|^2> matches a Layer-1 (3B) ground-truth patch; the
space- and time-correlation *shapes* come from the physics above.

Cost: O(M) to advance the latent state (particle-independent, hook-native),
O(N*M) to evaluate at N particle positions.  Mesh-free.

Internal units (kpc, Msun, Gyr; velocities km/s on input).
"""
from __future__ import annotations

import numpy as np

from .. import _core
from ..constants import KMS_TO_KPCGYR, hbar_over_m


class StochasticForceField:
    def __init__(self, m22, rho_mean, sigma_kms, n_modes=1024,
                 k_min=None, coherence_scale=None, seed=None, amp=1.0,
                 use_c=True):
        self.m22 = float(m22)
        self.rho_mean = float(rho_mean)
        self.sigma_kms = float(sigma_kms)
        self.M = int(n_modes)
        self.use_c = bool(use_c) and _core.available()
        self.hbar_m = hbar_over_m(self.m22, internal=True)  # kpc^2/Gyr
        sigma_int = self.sigma_kms * KMS_TO_KPCGYR
        self.k_sig = sigma_int / self.hbar_m  # 1/kpc

        # infrared cutoff: force power ~ 1/k^2 is IR-sensitive; cut at the scale
        # over which the local (constant sigma) approximation holds.
        if k_min is None:
            L_coh = coherence_scale if coherence_scale is not None else \
                6.0 / self.k_sig  # ~ a few de Broglie lengths
            k_min = 2.0 * np.pi / L_coh
        self.k_min = float(k_min)

        rng = np.random.default_rng(seed)
        # Sample from the FORCE power spectrum P_F(k) ~ exp(-k^2/4 k_sig^2)/k^2
        # with equal-weight modes: uniform direction on the sphere, |k| ~ the
        # radial pdf P_F(k) k^2 ~ exp(-k^2/4 k_sig^2) (half-normal, std=sqrt2 k_sig).
        # This puts modes where the force power lives (low k), capturing the
        # long-time correlation tail that P_drho sampling under-resolves.
        kmag = np.abs(rng.normal(0.0, np.sqrt(2.0) * self.k_sig, size=self.M))
        bad = kmag < self.k_min
        while np.any(bad):
            kmag[bad] = np.abs(rng.normal(0.0, np.sqrt(2.0) * self.k_sig, size=int(bad.sum())))
            bad = kmag < self.k_min
        dirs = rng.standard_normal((self.M, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.k = kmag[:, None] * dirs    # (M,3)
        self.kmag = kmag                 # (M,)
        self.khat = dirs                 # (M,3) unit

        # Per-mode frequency of the DENSITY/force field.  Because rho=|psi|^2 is
        # bilinear, the fast wavefunction carrier (hbar/m)k^2/2 CANCELS; the
        # dominant q~-k/2 pairing gives delta_rho_k(t) a slow beat centred at
        # w=0 with Gaussian width dw(k) = (hbar/m) |k| k_sig / sqrt(2).  Drawing
        # per-mode frequencies from N(0, dw^2) and advancing as pure phase gives
        # the correct slow, monotonic GAUSSIAN temporal decorrelation via dephasing.
        self.domega = self.hbar_m * self.kmag * self.k_sig / np.sqrt(2.0)  # 1/Gyr
        self.omega = self.domega * rng.standard_normal(self.M)             # centred at 0
        # optional slow damping (0 = frozen background); kept for future non-frozen use
        self.gamma = np.zeros(self.M)

        # equal-weight force amplitude along k-hat (all constants folded into C,
        # which is calibrated to the target force variance).
        self._C = float(amp)
        self.a = self._C * 1j * self.khat                        # (M,3) complex

        # latent state, unit stationary variance complex OU
        self._rng = rng
        self.b = (rng.standard_normal(self.M) + 1j * rng.standard_normal(self.M)) / np.sqrt(2.0)
        self.t = 0.0

    # --- latent-state dynamics (the hook advances this) ----------------------
    def advance(self, dt):
        """Advance the latent state by dt [Gyr].

        Frozen background (gamma=0): pure per-mode phase rotation b_j *= e^{-i w_j dt}
        (deterministic, reproducible).  If gamma>0, an exact complex-OU update
        with matching stationary variance is applied (for future non-frozen use).
        """
        decay = np.exp((-1j * self.omega - self.gamma) * dt)
        self.b = decay * self.b
        if np.any(self.gamma > 0):
            var_noise = 1.0 - np.exp(-2.0 * self.gamma * dt)
            noise = (self._rng.standard_normal(self.M)
                     + 1j * self._rng.standard_normal(self.M)) / np.sqrt(2.0)
            self.b = self.b + np.sqrt(var_noise) * noise
        self.t += dt

    def state_at(self, t):
        """Deterministic phase-advanced state at absolute time t [Gyr]."""
        return self.b * np.exp((-1j * self.omega) * (t - self.t))

    # --- field evaluation ----------------------------------------------------
    def force(self, pos, b=None):
        """Force at positions ``pos`` (N,3) [kpc] given latent state ``b``.

        Returns (N,3) in kpc/Gyr^2.  If ``b`` is None, uses the current state.
        """
        pos = np.atleast_2d(np.asarray(pos, float))
        b = self.b if b is None else b
        ab = self.a * b[:, None]                     # (M,3) complex
        if self.use_c:
            return _core.force_eval(pos, self.k,
                                    np.ascontiguousarray(ab.real),
                                    np.ascontiguousarray(ab.imag))
        E = np.exp(1j * (pos @ self.k.T))            # (N,M)
        return np.real(E @ ab)                        # (N,3)

    def potential(self, pos, b=None):
        """Fluctuating potential delta_Phi at positions ``pos`` [(kpc/Gyr)^2].

        Consistency F_k = -i k_vec delta_Phi_k with a_j = C i k-hat_j gives the
        scalar mode amplitude p_j = -C / |k_j| (so that -i k_vec p_j = a_j).
        """
        pos = np.atleast_2d(np.asarray(pos, float))
        b = self.b if b is None else b
        pb = (-self._C / self.kmag) * b                          # (M,) complex
        if self.use_c:
            return _core.potential_eval(pos, self.k,
                                        np.ascontiguousarray(pb.real),
                                        np.ascontiguousarray(pb.imag))
        return np.real(np.exp(1j * (pos @ self.k.T)) @ pb)

    # --- calibration ---------------------------------------------------------
    def force_variance(self, n_sample=4096, seed=0):
        """Monte-Carlo <|F|^2> over random positions at the current state."""
        rng = np.random.default_rng(seed)
        L = 2.0 * np.pi / self.k_min
        pos = rng.uniform(0, L, size=(n_sample, 3))
        F = self.force(pos)
        return float(np.mean(np.sum(F**2, axis=1)))

    def calibrate_amplitude(self, target_force_var, n_sample=4096):
        """Rescale overall amplitude C so <|F|^2> == target_force_var."""
        cur = self.force_variance(n_sample=n_sample)
        scale = np.sqrt(target_force_var / cur)
        self._C *= scale
        self.a *= scale
        return self._C
