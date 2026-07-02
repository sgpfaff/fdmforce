"""Validate the 3B local-GRF patch against analytic granule predictions."""
import numpy as np

from fdmforce import FDMBackground
from fdmforce.engines import LocalGRFPatch
from fdmforce.constants import KMS_TO_KPCGYR, G_INTERNAL

bg = FDMBackground(m22=1.0, M_halo=1e10)
r0 = bg.r_s
sigma0 = float(bg.sigma(r0))
rho0 = float(bg.density(r0))
print(f"halo: m22=1, M_h=1e10 | r_s={r0:.3f} kpc, sigma={sigma0:.2f} km/s, rho={rho0:.3e}")

lam = bg.lambda_db(r0)
L = 12.0 * lam
patch = LocalGRFPatch(m22=1.0, rho_mean=rho0, sigma_kms=sigma0, L=L, N=96, seed=1)
print(f"patch: L={L:.2f} kpc, N=96, dx={patch.dx:.3f} kpc, lambda_db={lam:.3f} kpc")

# --- density statistics ---
rho = patch.density(0.0)
print(f"<rho>={rho.mean():.3e} (target {rho0:.3e}), "
      f"std/mean={rho.std()/rho.mean():.3f} (expect ~1 for |GRF|^2)")

# --- granule size vs lambda_db ---
gs = patch.granule_size(0.0)
print(f"granule size (rho autocorr half-width) = {gs:.3f} kpc = {gs/lam:.2f} lambda_db")

# --- Poisson residual check: laplacian(Phi) vs 4 pi G delta_rho ---
phi, F = patch.potential_force(0.0)
phi_k = np.fft.fftn(phi)
lap = np.real(np.fft.ifftn(-patch.k2 * phi_k))
rhs = 4 * np.pi * G_INTERNAL * patch.delta_rho(0.0)
resid = np.linalg.norm(lap - rhs) / np.linalg.norm(rhs)
print(f"Poisson relative residual = {resid:.2e}")
print(f"RMS |F| = {np.sqrt((F**2).sum(0)).mean():.3e} kpc/Gyr^2")

# --- coherence time via time autocorrelation of rho at fixed points ---
tau_pred = patch.coherence_time_pred()
tmax = 6 * tau_pred
ts = np.linspace(0, tmax, 60)
# sample a subset of grid points
idx = (slice(None, None, 8),) * 3
series = np.array([patch.density(t)[idx].ravel() for t in ts])  # (nt, npts)
series -= series.mean(0, keepdims=True)
ac = (series[0] * series).mean(1)
ac /= ac[0]
# 1/e crossing
below = np.argmax(ac < np.exp(-1))
tau_meas = ts[below] if below > 0 else np.nan
print(f"coherence time: predicted {tau_pred:.4f} Gyr, measured(1/e) {tau_meas:.4f} Gyr "
      f"= {tau_meas/tau_pred:.2f} x")

# --- force autocorrelation (for the surrogate decision) ---
Fseries = []
for t in ts:
    _, Ft = patch.potential_force(t)
    Fseries.append(Ft[(slice(None),) + idx].reshape(3, -1))
Fseries = np.array(Fseries)  # (nt,3,npts)
Fac = (Fseries[0] * Fseries).sum(1).mean(1)
Fac /= Fac[0]
fbelow = np.argmax(Fac < np.exp(-1))
tauF = ts[fbelow] if fbelow > 0 else np.nan
print(f"force autocorr time (1/e) = {tauF:.4f} Gyr = {tauF/tau_pred:.2f} x tau_coh")
# crude velocity-diffusion coefficient D ~ int <F.F> dtau
FdotF0 = (Fseries[0] ** 2).sum(0).mean()
D = FdotF0 * np.trapz(Fac, ts)
print(f"<|F|^2>(0) = {FdotF0:.3e} (kpc/Gyr^2)^2 ; D ~ {D:.3e} (kpc/Gyr^2)^2 Gyr")
