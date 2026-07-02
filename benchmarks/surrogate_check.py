"""Validate surrogate B against the 3B ground-truth patch: force ACF shapes + speed."""
import time
import numpy as np

from fdmforce import FDMBackground
from fdmforce.engines import LocalGRFPatch
from fdmforce.surrogate import StochasticForceField

bg = FDMBackground(m22=1.0, M_halo=1e10)
r0 = bg.r_s
sigma0 = float(bg.sigma(r0)); rho0 = float(bg.density(r0))
lam = bg.lambda_db(r0)
L = 12.0 * lam
N = 96

# --- 3B ground truth --------------------------------------------------------
patch = LocalGRFPatch(m22=1.0, rho_mean=rho0, sigma_kms=sigma0, L=L, N=N, seed=3)

# force variance from a snapshot
_, F0 = patch.potential_force(0.0)
var_3B = float(np.mean(np.sum(F0**2, axis=0)))

# temporal force ACF (subset of points over time)
tau_coh = patch.coherence_time_pred()
ts = np.linspace(0, 8 * tau_coh, 80)
idx = (slice(None, None, 8),) * 3
Fser = []
for t in ts:
    _, Ft = patch.potential_force(t)
    Fser.append(Ft[(slice(None),) + idx].reshape(3, -1))
Fser = np.array(Fser)                       # (nt,3,npts)
acf_3B = (Fser[0] * Fser).sum(1).mean(1); acf_3B /= acf_3B[0]

# spatial force ACF along x (FFT autocorrelation, summed over components)
def spatial_acf(F):
    ac = np.zeros(F.shape[1])
    for d in range(3):
        fk = np.fft.fftn(F[d])
        ac += np.real(np.fft.ifftn(np.abs(fk) ** 2))[:, 0, 0]
    return ac / ac[0]
sac_3B = spatial_acf(F0)
lags = np.arange(N) * patch.dx

# --- surrogate B ------------------------------------------------------------
sf = StochasticForceField(m22=1.0, rho_mean=rho0, sigma_kms=sigma0,
                          n_modes=2048, coherence_scale=L, seed=7)
sf.calibrate_amplitude(var_3B)
print(f"halo r_s: sigma={sigma0:.1f} km/s, lambda_db={lam:.3f} kpc, patch L={L:.2f} kpc")
print(f"<|F|^2>: 3B={var_3B:.3e}, surrogate(after calib)={sf.force_variance():.3e}")

# analytic surrogate temporal ACF (pure tones at the spread frequencies)
amp2 = np.sum(np.abs(sf.a) ** 2, axis=1)   # (M,) vector power per mode
acf_sur = np.array([np.sum(amp2 * np.cos(sf.omega * t)) for t in ts])
acf_sur /= acf_sur[0]

# analytic surrogate spatial ACF along x
sac_sur = np.array([np.sum(amp2 * np.cos(sf.k[:, 0] * r)) for r in lags])
sac_sur /= sac_sur[0]

# correlation time / length (1/e) helper
def cross(x, y, thr=np.exp(-1)):
    i = np.argmax(y < thr)
    if i == 0:
        return np.nan
    return x[i - 1] + (thr - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])

print(f"force ACF time (1/e):  3B={cross(ts,acf_3B):.4f} Gyr, "
      f"surrogate={cross(ts,acf_sur):.4f} Gyr")
print(f"force corr length (1/e): 3B={cross(lags,sac_3B):.3f} kpc, "
      f"surrogate={cross(lags,sac_sur):.3f} kpc  (lambda_db={lam:.3f})")
# shape agreement metric
m = ts <= 4 * tau_coh
print(f"temporal ACF RMS diff (t<4 tau) = {np.sqrt(np.mean((acf_3B[m]-acf_sur[m])**2)):.3f}")
ms = lags <= 3 * lam
print(f"spatial  ACF RMS diff (r<3 lam) = {np.sqrt(np.mean((sac_3B[ms]-sac_sur[ms])**2)):.3f}")

# --- speed: force eval for N particles + state advance ----------------------
Npart = 20000
pos = np.random.default_rng(0).uniform(0, L, size=(Npart, 3))
t0 = time.time()
for _ in range(50):
    sf.advance(1e-3)
    _ = sf.force(pos)
dt = (time.time() - t0) / 50
print(f"surrogate: {Npart} particles, advance+eval = {dt*1e3:.2f} ms/step "
      f"({Npart/dt:.2e} force-evals/s)")
