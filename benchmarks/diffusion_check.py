"""Validate surrogate B vs 3B on the physical observable: velocity-diffusion
coefficient D = <|F|^2> * int_0^inf C_F(tau) dtau, at L_coh = local scale height,
across halo radii."""
import numpy as np
from scipy.integrate import trapezoid

from fdmforce import FDMBackground
from fdmforce.engines import LocalGRFPatch
from fdmforce.surrogate import StochasticForceField

bg = FDMBackground(m22=1.0, M_halo=1e10)

def measure_3B(rho, sigma, lam, L_coh, seed=3):
    # box IR cutoff = L_coh; resolve granules with dx < lam/4
    N = int(np.clip(2 * np.ceil(L_coh / (lam / 4) / 2), 48, 160))
    patch = LocalGRFPatch(m22=1.0, rho_mean=rho, sigma_kms=sigma, L=L_coh, N=N, seed=seed)
    _, F0 = patch.potential_force(0.0)
    var = float(np.mean(np.sum(F0**2, axis=0)))
    tc = patch.coherence_time_pred()
    ts = np.linspace(0, 10 * tc, 100)
    idx = (slice(None, None, max(1, N // 12)),) * 3
    F = []
    for t in ts:
        _, Ft = patch.potential_force(t)
        F.append(Ft[(slice(None),) + idx].reshape(3, -1))
    F = np.array(F)
    acf = (F[0] * F).sum(1).mean(1); acf /= acf[0]
    D = var * trapezoid(acf, ts)
    return var, D, trapezoid(acf, ts), N

def measure_surrogate(rho, sigma, L_coh, var_target, seed=7):
    sf = StochasticForceField(m22=1.0, rho_mean=rho, sigma_kms=sigma,
                              n_modes=4096, coherence_scale=L_coh, seed=seed)
    sf.calibrate_amplitude(var_target)
    amp2 = np.sum(np.abs(sf.a) ** 2, axis=1)
    tc = 1.0 / (sf.k_sig**2 * sf.hbar_m)  # tau_coh
    ts = np.linspace(0, 12 * tc, 200)
    acf = np.array([np.sum(amp2 * np.cos(sf.omega * t)) for t in ts]); acf /= acf[0]
    return var_target * trapezoid(acf, ts), trapezoid(acf, ts)

print("Validity of local-GRF/surrogate: need L_coh >> 2*pi*lam_db (full wavelength),")
print("i.e. L_coh/lam_db >~ 8  <=>  k_min/k_sig << 1.\n")
print(f"{'r/kpc':>8} {'sigma':>7} {'lam_db':>7} {'L_coh':>7} {'L/lam':>6} {'kmin/ksig':>9} "
      f"{'D_3B':>10} {'D_sur':>10} {'ratio':>6}")
for rr in [bg.r_s, 2 * bg.r_s, 0.25 * bg.r_vir, 0.5 * bg.r_vir, bg.r_vir]:
    sigma = float(bg.sigma(rr)); rho = float(bg.density(rr))
    lam = float(bg.lambda_db(rr)); L_coh = float(bg.scale_height(rr))
    k_sig = (sigma * 1.022712165045695) / (19.1715 * 1.022712165045695)  # sigma_int/hbar_m
    kmin_ksig = (2 * np.pi / L_coh) / k_sig
    if L_coh / lam < 8:
        print(f"{rr:8.2f} {sigma:7.1f} {lam:7.3f} {L_coh:7.2f} {L_coh/lam:6.1f} {kmin_ksig:9.2f} "
              f"  local approx breaks down (soliton/coherent regime) -> skip")
        continue
    var, D3, I3, N = measure_3B(rho, sigma, lam, L_coh)
    Ds, Is = measure_surrogate(rho, sigma, L_coh, var)
    print(f"{rr:8.2f} {sigma:7.1f} {lam:7.3f} {L_coh:7.2f} {L_coh/lam:6.1f} {kmin_ksig:9.2f} "
          f"{D3:10.3e} {Ds:10.3e} {Ds/D3:6.2f}")
