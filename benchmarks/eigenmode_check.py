"""Validate 3A eigenmode: does DF-populated mode sum reproduce rho(r)? + spectrum."""
import time
import numpy as np

from fdmforce import FDMBackground
from fdmforce.engines.eigenmode import EigenmodeHalo

bg = FDMBackground(m22=1.0, M_halo=1e10)
t0 = time.time()
em = EigenmodeHalo(bg, l_max=25, n_r=3000, seed=1)
build = time.time() - t0
nnl, nm = em.n_modes()
print(f"built in {build:.2f}s | (n,l) modes={nnl}, total m-states={nm}")
print(f"eps range: [{em.eps_arr.min():.3e}, {em.eps_arr.max():.3e}] (kpc/Gyr)^2, "
      f"all bound={np.all(em.eps_arr<0)}")

# density profile reconstruction vs target
r, rho_rec, rho_tgt = em.density_profile_recon()
for rr in [bg.r_c, bg.r_t, bg.r_s, 2*bg.r_s, bg.r_vir, 2*bg.r_vir]:
    i = np.argmin(np.abs(r - rr))
    print(f"  r={rr:8.3f} kpc  rho_recon/rho_target = {rho_rec[i]/rho_tgt[i]:.3f}")

# beat-frequency spectrum at r_s (informs surrogate choice)
dom, wt = em.beat_frequencies(r0=bg.r_s)
wt = wt / wt.sum()
# occupation-weighted percentiles of beat frequency
osel = np.argsort(dom)
cdf = np.cumsum(wt[osel])
def pct(p): return dom[osel][np.argmax(cdf >= p)]
print(f"beat-freq |eps_i-eps_j|/b [1/Gyr]: median={pct(0.5):.2f}, "
      f"90th={pct(0.9):.2f}, max={dom.max():.1f}")
# effective number of dominant frequencies (participation ratio)
pr = 1.0 / np.sum(wt**2)
print(f"participation ratio of beat spectrum = {pr:.1f} "
      f"(low => few tones => POD/stochastic both compact)")
tau_coh = bg.coherence_time(bg.r_s)
print(f"1/median-beat = {1/pct(0.5):.4f} Gyr vs granule tau_coh(r_s)={tau_coh:.4f} Gyr")
