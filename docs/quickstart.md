# Quickstart

## Install

```bash
git clone https://github.com/sgpfaff/fdmforce
cd fdmforce
pip install -e .            # numpy + scipy only
pip install -e '.[dev]'    # + pytest, matplotlib, astropy
```

The optional C/OpenMP backend is compiled automatically on first use; if no working compiler is
found, `fdmforce` transparently falls back to numpy (nothing breaks).

## Build a background halo

```python
from fdmforce import FDMBackground

bg = FDMBackground(m22=1.0, M_halo=1e10)   # axion mass 1e-22 eV, halo 1e10 Msun
print(bg.summary())
# -> soliton core (Schive relation), NFW envelope, Jeans sigma(r), scale height
```

All lengths are kpc, masses $M_\odot$, times Gyr, potentials/dispersions km/s — matching tambora.

## Evaluate fluctuating granular forces

```python
import numpy as np
from fdmforce.surrogate import StochasticForceField

r = 22.0                                    # kpc, in the outer halo (local approx valid)
sf = StochasticForceField(
    m22=1.0,
    rho_mean=float(bg.density(r)),
    sigma_kms=float(bg.sigma(r)),
    coherence_scale=float(bg.scale_height(r)),   # physical IR cutoff L_coh
    n_modes=2048,
    seed=0,
)

pos = np.random.uniform(-1, 1, size=(1000, 3)) + [r, 0, 0]
F = sf.force(pos)          # (1000, 3), kpc/Gyr^2 — the fluctuating granular force
Phi = sf.potential(pos)    # (1000,), (kpc/Gyr)^2

sf.advance(1e-3)           # step the field forward by 1 Myr (hook-friendly)
F_next = sf.force(pos)
```

`advance(dt)` rotates each mode's phase — $O(M)$, independent of the number of particles. This is
the object you hand to an integrator: advance the field once per step, then evaluate the force at
your particles.

## Calibrate the amplitude to a ground-truth patch

The surrogate's overall amplitude is fixed by matching the force variance of a cheap Layer-1
Gaussian-random-field patch at the same $(\rho,\sigma)$:

```python
from fdmforce.engines import LocalGRFPatch

lam = bg.lambda_db(r)
patch = LocalGRFPatch(m22=1.0, rho_mean=float(bg.density(r)),
                      sigma_kms=float(bg.sigma(r)), L=float(bg.scale_height(r)),
                      N=96, seed=1)
_, Fgrid = patch.potential_force(0.0)
target_var = float(np.mean((Fgrid**2).sum(0)))

sf.calibrate_amplitude(target_var)
```

## Notes on validity

- Use the surrogate where $L_{\rm coh}\gtrsim 8\,\lambda_{\rm dB}$ (outer halo). Closer in, the
  granule becomes comparable to the scale height and the coherent-soliton regime takes over.
- The realization is reproducible from `seed`; share the seed to get identical forces across
  galpy and tambora runs.
