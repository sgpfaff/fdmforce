# fdmforce

**Fast Fuzzy Dark Matter potentials and forces for orbit integration** — built to plug into
[galpy](https://docs.galpy.org) and [tambora](https://tambora.readthedocs.io).

Fuzzy Dark Matter (FDM) halos have a solitonic core plus a granular, time-fluctuating envelope
whose interference produces stochastic gravitational forces (the source of dynamical heating of
streams, discs, and dwarfs). Simulating this directly with a Schrödinger–Poisson solver costs
roughly **∝ m⁴ M_h^{8/3}** — prohibitive for axion masses m ≳ 10⁻²² eV and halos ≳ 10¹⁰ M_⊙.

`fdmforce` instead **constructs the field's statistics once and evaluates a compact surrogate on
the hot path**, so you get physically-correct, time-dependent FDM forces cheaply enough for Gyr
orbit integration.

## Approach (two layers)

1. **Generation** (offline, once per halo) — build the granule field or its statistics:
   - `engines.LocalGRFPatch` (**3B**): local Gaussian-random-field patch (fast, on-demand).
   - `engines.eigenmode.EigenmodeHalo` (**3A**): global eigenmode construction (reference oracle).
2. **Fast evaluation** (hot path) — `surrogate.StochasticForceField` (**B**): a mesh-free
   random-Fourier spectral model. Each mode is a plane wave advanced by pure phase rotation
   (O(M) per step, streams naturally into an integrator hook), reproducing the space- and
   time-correlation of the granular force.

The background (`backgrounds.FDMBackground`) is a Schive soliton core + NFW envelope with an
isotropic-Jeans velocity dispersion, in tambora-native units (kpc, M⊙, Gyr).

## Status

Validated against the local-GRF ground truth (m22=1, M_h=10¹⁰):

| Quantity | Agreement |
|---|---|
| Granule size / coherence time (3B vs theory) | ~10% |
| Surrogate force variance / spatial ACF / temporal ACF | matched / RMS 0.069 / RMS 0.058 |
| Surrogate velocity-diffusion coefficient (outer halo) | ~20% |

The surrogate is valid where the local approximation holds, `L_coh ≳ 8 λ_db` (outer halo); the
inner/coherent (soliton) region is handled by a dedicated component (in progress).

See [`docs/`](docs/) for the methodology and design rationale.

## Install (development)

```bash
pip install -e .            # numpy + scipy
pip install -e '.[dev]'     # + pytest, matplotlib, astropy
```

## Quickstart

```python
import numpy as np
from fdmforce import FDMBackground
from fdmforce.surrogate import StochasticForceField

bg = FDMBackground(m22=1.0, M_halo=1e10)          # soliton + NFW + sigma(r)
r = 22.0                                           # kpc (outer halo)
sf = StochasticForceField(
    m22=1.0, rho_mean=float(bg.density(r)), sigma_kms=float(bg.sigma(r)),
    coherence_scale=float(bg.scale_height(r)), n_modes=2048, seed=0,
)

pos = np.random.uniform(-1, 1, size=(1000, 3)) + [r, 0, 0]
F = sf.force(pos)          # (1000, 3) kpc/Gyr^2  — fluctuating granular force
sf.advance(1e-3)           # step the field by 1 Myr (hook-friendly)
```

## License

MIT
