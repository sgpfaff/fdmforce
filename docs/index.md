# fdmforce

**Fast Fuzzy Dark Matter potentials and forces for orbit integration** — built to plug into
[galpy](https://docs.galpy.org) and [tambora](https://tambora.readthedocs.io).

---

## The problem

Fuzzy Dark Matter (FDM) — an ultralight boson with mass $m \sim 10^{-22}$ eV and a kpc-scale de
Broglie wavelength — forms halos with a **solitonic core** and a **granular, time-fluctuating
envelope**. The granule interference produces stochastic gravitational forces that dynamically
heat streams, discs, and dwarf galaxies. Capturing this by directly evolving the
Schrödinger–Poisson system costs roughly

$$\text{compute} \propto m^4\,M_h^{8/3},\qquad \text{memory}\propto m^3 M_h^2,$$

which is prohibitive for the interesting regime $m \gtrsim 10^{-22}$ eV and $M_h \gtrsim 10^{10}\,M_\odot$,
and completely out of reach for a *scan* over masses.

## The idea

Don't evolve the field — **construct its statistics once, then evaluate a compact surrogate on
the hot path**. In a frozen background the fluctuating potential has exact hidden structure,

$$\delta\Phi(\mathbf{x},t)=\sum_{jk} a_j a_k^{*}\,\Phi_{jk}(\mathbf{x})\,e^{-i\omega_{jk}t},$$

a sum of fixed spatial fields modulated by pure tones. That lets us solve Poisson **offline** and
reduce every force evaluation to a cheap sum.

## Two layers

<div class="grid cards" markdown>

- **Layer 1 — Generation** (offline, once per halo)

    Build the granule field or its statistics.

    - [`LocalGRFPatch`][fdmforce.engines.grf.LocalGRFPatch] (**3B**) — local Gaussian-random-field patch; fast, on-demand.
    - [`EigenmodeHalo`][fdmforce.engines.eigenmode.EigenmodeHalo] (**3A**) — global eigenmode construction; reference oracle.

- **Layer 2 — Fast evaluation** (hot path)

    - [`StochasticForceField`][fdmforce.surrogate.stochastic.StochasticForceField] (**B**) — mesh-free random-Fourier spectral force field. Each mode is a plane wave advanced by pure phase rotation: $O(M)$ per step, streams into an integrator hook, $O(N M)$ to evaluate at $N$ particles. Optional C/OpenMP backend.

</div>

The background — [`FDMBackground`][fdmforce.backgrounds.halo.FDMBackground] — is a Schive soliton
core + NFW envelope with an isotropic-Jeans velocity dispersion, in tambora-native units
(kpc, $M_\odot$, Gyr).

## Status

Validated against the local-GRF ground truth ($m_{22}=1$, $M_h=10^{10}\,M_\odot$):

| Quantity | Result |
|---|---|
| Granule size / coherence time (3B vs theory) | 0.88 $\lambda_{\rm dB}$ / 0.92× |
| Surrogate force variance / spatial ACF / temporal ACF | matched / RMS 0.069 / RMS 0.058 |
| Surrogate velocity-diffusion coefficient (outer halo) | ~20% |
| C/OpenMP speedup (15 cores) | 15–18× vs numpy |

The surrogate is valid where the local approximation holds, $L_{\rm coh}\gtrsim 8\,\lambda_{\rm dB}$
(outer halo); the inner/coherent (soliton) region needs a dedicated component (in progress —
see the [roadmap](roadmap.md)).

Start with the [Quickstart](quickstart.md), or read the [Methodology](methodology.md).
