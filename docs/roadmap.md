# Roadmap

Current state: a validated end-to-end method — background → generation → fast surrogate, with a
C/OpenMP hot loop. What's next, roughly in priority order.

## Soliton / coherent-core component
The surrogate is valid in the outer halo ($L_{\rm coh}\gtrsim 8\,\lambda_{\rm dB}$). The inner
region — where the granule is comparable to the scale height — is dominated by the coherent
soliton, whose **breathing** and **random walk** are a leading source of central heating. Add
this as a separate structured component (analytic soliton profile + its known oscillation/wander
statistics) layered under the granule field.

## galpy and tambora adapters
- **tambora**: a native `ExternalConservativeForce` subclass exposing `acc(pos, t)` and
  `potential(pos, t)` in internal units, backed directly by the surrogate (bypassing galpy
  overhead and the potential whitelist). The realization advances with `t`; once tambora's
  mutating hooks land, the field's phase state can be advanced once per step by the hook.
  *Deliberately not modelled on the existing `WidrowKaiserFDM` stub, which re-solves Poisson
  every step.*
- **galpy**: a `Potential` subclass over the same core, sharing the seed so realizations are
  identical across the two codes.

## First-principles amplitude
Currently the surrogate's overall amplitude is calibrated to a cheap 3B patch. Derive the
absolute normalization from the density fluctuation amplitude ($\langle\delta\rho^2\rangle=\rho^2$
for $\lvert\text{GRF}\rvert^2$) and the force power spectrum, so no per-halo calibration is
needed.

## Position-dependent field for whole-halo orbits
Stitch the locally-homogeneous surrogate across radii (position-dependent $\sigma(r)$, $\rho(r)$,
$L_{\rm coh}(r)$) so a single orbit can traverse the halo and feel the correct local statistics
everywhere, with the soliton component near the centre.

## Quantitative 3A oracle
Finish the eigenmode density-profile population (mode completeness, larger $r_{\max}$, separate
soliton ground-state handling) so 3A becomes a fully quantitative cross-check, not just a
timescale oracle.

## Packaging & polish
Wheels with the compiled backend, example notebooks (stream heating, disc heating), a proper
Schrödinger–Poisson anchor comparison, and CI running the benchmark suite.
