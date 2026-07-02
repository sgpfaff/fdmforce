# Fast Fuzzy Dark Matter Potentials & Forces — Method Selection / Design Doc

_Draft v0.1 — 2026-07-01. Working document; iterate freely._

## 1. Goal & scope

Provide **time-dependent** FDM gravitational potential Φ(x, t) and force F(x, t) = −∇Φ that
can be evaluated cheaply inside orbit/N-body integrators, over the parameter range:

- axion mass **m ≥ 1e-22 eV** (m₂₂ ≡ m/1e-22 eV ≥ 1)
- halo mass **M_h ≥ 1e10 M_⊙**

Delivered as a Python package with a friendly API + examples; hot loops in C.
Integration targets: **galpy** (`Potential` subclass) and **tambora** (custom force via its
between-snapshot hook).

Use cases (both, treated as one superset):
- Test-particle / N-body orbits through a realistic fluctuating halo.
- Heating / velocity-diffusion statistics (a derived product of the same field).

Decided constraints:
- **Frozen background**: granules evolve in time (via mode phases), but the mean potential
  they live in (soliton core + NFW-like envelope) is fixed. No self-gravity feedback.
- Time-dependence is **required** (orbits must feel stochastic forcing over ~Gyr).

## 2. Why not direct Schrödinger–Poisson (the thing to beat)

Resolving ƛ = ℏ/(mv) at CFL Δt ≲ mΔx²/(πℏ) gives, with halo scalings
(V_vir, R_vir ∝ M_h^{1/3}, box ∝ R_vir, fixed # dynamical times):

    memory  ∝ N_cell ∝ m^3 M_h^2
    compute ∝ N_cell·N_t ∝ m^4 M_h^{8/3}

Anchored to ~1 GPU-day at (m₂₂=1, M_h=1e9): a single live halo at (m₂₂=1, M_h=1e10) is
already multi-GPU/multi-day; +460×/decade in M_h, +1e4×/decade in m. A mass **scan** is
impossible. => Use direct SP only as a **ground-truth anchor** at one point, never production.

## 3. Chosen approach: construct the field, don't evolve it

Two frozen-background construction families:

### 3A. Global eigenmode construction (Yavetz–Li–Hui 2022; Lin+ 2018)
- Fix spherical background Φ₀ = soliton core + NFW envelope.
- Solve radial Schrödinger eigenproblem in Φ₀ → eigenfns R_{nl}(r), energies E_{nl}.
- ψ(x,t) = Σ a_{nlm} R_{nl}(r) Y_{lm}(θ,φ) e^{−i E_{nl} t/ℏ}, with |a_{nlm}|² set by
  Eddington inversion of a target distribution function; phases random (Gaussian a's ⇒ GRF).
- Time-dependence free from phases; no CFL. Self-consistent spherical structure, globally
  coherent modes.
- **Problem in our regime**: # occupied modes ~ (R_vir/ƛ)³.
    M_h=1e10: R_vir≈44 kpc, ƛ≈0.6 kpc → ~3e5 modes.
    m₂₂=10 (m=1e-21):        ƛ≈0.06 kpc → ~3e8 modes.  ← memory/eval killer.

### 3B. Local Gaussian-random-field envelope, generated on demand  ← PROPOSED PRIMARY
- Represent the envelope wavefunction locally as a superposition of plane waves:
      ψ(x,t) = Σ_k c_k exp[i(k·x − ω_k t)],   ω_k = ℏ k²/(2m) (+ mean-field shift),
  with c_k complex Gaussian, variance ∝ f_local(v = ℏk/m) — the LOCAL velocity distribution
  (from the DF at that radius). Granule size ~ π/k_typ, coherence time ~ ℏ/(mσ²).
- Generate only in patches around where particles currently are (a few granules wide):
    1. draw ρ(x,t) = m|ψ|² on a small FFT patch,
    2. solve Poisson in-patch: δΦ_k = −4πG ρ̃_k / k², F_k = −i k δΦ_k,
    3. inverse-FFT / interpolate to particle positions; advance phases by ω_k Δt for time-dep.
  All O(N log N) per patch per snapshot. Cost tied to sampled volume × granule density,
  **not** whole-halo mode count. Scales gently with m.
- Trade-off: loses exact global mode coherence & spherical self-consistency; keeps correct
  LOCAL granule statistics (size, coherence time, force correlation) — which is what governs
  heating and short-range orbit perturbations.

### Total potential/force
    Φ(x,t) = Φ_soliton(r) [analytic] + Φ_NFW(r) [analytic mean] + δΦ_granule(x,t) [3A or 3B]
    F = −∇Φ (each term analytic or from in-patch Poisson).
Optional (later): soliton core breathing + random-walk with known statistics (dominant central
heating source); off by default under "frozen background".

## 4. Two-layer architecture: GENERATION vs FAST EVALUATION  (the creative speedup)

Explicitly REJECT the tambora `WidrowKaiserFDM` pattern (re-realize density → build
`MultipoleExpansionPotential` → subtract mean, every timestep). That re-solves Poisson on a
grainy field per step and pays galpy overhead per force call — the slow thing we are replacing.

Split the problem into two layers:

### Layer 1 — GENERATION (physics truth, run ONCE per halo)
3A eigenmode and/or 3B GRF (Sec 3). Purpose: obtain a physically-correct field or its exact
statistics; validate vs the SP anchor. Not on the hot path.

### Exact structural fact we exploit (frozen background)
    ψ(x,t) = Σ_j a_j φ_j(x) e^{−iE_j t/ℏ}
    δρ(x,t) = m Σ_{j≠k} a_j a_k^* φ_j φ_k^* e^{−iω_{jk} t},   ω_{jk} = (E_j−E_k)/ℏ
    ∇²δΦ = 4πG δρ  ⇒  δΦ(x,t) = Σ_{jk} a_j a_k^* Φ_{jk}(x) e^{−iω_{jk} t},
    with ∇²Φ_{jk} = 4πG m φ_j φ_k^*.
So δΦ is a sum of FIXED spatial fields modulated by pure tones. Band-limited: |ω| ≲ few·mσ²/ℏ
(= inverse granule coherence time). This structure is what makes fast evaluation possible.

### Layer 2 — FAST EVALUATION (production hot path). Two candidate surrogates:

**A. Deterministic POD / reduced-order model (reproducible, resonance-faithful).**
- Compress δΦ(x,t) ≈ Σ_{p=1}^r u_p(x) v_p(t), r ~ O(10–50) via SVD/POD of a modest space-time
  sample (or directly from the (Φ_{jk}, ω_{jk}) list).
- v_p(t) are sums of the known tones ω_{jk} ⇒ analytically continuable to ANY t (quasi-periodic):
  one compression serves arbitrarily long integrations.
- Eval: F(x,t) = −Σ_p ∇u_p(x) v_p(t). Store u_p on radial×Y_lm or grid+interp. No Poisson,
  no density, ever. Fully deterministic & seed-reproducible (good for convergence tests and
  cross-code orbit comparison). Keeps resonant phase coherence.

**B. Stochastic state-space surrogate (fastest, novel, tambora-hook-native).**
- Don't store the field. Model F(x,t) as a Gaussian field with the theoretical space–time
  correlation K(x,x',t−t') (from the DF; cf. Bar-Or–Fouvry–Pichon force correlation).
- Temporal correlation ≈ sum of a few damped/oscillatory exponentials ⇒ realize as a low-order
  linear SDE (OU / state-space) driven by white noise. Spatial correlation ⇒ low-rank kernel via
  a fixed dictionary g_p(x) (random Fourier features matched to the granule spectrum, or the u_p).
- F(x,t) = Σ_p g_p(x) s_p(t), latent s_p obey ds = A s dt + B dW (few coupled OU modes) tuned so
  stationary covariance & correlation time reproduce K.
- Per step: advance the r latent scalars (O(r), independent of particle count), then evaluate
  Σ_p g_p(x) s_p(t). m enters ONLY via correlation length/time ⇒ scales trivially across the
  whole mass range. **Maps directly onto tambora's mutating hook**: s_p is extra sim state the
  hook advances each step. Reproduces 2-pt (Gaussian) stats exactly; add explicit structured
  components (soliton wobble/random-walk mode) for non-Gaussian central heating if needed.

Generation → surrogate: 3A gives (ω_{jk}, Φ_{jk}) → literally Layer-A, and exact K for B.
3B gives realizations/statistics → POD for A, or fit K for B.

### Adapters
- galpy: A is a genuine time-dependent `Potential` (our own subclass; note it is NOT on tambora's
  whitelist, so for tambora we go native). B needs a stochastic-state wrapper.
- tambora: native `ExternalConservativeForce` subclass calling our evaluator directly (bypass
  galpy overhead & whitelist). B's latent state advanced via mutating hook (WIP feature); until
  then, advance inside `acc(pos,t)` keyed on t. C core + thin cython/ctypes for the hot path.

## 5. Validation plan

1. **Analytic checks**: force auto-correlation ∫⟨F(t)·F(t+τ)⟩dτ vs Dalal–Kravtsov /
   Bar-Or–Fouvry–Pichon diffusion coefficient; granule size & coherence time vs theory;
   density power spectrum shape.
2. **SP anchor run**: one live PyUltraLight/GAMER run at (m₂₂=1, M_h=1e10); compare density
   PS, granule stats, and test-particle heating rate against 3A and 3B.
3. **Cross-method**: 3B (local GRF) vs 3A (global eigenmode) on a small halo where 3A is affordable.

## 6. Python API sketch

    halo = FDMHalo(m_axion=1e-22*u.eV, M_halo=1e10*u.Msun,
                   profile="soliton+nfw", method="local_grf", seed=0)
    halo.potential(xyz, t)      # -> array
    halo.force(xyz, t)          # -> (...,3)
    halo.density(xyz, t)
    halo.diffusion_coefficient(r)          # analytic + measured
    halo.force_correlation(r, tau)

    # adapters
    galpy_pot = halo.as_galpy_potential()          # galpy Potential subclass
    tambora_force = halo.as_tambora_force()         # tambora ExternalConservativeForce

### tambora binding (confirmed from docs)
- Custom force = subclass of `tambora.dynamics.forces.Force`; for us the right base is
  `ExternalConservativeForce` with signature `acc(pos, t) -> ndarray (N,3)` in tambora
  INTERNAL UNITS.
- Mixin `Conservative` and implement `potential(pos, t)`; override `acc_and_potential(pos, t)`
  so we return force+potential from a single basis evaluation (our evaluator computes both).
- Compose with `+` (auto `CompositeForce`) alongside a disk/bulge etc.
- **Hook** (unpublished, between-snapshot): use it to RE-CENTER/refresh the local-GRF patch
  caches (3B) around current particle positions and advance granule phases by Δt_snap; the
  per-step time dependence itself is already carried by the `t` argument of `acc`.
- Unit adapter: map astropy units <-> tambora internal units at construction; store seed for
  reproducible realizations shared with the galpy path.

    halo.save("halo.h5"); FDMHalo.load("halo.h5")  # reproducible realizations

## 7. Package / tooling

- `pyproject.toml`, C extension (meson-python or scikit-build-core), numpy/scipy, astropy units.
- `src/fdmforce/` (python) + `src/fdmforce/_core/` (C). `examples/` notebooks. `tests/` (pytest).
- Optional galpy/tambora extras. Docs via mkdocs. CI later.

## 7b. DECISION (v0.1): prototype BOTH, benchmark, then pick

Build minimal pure-Python/numpy prototypes of 3A (global eigenmode) and 3B (local GRF) on a
SHARED spherical halo (m₂₂=1, M_h=1e10), benchmark head-to-head, then commit the winner as the
C-backed production engine; the loser (if useful) survives as a validation oracle.

Benchmark metrics (shared halo, same seed where possible):
1. Physical fidelity: radial density profile; density power spectrum P(k); granule size
   (autocorrelation length of ρ); coherence time (τ where ρ autocorr in time decays); force
   auto-correlation ∫⟨F·F⟩dτ and implied velocity-diffusion coefficient — all vs analytic theory.
2. Cost: build time; memory footprint; force evals/sec (single core); scaling probe at m₂₂=3.
3. Practicality: ease of the galpy/tambora `acc(pos,t)` binding; reproducibility from seed.
Decision rule: pick the engine that hits fidelity tolerances AND scales to m₂₂≥1 / M_h≥1e10;
prefer 3B unless it fails a fidelity check that 3A passes.

## 7c. GENERATION BENCHMARK RESULTS (v0.1 prototypes, shared halo m22=1, M_h=1e10)

Background validated: V_vir=32.6 km/s, M(<r_vir)/M_h≈1.1, lambda_db(r_s)=0.55 kpc,
tau_coh(r_s)=0.0151 Gyr — all physically sensible across (m22, M_h) in target range.

3B local GRF (engines/grf.py):
  <rho>=target exactly; density std/mean=1.014 (=1 expected for |GRF|^2 -> Gaussian confirmed);
  granule size=0.88 lambda_db; Poisson residual 1e-13; coherence time 0.0138 Gyr (0.92x pred);
  force autocorr time ~1.7 tau_coh (~26 Myr), single-timescale decay. Build+eval: sub-second.
  => clean, fast, matches theory out of the box. SCALES gently in m.

3A eigenmode (engines/eigenmode.py):
  2864 (n,l) modes / 72k m-states, built in 2.5 s. Radial eigenproblem + Eddington DF work.
  Overlap-weighted beat spectrum at r_s: median 61/Gyr -> 1/median=0.0163 Gyr ≈ tau_coh (CROSS-
  VALIDATES 3B's granule timescale independently). BUT density profile only ~2x correct
  (overshoot mid, undershoot past r_vir): needs r_max extension, proper mode-population
  normalization (completeness), and separate soliton-ground-state handling. Fiddly, as expected.

DECISION (data-driven, per the "decide after benchmark" gate):
- Production GENERATION = **3B local GRF** (clean, fast, validated, scalable).
- 3A = reference oracle; already cross-validates the granule timescale; profile-population
  refinement deferred (only needed if we want 3A as a quantitative check).
- The local beat/force spectrum is dominated by ONE characteristic timescale (tau_coh) with a
  moderate participation ratio => Layer-2 **stochastic state-space surrogate (B)** is the natural
  production evaluator (low-order OU reproduces the force correlation; hook-native). POD (A)
  remains the reproducible/resonance-faithful option, derivable from the same statistics.

## 7d. SURROGATE B RESULTS + a key physics finding (v0.1)

Surrogate B (surrogate/stochastic.py): random-Fourier modes k_j ~ force power
spectrum P_F(k)=P_drho/k^2 (uniform direction, |k| half-normal, IR cutoff k_min),
each a plane wave e^{i(k_j.x - w_j t)} with w_j = (hbar/m)k_j^2/2 + N(0,(hbar/m k k_sig)^2).
Mesh-free; advance = per-mode phase rotation (O(M), hook-native); eval O(N*M).

Validation vs 3B (m22=1, M_h=1e10, r_s):
- <|F|^2>: matched by 1-parameter amplitude calibration.
- Spatial force correlation length: 0.87 vs 0.99 kpc; ACF RMS 0.069  ✓ (physics-correct).
- Temporal line SHAPE: first-order OU gives an exponential/Lorentzian ACF, but the true
  FDM density mode decorrelates as a GAUSSIAN (sum of q.k tones). Fixed by drawing per-mode
  frequency from the Gaussian line and advancing as PURE PHASE (dephasing => Gaussian ACF).
  => the "stochastic state-space" collapses to a reproducible quasi-periodic SPECTRAL model
  (A and B merge); the damped-OU path (gamma>0) is retained for future non-frozen/decoherence.

KEY FINDING — force temporal correlation is INFRARED-sensitive:
  3B force ACF 1/e time grows with box: 1.58, 1.80, 2.65, 2.73 tau_coh for L=8,12,18,26 lambda_db
  (saturating ~20 lambda_db). The 1/k^2 force is dominated by the largest coherent scale, so the
  force decorrelation time (and hence heating/velocity-diffusion) depends on the physical
  COHERENCE SCALE L_coh, not an arbitrary box.  L_coh must be set physically: ~ local density
  scale height / orbital radius / region where the DF stays ~Maxwellian.  This is a modeling
  parameter with real consequences for heating rates and is exposed as k_min = 2 pi / L_coh.

Frequency-centering fix (found via diffusion validation): density/force modes are centred at
w=0 (fast psi carrier cancels in |psi|^2), width dw=(hbar/m)|k|k_sig/sqrt2. After this,
temporal ACF RMS vs 3B = 0.058, spatial 0.069.

DIFFUSION-COEFFICIENT VALIDATION (L_coh = local scale height H=rho/|drho/dr|):
  Validity: local-GRF/surrogate need L_coh >> 2*pi*lam_db (full wavelength) i.e. L_coh/lam >~ 8
  (k_min/k_sig << 1).  Results (m22=1, M_h=1e10):
    r=22 kpc (L/lam=8.9): D_sur/D_3B = 0.86
    r=44 kpc (L/lam=14):  D_sur/D_3B = 0.76
    r<=11 kpc (L/lam<6):  local approx breaks (granule ~ scale height) -> SOLITON/coherent regime.
  => surrogate reproduces the velocity-diffusion coefficient to ~20% in the outer halo (great for
  heating), and self-identifies the inner region where the coherent-soliton component is required.

OPEN (roadmap): (1) C/numba backend for the O(N*M) hot loop (numpy exp bottleneck, ~2e4 evals/s);
(2) soliton dynamics component for the inner/coherent regime; (3) galpy + tambora adapters;
(4) absolute (uncalibrated) amplitude from first principles so no per-halo 3B calibration needed;
(5) packaging, tests, examples.

## 8. Open questions / risks (for iteration)

- **Primary method**: DECIDED to prototype both first (see §7b).
- Poisson non-locality: in-patch periodic Poisson mishandles long-wavelength δΦ; large-scale
  fluctuations may need a coarse global solve layered under local patches (multiscale).
- DF / core–halo relation choice (Schive+14 vs alternatives); anisotropy.
- Units & reproducibility (seed → identical realization across galpy/tambora).
- tambora interface: need repo/PyPI to match custom-force + hook signatures exactly.

## 9. Performance targets (straw man)

- Build a halo: seconds–minutes, m-independent for 3B.
- Force eval: ≳1e6 evals/s single core for orbit integration; scalable with particles.
- Memory: MB-scale for 3B (patch caches), independent of full-halo mode count.
