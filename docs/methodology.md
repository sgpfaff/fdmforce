# Methodology

## Why not simulate?

FDM dynamics obey the Schrödinger–Poisson system

$$i\hbar\,\partial_t\psi = -\frac{\hbar^2}{2m}\nabla^2\psi + m\Phi\psi,\qquad
\nabla^2\Phi = 4\pi G\, m|\psi|^2 .$$

Resolving the de Broglie scale $\lambda_{\rm dB}=\hbar/(m v)$ everywhere and stepping at the
kinetic CFL limit gives compute $\propto m^4 M_h^{8/3}$ and memory $\propto m^3 M_h^2$. A single
live halo at $m_{22}=1,\ M_h=10^{10}$ is already a multi-GPU, multi-day job; a mass scan is
impossible. So direct Schrödinger–Poisson is used here only as a one-point **ground-truth
anchor**, never as the production method.

## Layer 1 — Generation

Two ways to build a physically-correct frozen-background field:

### 3B — Local Gaussian random field

In a patch small compared to the halo, $\psi$ is a superposition of plane waves whose momenta
sample the local velocity distribution. For an isotropic Maxwellian of dispersion $\sigma$,

$$P(k)\propto \exp\!\left(-\frac{k^2}{2k_\sigma^2}\right),\qquad k_\sigma=\frac{\sigma}{\hbar/m}.$$

The density $\rho=m|\psi|^2$ is then a $|\text{GRF}|^2$ field; granules have size $\sim\lambda_{\rm dB}$
and coherence time $\sim\hbar/(m\sigma^2)$. The fluctuating potential comes from an in-patch FFT
Poisson solve. Cost is tied to the patch, not the whole halo, so it scales gently with $m$.

### 3A — Global eigenmode construction

Solve the radial Schrödinger eigenproblem in the fixed spherical background $\Phi_0(r)$,

$$-\tfrac{b^2}{2}u'' + \Big[\Phi_0(r) + \tfrac{b^2}{2}\tfrac{l(l+1)}{r^2}\Big]u = \varepsilon\,u,
\qquad b=\hbar/m,\ u=rR_{nl},$$

populate the modes from the Eddington distribution function, and reconstruct
$\psi=\sum a_{nlm}R_{nl}Y_{lm}e^{-i\varepsilon_{nl}t/b}$. Globally coherent and spherically
self-consistent, but the number of modes grows steeply with $m$, so it serves as a **reference
oracle** (it independently reproduces the granule coherence timescale).

## Layer 2 — Fast evaluation (surrogate B)

The key structural fact: a **density** Fourier mode is bilinear in $\psi$, so the fast
wavefunction carrier $(\hbar/m)k^2/2$ **cancels**. Working out
$\langle\delta\rho_k(0)\delta\rho_k^{*}(\tau)\rangle$, the dominant pairing centres the mode at
$\omega=0$ with a **Gaussian** line of width

$$\Delta\omega(k)=\frac{\hbar}{m}\,\frac{|k|\,k_\sigma}{\sqrt 2}.$$

The force mode follows from $F_k=-i\mathbf{k}\,\delta\Phi_k=i(4\pi G)\,(\mathbf{k}/k^2)\,\delta\rho_k$.
We therefore represent the force as $M$ random-Fourier modes drawn from the **force** power
spectrum $P_F(k)=P_{\delta\rho}(k)/k^2$,

$$\mathbf F(\mathbf x,t)=\mathrm{Re}\sum_{j=1}^{M} \mathbf a_j\,b_j(t)\,e^{i\mathbf k_j\cdot\mathbf x},
\qquad b_j(t)=b_j(0)\,e^{-i\omega_j t},\ \ \omega_j\sim\mathcal N\!\big(0,\Delta\omega(k_j)^2\big).$$

Advancing the field is a per-mode phase rotation ($O(M)$, particle-independent — ideal for an
integrator hook); evaluation is $O(NM)$ and mesh-free.

!!! note "Why the surrogate is deterministic"
    A first-order Ornstein–Uhlenbeck latent process gives an *exponential* (Lorentzian)
    autocorrelation, but the true FDM line shape is **Gaussian**. Pure phase-rotation with a
    spread of frequencies reproduces it via dephasing, so the "stochastic state-space" model
    collapses into a reproducible **spectral** one. A damped-OU path (`gamma > 0`) is retained
    for future genuinely non-frozen / decohering physics.

## The infrared / coherence scale

Because the force carries a $1/k^2$ weight, its temporal correlation is dominated by the largest
coherent scale and is **infrared-sensitive**: in a periodic box the force-ACF time grows with box
size (measured: 1.6 → 2.7 $\tau_{\rm coh}$ from $L=8$ to $26\,\lambda_{\rm dB}$). There is no single
"force correlation time" — it depends on a **physical coherence scale** $L_{\rm coh}$, exposed as
$k_{\min}=2\pi/L_{\rm coh}$.

`fdmforce` takes $L_{\rm coh}$ to be the local density scale height
$H=\rho/|\mathrm d\rho/\mathrm dr|$ — the region over which the homogeneous/Maxwellian
approximation holds. This also defines the surrogate's regime of validity: it requires
$L_{\rm coh}\gg 2\pi\lambda_{\rm dB}$, i.e. $L_{\rm coh}/\lambda_{\rm dB}\gtrsim 8$ (outer halo).
Inside that, the granule is comparable to the scale height and the coherent-soliton component is
required.

See [Validation](validation.md) for the numbers, or the raw [Design notes](design.md).
