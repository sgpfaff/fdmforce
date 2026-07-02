"""Generate figures and animations for the docs (into docs/assets/).

Run:  PYTHONPATH=../src python make_figures.py   (from docs/)
   or: PYTHONPATH=src python docs/make_figures.py

Style: clean white background, Okabe-Ito colorblind-safe palette, thin marks,
direct labels. Mix of informative (profiles, ACFs) and intuitive (granule field
snapshot + animation) visuals.
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from fdmforce import FDMBackground
from fdmforce.engines import LocalGRFPatch
from fdmforce.surrogate import StochasticForceField

# --- Okabe-Ito palette (colorblind-safe by construction) --------------------
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442")
INK, MUTED = "#1a1a1a", "#8a8a8a"

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)


def fig_halo_structure():
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    r = np.logspace(-1.5, np.log10(3 * bg.r_vir), 500)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.2))

    a0.loglog(r, bg._rho_soliton(r), "--", color=ORANGE, lw=2, label="soliton core")
    a0.loglog(r, bg._rho_nfw(r), "--", color=SKY, lw=2, label="NFW envelope")
    a0.loglog(r, bg.density(r), "-", color=BLUE, lw=2.4, label="total")
    for x, lab, c in [(bg.r_c, "$r_c$", ORANGE), (bg.r_t, "$r_t$", GREEN),
                      (bg.r_s, "$r_s$", MUTED), (bg.r_vir, "$r_{vir}$", MUTED)]:
        a0.axvline(x, color=c, lw=0.8, ls=":", alpha=0.7)
        a0.text(x, a0.get_ylim()[1], lab, color=c, fontsize=9, ha="center", va="bottom")
    a0.set_xlabel("radius [kpc]"); a0.set_ylabel(r"$\rho\ [M_\odot\,\mathrm{kpc}^{-3}]$")
    a0.set_title("Soliton core + NFW envelope", color=INK, loc="left", fontweight="bold")
    a0.legend(frameon=False, fontsize=9)

    a1.semilogx(r, bg.sigma(r), "-", color=VERM, lw=2.4, label=r"$\sigma(r)$ [km/s]")
    a1b = a1.twinx() if False else a1  # single axis rule: put lambda_db on its own scaling
    lam = np.array([bg.lambda_db(ri) for ri in r])
    a1.semilogx(r, lam * 20, "-", color=PURPLE, lw=2,
                label=r"$20\times\lambda_{\rm dB}(r)$ [kpc]")
    a1.set_xlabel("radius [kpc]"); a1.set_ylabel("km/s   /   scaled kpc")
    a1.set_title("Velocity dispersion & granule scale", color=INK, loc="left",
                 fontweight="bold")
    a1.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "halo_structure.png"),
                                    bbox_inches="tight"); plt.close(fig)
    print("wrote halo_structure.png")


def _patch(N=140):
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    r0 = bg.r_s
    lam = bg.lambda_db(r0)
    L = 10 * lam
    p = LocalGRFPatch(m22=1.0, rho_mean=float(bg.density(r0)),
                      sigma_kms=float(bg.sigma(r0)), L=L, N=N, seed=4)
    return bg, p, lam, L


def fig_granule_snapshot():
    bg, p, lam, L = _patch()
    rho = p.density(0.0)[:, :, p.N // 2]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(rho.T / rho.mean(), origin="lower", extent=[0, L, 0, L],
                   cmap="magma", vmin=0, vmax=4)
    # de Broglie scale bar
    ax.plot([0.6, 0.6 + 2 * np.pi * lam], [0.6, 0.6], color="white", lw=3)
    ax.text(0.6, 0.6 + 0.25, r"$2\pi\lambda_{\rm dB}$", color="white", fontsize=10)
    ax.set_xlabel("kpc"); ax.set_ylabel("kpc")
    ax.set_title(r"FDM granule field  $\rho/\langle\rho\rangle$  ($m_{22}=1$, $r=r_s$)",
                 color=INK, loc="left", fontweight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\rho/\langle\rho\rangle$")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "granule_snapshot.png"),
                                    bbox_inches="tight"); plt.close(fig)
    print("wrote granule_snapshot.png")


def anim_granules():
    bg, p, lam, L = _patch(N=120)
    tau = p.coherence_time_pred()
    times = np.linspace(0, 6 * tau, 48)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    rho0 = p.density(0.0)[:, :, p.N // 2]
    im = ax.imshow(rho0.T / rho0.mean(), origin="lower", extent=[0, L, 0, L],
                   cmap="magma", vmin=0, vmax=4)
    ax.plot([0.5, 0.5 + 2 * np.pi * lam], [0.5, 0.5], color="white", lw=3)
    ax.text(0.5, 0.5 + 0.25, r"$2\pi\lambda_{\rm dB}$", color="white", fontsize=9)
    ax.set_xlabel("kpc"); ax.set_ylabel("kpc")
    ttl = ax.set_title("", color=INK, loc="left", fontweight="bold", fontsize=11)

    def update(i):
        rho = p.density(times[i])[:, :, p.N // 2]
        im.set_data(rho.T / rho.mean())
        ttl.set_text(f"Granules flickering & drifting   t = {times[i]*1e3:4.1f} Myr "
                     f"($\\tau_{{coh}}$={tau*1e3:.1f} Myr)")
        return im, ttl

    an = FuncAnimation(fig, update, frames=len(times), blit=False)
    out = os.path.join(ASSETS, "granules.gif")
    an.save(out, writer=PillowWriter(fps=10)); plt.close(fig)
    print("wrote granules.gif")


def fig_surrogate_validation():
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    r0 = bg.r_s; sigma = float(bg.sigma(r0)); rho = float(bg.density(r0))
    lam = bg.lambda_db(r0); L = 12 * lam; N = 96
    patch = LocalGRFPatch(m22=1.0, rho_mean=rho, sigma_kms=sigma, L=L, N=N, seed=3)
    _, F0 = patch.potential_force(0.0)
    var = float(np.mean(np.sum(F0**2, axis=0)))
    tc = patch.coherence_time_pred(); ts = np.linspace(0, 6 * tc, 70)
    idx = (slice(None, None, 8),) * 3
    Fser = np.array([patch.potential_force(t)[1][(slice(None),) + idx].reshape(3, -1)
                     for t in ts])
    acf_gt = (Fser[0] * Fser).sum(1).mean(1); acf_gt /= acf_gt[0]
    # spatial acf along x
    def sacf(F):
        ac = np.zeros(F.shape[1])
        for d in range(3):
            fk = np.fft.fftn(F[d]); ac += np.real(np.fft.ifftn(np.abs(fk)**2))[:, 0, 0]
        return ac / ac[0]
    sac_gt = sacf(F0); lags = np.arange(N) * patch.dx

    sf = StochasticForceField(m22=1.0, rho_mean=rho, sigma_kms=sigma,
                              n_modes=4096, coherence_scale=L, seed=7)
    sf.calibrate_amplitude(var)
    amp2 = np.sum(np.abs(sf.a)**2, axis=1)
    acf_s = np.array([np.sum(amp2 * np.cos(sf.omega * t)) for t in ts]); acf_s /= acf_s[0]
    sac_s = np.array([np.sum(amp2 * np.cos(sf.k[:, 0] * r)) for r in lags]); sac_s /= sac_s[0]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.2))
    a0.plot(ts * 1e3, acf_gt, "-", color=BLUE, lw=2.6, label="3B ground truth")
    a0.plot(ts * 1e3, acf_s, "--", color=ORANGE, lw=2.2, label="surrogate B")
    a0.axhline(0, color=MUTED, lw=0.6)
    a0.set_xlabel("lag [Myr]"); a0.set_ylabel(r"force ACF $C_F(\tau)$")
    a0.set_title("Temporal force correlation", color=INK, loc="left", fontweight="bold")
    a0.legend(frameon=False, fontsize=9)
    a1.plot(lags, sac_gt, "-", color=BLUE, lw=2.6, label="3B ground truth")
    a1.plot(lags, sac_s, "--", color=ORANGE, lw=2.2, label="surrogate B")
    a1.axvline(lam, color=GREEN, lw=0.8, ls=":"); a1.text(lam, 0.8, r"$\lambda_{\rm dB}$",
                                                          color=GREEN, fontsize=9)
    a1.axhline(0, color=MUTED, lw=0.6); a1.set_xlim(0, 4 * lam)
    a1.set_xlabel("separation [kpc]"); a1.set_ylabel(r"force ACF $C_F(\Delta x)$")
    a1.set_title("Spatial force correlation", color=INK, loc="left", fontweight="bold")
    a1.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "surrogate_validation.png"),
                                    bbox_inches="tight"); plt.close(fig)
    print("wrote surrogate_validation.png")


if __name__ == "__main__":
    fig_halo_structure()
    fig_granule_snapshot()
    fig_surrogate_validation()
    anim_granules()
    print("done ->", ASSETS)
