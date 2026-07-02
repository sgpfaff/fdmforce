import numpy as np
import pytest

from fdmforce import FDMBackground, constants
from fdmforce.engines import LocalGRFPatch
from fdmforce.surrogate import StochasticForceField


def _sf(**kw):
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    r = 22.0
    base = dict(m22=1.0, rho_mean=float(bg.density(r)), sigma_kms=float(bg.sigma(r)),
                coherence_scale=float(bg.scale_height(r)), n_modes=1024, seed=0)
    base.update(kw)
    return StochasticForceField(**base)


def test_background_physical():
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    # virial velocity ~ 30 km/s for a 1e10 halo
    assert 25 < float(bg.v_circ(bg.r_vir)) < 40
    # NFW carries ~the halo mass (soliton adds a bit)
    assert 0.9 < float(bg.enclosed_mass(bg.r_vir)) / 1e10 < 1.3
    assert float(bg.scale_height(bg.r_s)) > 0


def test_constants_de_broglie():
    # hbar/m ~ 19.17 kpc km/s for m22=1
    assert abs(constants.HBAR_OVER_M_KPC_KMS_M22_1 - 19.17) < 0.05
    assert constants.de_broglie_length(1.0, 100.0) == pytest.approx(0.1917, rel=1e-2)


def test_grf_matches_theory():
    bg = FDMBackground(m22=1.0, M_halo=1e10)
    r = bg.r_s
    lam = bg.lambda_db(r)
    p = LocalGRFPatch(m22=1.0, rho_mean=float(bg.density(r)),
                      sigma_kms=float(bg.sigma(r)), L=12 * lam, N=64, seed=1)
    rho = p.density(0.0)
    assert rho.mean() == pytest.approx(float(bg.density(r)), rel=1e-6)
    assert abs(rho.std() / rho.mean() - 1.0) < 0.1   # |GRF|^2 -> exponential PDF
    assert 0.6 < p.granule_size(0.0) / lam < 1.1


def test_c_matches_numpy():
    sf_c = _sf(use_c=True)
    sf_n = _sf(use_c=False)
    sf_n.b = sf_c.b.copy(); sf_n._C = sf_c._C; sf_n.a = sf_c.a.copy()
    pos = np.random.default_rng(0).uniform(0, 10, size=(2000, 3))
    Fc, Fn = sf_c.force(pos), sf_n.force(pos)
    assert np.max(np.abs(Fc - Fn)) / np.std(Fn) < 1e-10
    Pc, Pn = sf_c.potential(pos), sf_n.potential(pos)
    assert np.max(np.abs(Pc - Pn)) / (np.std(Pn) + 1e-30) < 1e-10


def test_force_is_gradient_of_potential():
    # numeric check: F = -grad(Phi) for the surrogate
    sf = _sf(use_c=False)
    x0 = np.array([[22.0, 0.3, -0.2]])
    F = sf.force(x0)[0]
    h = 1e-4
    grad = np.zeros(3)
    for d in range(3):
        xp = x0.copy(); xm = x0.copy()
        xp[0, d] += h; xm[0, d] -= h
        grad[d] = (sf.potential(xp)[0] - sf.potential(xm)[0]) / (2 * h)
    assert np.allclose(F, -grad, rtol=1e-3, atol=1e-3 * np.linalg.norm(F))


def test_advance_is_stationary():
    sf = _sf()
    v0 = sf.force_variance(n_sample=2000)
    for _ in range(50):
        sf.advance(1e-3)
    v1 = sf.force_variance(n_sample=2000)
    assert 0.5 < v1 / v0 < 2.0   # variance roughly preserved (stationary field)
