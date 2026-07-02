import numpy as np
import pytest
from fdmforce import FDMHalo

halo = FDMHalo(m22=1.0, M_halo=1e10, r_fluct=22.0, n_modes=512, seed=0)


def test_halo_force_is_mean_plus_fluct():
    pos = np.array([[20.0, 4.0, 3.0]])
    F = halo.force(pos, 0.0)
    Fm = halo.mean_force(pos)
    assert F.shape == (1, 3)
    assert not np.allclose(F, Fm)          # granular adds structure
    assert np.allclose(halo.force(pos, 0.0, granular=False), Fm)


def test_galpy_adapter_matches_halo():
    galpy = pytest.importorskip("galpy")
    from galpy.potential import evaluateRforces
    from fdmforce.constants import KMS_TO_KPCGYR
    ro, vo = 8.0, 220.0
    D = (vo * KMS_TO_KPCGYR) ** 2
    pot = halo.as_galpy_potential(ro=ro, vo=vo)
    xyz = np.array([[20.0, 4.0, 3.0]])
    R = np.hypot(xyz[0, 0], xyz[0, 1]); phi = np.arctan2(xyz[0, 1], xyz[0, 0]); z = xyz[0, 2]
    Rf = evaluateRforces(pot, R / ro, z / ro, phi=phi, t=0.0, use_physical=False)
    F = halo.force(xyz, 0.0)[0]
    FR = F[0] * np.cos(phi) + F[1] * np.sin(phi)
    assert Rf * D / ro == pytest.approx(FR, rel=1e-6)


def test_tambora_adapter_acc():
    pytest.importorskip("tambora")
    force = halo.as_tambora_force()
    pos = np.random.default_rng(0).uniform(-1, 1, size=(50, 3)) + np.array([22., 0, 0])
    a = force.acc(pos, 0.1)
    assert a.shape == (50, 3)
    assert np.all(np.isfinite(a))
