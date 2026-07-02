"""Physical constants and unit system for fdmforce.

Internal units match **tambora** exactly so hand-off is unit-trivial:

    length   kpc
    mass     Msun
    time     Gyr
    velocity kpc/Gyr   (user-facing velocities are km/s)
    G        4.498502151469554e-06  kpc^3 Msun^-1 Gyr^-2

The single FDM-specific quantity is ``hbar/m`` (dimensions length*velocity =
specific angular momentum), which sets the de Broglie scale.  For m = 1e-22 eV
it is ~19.17 kpc*km/s.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as _sc

# --- gravitational constant --------------------------------------------------
#: G in tambora internal units [kpc^3 Msun^-1 Gyr^-2]
G_INTERNAL = 4.498502151469554e-06
#: G in mixed units [kpc (km/s)^2 Msun^-1]
G_KPC_KMS = 4.300917270036279e-06

# --- unit conversions (match tambora) ---------------------------------------
#: 1 km/s in kpc/Gyr
KMS_TO_KPCGYR = 1.022712165045695
#: 1 kpc/Gyr in km/s
KPCGYR_TO_KMS = 1.0 / KMS_TO_KPCGYR

_KPC_IN_M = 3.0856775814913673e19  # meters
_MSUN_IN_KG = 1.988409870698051e30  # kg (not used on hot path; for reference)


def _hbar_over_m_kpc_kms(m22: float) -> float:
    """hbar / m in units of kpc*km/s, for axion mass m = m22 * 1e-22 eV.

    Derived from fundamentals so there is no transcription error:
        hbar/m = hbar / (m_ev / c^2),  m_ev in Joules.
    """
    m_ev_joule = m22 * 1e-22 * _sc.eV                 # rest energy [J]
    m_kg = m_ev_joule / _sc.c**2                       # mass [kg]
    hbar_over_m_si = _sc.hbar / m_kg                    # [m^2/s]
    kpc_kms_in_si = _KPC_IN_M * 1e3                     # 1 kpc*km/s in m^2/s
    return hbar_over_m_si / kpc_kms_in_si


#: hbar/m for m22 = 1, in kpc*km/s  (~19.17)
HBAR_OVER_M_KPC_KMS_M22_1 = _hbar_over_m_kpc_kms(1.0)


def hbar_over_m(m22: float, internal: bool = True) -> float:
    """hbar/m for axion mass m22*1e-22 eV.

    Parameters
    ----------
    m22 : float
        Axion mass in units of 1e-22 eV.
    internal : bool
        If True return kpc^2/Gyr (tambora internal, velocity=kpc/Gyr);
        else return kpc*km/s.
    """
    val = HBAR_OVER_M_KPC_KMS_M22_1 / m22
    return val * KMS_TO_KPCGYR if internal else val


def de_broglie_length(m22: float, v_kms: float) -> float:
    """Reduced de Broglie wavelength lambda_bar = (hbar/m)/v, in kpc.

    (Full wavelength is 2*pi times this.)  v in km/s.
    """
    return HBAR_OVER_M_KPC_KMS_M22_1 / m22 / v_kms


def coherence_time_gyr(m22: float, sigma_kms: float) -> float:
    """Granule coherence time ~ hbar/(m sigma^2), in Gyr.

    tau = (hbar/m) / sigma^2.  (hbar/m in kpc*km/s, sigma in km/s) -> kpc/(km/s);
    convert kpc/(km/s) to Gyr.
    """
    tau_kpc_per_kms = HBAR_OVER_M_KPC_KMS_M22_1 / m22 / sigma_kms**2  # kpc/(km/s)
    # kpc/(km/s) -> Gyr:  1 kpc / (1 km/s) = 1/KMS_TO_KPCGYR Gyr
    return tau_kpc_per_kms / KMS_TO_KPCGYR
