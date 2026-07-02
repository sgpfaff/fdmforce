"""Optional C backend for the random-Fourier force/potential hot loop.

The shared library is compiled on first use (cached in this directory).  If no
working compiler is found, callers transparently fall back to the numpy path, so
importing fdmforce never fails for lack of a C toolchain.

Public surface:
    build(force=False) -> path            # (re)compile the shared library
    available() -> bool                   # is the C backend usable?
    has_openmp() -> bool                  # was it built with OpenMP?
    force_eval(pos, k, zr, zi) -> (N,3)   # F_d = sum_j zr_jd cos - zi_jd sin
    potential_eval(pos, k, pr, pim) -> (N,)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "force.c")
_LIB = os.path.join(_HERE, "libfdmforce_core.so")


def _flag_sets():
    """Compile flag-sets tried in order; OpenMP variants first.

    Uses the active Python environment's lib/include (e.g. a conda prefix) so
    libomp is found, and embeds an rpath so the shared object locates it at load.
    """
    libdir = os.path.join(sys.prefix, "lib")
    incdir = os.path.join(sys.prefix, "include")
    base = ["-O3", "-fPIC", "-shared", "-funroll-loops"]
    envomp = [f"-I{incdir}", f"-L{libdir}", f"-Wl,-rpath,{libdir}"]
    sets = []
    for march in (["-march=native"], []):
        # macOS clang + libomp (with env paths + rpath)
        sets.append(base + march + envomp + ["-Xpreprocessor", "-fopenmp", "-lomp"])
        # GNU gcc/clang -fopenmp (libgomp), env paths + rpath
        sets.append(base + march + [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-fopenmp"])
        # plain -fopenmp (toolchain-native)
        sets.append(base + march + ["-fopenmp"])
    for march in (["-march=native"], []):
        sets.append(base + march)  # no-OpenMP fallbacks (still -O3 C)
    return sets

_handle = None          # ctypes CDLL once loaded
_tried_build = False     # only attempt the build once per process


def _compiler():
    return os.environ.get("CC", "cc")


def build(force: bool = False) -> str:
    """Compile force.c to a shared library; return its path. Raises on failure."""
    if (os.path.exists(_LIB) and not force
            and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC)):
        return _LIB
    cc = _compiler()
    errors = []
    for flags in _flag_sets():
        cmd = [cc, *flags, _SRC, "-o", _LIB, "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return _LIB
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errors.append((" ".join(flags), getattr(e, "stderr", b"")))
            if os.path.exists(_LIB):
                os.remove(_LIB)
    raise RuntimeError("fdmforce C backend failed to build:\n" +
                       "\n".join(f"  [{f}] {err!r}" for f, err in errors[-2:]))


def _load():
    global _handle, _tried_build
    if _handle is not None:
        return _handle
    if _tried_build:
        return None
    _tried_build = True
    try:
        path = build()
        lib = ctypes.CDLL(path)
    except Exception as e:  # pragma: no cover - environment dependent
        warnings.warn(f"fdmforce C backend unavailable ({e}); using numpy fallback.")
        return None
    d = ctypes.POINTER(ctypes.c_double)
    lib.fdm_force.restype = None
    lib.fdm_force.argtypes = [ctypes.c_long, ctypes.c_long] + [d] * 15
    lib.fdm_potential.restype = None
    lib.fdm_potential.argtypes = [ctypes.c_long, ctypes.c_long] + [d] * 9
    lib.fdm_has_openmp.restype = ctypes.c_int
    _handle = lib
    return lib


def available() -> bool:
    return _load() is not None


def has_openmp() -> bool:
    lib = _load()
    return bool(lib.fdm_has_openmp()) if lib is not None else False


def _c(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def force_eval(pos, k, zr, zi):
    """F_d(x_i) = sum_j [ zr_{jd} cos(k_j.x_i) - zi_{jd} sin(k_j.x_i) ], shape (N,3)."""
    lib = _load()
    if lib is None:
        raise RuntimeError("C backend not available")
    N, M = pos.shape[0], k.shape[0]
    _px, px = _c(pos[:, 0]); _py, py = _c(pos[:, 1]); _pz, pz = _c(pos[:, 2])
    _kx, kx = _c(k[:, 0]); _ky, ky = _c(k[:, 1]); _kz, kz = _c(k[:, 2])
    _zrx, zrx = _c(zr[:, 0]); _zry, zry = _c(zr[:, 1]); _zrz, zrz = _c(zr[:, 2])
    _zix, zix = _c(zi[:, 0]); _ziy, ziy = _c(zi[:, 1]); _ziz, ziz = _c(zi[:, 2])
    Fx = np.empty(N); Fy = np.empty(N); Fz = np.empty(N)
    fx = Fx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fy = Fy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fz = Fz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.fdm_force(N, M, px, py, pz, kx, ky, kz,
                  zrx, zry, zrz, zix, ziy, ziz, fx, fy, fz)
    return np.stack([Fx, Fy, Fz], axis=1)


def potential_eval(pos, k, pr, pim):
    """Phi(x_i) = sum_j [ pr_j cos(k_j.x_i) - pim_j sin(k_j.x_i) ], shape (N,)."""
    lib = _load()
    if lib is None:
        raise RuntimeError("C backend not available")
    N, M = pos.shape[0], k.shape[0]
    _px, px = _c(pos[:, 0]); _py, py = _c(pos[:, 1]); _pz, pz = _c(pos[:, 2])
    _kx, kx = _c(k[:, 0]); _ky, ky = _c(k[:, 1]); _kz, kz = _c(k[:, 2])
    _pr, pr_ = _c(pr); _pim, pim_ = _c(pim)
    Phi = np.empty(N)
    ph = Phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.fdm_potential(N, M, px, py, pz, kx, ky, kz, pr_, pim_, ph)
    return Phi
