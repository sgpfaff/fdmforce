"""Adapters that expose an :class:`~fdmforce.halo.FDMHalo` to external codes.

Imports of galpy / tambora are deferred to the factory functions so that
``import fdmforce`` never requires either package.
"""
from .tambora import make_tambora_force
from .galpy import make_galpy_potential

__all__ = ["make_tambora_force", "make_galpy_potential"]
