# API reference

## Background

::: fdmforce.backgrounds.halo.FDMBackground

## Generation engines

### 3B — Local Gaussian random field

::: fdmforce.engines.grf.LocalGRFPatch

### 3A — Global eigenmode construction

::: fdmforce.engines.eigenmode.EigenmodeHalo

## Surrogate (fast evaluation)

::: fdmforce.surrogate.stochastic.StochasticForceField

## Constants and units

::: fdmforce.constants
    options:
      members:
        - hbar_over_m
        - de_broglie_length
        - coherence_time_gyr

## C backend

::: fdmforce._core
    options:
      members:
        - available
        - has_openmp
        - build
        - force_eval
        - potential_eval
