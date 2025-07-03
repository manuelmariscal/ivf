"""
2_analytic_functions.py
--------------------------------------------
Implementa las curvas hormonales como
clases polimórficas.

* HormoneFunction (ABC)
* GaussianSumHormone -> suma de una o más gauss
  - Subclases FSHCurve, LHCurve, E2Curve, PGCurve

--------------------------------------------
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class HormoneFunction(ABC):
    """Interfaz: todas las curvas devuelven concentración dado un día t."""
    @abstractmethod
    def __call__(self, t: float | np.ndarray) -> np.ndarray: ...

# ---------- implementaciones -------------

class GaussianSumHormone(HormoneFunction):
    """
    Representa Σ A * exp(-0.5 * ((t-μ)/σ)**2) + baseline
    """
    def __init__(self,
                 mus: list[float],
                 sigmas: list[float],
                 amps: list[float],
                 baseline: float = 0.0) -> None:
        assert len(mus) == len(sigmas) == len(amps)
        self.mus = mus
        self.sigmas = sigmas
        self.amps = amps
        self.baseline = baseline

    def __call__(self, t):
        t = np.asarray(t)
        total = np.zeros_like(t, dtype=float)
        for μ, σ, A in zip(self.mus, self.sigmas, self.amps):
            total += A * np.exp(-0.5 * ((t-μ)/σ)**2)
        return total + self.baseline

# Curvas concretas (polimorfismo)
class FSHCurve(GaussianSumHormone):
    def __init__(self) -> None:
        super().__init__(mus=[4, 13],
                         sigmas=[1.3, 1.0],
                         amps=[7, 6],
                         baseline=2)

class LHCurve(GaussianSumHormone):
    def __init__(self) -> None:
        super().__init__(mus=[14],
                         sigmas=[0.6],
                         amps=[50],
                         baseline=0.5)

class E2Curve(GaussianSumHormone):
    def __init__(self) -> None:
        super().__init__(mus=[13, 21],
                         sigmas=[2.0, 3.0],
                         amps=[30, 12],
                         baseline=10)

class PGCurve(GaussianSumHormone):
    def __init__(self) -> None:
        super().__init__(mus=[21],
                         sigmas=[3.5],
                         amps=[25],
                         baseline=0.2)

# Diccionario de conveniencia
CURVES = {
    "FSH": FSHCurve(),
    "LH":  LHCurve(),
    "E2":  E2Curve(),
    "PG":  PGCurve()
}
