"""
Genera un CSV sintético realista (log-escala en todas las hormonas)
-------------------------------------------------------------------
Columnas objetivo:  mu_real,  logA,  logσ
-------------------------------------------------------------------
"""

from pathlib import Path
import numpy as np, polars as pl
from utils import Logger
from analytic_functions import FSHCurve, LHCurve, E2Curve, PGCurve

RNG = np.random.default_rng(42)


class SyntheticCycleGenerator:
    def __init__(self,
                 n_cycles: int = 25,
                 out_csv: str | Path = "data/synthetic_cycles.csv",
                 cv_pct: float = 0.08,
                 outlier_rate: float = 0.04) -> None:
        self.n_cycles = n_cycles
        self.out_csv = Path(out_csv)
        self.cv_pct = cv_pct
        self.outlier_rate = outlier_rate
        self.fsh, self.lh, self.e2, self.pg = (
            FSHCurve(), LHCurve(), E2Curve(), PGCurve()
        )

    # ---------- helpers ----------
    def _noise(self, val: float) -> float:
        return val * RNG.normal(loc=1.0, scale=self.cv_pct)

    def _log_h(self, x):
        return np.log10(max(x, 0.5))   # evita log(0)

    # ---------- genera fila ----------
    def _generate_row(self, cid: int) -> dict:
        mu_shift = RNG.uniform(-2, 2)
        is_out   = RNG.random() < self.outlier_rate
        if is_out:
            mu_shift += RNG.uniform(-4, 4)

        # amplitud / anchura
        A_real = float(RNG.lognormal(np.log(50), 0.6))        # 25-200
        if is_out:
            A_real *= RNG.uniform(2, 6)                       # 50-1200
        sigma_real = float(np.clip(RNG.normal(2, 0.4), 1.2, 3.5))
        # días de extracción
        s1_day, s2_day = RNG.integers(2, 5), RNG.integers(6, 9)

        def e2_val(day):
            return A_real * np.exp(-0.5 * ((day + mu_shift - 13) / sigma_real) ** 2) + 10
        def eval_curve(curve, d):
            return self._noise(curve(d + mu_shift))

        return {
            "cycle_id": cid,
            "s1_day": s1_day,
            "s1_fsh": self._log_h(eval_curve(self.fsh, s1_day)),
            "s1_lh":  self._log_h(eval_curve(self.lh,  s1_day)),
            "s1_e2":  self._log_h(self._noise(e2_val(s1_day))),
            "s1_pg":  self._log_h(eval_curve(self.pg,  s1_day)),
            "s2_day": s2_day,
            "s2_fsh": self._log_h(eval_curve(self.fsh, s2_day)),
            "s2_lh":  self._log_h(eval_curve(self.lh,  s2_day)),
            "s2_e2":  self._log_h(self._noise(e2_val(s2_day))),
            "s2_pg":  self._log_h(eval_curve(self.pg,  s2_day)),
            # objetivos
            "mu_real":  13 + mu_shift,
            "logA":     np.log10(A_real),
            "logσ":     np.log10(sigma_real)
        }

    # ---------- API ----------
    def build(self) -> pl.DataFrame:
        Logger.info(f"Generando {self.n_cycles} ciclos sintéticos …")
        df = pl.from_dicts([self._generate_row(i+1) for i in range(self.n_cycles)])
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(self.out_csv)
        Logger.success(f"Dataset guardado en «{self.out_csv}» ({self.n_cycles} filas)")
        return df
