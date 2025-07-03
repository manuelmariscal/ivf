"""
HONN orden-2 · Ridge-SVD multi-target con clip ±5
─────────────────────────────────────────────────
target = [μ_real, logA, logσ]
"""

from __future__ import annotations
import numpy as np, polars as pl, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List
from utils import Logger


@dataclass
class CurveParams:
    mu: float
    A: float
    sigma: float


class HONNTrainer:
    def __init__(self,
                 kappa: float = 2.8,        # tanh(X / κσj)
                 clip_phi: float = 4.70,     # Φn ∈ [-clip, +clip]
                 lambda_alpha: float = 0.8, # λ = scale·trace / m
                 debug: bool = False) -> None:

        self.kappa = kappa
        self.clip_phi = clip_phi
        self.lambda_alpha = lambda_alpha
        self.debug = debug

        self.feature_cols = [
            "s1_day","s1_fsh","s1_lh","s1_e2","s1_pg",
            "s2_day","s2_fsh","s2_lh","s2_e2","s2_pg"]
        self.target_cols  = ["mu_real","logA","logσ"]
        self._fitted = False

    # ---------- helpers ----------
    @staticmethod
    def _poly2(X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        cross = [(X[:,i]*X[:,j])[:,None]
                 for i in range(d) for j in range(i+1,d)]
        return np.hstack([X, X**2, *cross])       # (n,65)

    # tanh / κσj
    def _scale_fit(self, X):
        self.scale_ = self.kappa * np.maximum(X.std(0,keepdims=True),1e-2)

    def _scale_apply(self, X):          # tanh in (-1,1)
        return np.tanh(X / self.scale_)

    # ---------- train ----------
    def train(self, df: pl.DataFrame):
        Logger.info("Entrenando HONN (SVD)…")
        Xraw = df.select(self.feature_cols).to_numpy().astype(float)
        self._scale_fit(Xraw)
        X = self._scale_apply(Xraw)

        Y = df.select(self.target_cols).to_numpy().astype(float)   # (n,3)
        Φ = self._poly2(X)
        self.mu_x   = Φ.mean(0, keepdims=True)
        self.sigma_x= np.maximum(Φ.std(0, keepdims=True), 1e-2)
        Φn = np.clip((Φ - self.mu_x) / self.sigma_x,
                     -self.clip_phi, self.clip_phi)                # (n,65)

        # ----- Ridge via SVD -----
        U, S, Vt = np.linalg.svd(Φn, full_matrices=False)
        λ = self.lambda_alpha * np.trace(np.diag(S**2)) / S.size
        d = S / (S**2 + λ)
        W = Vt.T @ (d[:,None] * (U.T @ Y))         # (65,3)
        b = Y.mean(0) - Φn.mean(0) @ W

        self.W, self.b = W, b
        self._fitted = True

        Y_hat = Φn @ W + b
        mae = np.mean(np.abs(Y_hat - Y), 0)
        Logger.success(f"MAE μ={mae[0]:.3f} logA={mae[1]:.3f} logσ={mae[2]:.3f}")
        if self.debug:
            Logger.info(f"λ={λ:.1e}  ||W||_F={np.linalg.norm(W):.3f}")

    # ---------- predict ----------
    def predict(self, df: pl.DataFrame) -> np.ndarray:   # (n,3)
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado")
        X = self._scale_apply(df.select(self.feature_cols).to_numpy().astype(float))
        Φn = np.clip((self._poly2(X) - self.mu_x) / self.sigma_x,
                     -self.clip_phi, self.clip_phi)
        return np.nan_to_num(Φn @ self.W + self.b)

    def predict_params(self, df: pl.DataFrame) -> List[CurveParams]:
        Y = self.predict(df)
        return [CurveParams(mu=float(mu), A=float(10**logA), sigma=float(10**logσ))
                for mu, logA, logσ in Y]

    # ---------- save / load ----------
    def save(self, file: Path):
        pickle.dump(self.__dict__, file.open("wb"))

    @classmethod
    def load(cls, file: Path, debug=False):
        inst = cls(debug=debug)
        inst.__dict__.update(pickle.load(file.open("rb")))
        return inst
