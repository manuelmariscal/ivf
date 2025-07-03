"""
visualization.py
--------------------------------------------
• gold_standard()
• scatter_pred()
• residual_hist()
• single_cycle()
• show_all() → compone fig 1-3 y, si recibe params,
               añade fig4 con la curva de ese ciclo
--------------------------------------------
"""

import math, numpy as np, matplotlib.pyplot as plt, polars as pl
from analytic_functions import CURVES, E2Curve
from honn_trainer import CurveParams
from utils import Logger

# --- 1
def gold_standard(ax=None):
    t = np.linspace(0, 28, 500); ax = ax or plt.gca()
    for n,c in CURVES.items(): ax.plot(t, c(t), label=n)
    ax.set_title("Gold-standard analítico"); ax.legend()
    ax.set_xlabel("Día"); ax.set_ylabel("Concentración")

# --- 2
def scatter_pred(df: pl.DataFrame, y_pred: np.ndarray, ax=None):
    ax = ax or plt.gca(); y = df["peak_day"].to_numpy()
    ax.scatter(y, y_pred, alpha=.4, s=10)
    lim = [y.min()-1, y.max()+1]; ax.plot(lim, lim, "--k"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Pred vs. True"); ax.set_xlabel("Real"); ax.set_ylabel("Predicho")

# --- 3
def residual_hist(df: pl.DataFrame, y_pred: np.ndarray, ax=None):
    ax = ax or plt.gca(); r = y_pred - df["peak_day"].to_numpy()
    ax.hist(r, bins=30, edgecolor="k"); ax.axvline(0,color="red")
    ax.set_title(f"Residuales σ={r.std():.3f}"); ax.set_xlabel("Error (día)")

# --- 4
def single_cycle(params: CurveParams, ax=None):
    ax = ax or plt.gca(); t = np.linspace(0,28,400)
    e2_true = E2Curve()(t)
    e2_pred = params.A*np.exp(-0.5*((t-params.mu)/params.sigma)**2)+10
    trig_t  = params.mu - params.sigma*np.sqrt(-2*np.log(0.75))

    ax.plot(t, e2_true, label="E₂ real")
    ax.plot(t, e2_pred, "--", label="E₂ pred")
    ax.axvline(trig_t, color="red", ls=":", label="Trigger 75 %")
    ax.set_title(f"Ciclo ejemplo μ̂={params.mu:.2f}")
    ax.set_xlabel("Día del ciclo"); ax.set_ylabel("E₂ (u. rel.)")
    ax.legend()

# --- wrapper
def show_all(df: pl.DataFrame, y_pred: np.ndarray,
             params: CurveParams | None = None) -> None:
    Logger.info("Mostrando figuras …")
    ncols = 3 + (params is not None)
    fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    if ncols==3: axs1,axs2,axs3 = axs
    else: axs1,axs2,axs3,axs4 = axs
    gold_standard(axs1); scatter_pred(df,y_pred,axs2); residual_hist(df,y_pred,axs3)
    if params is not None: single_cycle(params, axs4)
    plt.tight_layout(); plt.show()
