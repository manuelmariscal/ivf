"""
diagnostics.py
--------------
Herramientas para inspeccionar la HONN entrenada.

Uso:
    from diagnostics import plot_spectrum, plot_weights, animate_lambda

    trainer = HONNTrainer(lambda_l2=5.0, debug=False)
    trainer.train(df)

    plot_spectrum(trainer)
    plot_weights(trainer, top_k=15)
    # animate_lambda(df, idx=0, lambdas=[0.1,1,5,20,100])
"""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from honn_trainer import HONNTrainer, CurveParams
from analytic_functions import E2Curve


# ---------- 1. Singular spectrum ----------
def plot_spectrum(trainer: HONNTrainer):
    S = trainer.singular_vals
    plt.figure(figsize=(5,4))
    plt.semilogy(np.arange(1, len(S)+1), S, marker='o')
    plt.title("Espectro de valores singulares")
    plt.xlabel("Índice"); plt.ylabel("S (escala log)")
    plt.tight_layout(); plt.show()


# ---------- 2. Pesos más importantes ----------
def plot_weights(trainer: HONNTrainer, top_k: int = 15):
    w = trainer.w
    idx_sorted = np.argsort(np.abs(w))[::-1][:top_k]
    labels = [f"w{idx}" for idx in idx_sorted]
    vals   = w[idx_sorted]

    plt.figure(figsize=(6,4))
    plt.barh(labels, vals)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_k} pesos absolutos")
    plt.tight_layout(); plt.show()


# ---------- 3. Animación barriendo λ ----------
def animate_lambda(df: pl.DataFrame, idx: int,
                   lambdas = [0.1,1,5,20,100], save_gif: str | None = None):
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(5,4))
    e2_true = E2Curve()

    t = np.linspace(0,28,400)

    def frame(l):
        ax.clear()
        trainer = HONNTrainer(lambda_l2=l)
        trainer.train(df)
        params: CurveParams = trainer.predict_params(df[idx:idx+1])[0]
        e2_pred = params.A*np.exp(-0.5*((t-params.mu)/params.sigma)**2) + 10
        ax.plot(t, e2_true(t), label="E₂ real")
        ax.plot(t, e2_pred, "--", label=f"E₂ pred  λ={l}")
        ax.set_ylim(0,45); ax.set_xlim(0,28)
        ax.set_title(f"Ciclo idx {idx}, μ̂={params.mu:.2f}")
        ax.legend()

    ani = animation.FuncAnimation(fig, frame, frames=lambdas, repeat=False)
    if save_gif:
        ani.save(save_gif, writer="pillow", fps=1)
    plt.show()
