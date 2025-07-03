"""
infer_dashboard.py  –  muestra gráficas para uno o varios pacientes
===================================================================
• Lee un JSON con ≥1 pacientes
• Carga el modelo HONN multi-target (mu, logA, logSigma)
• Para cada paciente dibuja:
      1) Dispersión mu_pred vs mu_real (set entrenamiento)
      2) Curva E2 predicha + línea de trigger + E2 gold-standard
      3) Panel con números
"""

from pathlib import Path
import json, math, numpy as np, matplotlib.pyplot as plt, polars as pl
from honn_trainer import HONNTrainer, CurveParams
from analytic_functions import E2Curve           # gold-standard
from utils import Logger

# ---------------- Rutas ----------------
JSON_FILE  = Path("data/json/patients.json")
MODEL_FILE = Path("models/honn.pkl")
TRAIN_CSV  = Path("data/synthetic_cycles.csv")

# ---------------- Modelo ---------------
model = HONNTrainer.load(MODEL_FILE)

# --------------- Pacientes -------------
entries = json.loads(JSON_FILE.read_text())
if isinstance(entries, dict):           # permitir un solo paciente
    entries = [entries]

def to_df(entry: dict) -> pl.DataFrame:
    """Convierte el dict del paciente en DataFrame con las 10 columnas requeridas."""
    return pl.DataFrame([{col: entry[col] for col in model.feature_cols}])

# ------------- Scatter global -----------
df_train  = pl.read_csv(TRAIN_CSV)
target_col = "mu_real" if "mu_real" in df_train.columns else "peak_day"
y_true      = df_train[target_col].to_numpy()
y_pred_mu   = model.predict(df_train)[:, 0]   # primera salida = mu_hat

def add_scatter(ax, y_true, y_pred):
    """Dispersión mu_pred vs mu_real, forzando ejes a arrancar en 0."""
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    lim_inf = 0.0
    lim_sup = max(float(y_true.max()), float(y_pred.max())) + 1.0
    ax.plot([lim_inf, lim_sup], [lim_inf, lim_sup], "--k")
    ax.set_xlim(lim_inf, lim_sup)
    ax.set_ylim(lim_inf, lim_sup)
    ax.set_xlabel("μ real (día)")
    ax.set_ylabel("μ pred (día)")
    ax.set_title("Pred vs. True  (diagonal = ideal)")

# ---------- Gold-standard E2 -------------
e2_gold = E2Curve()                     # amplitud 30, pico día 13, sigma 2
t_grid   = np.linspace(0, 28, 400)
e2_gold_vals = e2_gold(t_grid)

# ------------- Bucle sobre pacientes -------------
for idx, entry in enumerate(entries, 1):
    pid    = entry.get("patient_id", f"#{idx}")
    df_one = to_df(entry)

    params: CurveParams = model.predict_params(df_one)[0]
    mu_hat, A_hat, sigma_hat = params.mu, params.A, params.sigma

    # -------- validación rápida ----------
    if not np.isfinite([mu_hat, A_hat, sigma_hat]).all() or A_hat <= 0 or sigma_hat <= 0:
        Logger.warn(f"{pid}: parámetros no válidos, se omite gráfica")
        continue

    trigger = mu_hat - 0.754 * sigma_hat           # factor 0.754 = sqrt(-2 ln 0.75)

    # --------------- Figuras ---------------
    fig = plt.figure(figsize=(12, 4))

    # (1) Scatter global
    ax1 = fig.add_subplot(1, 3, 1)
    add_scatter(ax1, y_true, y_pred_mu)

    # (2) Curva E2 predicha + gold-standard
    e2_pred = A_hat * np.exp(-0.5 * ((t_grid - mu_hat) / sigma_hat) ** 2) + 10
    e2_pred = np.nan_to_num(e2_pred, nan=0.0, posinf=0.0, neginf=0.0)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(t_grid, e2_pred,  "--", label="E₂ predicha")
    ax2.plot(t_grid, e2_gold_vals, color="gray",  lw=1.2, label="E₂ gold-standard")
    ax2.axvline(trigger, color="red", ls=":", label="Trigger 75 %")
    ax2.set_xlim(0, 28)
    ax2.set_ylim(0, max(e2_pred.max(), e2_gold_vals.max()) + 10)
    ax2.set_xlabel("Día")
    ax2.set_ylabel("E₂ (u. rel.)")
    ax2.set_title(f"Ciclo {pid}")
    ax2.legend()

    # (3) Panel numérico
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    txt = (
        f"Paciente : {pid}\n"
        f"μ̂ (pico) : {mu_hat:.2f}\n"
        f"Â        : {A_hat:.1f}\n"
        f"σ̂        : {sigma_hat:.2f}\n"
        f"Trigger   : {trigger:.2f}"
    )
    ax3.text(0.0, 0.6, txt, fontsize=12, family="monospace")

    # --------- Consola ----------
    print(txt.replace("\n", " | "))

    plt.tight_layout()
    plt.show()
