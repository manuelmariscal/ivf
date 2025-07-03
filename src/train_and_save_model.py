"""
train_and_save_model.py
───────────────────────
CLI:
    python src/train_and_save_model.py  --n 500000  --regen

  --n      Nº de ciclos sintéticos (default 50_000)
  --regen  Fuerza regenerar data/synthetic_cycles.csv
"""

import argparse, sys, traceback
from pathlib import Path
import polars as pl
from utils import Logger
from dataset_generator import SyntheticCycleGenerator
from data_loader import SyntheticDataLoader
from honn_trainer import HONNTrainer

DATA_CSV   = Path("data/synthetic_cycles.csv")
MODEL_PKL  = Path("models/honn.pkl")


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000,
                    help="Número de ciclos sintéticos")
    ap.add_argument("--regen", action="store_true",
                    help="Regenera el CSV antes de entrenar")
    ap.add_argument("--debug", action="store_true",
                    help="Modo verboso en HONNTrainer")
    return ap.parse_args()


def ensure_dataset(n_rows: int, force_regen: bool):
    if force_regen or not DATA_CSV.exists():
        SyntheticCycleGenerator(n_cycles=n_rows, out_csv=DATA_CSV).build()
    else:
        Logger.info(f"Usando dataset existente «{DATA_CSV}»")


def main():
    a = args()

    # 1) dataset
    ensure_dataset(a.n, a.regen)
    df = SyntheticDataLoader(DATA_CSV).load()

    # 2) modelo
    model = HONNTrainer(lambda_alpha=1e-1, debug=True)

    model.train(df)

    # 3) guardar
    model.save(MODEL_PKL)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        Logger.error("💥 Memoria insuficiente. Reduce --n o usa máquinas con más RAM.")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
