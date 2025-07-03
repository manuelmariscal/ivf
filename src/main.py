"""
main.py – CLI
"""

import argparse, random
from pathlib import Path
from utils import Logger
from data_loader import SyntheticDataLoader, RealDataLoader
from dataset_generator import SyntheticCycleGenerator
from honn_trainer import HONNTrainer
from diagnostics import plot_spectrum, plot_weights, animate_lambda
import visualization              

DATA_PATH = Path("data/synthetic_cycles.csv")

def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["syn","real"], default="syn")
    ap.add_argument("--n_syn", type=int, default=50)
    ap.add_argument("--engine", choices=["honn","skl"], default="honn")
    ap.add_argument("--debug", action="store_true", help="modo verboso")
    return ap.parse_args()

def main():
    a = _args()

    # dataset
    if a.data=="syn":
        SyntheticCycleGenerator(a.n_syn, DATA_PATH).build()
        df = SyntheticDataLoader(DATA_PATH).load()
    else:
        df = RealDataLoader("data/real_cycles.csv").load()

    # modelo
    if a.engine == "honn":
        trainer = HONNTrainer(lambda_alpha=4.7, debug=a.debug)

    trainer.train(df)
    y_hat = trainer.predict(df)

    # un ciclo para curva individual
    idx = random.randint(0,len(df)-1)
    params = trainer.predict_params(df[idx:idx+1])[0]

    visualization.show_all(df, y_hat, params)
    plot_spectrum(trainer)
    plot_weights(trainer, top_k=20)
    animate_lambda(df, idx=3, lambdas=[0.1,1,5,20,100], save_gif="lambda_sweep.gif")

if __name__ == "__main__":
    main()
    

