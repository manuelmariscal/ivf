from pathlib import Path
import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from utils import Logger, set_debug
from data_loader import RealDataLoader

DEFAULT_DB = Path('data/e2.db')
MODEL_FILE = Path('models/peak_day_model.pkl')

class ModelTrainer:
    def __init__(self, db_path: Path = DEFAULT_DB, epochs: int = 200, debug: bool = False) -> None:
        set_debug(debug)
        self.db_path = db_path
        self.epochs = epochs
        self.model = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=self.epochs, random_state=42)

    def _prepare_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        loader = RealDataLoader(self.db_path)
        df = loader.load().to_pandas()
        rows = []
        for pid, group in df.groupby('patientId'):
            group_sorted = group.sort_values('stimDay')
            first = group_sorted.iloc[0]
            peak_day = group_sorted.loc[group_sorted['e2'].idxmax(), 'stimDay']
            first_dict = first.drop(labels=['patientId']).to_dict()
            rows.append((first_dict, peak_day))
        if not rows:
            raise ValueError('No data available')
        X = pd.DataFrame([r[0] for r in rows])
        y = pd.Series([r[1] for r in rows])
        return X, y

    def train(self) -> None:
        X, y = self._prepare_dataset()
        Logger.info(f'Training on {len(X)} samples for {self.epochs} epochs')
        self.model.max_iter = self.epochs
        self.model.fit(X, y)
        Logger.success('Model trained')

    def save(self, path: Path = MODEL_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        Logger.success(f'Model saved to {path}')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=DEFAULT_DB)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    trainer = ModelTrainer(db_path=args.db, epochs=args.epochs, debug=args.debug)
    trainer.train()
    trainer.save()
