from pathlib import Path
import json
import pandas as pd
import joblib
from utils import Logger, set_debug

MODEL_FILE = Path('models/peak_day_model.pkl')

def load_model(path: Path = MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}')
    Logger.info(f'Loading model from {path}')
    return joblib.load(path)

def predict_from_json(json_file: Path, model_path: Path = MODEL_FILE, debug: bool = False) -> float:
    set_debug(debug)
    model = load_model(model_path)
    entry = json.loads(json_file.read_text())
    X = pd.DataFrame([entry])
    pred = model.predict(X)[0]
    Logger.success(f'Predicted peak day: {pred:.2f}')
    trigger = pred - 1
    Logger.info(f'Suggested extraction day: {trigger:.2f}')
    return pred

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('json', type=Path, help='JSON file with a measurement entry')
    ap.add_argument('--model', type=Path, default=MODEL_FILE)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    predict_from_json(args.json, model_path=args.model, debug=args.debug)
