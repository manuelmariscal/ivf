# IVF Predictor

This repository contains a small experiment to predict the ideal follicle extraction day from real hormone measurements.

## Workflow
1. **Convert the CSV file to SQLite**
   ```bash
   python -m pip install -r requirements.txt
   python src/db_builder.py
   ```
   This reads `data/E2_develop.csv`, standardises the headers and stores the result in `data/e2.db`.

2. **Train the model**
   ```bash
   python src/trainer.py --epochs 300
   ```
   The number of epochs is configurable. The trained model is saved to `models/peak_day_model.pkl`.

3. **Predict for a single measurement**
   ```bash
   python src/predictor.py path/to/patient.json
   ```
   The JSON file must contain the same fields as the database table (except `patientId`).

Use `--debug` on any script for verbose output.
