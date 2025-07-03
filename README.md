# IVF-Predictor
### High-Order Neural Network (HONN) for choosing the trigger day

Author: Víctor Manuel Mariscal Cervantes  
Programa de Maestría en Cómputo Aplicado – Universidad de Guadalajara  

---

## What the project does

* **Input:** two early hormone panels (days 2-4 and 6-8 of the same
  menstrual cycle).  
* **Output:**  
  * The day when estradiol will reach its maximum (mu_hat).  
  * The height of that peak (A_hat, estradiol units).  
  * The width of the peak (sigma_hat, days).  
  * **Trigger day** = mu_hat minus three-quarters of one sigma_hat  
    (≈ one day before the real peak), which is the best moment to
    retrieve oocytes for IVF.

Everything runs locally in less than a second, even with six hundred
thousand synthetic cycles for training.

---

## Directory layout

```

data/
│ synthetic_cycles.csv     ← huge training set (auto-generated)
│ json/patients.json       ← example patients for inference
models/
│ honn.pkl                 ← trained network (saved automatically)
src/
│ analytic_functions.py    ← ideal hormone curves
│ dataset_generator.py     ← makes synthetic\_cycles.csv
│ honn_trainer.py          ← the HONN itself
│ train_and_save_model.py  ← command-line trainer
│ infer_dashboard.py       ← pretty plots for several patients
│ utils.py                 ← coloured logs, timers, …

````

---

## Quick start

```bash
# 1. set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. create / regenerate 600 000 synthetic cycles and train the model
python src/train_and_save_model.py --n 600000 --regen 

# 3. run inference for every patient in patients.json
python src/infer_dashboard.py
````

`infer_dashboard.py` pops up one figure per patient showing

* the two measured points,
* the predicted estradiol curve,
* the trigger day marked with a vertical dashed line.

It also prints something like:

```
Patient P-001
mu_hat      : 12.5  days
A_hat       :  22   estradiol units (relative)
sigma_hat   :   1.4 days
Trigger day : 11.5 days
```

---

## How the HONN works 

| Stage                        | What happens                                                                                                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Log scale**             | Hormone values are converted to base-10 logarithm so huge numbers become small.                                                                                                           |
| **2. Tanh scaling**          | Every column is divided by `kappa × its own standard deviation`, then fed into the tanh function. That squeezes all inputs into the range −1 to +1.                                       |
| **3. Polynomial expansion**  | The ten inputs become sixty-five: the ten originals, their squares, and every pairwise product. This lets the network learn bends and interactions without a deep net.                    |
| **4. Centre, scale, clip**   | Each of the sixty-five columns is centred to zero mean, scaled to unit variance, then clipped between `−clip_phi` and `+clip_phi` so nothing blows up later.                              |
| **5. Ridge training by SVD** | A closed-form formula (no back-prop) finds a 65×3 weight matrix `W` that links the sixty-five features to the three targets; a penalty `lambda_scale` keeps the weights small and stable. |
| **6. Prediction**            | For any new patient we repeat steps 1-4, multiply by `W`, add a small bias vector, and get `mu_hat`, `logA_hat`, `logSigma_hat`.                                                          |
| **7. Post-processing**       | `A_hat` = 10 to the power of `logA_hat`; `sigma_hat` = 10 to the power of `logSigma_hat`; Trigger = `mu_hat` minus 0.754 × `sigma_hat`.                                                   |

Typical error after training with six hundred thousand cycles
(*Mean Absolute Error, or average absolute difference between prediction
and truth*):

* mu\_hat ± 0.3-0.6 days
* logA\_hat ± 0.22 (that is, a ×1.6 factor in real units)
* logSigma\_hat ± 0.06 (±15 %)

---

## Tuning tips

* **clip\_phi** 4.5 to 6 lower → safer numbers, higher → more
  sensitivity.
* **lambda\_scale** 0.8 to 1.2 higher → smoother predictions, lower → more
  variance.
* **kappa** 2.5 to 3.2 smaller → sharper contrasts between hormone levels.

If mu\_hat drifts above 15 days, raise lambda\_scale or lower clip\_phi.
If A\_hat never goes above 100, raise clip\_phi slightly.

---

