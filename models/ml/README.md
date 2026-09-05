# Phase 3 ML training

Run every command from the repository root. Training is CPU-intensive, so the
recommended eight workers leave two logical cores free on the current machine.

## 1. Install the locked packages

```bash
./venv/bin/python -m pip install scikit-learn==1.9.0 xgboost==3.4.1 shap==0.52.0
./venv/bin/python -m pip freeze > requirements.txt
```

## 2. Run the no-training preflight

```bash
./venv/bin/python models/ml/train_models.py check
```

Expected sample counts are 859,540 tuning-training rows, 238,982 validation
rows, 1,108,846 final-refit rows, and 1,234,643 test rows.

## 3. Tune on training/validation only

Run the models sequentially. On macOS, `caffeinate -i` prevents sleep while a
command is active. Each completed candidate is written immediately to its
selection JSON; rerunning an interrupted command skips completed candidates.

```bash
mkdir -p models/ml/logs
caffeinate -i ./venv/bin/python -u models/ml/train_models.py tune --model random_forest --workers 8 2>&1 | tee models/ml/logs/random_forest_tune.log
caffeinate -i ./venv/bin/python -u models/ml/train_models.py tune --model xgboost --workers 8 2>&1 | tee models/ml/logs/xgboost_tune.log
```

Do not run the final stage until both selection files have been reviewed and
committed. Tuning writes:

- `models/ml/results/random_forest_selection.json`
- `models/ml/results/xgboost_selection.json`

## 4. Final refit and frozen test predictions

After selection is frozen, run:

```bash
caffeinate -i ./venv/bin/python -u models/ml/train_models.py final --model random_forest --workers 8 2>&1 | tee models/ml/logs/random_forest_final.log
caffeinate -i ./venv/bin/python -u models/ml/train_models.py final --model xgboost --workers 8 2>&1 | tee models/ml/logs/xgboost_final.log
```

The final stage saves model artifacts for SHAP and test forecasts for later
evaluation. It intentionally does not calculate or display test-period losses.

