# SHAP interpretability analysis

Run these commands from the repository root after the final evaluation result
has been committed. This stage does not refit either model or alter any test
forecast.

The locked population is the exact five-model final comparison panel. The
script selects 10,000 rows without replacement, proportionally stratified by
calendar year and VIX regime, with at least one row from every populated
stratum and seed 42.

## 1. Run the no-SHAP preflight

```bash
./venv/bin/python interpretability/shap_analysis.py check
```

This validates the matrix, panel, result, model-selection metadata, hashes,
sample allocation, and sample key hash. It does not load the models or compute
SHAP values. The locked sample key SHA-256 is
`5938bd766426e8b42b10b1f54a06bcb669f07e21032480a6054d035c62a6f185`.

## 2. Commit the pipeline before revealing interpretations

Commit the protocol, script, tests, and this README after the preflight and
test suite pass. Do not run the analysis until that commit exists.

## 3. Run SHAP once

```bash
mkdir -p interpretability/logs
caffeinate -i ./venv/bin/python -u interpretability/shap_analysis.py analyze 2>&1 | tee interpretability/logs/shap_analysis.log
```

The analysis calculates exact TreeSHAP contributions on the models' raw
log-volatility outputs. It verifies SHAP additivity and requires the loaded
models to reproduce the frozen forecasts.

Outputs:

- `interpretability/results/shap_summary.json`: aggregate feature and grouped
  importance, plus stability between 2015-2019 and 2020-2024. Commit this file
  after reviewing it.
- `interpretability/results/shap_values.parquet`: row-level sample features and
  SHAP values used to build figures. Parquet files are ignored by Git; retain
  this file locally for the reporting phase.
- `interpretability/logs/shap_analysis.log`: execution log, ignored by Git.

Interpret the grouped results as descriptive associations with model output,
not causal effects. Individual rankings among correlated volatility features
are secondary.
