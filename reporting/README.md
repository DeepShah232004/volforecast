# Reproducible reporting assets

This stage converts the two committed result artifacts into paper-ready tables
and figures. It does not load raw WRDS data, refit a model, recalculate a
forecast, or change a hypothesis decision.

Run from the repository root:

```bash
./venv/bin/python reporting/build_report_assets.py
```

The script verifies the SHA-256 hashes of:

- `evaluation/results/final_evaluation.json`
- `interpretability/results/shap_summary.json`

It then writes:

- `reporting/tables/results_tables.md`
- Five figures in both 300-DPI PNG and vector PDF formats under
  `reporting/figures/`

The figures cover overall accuracy, the preregistered regime comparisons, the
earnings-window penalty, grouped SHAP importance, and SHAP stability. Every
number comes from a locked result artifact.
