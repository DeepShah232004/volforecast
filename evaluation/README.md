# Final evaluation

Run these commands from the repository root only after the final-refit metadata
and this evaluation code have been committed.

Build the common five-model panel without calculating losses:

```bash
./venv/bin/python evaluation/build_final_panel.py
```

Run the no-loss preflight:

```bash
./venv/bin/python evaluation/final_evaluation.py check
```

The panel should contain 1,206,821 rows across 682 PERMNOs. Once that check
passes, reveal the registered test results exactly once:

```bash
./venv/bin/python evaluation/final_evaluation.py evaluate
```

The evaluation writes `evaluation/results/final_evaluation.json`. It refuses
to overwrite an existing result unless `--force` is supplied; any forced rerun
must be documented.

