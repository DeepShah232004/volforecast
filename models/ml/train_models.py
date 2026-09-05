"""Tune and fit the locked Random Forest and XGBoost specifications.

The command has separate check, tune, and final stages so validation choices
can be frozen before any test-period evaluation. Final training saves test
predictions but deliberately does not calculate test losses.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_common import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    MATRIX_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    extract_xy,
    load_and_validate_matrix,
    matrix_sha256,
    read_json,
    save_forecasts,
    validation_metrics,
    volatility_from_log_prediction,
    write_json_atomic,
)


RF_GRID = [
    {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    for max_depth in (8, 16, None)
    for min_samples_leaf in (20, 100)
    for max_features in ("sqrt", 0.5)
]

XGB_GRID = [
    {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
    }
    for learning_rate in (0.03, 0.08)
    for max_depth in (3, 6)
    for min_child_weight in (5, 20)
]

RF_TREES = 400
XGB_MAX_ROUNDS = 1500
XGB_EARLY_STOPPING_ROUNDS = 75


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_id(model_name: str, position: int) -> str:
    return f"{model_name}_{position + 1:02d}"


def new_run_record(
    model_name: str,
    grid: list[dict[str, Any]],
    matrix_hash: str,
    sample_counts: dict[str, int],
    library_name: str,
    library_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model_name,
        "created_utc": utc_now(),
        "matrix_sha256": matrix_hash,
        "feature_columns": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "sample_counts": sample_counts,
        "library": {"name": library_name, "version": library_version},
        "grid": [
            {"candidate_id": candidate_id(model_name, position), "params": params}
            for position, params in enumerate(grid)
        ],
        "candidates": [],
        "selected": None,
        "final_fit": None,
    }


def load_or_create_run(
    result_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not result_path.exists():
        write_json_atomic(result_path, expected)
        return expected

    saved = read_json(result_path)
    identity_fields = [
        "schema_version",
        "model",
        "matrix_sha256",
        "feature_columns",
        "random_seed",
        "sample_counts",
        "library",
        "grid",
    ]
    mismatches = [field for field in identity_fields if saved.get(field) != expected[field]]
    if mismatches:
        raise ValueError(
            f"Existing checkpoint {result_path} is incompatible in: {mismatches}. "
            "Move it aside before beginning a new run."
        )
    return saved


def finish_selection(run: dict[str, Any], result_path: Path) -> dict[str, Any]:
    expected_ids = {candidate["candidate_id"] for candidate in run["grid"]}
    completed_ids = {candidate["candidate_id"] for candidate in run["candidates"]}
    if completed_ids != expected_ids:
        missing = sorted(expected_ids.difference(completed_ids))
        raise RuntimeError(f"Cannot select a model; unfinished candidates: {missing}")

    winner = min(
        run["candidates"],
        key=lambda candidate: (
            candidate["validation_metrics"]["qlike"],
            candidate["candidate_id"],
        ),
    )
    run["selected"] = {
        "candidate_id": winner["candidate_id"],
        "params": winner["params"],
        "validation_metrics": winner["validation_metrics"],
        **(
            {
                "best_rounds": winner["best_rounds"],
                "early_stopping_log_rmse": winner["early_stopping_log_rmse"],
            }
            if "best_rounds" in winner
            else {}
        ),
        "selected_utc": utc_now(),
    }
    write_json_atomic(result_path, run)
    print(
        f"Selected {winner['candidate_id']} with validation "
        f"QLIKE={winner['validation_metrics']['qlike']:.9f}"
    )
    return run


def tune_random_forest(matrix, masks, workers: int, matrix_hash: str) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor

    result_path = RESULTS_DIR / "random_forest_selection.json"
    expected = new_run_record(
        "random_forest",
        RF_GRID,
        matrix_hash,
        masks.counts(),
        "scikit-learn",
        importlib.metadata.version("scikit-learn"),
    )
    run = load_or_create_run(result_path, expected)
    completed_ids = {candidate["candidate_id"] for candidate in run["candidates"]}

    train_x, train_y = extract_xy(matrix, masks.tuning_train)
    validation_x, _ = extract_xy(matrix, masks.validation)
    validation_actual = matrix.loc[
        masks.validation, "target_forward_21d"
    ].to_numpy(dtype="float64")

    for position, params in enumerate(RF_GRID):
        current_id = candidate_id("random_forest", position)
        if current_id in completed_ids:
            print(f"Skipping completed candidate {current_id}.")
            continue

        print(f"Starting {current_id} ({position + 1}/{len(RF_GRID)}): {params}")
        started = time.perf_counter()
        model = RandomForestRegressor(
            n_estimators=RF_TREES,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=workers,
            verbose=1,
        )
        model.fit(train_x, train_y)
        validation_log_prediction = model.predict(validation_x)
        metrics = validation_metrics(validation_actual, validation_log_prediction)
        elapsed = time.perf_counter() - started
        run["candidates"].append(
            {
                "candidate_id": current_id,
                "params": params,
                "validation_metrics": metrics,
                "fit_and_score_seconds": round(elapsed, 3),
                "completed_utc": utc_now(),
            }
        )
        write_json_atomic(result_path, run)
        print(
            f"Completed {current_id} in {elapsed / 60:.1f} minutes: "
            f"QLIKE={metrics['qlike']:.9f}, RMSE={metrics['rmse']:.9f}, "
            f"MAE={metrics['mae']:.9f}"
        )
        del model, validation_log_prediction
        gc.collect()

    return finish_selection(run, result_path)


def tune_xgboost(matrix, masks, workers: int, matrix_hash: str) -> dict[str, Any]:
    import xgboost as xgb

    result_path = RESULTS_DIR / "xgboost_selection.json"
    expected = new_run_record(
        "xgboost",
        XGB_GRID,
        matrix_hash,
        masks.counts(),
        "xgboost",
        importlib.metadata.version("xgboost"),
    )
    run = load_or_create_run(result_path, expected)
    completed_ids = {candidate["candidate_id"] for candidate in run["candidates"]}

    train_x, train_y = extract_xy(matrix, masks.tuning_train)
    validation_x, validation_y = extract_xy(matrix, masks.validation)
    validation_actual = matrix.loc[
        masks.validation, "target_forward_21d"
    ].to_numpy(dtype="float64")
    train_matrix = xgb.DMatrix(train_x, label=train_y, feature_names=FEATURE_COLUMNS)
    validation_matrix = xgb.DMatrix(
        validation_x,
        label=validation_y,
        feature_names=FEATURE_COLUMNS,
    )
    del train_x, train_y, validation_x, validation_y
    gc.collect()

    for position, params in enumerate(XGB_GRID):
        current_id = candidate_id("xgboost", position)
        if current_id in completed_ids:
            print(f"Skipping completed candidate {current_id}.")
            continue

        print(f"Starting {current_id} ({position + 1}/{len(XGB_GRID)}): {params}")
        started = time.perf_counter()
        booster = xgb.train(
            params={
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "eta": params["learning_rate"],
                "max_depth": params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "seed": RANDOM_SEED,
                "nthread": workers,
            },
            dtrain=train_matrix,
            num_boost_round=XGB_MAX_ROUNDS,
            evals=[(validation_matrix, "validation")],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            verbose_eval=25,
        )
        best_iteration = int(booster.best_iteration)
        validation_log_prediction = booster.predict(
            validation_matrix,
            iteration_range=(0, best_iteration + 1),
        )
        metrics = validation_metrics(validation_actual, validation_log_prediction)
        elapsed = time.perf_counter() - started
        run["candidates"].append(
            {
                "candidate_id": current_id,
                "params": params,
                "best_rounds": best_iteration + 1,
                "early_stopping_log_rmse": float(booster.best_score),
                "validation_metrics": metrics,
                "fit_and_score_seconds": round(elapsed, 3),
                "completed_utc": utc_now(),
            }
        )
        write_json_atomic(result_path, run)
        print(
            f"Completed {current_id} in {elapsed / 60:.1f} minutes at "
            f"{best_iteration + 1} rounds: QLIKE={metrics['qlike']:.9f}, "
            f"RMSE={metrics['rmse']:.9f}, MAE={metrics['mae']:.9f}"
        )
        del booster, validation_log_prediction
        gc.collect()

    return finish_selection(run, result_path)


def require_completed_selection(
    model_name: str,
    grid: list[dict[str, Any]],
    matrix_hash: str,
    sample_counts: dict[str, int],
    library_name: str,
    library_version: str,
) -> tuple[dict[str, Any], Path]:
    result_path = RESULTS_DIR / f"{model_name}_selection.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"No completed tuning result at {result_path}. Run the tune stage first."
        )
    expected = new_run_record(
        model_name,
        grid,
        matrix_hash,
        sample_counts,
        library_name,
        library_version,
    )
    run = load_or_create_run(result_path, expected)
    expected_ids = {
        candidate_id(model_name, position) for position in range(len(grid))
    }
    completed_ids = {candidate["candidate_id"] for candidate in run["candidates"]}
    if run.get("selected") is None or completed_ids != expected_ids:
        raise RuntimeError("Tuning is incomplete; final fitting is not allowed yet.")
    return run, result_path


def finalize_random_forest(
    matrix,
    masks,
    workers: int,
    force: bool,
    matrix_hash: str,
) -> None:
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    run, result_path = require_completed_selection(
        "random_forest",
        RF_GRID,
        matrix_hash,
        masks.counts(),
        "scikit-learn",
        importlib.metadata.version("scikit-learn"),
    )
    model_path = ARTIFACTS_DIR / "random_forest_final.joblib"
    forecast_path = ARTIFACTS_DIR.parent / "forecasts_random_forest.parquet"
    if run.get("final_fit") and model_path.exists() and forecast_path.exists() and not force:
        print("Final Random Forest artifacts already exist; use --force-final to rebuild.")
        return

    final_x, final_y = extract_xy(matrix, masks.final_refit)
    test_x, _ = extract_xy(matrix, masks.test)
    params = run["selected"]["params"]
    print(
        f"Fitting final Random Forest on {len(final_y):,} rows and generating "
        f"{len(test_x):,} test forecasts. Test losses will not be calculated."
    )
    started = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=workers,
        verbose=1,
    )
    model.fit(final_x, final_y)
    forecast = volatility_from_log_prediction(model.predict(test_x))
    save_forecasts(
        matrix,
        masks.test,
        forecast,
        "forecast_random_forest",
        forecast_path,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    elapsed = time.perf_counter() - started
    run["final_fit"] = {
        "rows": int(masks.final_refit.sum()),
        "test_predictions": int(masks.test.sum()),
        "fit_predict_save_seconds": round(elapsed, 3),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "forecast_path": str(forecast_path.relative_to(PROJECT_ROOT)),
        "completed_utc": utc_now(),
    }
    write_json_atomic(result_path, run)
    print(f"Saved final model to {model_path}")
    print(f"Saved frozen test forecasts to {forecast_path}")


def finalize_xgboost(
    matrix,
    masks,
    workers: int,
    force: bool,
    matrix_hash: str,
) -> None:
    import xgboost as xgb

    run, result_path = require_completed_selection(
        "xgboost",
        XGB_GRID,
        matrix_hash,
        masks.counts(),
        "xgboost",
        importlib.metadata.version("xgboost"),
    )
    model_path = ARTIFACTS_DIR / "xgboost_final.ubj"
    forecast_path = ARTIFACTS_DIR.parent / "forecasts_xgboost.parquet"
    if run.get("final_fit") and model_path.exists() and forecast_path.exists() and not force:
        print("Final XGBoost artifacts already exist; use --force-final to rebuild.")
        return

    final_x, final_y = extract_xy(matrix, masks.final_refit)
    test_x, _ = extract_xy(matrix, masks.test)
    params = run["selected"]["params"]
    best_rounds = int(run["selected"]["best_rounds"])
    print(
        f"Fitting final XGBoost for {best_rounds} rounds on {len(final_y):,} rows "
        f"and generating {len(test_x):,} test forecasts. Test losses will not "
        "be calculated."
    )
    started = time.perf_counter()
    final_matrix = xgb.DMatrix(final_x, label=final_y, feature_names=FEATURE_COLUMNS)
    test_matrix = xgb.DMatrix(test_x, feature_names=FEATURE_COLUMNS)
    del final_x, final_y, test_x
    gc.collect()
    booster = xgb.train(
        params={
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eta": params["learning_rate"],
            "max_depth": params["max_depth"],
            "min_child_weight": params["min_child_weight"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "seed": RANDOM_SEED,
            "nthread": workers,
        },
        dtrain=final_matrix,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )
    forecast = volatility_from_log_prediction(booster.predict(test_matrix))
    save_forecasts(
        matrix,
        masks.test,
        forecast,
        "forecast_xgboost",
        forecast_path,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    elapsed = time.perf_counter() - started
    run["final_fit"] = {
        "rows": int(masks.final_refit.sum()),
        "boosting_rounds": best_rounds,
        "test_predictions": int(masks.test.sum()),
        "fit_predict_save_seconds": round(elapsed, 3),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "forecast_path": str(forecast_path.relative_to(PROJECT_ROOT)),
        "completed_utc": utc_now(),
    }
    write_json_atomic(result_path, run)
    print(f"Saved final model to {model_path}")
    print(f"Saved frozen test forecasts to {forecast_path}")


def print_preflight(matrix, masks, matrix_path: Path) -> None:
    print(f"Validated matrix: {matrix_path}")
    print(f"Rows: {len(matrix):,}; features: {len(FEATURE_COLUMNS)}")
    for label, count in masks.counts().items():
        print(f"{label}: {count:,}")
    reintroduced = int(
        (
            masks.final_refit
            & matrix["crosses_split_boundary"].to_numpy(bool)
            & matrix["split"].eq("train").to_numpy()
        ).sum()
    )
    print(f"Late-2012 rows re-admitted only for final refit: {reintroduced:,}")
    for package in ("scikit-learn", "xgboost", "shap"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        print(f"{package}: {version}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "tune", "final"))
    parser.add_argument("--model", choices=("random_forest", "xgboost"))
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="CPU threads/processes used by the estimator (default: CPU count minus 2).",
    )
    parser.add_argument(
        "--force-final",
        action="store_true",
        help="Rebuild already-completed final model and forecast artifacts.",
    )
    args = parser.parse_args()
    if args.stage != "check" and args.model is None:
        parser.error("--model is required for tune and final stages")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    matrix, masks = load_and_validate_matrix(args.matrix)
    if args.stage == "check":
        print_preflight(matrix, masks, args.matrix)
        return

    matrix_hash = matrix_sha256(args.matrix)
    print(f"Matrix SHA-256: {matrix_hash}")
    print(f"Workers: {args.workers}")
    if args.stage == "tune" and args.model == "random_forest":
        tune_random_forest(matrix, masks, args.workers, matrix_hash)
    elif args.stage == "tune" and args.model == "xgboost":
        tune_xgboost(matrix, masks, args.workers, matrix_hash)
    elif args.stage == "final" and args.model == "random_forest":
        finalize_random_forest(
            matrix, masks, args.workers, args.force_final, matrix_hash
        )
    elif args.stage == "final" and args.model == "xgboost":
        finalize_xgboost(matrix, masks, args.workers, args.force_final, matrix_hash)


if __name__ == "__main__":
    main()
