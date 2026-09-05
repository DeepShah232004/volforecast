"""Shared data validation and bookkeeping for Phase 3 model training."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_ml_features import FEATURE_COLUMNS, VALIDATION_END
from evaluation.metrics import mae, qlike, rmse


RANDOM_SEED = 42
MATRIX_PATH = PROJECT_ROOT / "data/ml_feature_matrix_sp500_2005_2024.parquet"
RESULTS_DIR = PROJECT_ROOT / "models/ml/results"
ARTIFACTS_DIR = PROJECT_ROOT / "models/ml/artifacts"

MATRIX_COLUMNS = [
    "permno",
    "gvkey",
    "dlycaldt",
    "in_sp500_membership",
    *FEATURE_COLUMNS,
    "target_forward_21d",
    "log_target_forward_21d",
    "target_end_date",
    "split",
    "crosses_split_boundary",
    "eligible_ml",
]


@dataclass(frozen=True)
class SampleMasks:
    tuning_train: np.ndarray
    validation: np.ndarray
    final_refit: np.ndarray
    test: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "tuning_train": int(self.tuning_train.sum()),
            "validation": int(self.validation.sum()),
            "final_refit": int(self.final_refit.sum()),
            "test": int(self.test.sum()),
        }


def matrix_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_matrix(path: Path = MATRIX_PATH) -> tuple[pd.DataFrame, SampleMasks]:
    """Load the canonical matrix and enforce all sample-boundary invariants."""
    if not path.exists():
        raise FileNotFoundError(
            f"ML feature matrix not found at {path}. Run data/build_ml_features.py first."
        )

    matrix = pd.read_parquet(path, columns=MATRIX_COLUMNS)
    matrix["dlycaldt"] = pd.to_datetime(matrix["dlycaldt"])
    matrix["target_end_date"] = pd.to_datetime(matrix["target_end_date"])

    duplicate_count = int(matrix.duplicated(["permno", "dlycaldt"]).sum())
    if duplicate_count:
        raise ValueError(f"ML matrix contains {duplicate_count} duplicate keys.")
    if not matrix["in_sp500_membership"].fillna(False).all():
        raise ValueError("ML matrix contains non-membership forecast origins.")
    if matrix["split"].isna().any():
        raise ValueError("ML matrix contains dates outside the locked data splits.")

    model_values = matrix[FEATURE_COLUMNS + ["log_target_forward_21d"]].to_numpy(
        dtype="float64"
    )
    finite = np.isfinite(model_values).all(axis=1)
    expected_eligible = finite & ~matrix["crosses_split_boundary"].to_numpy(bool)
    stored_eligible = matrix["eligible_ml"].to_numpy(bool)
    if not np.array_equal(expected_eligible, stored_eligible):
        mismatch_count = int(np.count_nonzero(expected_eligible != stored_eligible))
        raise ValueError(
            f"eligible_ml disagrees with locked rules on {mismatch_count} rows."
        )

    positive_target = matrix["target_forward_21d"].gt(0).to_numpy()
    if not positive_target[finite].all():
        raise ValueError("Finite modeling rows contain a non-positive volatility target.")
    if not np.allclose(
        matrix.loc[finite, "log_target_forward_21d"].to_numpy(),
        np.log(matrix.loc[finite, "target_forward_21d"].to_numpy()),
    ):
        raise ValueError("Saved log targets do not match the volatility targets.")

    split = matrix["split"]
    tuning_train = stored_eligible & split.eq("train").to_numpy()
    validation = stored_eligible & split.eq("validation").to_numpy()
    test = stored_eligible & split.eq("test").to_numpy()

    # Once hyperparameters are selected, the train/validation boundary is no
    # longer relevant. Re-admit finite late-2012 rows whose labels end in 2013,
    # but never admit a label reaching the test period.
    final_refit = (
        finite
        & matrix["dlycaldt"].lt(VALIDATION_END).to_numpy()
        & matrix["target_end_date"].lt(VALIDATION_END).to_numpy()
    )

    if matrix.loc[tuning_train, "target_end_date"].ge("2013-01-01").any():
        raise ValueError("A tuning-training target crosses into validation.")
    if matrix.loc[validation, "target_end_date"].ge(VALIDATION_END).any():
        raise ValueError("A validation target crosses into the test period.")
    if matrix.loc[final_refit, "target_end_date"].ge(VALIDATION_END).any():
        raise ValueError("A final-refit target crosses into the test period.")
    if matrix.loc[test, "dlycaldt"].lt(VALIDATION_END).any():
        raise ValueError("The test sample contains a pre-2015 origin.")

    masks = SampleMasks(tuning_train, validation, final_refit, test)
    if any(count == 0 for count in masks.counts().values()):
        raise ValueError(f"At least one locked sample is empty: {masks.counts()}")
    return matrix, masks


def extract_xy(
    matrix: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact float32 features and log-volatility labels."""
    features = matrix.loc[mask, FEATURE_COLUMNS].to_numpy(dtype="float32", copy=True)
    target = matrix.loc[mask, "log_target_forward_21d"].to_numpy(
        dtype="float32", copy=True
    )
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("A selected modeling sample contains non-finite values.")
    return features, target


def volatility_from_log_prediction(log_prediction: np.ndarray) -> np.ndarray:
    forecast = np.exp(np.asarray(log_prediction, dtype="float64"))
    if not np.isfinite(forecast).all() or np.any(forecast <= 0):
        raise ValueError("Model produced a non-finite or non-positive forecast.")
    return forecast


def validation_metrics(
    actual_volatility: np.ndarray,
    log_prediction: np.ndarray,
) -> dict[str, float]:
    forecast = volatility_from_log_prediction(log_prediction)
    actual = np.asarray(actual_volatility, dtype="float64")
    return {
        "qlike": float(qlike(actual, forecast)),
        "rmse": float(rmse(actual, forecast)),
        "mae": float(mae(actual, forecast)),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_forecasts(
    matrix: pd.DataFrame,
    test_mask: np.ndarray,
    forecast: np.ndarray,
    forecast_column: str,
    output_path: Path,
) -> None:
    if len(forecast) != int(test_mask.sum()):
        raise ValueError("Forecast length does not match the locked test sample.")
    if not np.isfinite(forecast).all() or np.any(forecast <= 0):
        raise ValueError("Refusing to save invalid volatility forecasts.")

    output = matrix.loc[
        test_mask,
        ["permno", "gvkey", "dlycaldt", "target_forward_21d"],
    ].copy()
    output[forecast_column] = forecast
    if output.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Final forecasts contain duplicate security-date keys.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    saved = pd.read_parquet(output_path)
    if len(saved) != len(output):
        raise RuntimeError("Saved forecasts failed round-trip row-count validation.")
