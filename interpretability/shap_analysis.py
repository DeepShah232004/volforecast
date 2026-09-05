"""Run the locked SHAP analysis for the frozen Random Forest and XGBoost models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ml.training_common import (
    FEATURE_COLUMNS,
    load_and_validate_matrix,
    read_json,
    write_json_atomic,
)


MATRIX_PATH = PROJECT_ROOT / "data/ml_feature_matrix_sp500_2005_2024.parquet"
PANEL_PATH = PROJECT_ROOT / "data/final_forecast_panel_sp500_2015_2024.parquet"
EVALUATION_PATH = PROJECT_ROOT / "evaluation/results/final_evaluation.json"
RF_SELECTION_PATH = PROJECT_ROOT / "models/ml/results/random_forest_selection.json"
XGB_SELECTION_PATH = PROJECT_ROOT / "models/ml/results/xgboost_selection.json"
RF_MODEL_PATH = PROJECT_ROOT / "models/ml/artifacts/random_forest_final.joblib"
XGB_MODEL_PATH = PROJECT_ROOT / "models/ml/artifacts/xgboost_final.ubj"
SUMMARY_PATH = PROJECT_ROOT / "interpretability/results/shap_summary.json"
VALUES_PATH = PROJECT_ROOT / "interpretability/results/shap_values.parquet"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42
ADDITIVITY_TOLERANCE = 1e-4
EXPECTED_POPULATION_ROWS = 1_206_821
EXPECTED_SAMPLE_KEY_SHA256 = (
    "5938bd766426e8b42b10b1f54a06bcb669f07e21032480a6054d035c62a6f185"
)
KEY_COLUMNS = ["permno", "dlycaldt"]
REGIME_ORDER = ["low", "middle", "high"]
MODEL_FORECAST_COLUMNS = {
    "random_forest": "forecast_random_forest",
    "xgboost": "forecast_xgboost",
}
FEATURE_GROUPS = {
    "trailing_volatility": [
        "return_std_5d",
        "return_std_21d",
        "return_std_63d",
        "return_std_126d",
        "return_std_252d",
        "mean_abs_return_5d",
        "mean_abs_return_21d",
        "mean_abs_return_63d",
        "ewma_volatility_094",
    ],
    "compounded_returns": [
        "compounded_return_5d",
        "compounded_return_21d",
        "compounded_return_63d",
    ],
    "vix_level": ["vix_close"],
    "vix_changes": [
        "vix_log_change_1d",
        "vix_log_change_5d",
        "vix_log_change_21d",
    ],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_feature_groups() -> None:
    grouped = [feature for features in FEATURE_GROUPS.values() for feature in features]
    if len(grouped) != len(set(grouped)):
        raise ValueError("A feature appears in more than one SHAP group.")
    if set(grouped) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS).difference(grouped))
        extra = sorted(set(grouped).difference(FEATURE_COLUMNS))
        raise ValueError(
            f"SHAP feature groups are incomplete: missing={missing}, extra={extra}"
        )


def allocate_stratum_counts(
    stratum_sizes: pd.Series,
    sample_size: int,
) -> pd.Series:
    """Allocate a proportional sample with one row per populated stratum."""
    sizes = stratum_sizes.astype("int64")
    if sizes.empty or (sizes <= 0).any():
        raise ValueError("Every supplied stratum must be populated.")
    if sample_size < len(sizes):
        raise ValueError("Sample size is too small to represent every stratum.")
    if sample_size > int(sizes.sum()):
        raise ValueError("Sample size exceeds the available population.")
    if sample_size == int(sizes.sum()):
        return sizes.copy()

    allocation = pd.Series(1, index=sizes.index, dtype="int64")
    remaining = sample_size - len(sizes)
    if remaining == 0:
        return allocation

    capacity = sizes - allocation
    quotas = remaining * capacity / int(capacity.sum())
    floors = np.floor(quotas).astype("int64")
    allocation += floors
    leftover = sample_size - int(allocation.sum())

    if leftover:
        candidates = pd.DataFrame(
            {
                "fraction": (quotas - floors).to_numpy(dtype="float64"),
                "position": np.arange(len(sizes)),
                "capacity_left": (sizes - allocation).to_numpy(dtype="int64"),
            },
            index=sizes.index,
        )
        candidates = candidates.loc[candidates["capacity_left"].gt(0)].sort_values(
            ["fraction", "position"],
            ascending=[False, True],
            kind="stable",
        )
        if leftover > len(candidates):
            raise RuntimeError("Largest-remainder allocation did not converge.")
        allocation.loc[candidates.index[:leftover]] += 1

    if int(allocation.sum()) != sample_size or (allocation > sizes).any():
        raise RuntimeError("Invalid stratified-sample allocation.")
    return allocation


def build_analysis_population(
    matrix: pd.DataFrame,
    test_mask: np.ndarray,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Join locked features to the exact five-model common test population."""
    matrix_required = {*KEY_COLUMNS, *FEATURE_COLUMNS}
    panel_required = {
        *KEY_COLUMNS,
        "vix_regime",
        *MODEL_FORECAST_COLUMNS.values(),
    }
    missing_matrix = matrix_required.difference(matrix.columns)
    missing_panel = panel_required.difference(panel.columns)
    if missing_matrix:
        raise KeyError(f"ML matrix is missing columns: {sorted(missing_matrix)}")
    if missing_panel:
        raise KeyError(f"Final panel is missing columns: {sorted(missing_panel)}")
    if len(test_mask) != len(matrix):
        raise ValueError("Test mask length does not match the ML matrix.")

    features = matrix.loc[test_mask, KEY_COLUMNS + FEATURE_COLUMNS].copy()
    locked_panel = panel[
        KEY_COLUMNS + ["vix_regime", *MODEL_FORECAST_COLUMNS.values()]
    ].copy()
    for frame, label in ((features, "ML test features"), (locked_panel, "Final panel")):
        frame["dlycaldt"] = pd.to_datetime(frame["dlycaldt"])
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{label} contains duplicate security-date keys.")

    if not locked_panel["vix_regime"].isin(REGIME_ORDER).all():
        raise ValueError("Final panel contains an invalid VIX regime.")
    population = locked_panel.merge(
        features,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not population["_merge"].eq("both").all():
        missing_count = int(population["_merge"].ne("both").sum())
        raise ValueError(
            f"{missing_count} final-panel rows are absent from the ML test matrix."
        )
    population = population.drop(columns="_merge")

    numeric = population[
        FEATURE_COLUMNS + list(MODEL_FORECAST_COLUMNS.values())
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise ValueError("The SHAP population contains non-finite features or forecasts.")
    if (population[list(MODEL_FORECAST_COLUMNS.values())] <= 0).any().any():
        raise ValueError("The SHAP population contains a non-positive forecast.")

    population["year"] = population["dlycaldt"].dt.year.astype("int16")
    if not population["year"].between(2015, 2024).all():
        raise ValueError("The SHAP population is not confined to the locked test years.")
    population["vix_regime"] = pd.Categorical(
        population["vix_regime"], categories=REGIME_ORDER, ordered=True
    )
    return population.sort_values(KEY_COLUMNS).reset_index(drop=True)


def draw_stratified_sample(
    population: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the fixed year-by-VIX-regime sample without replacement."""
    required = {*KEY_COLUMNS, "year", "vix_regime"}
    missing = required.difference(population.columns)
    if missing:
        raise KeyError(f"Sampling population is missing columns: {sorted(missing)}")
    ordered = population.sort_values(KEY_COLUMNS).reset_index(drop=True)
    sizes = ordered.groupby(["year", "vix_regime"], observed=True).size()
    allocation = allocate_stratum_counts(sizes, sample_size)
    rng = np.random.default_rng(seed)
    selected_positions: list[np.ndarray] = []
    records: list[dict[str, int | str]] = []

    for (year, regime), population_count in sizes.items():
        requested = int(allocation.loc[(year, regime)])
        positions = np.flatnonzero(
            ordered["year"].eq(year).to_numpy()
            & ordered["vix_regime"].eq(regime).to_numpy()
        )
        selected_positions.append(rng.choice(positions, size=requested, replace=False))
        records.append(
            {
                "year": int(year),
                "vix_regime": str(regime),
                "population_rows": int(population_count),
                "sample_rows": requested,
            }
        )

    sample = ordered.iloc[np.concatenate(selected_positions)].copy()
    sample = sample.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if len(sample) != sample_size or sample.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("The fixed SHAP sample failed size or uniqueness checks.")
    strata = pd.DataFrame.from_records(records)
    return sample, strata


def sample_key_sha256(sample: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    keys = sample[KEY_COLUMNS + ["vix_regime"]].sort_values(KEY_COLUMNS)
    for row in keys.itertuples(index=False):
        line = f"{int(row.permno)}|{pd.Timestamp(row.dlycaldt).date()}|{row.vix_regime}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def normalize_explanation(
    values: np.ndarray,
    base_values: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    shap_values = np.asarray(values, dtype="float64")
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != (n_rows, len(FEATURE_COLUMNS)):
        raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")

    bases = np.asarray(base_values, dtype="float64")
    if bases.size == 1:
        bases = np.full(n_rows, float(bases.reshape(-1)[0]))
    else:
        bases = bases.reshape(n_rows, -1)
        if bases.shape[1] != 1:
            raise ValueError(f"Unexpected SHAP base-value shape: {bases.shape}")
        bases = bases[:, 0]
    if not np.isfinite(shap_values).all() or not np.isfinite(bases).all():
        raise ValueError("SHAP returned non-finite values.")
    return shap_values, bases


def compute_tree_shap(
    model_name: str,
    model,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact TreeSHAP values, base values, and model log predictions."""
    import shap

    feature_values = np.asarray(features, dtype="float32")
    if model_name == "random_forest":
        predictions = np.asarray(model.predict(feature_values), dtype="float64")
    elif model_name == "xgboost":
        import xgboost as xgb

        predictions = np.asarray(
            model.predict(xgb.DMatrix(feature_values, feature_names=FEATURE_COLUMNS)),
            dtype="float64",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    explanation = shap.TreeExplainer(model)(feature_values, check_additivity=True)
    shap_values, base_values = normalize_explanation(
        explanation.values, explanation.base_values, len(feature_values)
    )
    reconstructed = base_values + shap_values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstructed - predictions)))
    if max_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"{model_name} SHAP additivity error {max_error:.8g} exceeds "
            f"{ADDITIVITY_TOLERANCE:.8g}."
        )
    return shap_values, base_values, predictions


def ranked_feature_importance(values: np.ndarray) -> list[dict[str, float | int | str]]:
    means = np.mean(np.abs(values), axis=0)
    signed_means = np.mean(values, axis=0)
    total = float(means.sum())
    if total <= 0:
        raise ValueError("SHAP feature importance is identically zero.")
    order = sorted(
        range(len(FEATURE_COLUMNS)),
        key=lambda index: (-means[index], FEATURE_COLUMNS[index]),
    )
    return [
        {
            "rank": rank,
            "feature": FEATURE_COLUMNS[index],
            "mean_absolute_shap": float(means[index]),
            "normalized_share": float(means[index] / total),
            "mean_signed_shap": float(signed_means[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def ranked_group_importance(values: np.ndarray) -> list[dict[str, object]]:
    feature_index = {feature: index for index, feature in enumerate(FEATURE_COLUMNS)}
    records: list[dict[str, object]] = []
    for group, features in FEATURE_GROUPS.items():
        indices = [feature_index[feature] for feature in features]
        gross_importance = float(np.mean(np.abs(values[:, indices]), axis=0).sum())
        net_importance = float(np.mean(np.abs(values[:, indices].sum(axis=1))))
        records.append(
            {
                "group": group,
                "features": features,
                "mean_absolute_shap": gross_importance,
                "mean_absolute_net_contribution": net_importance,
            }
        )
    total = sum(float(record["mean_absolute_shap"]) for record in records)
    if total <= 0:
        raise ValueError("SHAP group importance is identically zero.")
    records.sort(
        key=lambda record: (
            -float(record["mean_absolute_shap"]),
            str(record["group"]),
        )
    )
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        record["normalized_share"] = float(record["mean_absolute_shap"]) / total
    return records


def rank_correlation(first: list[dict], second: list[dict], key: str) -> float | None:
    first_ranks = {str(record[key]): int(record["rank"]) for record in first}
    second_ranks = {str(record[key]): int(record["rank"]) for record in second}
    labels = sorted(first_ranks)
    if labels != sorted(second_ranks):
        raise ValueError("Cannot compare rankings with different labels.")
    left = pd.Series([first_ranks[label] for label in labels], dtype="float64")
    right = pd.Series([second_ranks[label] for label in labels], dtype="float64")
    correlation = left.corr(right, method="pearson")
    return None if pd.isna(correlation) else float(correlation)


def top_overlap(
    first: list[dict],
    second: list[dict],
    key: str,
    n: int,
) -> dict[str, object]:
    left = {str(record[key]) for record in first[:n]}
    right = {str(record[key]) for record in second[:n]}
    union = left | right
    return {
        "n": n,
        "shared": sorted(left & right),
        "jaccard": float(len(left & right) / len(union)),
    }


def summarize_model(values: np.ndarray, years: np.ndarray) -> dict[str, object]:
    segments = {
        "overall": np.ones(len(years), dtype=bool),
        "2015_2019": (years >= 2015) & (years <= 2019),
        "2020_2024": (years >= 2020) & (years <= 2024),
    }
    summaries: dict[str, dict[str, object]] = {}
    for label, mask in segments.items():
        if not mask.any():
            raise ValueError(f"SHAP stability segment {label} is empty.")
        summaries[label] = {
            "rows": int(mask.sum()),
            "features": ranked_feature_importance(values[mask]),
            "groups": ranked_group_importance(values[mask]),
        }

    early = summaries["2015_2019"]
    late = summaries["2020_2024"]
    stability = {
        "feature_rank_spearman": rank_correlation(
            early["features"], late["features"], "feature"
        ),
        "feature_top_5_overlap": top_overlap(
            early["features"], late["features"], "feature", 5
        ),
        "group_rank_spearman": rank_correlation(
            early["groups"], late["groups"], "group"
        ),
        "group_top_2_overlap": top_overlap(
            early["groups"], late["groups"], "group", 2
        ),
    }
    return {"segments": summaries, "stability": stability}


def validate_locked_metadata() -> dict[str, str]:
    required_paths = [
        MATRIX_PATH,
        PANEL_PATH,
        EVALUATION_PATH,
        RF_SELECTION_PATH,
        XGB_SELECTION_PATH,
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required locked artifacts are missing: {missing}")

    matrix_hash = file_sha256(MATRIX_PATH)
    panel_hash = file_sha256(PANEL_PATH)
    evaluation = read_json(EVALUATION_PATH)
    if evaluation.get("panel_sha256") != panel_hash:
        raise ValueError("Final panel hash does not match the locked evaluation result.")

    for path in (RF_SELECTION_PATH, XGB_SELECTION_PATH):
        selection = read_json(path)
        if selection.get("matrix_sha256") != matrix_hash:
            raise ValueError(f"Matrix hash does not match {path.name}.")
        if selection.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError(f"Feature order does not match {path.name}.")

    return {
        "matrix_sha256": matrix_hash,
        "panel_sha256": panel_hash,
        "random_forest_model_sha256": file_sha256(RF_MODEL_PATH),
        "xgboost_model_sha256": file_sha256(XGB_MODEL_PATH),
        "random_forest_selection_sha256": file_sha256(RF_SELECTION_PATH),
        "xgboost_selection_sha256": file_sha256(XGB_SELECTION_PATH),
        "evaluation_sha256": file_sha256(EVALUATION_PATH),
    }


def load_population_and_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix, masks = load_and_validate_matrix(MATRIX_PATH)
    panel = pd.read_parquet(
        PANEL_PATH,
        columns=KEY_COLUMNS + ["vix_regime", *MODEL_FORECAST_COLUMNS.values()],
    )
    population = build_analysis_population(matrix, masks.test, panel)
    del matrix, panel
    sample, strata = draw_stratified_sample(population)
    if len(population) != EXPECTED_POPULATION_ROWS:
        raise ValueError(
            f"Locked SHAP population has {len(population):,} rows; expected "
            f"{EXPECTED_POPULATION_ROWS:,}."
        )
    observed_hash = sample_key_sha256(sample)
    if observed_hash != EXPECTED_SAMPLE_KEY_SHA256:
        raise ValueError(
            f"Locked SHAP sample hash is {observed_hash}; expected "
            f"{EXPECTED_SAMPLE_KEY_SHA256}."
        )
    return population, sample, strata


def load_frozen_models():
    import joblib
    import xgboost as xgb

    random_forest = joblib.load(RF_MODEL_PATH)
    if int(random_forest.n_features_in_) != len(FEATURE_COLUMNS):
        raise ValueError("Random Forest artifact has the wrong feature count.")
    xgboost = xgb.Booster()
    xgboost.load_model(XGB_MODEL_PATH)
    if xgboost.feature_names != FEATURE_COLUMNS:
        raise ValueError("XGBoost artifact has the wrong feature order.")
    return {"random_forest": random_forest, "xgboost": xgboost}


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_analysis(
    sample: pd.DataFrame,
    strata: pd.DataFrame,
    hashes: dict[str, str],
) -> dict[str, object]:
    models = load_frozen_models()
    feature_values = sample[FEATURE_COLUMNS].to_numpy(dtype="float32", copy=True)
    years = sample["year"].to_numpy(dtype="int16")
    detailed = sample[
        KEY_COLUMNS
        + ["year", "vix_regime", *MODEL_FORECAST_COLUMNS.values()]
        + FEATURE_COLUMNS
    ].copy()
    model_results: dict[str, object] = {}

    for model_name, model in models.items():
        print(f"Computing exact TreeSHAP values for {model_name}...")
        values, base_values, log_predictions = compute_tree_shap(
            model_name, model, feature_values
        )
        forecast = np.exp(log_predictions)
        locked_forecast = sample[MODEL_FORECAST_COLUMNS[model_name]].to_numpy(
            dtype="float64"
        )
        if not np.allclose(forecast, locked_forecast, rtol=1e-6, atol=1e-10):
            raise RuntimeError(
                f"{model_name} artifact predictions do not reproduce frozen forecasts."
            )
        reconstructed = base_values + values.sum(axis=1)
        model_summary = summarize_model(values, years)
        model_summary.update(
            {
                "output_scale": "log forward 21-trading-day volatility",
                "max_absolute_additivity_error": float(
                    np.max(np.abs(reconstructed - log_predictions))
                ),
                "max_absolute_frozen_forecast_error": float(
                    np.max(np.abs(forecast - locked_forecast))
                ),
            }
        )
        model_results[model_name] = model_summary
        detailed[f"{model_name}_base_value"] = base_values
        detailed[f"{model_name}_log_prediction"] = log_predictions
        for index, feature in enumerate(FEATURE_COLUMNS):
            detailed[f"{model_name}_shap__{feature}"] = values[:, index]

    write_parquet_atomic(VALUES_PATH, detailed)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "algorithm": "exact TreeSHAP",
            "model_output": "log forward 21-trading-day volatility",
            "sample_design": "proportional year-by-VIX-regime sampling without replacement, with at least one row per populated stratum",
            "sample_size": SAMPLE_SIZE,
            "seed": RANDOM_SEED,
            "stability_periods": ["2015-2019", "2020-2024"],
            "feature_groups": FEATURE_GROUPS,
            "group_importance": "sum of member-feature mean absolute SHAP values; mean absolute net grouped contributions are also reported",
            "interpretation": "descriptive, not causal",
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
        "inputs": {
            "matrix_path": str(MATRIX_PATH.relative_to(PROJECT_ROOT)),
            "panel_path": str(PANEL_PATH.relative_to(PROJECT_ROOT)),
            "random_forest_model_path": str(RF_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "xgboost_model_path": str(XGB_MODEL_PATH.relative_to(PROJECT_ROOT)),
            **hashes,
        },
        "population": {
            "rows": int(strata["population_rows"].sum()),
            "strata": strata.to_dict(orient="records"),
        },
        "sample": {
            "rows": int(len(sample)),
            "permnos": int(sample["permno"].nunique()),
            "origin_start": str(sample["dlycaldt"].min().date()),
            "origin_end": str(sample["dlycaldt"].max().date()),
            "key_sha256": sample_key_sha256(sample),
            "values_path": str(VALUES_PATH.relative_to(PROJECT_ROOT)),
        },
        "models": model_results,
    }
    write_json_atomic(SUMMARY_PATH, result)
    return result


def print_preflight(population: pd.DataFrame, sample: pd.DataFrame, strata: pd.DataFrame) -> None:
    print(f"Validated common SHAP population: {len(population):,} rows")
    print(
        f"Fixed sample: {len(sample):,} rows, {sample['permno'].nunique():,} PERMNOs, "
        f"key SHA-256 {sample_key_sha256(sample)}"
    )
    print("Sample rows by year and VIX regime:")
    table = strata.pivot(index="year", columns="vix_regime", values="sample_rows")
    print(table.reindex(columns=REGIME_ORDER).fillna(0).astype(int).to_string())
    print("No SHAP values were calculated.")


def print_results(results: dict[str, object]) -> None:
    for model_name, model_result in results["models"].items():
        print(f"{model_name} overall SHAP groups")
        for record in model_result["segments"]["overall"]["groups"]:
            print(
                f"  {record['rank']}. {record['group']}: "
                f"{100 * record['normalized_share']:.2f}%"
            )
        stability = model_result["stability"]
        print(
            "  2015-2019 vs 2020-2024 feature-rank Spearman: "
            f"{stability['feature_rank_spearman']:.4f}"
        )
    print(f"Saved aggregate results to {SUMMARY_PATH}")
    print(f"Saved row-level SHAP values to {VALUES_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "analyze"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing SHAP outputs only for a documented rerun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_feature_groups()
    hashes = validate_locked_metadata()
    population, sample, strata = load_population_and_sample()
    print_preflight(population, sample, strata)
    if args.stage == "check":
        return
    existing = [path for path in (SUMMARY_PATH, VALUES_PATH) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"SHAP output already exists: {[str(path) for path in existing]}; "
            "use --force only for a documented rerun."
        )
    del population
    results = run_analysis(sample, strata, hashes)
    print_results(results)


if __name__ == "__main__":
    main()
