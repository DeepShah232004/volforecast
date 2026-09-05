"""Build the locked common-sample panel for all five forecast models."""

from pathlib import Path

import numpy as np
import pandas as pd


CLASSICAL_PATH = Path("data/classical_forecast_panel_sp500_2015_2024.parquet")
RANDOM_FOREST_PATH = Path("models/ml/forecasts_random_forest.parquet")
XGBOOST_PATH = Path("models/ml/forecasts_xgboost.parquet")
OUTPUT_PATH = Path("data/final_forecast_panel_sp500_2015_2024.parquet")

KEY_COLUMNS = ["permno", "dlycaldt"]
MODEL_FORECAST_COLUMNS = {
    "persistence": "forecast_naive",
    "ewma": "forecast_ewma",
    "garch": "forecast_garch",
    "random_forest": "forecast_random_forest",
    "xgboost": "forecast_xgboost",
}


def require_unique_keys(frame: pd.DataFrame, label: str) -> None:
    duplicates = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate keys.")


def require_same_ml_population(
    random_forest: pd.DataFrame,
    xgboost: pd.DataFrame,
) -> None:
    """Require both ML forecast files to contain identical keys and targets."""
    require_unique_keys(random_forest, "Random Forest forecasts")
    require_unique_keys(xgboost, "XGBoost forecasts")
    comparison = random_forest[
        KEY_COLUMNS + ["target_forward_21d"]
    ].merge(
        xgboost[KEY_COLUMNS + ["target_forward_21d"]],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_rf", "_xgb"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Random Forest and XGBoost forecast keys differ.")
    if not np.allclose(
        comparison["target_forward_21d_rf"],
        comparison["target_forward_21d_xgb"],
        equal_nan=True,
    ):
        raise ValueError("Random Forest and XGBoost targets differ.")


def build_final_panel(
    classical: pd.DataFrame,
    random_forest: pd.DataFrame,
    xgboost: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact common population without calculating forecast losses."""
    classical_required = {
        *KEY_COLUMNS,
        "gvkey",
        "target_forward_21d",
        "target_end_date",
        "vix_close",
        "vix_regime",
        "earnings_count_forward_21d",
        "has_earnings_forward_21d",
        "common_valid",
        "forecast_naive",
        "forecast_ewma",
        "forecast_garch",
    }
    rf_required = {
        *KEY_COLUMNS,
        "target_forward_21d",
        "forecast_random_forest",
    }
    xgb_required = {
        *KEY_COLUMNS,
        "target_forward_21d",
        "forecast_xgboost",
    }
    for label, frame, required in (
        ("Classical panel", classical, classical_required),
        ("Random Forest forecasts", random_forest, rf_required),
        ("XGBoost forecasts", xgboost, xgb_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} is missing columns: {sorted(missing)}")
        require_unique_keys(frame, label)

    require_same_ml_population(random_forest, xgboost)
    panel = classical.merge(
        random_forest[
            KEY_COLUMNS + ["target_forward_21d", "forecast_random_forest"]
        ],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_rf"),
        validate="one_to_one",
    ).merge(
        xgboost[KEY_COLUMNS + ["target_forward_21d", "forecast_xgboost"]],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_xgb"),
        validate="one_to_one",
    )

    if not np.allclose(
        panel["target_forward_21d"],
        panel["target_forward_21d_rf"],
        equal_nan=True,
    ) or not np.allclose(
        panel["target_forward_21d"],
        panel["target_forward_21d_xgb"],
        equal_nan=True,
    ):
        raise ValueError("ML targets do not match the classical panel target.")
    panel = panel.loc[panel["common_valid"]].copy()
    panel = panel.drop(
        columns=["target_forward_21d_rf", "target_forward_21d_xgb", "common_valid"]
    )

    required_complete = [
        "target_forward_21d",
        "target_end_date",
        "vix_close",
        "vix_regime",
        "earnings_count_forward_21d",
        "has_earnings_forward_21d",
        *MODEL_FORECAST_COLUMNS.values(),
    ]
    if panel[required_complete].isna().any().any():
        missing = panel[required_complete].isna().sum()
        raise ValueError(
            "Final common panel contains missing required values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )

    numeric_columns = ["target_forward_21d", *MODEL_FORECAST_COLUMNS.values()]
    numeric_values = panel[numeric_columns].to_numpy(dtype="float64")
    if not np.isfinite(numeric_values).all():
        raise ValueError("Final common panel contains non-finite targets or forecasts.")
    if np.any(numeric_values <= 0):
        raise ValueError("Final common panel requires positive targets and forecasts.")
    if not panel["vix_regime"].isin(["low", "middle", "high"]).all():
        raise ValueError("Final common panel contains an invalid VIX regime.")

    output_columns = [
        "permno",
        "gvkey",
        "dlycaldt",
        "target_end_date",
        "target_forward_21d",
        *MODEL_FORECAST_COLUMNS.values(),
        "vix_close",
        "vix_regime",
        "earnings_count_forward_21d",
        "has_earnings_forward_21d",
    ]
    panel = panel[output_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    require_unique_keys(panel, "Final all-model panel")
    return panel


def main() -> None:
    classical = pd.read_parquet(CLASSICAL_PATH)
    random_forest = pd.read_parquet(RANDOM_FOREST_PATH)
    xgboost = pd.read_parquet(XGBOOST_PATH)
    panel = build_final_panel(classical, random_forest, xgboost)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH)
    require_unique_keys(saved, "Saved all-model panel")
    if len(saved) != len(panel):
        raise RuntimeError("Saved panel failed round-trip row-count validation.")

    print(f"Saved all-model rows: {len(saved):,}")
    print(f"PERMNOs: {saved['permno'].nunique():,}")
    print(
        f"Origin dates: {saved['dlycaldt'].min().date()} through "
        f"{saved['dlycaldt'].max().date()}"
    )
    print("VIX-regime row counts (no losses calculated):")
    print(saved["vix_regime"].value_counts().sort_index().to_string())
    print("Earnings-window row counts (no losses calculated):")
    print(saved["has_earnings_forward_21d"].value_counts().to_string())
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
