"""Build the point-in-time feature matrix for the locked Phase 3 ML models.

Features are computed on each security's complete CRSP history through forecast
origin t.  Only origins that occurred during an actual S&P 500 membership spell
are retained for pooled model training and evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_HORIZON = 21
EWMA_LAMBDA = 0.94
EWMA_BURN_IN = 21

TRAIN_END = pd.Timestamp("2013-01-01")
VALIDATION_END = pd.Timestamp("2015-01-01")

TARGET_PATH = Path("data/target_variable_sp500_2005_2024.parquet")
VIX_PATH = Path("data/vix_2005_2024.parquet")
OUTPUT_PATH = Path("data/ml_feature_matrix_sp500_2005_2024.parquet")

KEY_COLUMNS = ["permno", "dlycaldt"]
RETURN_STD_WINDOWS = (5, 21, 63, 126, 252)
MEAN_ABS_RETURN_WINDOWS = (5, 21, 63)
COMPOUNDED_RETURN_WINDOWS = (5, 21, 63)
VIX_CHANGE_LAGS = (1, 5, 21)

FEATURE_COLUMNS = [
    *(f"return_std_{window}d" for window in RETURN_STD_WINDOWS),
    *(f"mean_abs_return_{window}d" for window in MEAN_ABS_RETURN_WINDOWS),
    *(f"compounded_return_{window}d" for window in COMPOUNDED_RETURN_WINDOWS),
    "ewma_volatility_094",
    "vix_close",
    *(f"vix_log_change_{lag}d" for lag in VIX_CHANGE_LAGS),
]

OUTPUT_COLUMNS = [
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


def require_unique_keys(frame: pd.DataFrame, label: str) -> None:
    duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} contains {duplicate_count} duplicate (PERMNO, date) keys."
        )


def add_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add locked trailing return features using information through each row."""
    required_columns = set(KEY_COLUMNS + ["dlyret"])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Return data are missing columns: {sorted(missing_columns)}")

    result = frame.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    require_unique_keys(result, "Return data")
    result["dlycaldt"] = pd.to_datetime(result["dlycaldt"])
    returns = pd.to_numeric(result["dlyret"], errors="coerce").astype("float64")
    result["dlyret"] = returns

    grouped_returns = result.groupby("permno", sort=False)["dlyret"]
    for window in RETURN_STD_WINDOWS:
        result[f"return_std_{window}d"] = grouped_returns.transform(
            lambda values, window=window: values.rolling(
                window=window,
                min_periods=window,
            ).std()
        )

    result["_absolute_return"] = returns.abs()
    grouped_absolute_returns = result.groupby("permno", sort=False)[
        "_absolute_return"
    ]
    for window in MEAN_ABS_RETURN_WINDOWS:
        result[f"mean_abs_return_{window}d"] = (
            grouped_absolute_returns.transform(
                lambda values, window=window: values.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
        )

    gross_returns = 1.0 + returns
    if (gross_returns < 0).any():
        raise ValueError("A daily return below -100% cannot be compounded.")

    # Log sums are equivalent to multiplying gross returns and avoid a slow
    # Python product for every rolling window.  A -100% return is handled as
    # an exact zero gross return rather than taking log(0).
    result["_log_gross_return"] = np.log(gross_returns.where(gross_returns > 0))
    result["_zero_gross_return"] = gross_returns.eq(0).astype("int8")
    grouped_log_gross = result.groupby("permno", sort=False)["_log_gross_return"]
    grouped_zero_gross = result.groupby("permno", sort=False)["_zero_gross_return"]
    for window in COMPOUNDED_RETURN_WINDOWS:
        rolling_log_sum = grouped_log_gross.transform(
            lambda values, window=window: values.rolling(
                window=window,
                min_periods=window,
            ).sum()
        )
        rolling_zero_count = grouped_zero_gross.transform(
            lambda values, window=window: values.rolling(
                window=window,
                min_periods=window,
            ).sum()
        )
        compounded = np.expm1(rolling_log_sum)
        compounded.loc[rolling_zero_count.gt(0)] = -1.0
        result[f"compounded_return_{window}d"] = compounded

    ewma_variance = grouped_returns.transform(
        lambda values: values.pow(2).ewm(
            alpha=1 - EWMA_LAMBDA,
            adjust=False,
        ).mean()
    )
    result["ewma_volatility_094"] = np.sqrt(ewma_variance)
    valid_return_count = grouped_returns.transform(lambda values: values.notna().cumsum())
    result.loc[
        valid_return_count.lt(EWMA_BURN_IN),
        "ewma_volatility_094",
    ] = np.nan

    return result.drop(
        columns=["_absolute_return", "_log_gross_return", "_zero_gross_return"]
    )


def build_vix_features(
    vix: pd.DataFrame,
    crsp_dates: pd.Series,
) -> pd.DataFrame:
    """Create origin VIX features on the CRSP trading-day calendar."""
    required_columns = {"date", "vix_close"}
    missing_columns = required_columns.difference(vix.columns)
    if missing_columns:
        raise KeyError(f"VIX data are missing columns: {sorted(missing_columns)}")

    vix_features = vix[["date", "vix_close"]].copy()
    vix_features["date"] = pd.to_datetime(vix_features["date"])
    if vix_features["date"].duplicated().any():
        raise ValueError("VIX data contain duplicate dates.")
    vix_features["vix_close"] = pd.to_numeric(
        vix_features["vix_close"], errors="coerce"
    )
    if vix_features["vix_close"].le(0).any():
        raise ValueError("VIX closes must be positive before taking log changes.")

    calendar = pd.DataFrame(
        {"dlycaldt": pd.to_datetime(crsp_dates).dropna().drop_duplicates().sort_values()}
    ).reset_index(drop=True)
    calendar = calendar.merge(
        vix_features.rename(columns={"date": "dlycaldt"}),
        on="dlycaldt",
        how="left",
        validate="one_to_one",
    )
    if calendar["vix_close"].isna().any():
        missing_dates = calendar.loc[calendar["vix_close"].isna(), "dlycaldt"]
        raise ValueError(
            "VIX is missing on CRSP trading dates: "
            f"{missing_dates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    log_vix = np.log(calendar["vix_close"])
    for lag in VIX_CHANGE_LAGS:
        calendar[f"vix_log_change_{lag}d"] = log_vix - log_vix.shift(lag)
    return calendar


def add_targets_and_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the exact forward target and purge labels crossing split boundaries."""
    required_columns = set(KEY_COLUMNS + ["volatility_21d"])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Target data are missing columns: {sorted(missing_columns)}")

    result = frame.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    require_unique_keys(result, "Feature data")
    result["dlycaldt"] = pd.to_datetime(result["dlycaldt"])
    grouped = result.groupby("permno", sort=False)
    result["target_forward_21d"] = grouped["volatility_21d"].shift(
        -FORECAST_HORIZON
    )
    result["target_end_date"] = grouped["dlycaldt"].shift(-FORECAST_HORIZON)
    if result["target_forward_21d"].lt(0).any():
        raise ValueError("Forward volatility targets cannot be negative.")
    result["log_target_forward_21d"] = np.log(
        result["target_forward_21d"].where(result["target_forward_21d"].gt(0))
    )

    result["split"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[result["dlycaldt"].lt(TRAIN_END), "split"] = "train"
    result.loc[
        result["dlycaldt"].ge(TRAIN_END)
        & result["dlycaldt"].lt(VALIDATION_END),
        "split",
    ] = "validation"
    result.loc[result["dlycaldt"].ge(VALIDATION_END), "split"] = "test"

    result["crosses_split_boundary"] = False
    result.loc[
        result["split"].eq("train")
        & result["target_end_date"].ge(TRAIN_END),
        "crosses_split_boundary",
    ] = True
    result.loc[
        result["split"].eq("validation")
        & result["target_end_date"].ge(VALIDATION_END),
        "crosses_split_boundary",
    ] = True
    return result


def build_ml_feature_matrix(target: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical member-origin matrix and its eligibility flag."""
    required_columns = set(
        KEY_COLUMNS
        + ["gvkey", "dlyret", "in_sp500_membership", "volatility_21d"]
    )
    missing_columns = required_columns.difference(target.columns)
    if missing_columns:
        raise KeyError(f"Target data are missing columns: {sorted(missing_columns)}")

    features = add_return_features(target)
    features = add_targets_and_splits(features)
    vix_features = build_vix_features(vix, features["dlycaldt"])
    features = features.merge(
        vix_features,
        on="dlycaldt",
        how="left",
        validate="many_to_one",
    )

    membership = features["in_sp500_membership"].fillna(False).astype(bool)
    features = features.loc[membership].copy()
    finite_model_values = np.isfinite(
        features[FEATURE_COLUMNS + ["log_target_forward_21d"]].to_numpy(
            dtype="float64"
        )
    ).all(axis=1)
    features["eligible_ml"] = (
        finite_model_values
        & ~features["crosses_split_boundary"]
    )
    features = features[OUTPUT_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    require_unique_keys(features, "Final ML feature matrix")
    return features


def main() -> None:
    target = pd.read_parquet(TARGET_PATH)
    vix = pd.read_parquet(VIX_PATH)
    print(
        f"Loaded {len(target):,} security-date rows for "
        f"{target['permno'].nunique():,} PERMNOs."
    )

    matrix = build_ml_feature_matrix(target, vix)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(OUTPUT_PATH, index=False)

    saved = pd.read_parquet(OUTPUT_PATH)
    require_unique_keys(saved, "Saved ML feature matrix")
    if len(saved) != len(matrix):
        raise RuntimeError("Saved matrix failed round-trip row-count validation.")

    eligible = saved.loc[saved["eligible_ml"]]
    print(f"Saved member-origin rows: {len(saved):,}")
    print(f"Eligible ML rows: {len(eligible):,}")
    print(f"Eligible PERMNOs: {eligible['permno'].nunique():,}")
    print("Eligible rows by split:")
    print(eligible["split"].value_counts().reindex(["train", "validation", "test"]).to_string())
    print("Purged rows crossing a train/validation boundary:")
    print(
        saved.loc[saved["crosses_split_boundary"], "split"]
        .value_counts()
        .reindex(["train", "validation"], fill_value=0)
        .to_string()
    )
    print("Missing values by locked input or target:")
    print(
        saved[FEATURE_COLUMNS + ["log_target_forward_21d"]]
        .isna()
        .sum()
        .to_string()
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
