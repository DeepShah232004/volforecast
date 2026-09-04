"""Build the canonical 2015-2024 evaluation panel for classical forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_START = pd.Timestamp("2005-01-01")
TEST_START = pd.Timestamp("2015-01-01")
LOW_VIX_QUANTILE = 1 / 3
HIGH_VIX_QUANTILE = 2 / 3
FORECAST_HORIZON = 21

TARGET_PATH = Path("data/target_variable_sp500_2005_2024.parquet")
EARNINGS_PATH = Path("data/earnings_dates_sp500_2005_2024.parquet")
VIX_PATH = Path("data/vix_2005_2024.parquet")
NAIVE_PATH = Path("models/classical/forecasts_historical_vol.parquet")
EWMA_PATH = Path("models/classical/forecasts_ewma.parquet")
GARCH_PATH = Path("models/classical/forecasts_garch.parquet")
OUTPUT_PATH = Path("data/classical_forecast_panel_sp500_2015_2024.parquet")

KEY_COLUMNS = ["permno", "dlycaldt"]
FORECAST_COLUMNS = ["forecast_naive", "forecast_ewma", "forecast_garch"]


def require_unique_keys(frame: pd.DataFrame, label: str) -> None:
    duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} contains {duplicate_count} duplicate (PERMNO, date) keys."
        )


def compute_vix_regime_thresholds(vix: pd.DataFrame) -> tuple[float, float]:
    """Compute fixed VIX tertiles using only the pre-test training period."""
    required_columns = {"date", "vix_close"}
    missing_columns = required_columns.difference(vix.columns)
    if missing_columns:
        raise KeyError(f"VIX data are missing required columns: {sorted(missing_columns)}")

    dates = pd.to_datetime(vix["date"])
    training_vix = pd.to_numeric(
        vix.loc[(dates >= SAMPLE_START) & (dates < TEST_START), "vix_close"],
        errors="coerce",
    ).dropna()
    if training_vix.empty:
        raise ValueError("No pre-2015 VIX observations are available for regime thresholds.")

    low_threshold = float(training_vix.quantile(LOW_VIX_QUANTILE))
    high_threshold = float(training_vix.quantile(HIGH_VIX_QUANTILE))
    if low_threshold >= high_threshold:
        raise ValueError("Training-period VIX regime thresholds are not ordered.")

    return low_threshold, high_threshold


def assign_vix_regime(
    vix_close: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    """Assign low/middle/high regimes using already-locked thresholds."""
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be below high_threshold.")

    values = pd.to_numeric(vix_close, errors="coerce")
    regimes = pd.Series(pd.NA, index=vix_close.index, dtype="string")
    valid = values.notna()
    regimes.loc[valid] = "middle"
    regimes.loc[valid & (values <= low_threshold)] = "low"
    regimes.loc[valid & (values >= high_threshold)] = "high"
    return regimes


def add_forward_earnings_flags(
    windows: pd.DataFrame,
    earnings: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Label announcements in each exact forward window, ``(t, t+21]``.

    Announcement dates equal to the forecast origin are excluded. Dates equal
    to the realized target window's final security trading date are included.
    Rows without a valid target end date receive missing event labels.
    """
    required_window_columns = {"permno", "dlycaldt", "target_end_date"}
    required_earnings_columns = {"permno", "anndats"}
    missing_window_columns = required_window_columns.difference(windows.columns)
    missing_earnings_columns = required_earnings_columns.difference(earnings.columns)
    if missing_window_columns:
        raise KeyError(
            f"Forecast windows are missing columns: {sorted(missing_window_columns)}"
        )
    if missing_earnings_columns:
        raise KeyError(
            f"Earnings data are missing columns: {sorted(missing_earnings_columns)}"
        )

    result = windows.copy().reset_index(drop=True)
    result["dlycaldt"] = pd.to_datetime(result["dlycaldt"])
    result["target_end_date"] = pd.to_datetime(result["target_end_date"])
    if result["dlycaldt"].isna().any():
        raise ValueError("Forecast windows contain a missing origin date.")

    valid_end = result["target_end_date"].notna()
    reversed_windows = valid_end & (
        result["target_end_date"] <= result["dlycaldt"]
    )
    if reversed_windows.any():
        raise ValueError("Forecast target end dates must be after origin dates.")

    earnings_unique = earnings[["permno", "anndats"]].copy()
    earnings_unique["anndats"] = pd.to_datetime(earnings_unique["anndats"])
    earnings_unique = earnings_unique.dropna(subset=["permno", "anndats"])
    before_deduplication = len(earnings_unique)
    earnings_unique = earnings_unique.drop_duplicates(["permno", "anndats"])
    duplicate_announcement_count = before_deduplication - len(earnings_unique)

    announcements_by_permno = {
        permno: np.sort(group["anndats"].to_numpy(dtype="datetime64[ns]"))
        for permno, group in earnings_unique.groupby("permno", sort=False)
    }

    event_counts = pd.array([pd.NA] * len(result), dtype="Int64")
    event_flags = pd.array([pd.NA] * len(result), dtype="boolean")

    for permno, row_index in result.groupby("permno", sort=False).groups.items():
        positions = np.asarray(row_index)
        group_valid_end = valid_end.iloc[positions].to_numpy()
        valid_positions = positions[group_valid_end]
        if len(valid_positions) == 0:
            continue

        announcements = announcements_by_permno.get(permno)
        if announcements is None:
            counts = np.zeros(len(valid_positions), dtype="int64")
        else:
            origin_dates = result.loc[valid_positions, "dlycaldt"].to_numpy(
                dtype="datetime64[ns]"
            )
            end_dates = result.loc[valid_positions, "target_end_date"].to_numpy(
                dtype="datetime64[ns]"
            )
            first_after_origin = np.searchsorted(
                announcements, origin_dates, side="right"
            )
            first_after_end = np.searchsorted(announcements, end_dates, side="right")
            counts = first_after_end - first_after_origin

        event_counts[valid_positions] = counts
        event_flags[valid_positions] = counts > 0

    result["earnings_count_forward_21d"] = event_counts
    result["has_earnings_forward_21d"] = event_flags
    return result, duplicate_announcement_count


def assert_same_keys_and_targets(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    label: str,
) -> None:
    """Require a candidate forecast file to match the baseline population."""
    require_unique_keys(baseline, "Naive forecast file")
    require_unique_keys(candidate, label)

    comparison = baseline[
        KEY_COLUMNS + ["target_forward_21d"]
    ].merge(
        candidate[KEY_COLUMNS + ["target_forward_21d"]],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError(f"{label} does not have the same forecast keys as the baseline.")

    baseline_target = comparison["target_forward_21d_baseline"].to_numpy()
    candidate_target = comparison["target_forward_21d_candidate"].to_numpy()
    if not np.allclose(baseline_target, candidate_target, equal_nan=True):
        raise ValueError(f"{label} target values differ from the baseline.")


def main() -> None:
    naive = pd.read_parquet(NAIVE_PATH)
    ewma = pd.read_parquet(EWMA_PATH)
    garch = pd.read_parquet(GARCH_PATH)
    target = pd.read_parquet(
        TARGET_PATH,
        columns=["permno", "dlycaldt", "volatility_21d"],
    )
    earnings = pd.read_parquet(EARNINGS_PATH)
    vix = pd.read_parquet(VIX_PATH)

    assert_same_keys_and_targets(naive, ewma, "EWMA forecast file")
    assert_same_keys_and_targets(naive, garch, "GARCH forecast file")

    panel = naive.merge(
        ewma[KEY_COLUMNS + ["forecast_ewma"]],
        on=KEY_COLUMNS,
        validate="one_to_one",
    ).merge(
        garch[KEY_COLUMNS + ["forecast_garch"]],
        on=KEY_COLUMNS,
        validate="one_to_one",
    )

    target = target.sort_values(KEY_COLUMNS).reset_index(drop=True)
    require_unique_keys(target, "Target data")
    target["target_end_date"] = target.groupby("permno")["dlycaldt"].shift(
        -FORECAST_HORIZON
    )
    target["target_forward_21d_check"] = target.groupby("permno")[
        "volatility_21d"
    ].shift(-FORECAST_HORIZON)

    panel = panel.merge(
        target[
            KEY_COLUMNS + ["target_end_date", "target_forward_21d_check"]
        ],
        on=KEY_COLUMNS,
        validate="one_to_one",
    )
    if not np.allclose(
        panel["target_forward_21d"],
        panel["target_forward_21d_check"],
        equal_nan=True,
    ):
        raise ValueError("Saved forecast targets do not match the rebuilt target windows.")
    panel = panel.drop(columns="target_forward_21d_check")

    vix = vix.rename(columns={"date": "dlycaldt"})
    if vix.duplicated("dlycaldt").any():
        raise ValueError("VIX data contain duplicate dates.")
    panel = panel.merge(vix, on="dlycaldt", how="left", validate="many_to_one")
    if panel["vix_close"].isna().any():
        missing_dates = panel.loc[panel["vix_close"].isna(), "dlycaldt"].unique()
        raise ValueError(f"Evaluation panel is missing VIX on: {missing_dates.tolist()}")

    low_vix_threshold, high_vix_threshold = compute_vix_regime_thresholds(
        pd.read_parquet(VIX_PATH)
    )
    panel["vix_regime"] = assign_vix_regime(
        panel["vix_close"],
        low_vix_threshold,
        high_vix_threshold,
    )

    panel, duplicate_announcement_count = add_forward_earnings_flags(panel, earnings)
    panel["common_valid"] = panel[
        ["target_forward_21d"] + FORECAST_COLUMNS
    ].notna().all(axis=1)

    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)
    require_unique_keys(panel, "Final evaluation panel")
    panel.to_parquet(OUTPUT_PATH, index=False)

    saved = pd.read_parquet(OUTPUT_PATH)
    require_unique_keys(saved, "Saved evaluation panel")
    if len(saved) != len(panel):
        raise RuntimeError("Saved evaluation panel failed round-trip row-count validation.")

    common = saved[saved["common_valid"]]
    print(f"Saved panel rows: {len(saved):,}")
    print(f"Common valid rows: {len(common):,}")
    print(f"Common valid PERMNOs: {common['permno'].nunique():,}")
    print(
        "Training-only VIX thresholds: "
        f"low <= {low_vix_threshold:.6f}, high >= {high_vix_threshold:.6f}"
    )
    print("Common-sample VIX regime counts:")
    print(common["vix_regime"].value_counts().sort_index().to_string())
    print(
        "Duplicate same-day earnings records collapsed before event labeling: "
        f"{duplicate_announcement_count:,}"
    )
    print("Common-sample earnings-window counts:")
    print(common["has_earnings_forward_21d"].value_counts().to_string())
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
