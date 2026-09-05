"""Evaluate the frozen five-model test panel and adjudicate H1-H3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from build_final_panel import MODEL_FORECAST_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_ROOT / "data/final_forecast_panel_sp500_2015_2024.parquet"
RESULT_PATH = PROJECT_ROOT / "evaluation/results/final_evaluation.json"

BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42
NEWEY_WEST_LAG = 20
H2_COMPARABILITY_BOUND = 0.05
BOOTSTRAP_BATCH_SIZE = 100


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "permno",
        "gvkey",
        "dlycaldt",
        "target_forward_21d",
        "vix_regime",
        "has_earnings_forward_21d",
        *MODEL_FORECAST_COLUMNS.values(),
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Final panel is missing columns: {sorted(missing)}")

    result = panel.copy()
    result["dlycaldt"] = pd.to_datetime(result["dlycaldt"])
    if result.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Final panel contains duplicate security-date keys.")
    required_complete = [
        "target_forward_21d",
        "vix_regime",
        "has_earnings_forward_21d",
        *MODEL_FORECAST_COLUMNS.values(),
    ]
    if result[required_complete].isna().any().any():
        raise ValueError("Final panel contains missing evaluation values.")
    numeric = result[
        ["target_forward_21d", *MODEL_FORECAST_COLUMNS.values()]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or np.any(numeric <= 0):
        raise ValueError("Evaluation targets and forecasts must be finite and positive.")
    if not result["vix_regime"].isin(["low", "middle", "high"]).all():
        raise ValueError("Final panel contains an invalid VIX regime.")

    per_date_regimes = result.groupby("dlycaldt")["vix_regime"].nunique()
    if per_date_regimes.max() != 1:
        raise ValueError("A forecast date has more than one VIX regime.")
    return result.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)


def qlike_loss(actual_volatility: np.ndarray, forecast_volatility: np.ndarray) -> np.ndarray:
    actual_variance = np.asarray(actual_volatility, dtype="float64") ** 2
    forecast_variance = np.asarray(forecast_volatility, dtype="float64") ** 2
    if np.any(actual_variance <= 0) or np.any(forecast_variance <= 0):
        raise ValueError("QLIKE requires positive actual and forecast variance.")
    ratio = actual_variance / forecast_variance
    return ratio - np.log(ratio) - 1.0


def calculate_losses(panel: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    actual = panel["target_forward_21d"].to_numpy(dtype="float64")
    losses: dict[str, dict[str, np.ndarray]] = {}
    for model, forecast_column in MODEL_FORECAST_COLUMNS.items():
        forecast = panel[forecast_column].to_numpy(dtype="float64")
        errors = actual - forecast
        losses[model] = {
            "squared_error": errors**2,
            "absolute_error": np.abs(errors),
            "qlike": qlike_loss(actual, forecast),
        }
    return losses


def moving_block_bootstrap_indices(
    n_dates: int,
    block_length: int = BLOCK_LENGTH,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Draw non-circular moving blocks and truncate to the original length."""
    if n_dates < block_length:
        raise ValueError("The date count must be at least the block length.")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    block_count = math.ceil(n_dates / block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0,
        n_dates - block_length + 1,
        size=(replicates, block_count),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets).reshape(replicates, -1)
    return indices[:, :n_dates]


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("No finite bootstrap replicates are available.")
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def aggregate_by_date(
    values: np.ndarray,
    row_mask: np.ndarray,
    date_codes: np.ndarray,
    n_dates: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_codes = date_codes[row_mask]
    selected_values = np.asarray(values, dtype="float64")[row_mask]
    sums = np.bincount(selected_codes, weights=selected_values, minlength=n_dates)
    counts = np.bincount(selected_codes, minlength=n_dates).astype("int64")
    return sums, counts


def bootstrap_pooled_metric(
    values: np.ndarray,
    row_mask: np.ndarray,
    date_codes: np.ndarray,
    n_dates: int,
    bootstrap_indices: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Bootstrap a row-pooled metric while sampling entire date clusters."""
    sums, counts = aggregate_by_date(values, row_mask, date_codes, n_dates)
    if counts.sum() == 0:
        raise ValueError("Cannot summarize an empty evaluation group.")
    point = sums.sum() / counts.sum()
    replicates = np.empty(len(bootstrap_indices), dtype="float64")
    for start in range(0, len(bootstrap_indices), BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, len(bootstrap_indices))
        sampled = bootstrap_indices[start:stop]
        denominators = counts[sampled].sum(axis=1)
        if np.any(denominators == 0):
            raise ValueError("A bootstrap replicate contains no subgroup observations.")
        replicates[start:stop] = sums[sampled].sum(axis=1) / denominators

    if transform is not None:
        point = float(transform(np.asarray(point)).item())
        replicates = transform(replicates)
    lower, upper = percentile_interval(replicates)
    return {
        "estimate": float(point),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_valid_replicates": int(np.isfinite(replicates).sum()),
    }


def daily_mean(
    values: np.ndarray,
    row_mask: np.ndarray,
    date_codes: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    sums, counts = aggregate_by_date(values, row_mask, date_codes, n_dates)
    result = np.full(n_dates, np.nan, dtype="float64")
    present = counts > 0
    result[present] = sums[present] / counts[present]
    return result


def bootstrap_daily_mean(
    daily_values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[float, np.ndarray]:
    values = np.asarray(daily_values, dtype="float64")
    point = float(np.nanmean(values))
    replicates = np.empty(len(bootstrap_indices), dtype="float64")
    for start in range(0, len(bootstrap_indices), BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, len(bootstrap_indices))
        sampled = values[bootstrap_indices[start:stop]]
        valid_counts = np.isfinite(sampled).sum(axis=1)
        if np.any(valid_counts == 0):
            raise ValueError("A bootstrap replicate contains no conditional dates.")
        replicates[start:stop] = np.nansum(sampled, axis=1) / valid_counts
    return point, replicates


def bootstrap_relative_improvement(
    daily_improvement: np.ndarray,
    daily_benchmark_loss: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[float, np.ndarray]:
    numerator_point, numerator_replicates = bootstrap_daily_mean(
        daily_improvement, bootstrap_indices
    )
    denominator_point, denominator_replicates = bootstrap_daily_mean(
        daily_benchmark_loss, bootstrap_indices
    )
    if denominator_point <= 0 or np.any(denominator_replicates <= 0):
        raise ValueError("Relative QLIKE improvement requires positive benchmark loss.")
    return numerator_point / denominator_point, numerator_replicates / denominator_replicates


def newey_west_mean_test(
    daily_difference: np.ndarray,
    lag: int = NEWEY_WEST_LAG,
) -> dict[str, float | int]:
    """Two-sided HAC test of whether a daily mean loss difference equals zero."""
    values = np.asarray(daily_difference, dtype="float64")
    values = values[np.isfinite(values)]
    n_obs = len(values)
    if n_obs < 2:
        raise ValueError("Newey-West inference requires at least two dates.")
    lag_used = min(lag, n_obs - 1)
    estimate = float(values.mean())
    centered = values - estimate
    long_run_variance = float(np.dot(centered, centered) / n_obs)
    for current_lag in range(1, lag_used + 1):
        covariance = float(
            np.dot(centered[current_lag:], centered[:-current_lag]) / n_obs
        )
        weight = 1.0 - current_lag / (lag_used + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / n_obs)
    if standard_error == 0:
        statistic = 0.0 if estimate == 0 else math.copysign(math.inf, estimate)
        p_value = 1.0 if estimate == 0 else 0.0
    else:
        statistic = estimate / standard_error
        p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {
        "mean_difference": estimate,
        "standard_error": float(standard_error),
        "z_statistic": float(statistic),
        "two_sided_p_value": float(p_value),
        "ci_lower": float(estimate - 1.96 * standard_error),
        "ci_upper": float(estimate + 1.96 * standard_error),
        "dates": int(n_obs),
        "lag": int(lag_used),
    }


def classify_h1(comparisons: dict[str, dict]) -> str:
    supported = sum(bool(result["supports_h1"]) for result in comparisons.values())
    if supported == len(comparisons):
        return "full_support"
    if supported == 1:
        return "partial_support"
    return "not_supported"


def positive_effect_supported(estimate: float, ci_lower: float) -> bool:
    return bool(estimate > 0 and ci_lower > 0)


def h2_practically_comparable(
    relative_improvement: float,
    ci_upper: float,
    bound: float = H2_COMPARABILITY_BOUND,
) -> bool:
    return bool(relative_improvement <= 0 or ci_upper < bound)


def classify_h2(comparisons: dict[str, dict]) -> str:
    if all(bool(result["practically_comparable"]) for result in comparisons.values()):
        return "full_support"
    return "not_supported"


def classify_h3(models: dict[str, dict]) -> str:
    if all(bool(result["supports_h3"]) for result in models.values()):
        return "full_support"
    return "not_supported"


def summarize_metric_groups(
    panel: pd.DataFrame,
    losses: dict[str, dict[str, np.ndarray]],
    date_codes: np.ndarray,
    dates: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, dict]:
    earnings = panel["has_earnings_forward_21d"].astype(bool).to_numpy()
    groups = {
        "overall": np.ones(len(panel), dtype=bool),
        "vix_low": panel["vix_regime"].eq("low").to_numpy(),
        "vix_middle": panel["vix_regime"].eq("middle").to_numpy(),
        "vix_high": panel["vix_regime"].eq("high").to_numpy(),
        "earnings": earnings,
        "no_earnings": ~earnings,
    }
    summaries: dict[str, dict] = {}
    for group_name, row_mask in groups.items():
        model_summaries: dict[str, dict] = {}
        for model, model_losses in losses.items():
            model_summaries[model] = {
                "rmse": bootstrap_pooled_metric(
                    model_losses["squared_error"],
                    row_mask,
                    date_codes,
                    len(dates),
                    bootstrap_indices,
                    transform=np.sqrt,
                ),
                "mae": bootstrap_pooled_metric(
                    model_losses["absolute_error"],
                    row_mask,
                    date_codes,
                    len(dates),
                    bootstrap_indices,
                ),
                "qlike": bootstrap_pooled_metric(
                    model_losses["qlike"],
                    row_mask,
                    date_codes,
                    len(dates),
                    bootstrap_indices,
                ),
            }
        summaries[group_name] = {
            "rows": int(row_mask.sum()),
            "dates": int(np.unique(date_codes[row_mask]).size),
            "permnos": int(panel.loc[row_mask, "permno"].nunique()),
            "models": model_summaries,
        }
    return summaries


def evaluate_hypotheses(
    panel: pd.DataFrame,
    losses: dict[str, dict[str, np.ndarray]],
    date_codes: np.ndarray,
    date_information: pd.DataFrame,
    bootstrap_indices: np.ndarray,
) -> dict[str, dict]:
    n_dates = len(date_information)
    daily_qlike = {
        model: daily_mean(
            model_losses["qlike"],
            np.ones(len(panel), dtype=bool),
            date_codes,
            n_dates,
        )
        for model, model_losses in losses.items()
    }

    high_dates = date_information["vix_regime"].eq("high").to_numpy()
    h1_comparisons: dict[str, dict] = {}
    for model in ("random_forest", "xgboost"):
        difference = daily_qlike["garch"] - daily_qlike[model]
        conditional = np.where(high_dates, difference, np.nan)
        estimate, replicates = bootstrap_daily_mean(conditional, bootstrap_indices)
        lower, upper = percentile_interval(replicates)
        h1_comparisons[f"{model}_vs_garch"] = {
            "daily_qlike_improvement": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "supports_h1": positive_effect_supported(estimate, lower),
            "dates": int(np.isfinite(conditional).sum()),
            "newey_west": newey_west_mean_test(conditional),
        }
    h1 = {
        "definition": "GARCH QLIKE minus ML QLIKE on high-VIX dates; positive favors ML.",
        "comparisons": h1_comparisons,
        "status": classify_h1(h1_comparisons),
    }

    low_dates = date_information["vix_regime"].eq("low").to_numpy()
    h2_comparisons: dict[str, dict] = {}
    for model in ("random_forest", "xgboost"):
        difference = np.where(
            low_dates,
            daily_qlike["garch"] - daily_qlike[model],
            np.nan,
        )
        benchmark = np.where(low_dates, daily_qlike["garch"], np.nan)
        relative, replicates = bootstrap_relative_improvement(
            difference, benchmark, bootstrap_indices
        )
        lower, upper = percentile_interval(replicates)
        garch_win = relative <= 0
        comparable = h2_practically_comparable(relative, upper)
        h2_comparisons[f"{model}_vs_garch"] = {
            "relative_ml_qlike_improvement": float(relative),
            "ci_lower": lower,
            "ci_upper": upper,
            "garch_point_estimate_win": bool(garch_win),
            "practically_comparable": bool(comparable),
            "comparability_bound": H2_COMPARABILITY_BOUND,
            "dates": int(np.isfinite(difference).sum()),
            "newey_west_absolute_difference": newey_west_mean_test(difference),
        }
    h2 = {
        "definition": "Relative low-VIX ML QLIKE improvement over GARCH; upper CI below 5% means comparable, and a GARCH point-estimate win also qualifies.",
        "comparisons": h2_comparisons,
        "status": classify_h2(h2_comparisons),
    }

    event_rows = panel["has_earnings_forward_21d"].astype(bool).to_numpy()
    h3_models: dict[str, dict] = {}
    for model, model_losses in losses.items():
        event_daily = daily_mean(
            model_losses["qlike"], event_rows, date_codes, n_dates
        )
        non_event_daily = daily_mean(
            model_losses["qlike"], ~event_rows, date_codes, n_dates
        )
        difference = event_daily - non_event_daily
        estimate, replicates = bootstrap_daily_mean(difference, bootstrap_indices)
        lower, upper = percentile_interval(replicates)
        h3_models[model] = {
            "earnings_minus_non_earnings_daily_qlike": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "supports_h3": positive_effect_supported(estimate, lower),
            "dates": int(np.isfinite(difference).sum()),
            "newey_west": newey_west_mean_test(difference),
        }
    h3 = {
        "definition": "Within-date mean earnings-window QLIKE minus non-earnings QLIKE; positive means accuracy degrades around earnings.",
        "models": h3_models,
        "status": classify_h3(h3_models),
    }
    return {"H1": h1, "H2": h2, "H3": h3}


def evaluate(panel: pd.DataFrame, panel_path: Path) -> dict:
    dates = np.sort(panel["dlycaldt"].unique())
    date_codes = pd.Categorical(panel["dlycaldt"], categories=dates).codes
    if np.any(date_codes < 0):
        raise ValueError("Failed to assign every panel row to a forecast date.")
    date_information = (
        panel[["dlycaldt", "vix_regime"]]
        .drop_duplicates()
        .sort_values("dlycaldt")
        .reset_index(drop=True)
    )
    if len(date_information) != len(dates):
        raise ValueError("Date-level VIX information is not unique.")

    bootstrap_indices = moving_block_bootstrap_indices(len(dates))
    losses = calculate_losses(panel)
    metric_groups = summarize_metric_groups(
        panel, losses, date_codes, dates, bootstrap_indices
    )
    hypotheses = evaluate_hypotheses(
        panel, losses, date_codes, date_information, bootstrap_indices
    )
    hypotheses["H3"]["earnings_qlike_ranking"] = sorted(
        MODEL_FORECAST_COLUMNS,
        key=lambda model: metric_groups["earnings"]["models"][model]["qlike"][
            "estimate"
        ],
    )
    hypotheses["H3"]["non_earnings_qlike_ranking"] = sorted(
        MODEL_FORECAST_COLUMNS,
        key=lambda model: metric_groups["no_earnings"]["models"][model]["qlike"][
            "estimate"
        ],
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path.relative_to(PROJECT_ROOT)),
        "panel_sha256": file_sha256(panel_path),
        "sample": {
            "rows": int(len(panel)),
            "permnos": int(panel["permno"].nunique()),
            "dates": int(len(dates)),
            "origin_start": str(pd.Timestamp(dates[0]).date()),
            "origin_end": str(pd.Timestamp(dates[-1]).date()),
        },
        "inference": {
            "bootstrap": "non-circular moving-block bootstrap over the full test trading-date calendar",
            "block_length": BLOCK_LENGTH,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "confidence_level": 0.95,
            "newey_west_lag": NEWEY_WEST_LAG,
        },
        "metric_groups": metric_groups,
        "hypotheses": hypotheses,
    }


def write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def print_results(results: dict) -> None:
    print("Final overall test metrics (95% moving-block bootstrap CI)")
    print("model             RMSE          MAE           QLIKE")
    for model in MODEL_FORECAST_COLUMNS:
        metrics = results["metric_groups"]["overall"]["models"][model]
        print(
            f"{model:<17} "
            f"{metrics['rmse']['estimate']:.9f}  "
            f"{metrics['mae']['estimate']:.9f}  "
            f"{metrics['qlike']['estimate']:.9f}"
        )
    print("Hypothesis decisions")
    for hypothesis in ("H1", "H2", "H3"):
        print(f"{hypothesis}: {results['hypotheses'][hypothesis]['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "evaluate"))
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing final evaluation result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        raise FileNotFoundError(
            f"Final panel not found at {args.panel}. Run evaluation/build_final_panel.py first."
        )
    panel = validate_panel(pd.read_parquet(args.panel))
    print(f"Validated final panel: {len(panel):,} rows, {panel['permno'].nunique():,} PERMNOs")
    print(
        f"Inference lock: {BOOTSTRAP_REPLICATES:,} replicates, "
        f"block length {BLOCK_LENGTH}, seed {RANDOM_SEED}, "
        f"Newey-West lag {NEWEY_WEST_LAG}."
    )
    if args.stage == "check":
        print("No forecast losses were calculated.")
        return
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Final results already exist at {args.output}; use --force only if a "
            "documented rerun is required."
        )
    results = evaluate(panel, args.panel)
    write_results(args.output, results)
    print_results(results)
    print(f"Saved full results to {args.output}")


if __name__ == "__main__":
    main()
