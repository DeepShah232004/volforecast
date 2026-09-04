"""Numerical recursion used by the expanding-window GARCH baseline."""

from collections.abc import Mapping

import numpy as np


GarchParams = tuple[float, float, float]


def compute_garch_path_and_forecasts(
    returns_pct: np.ndarray,
    years: np.ndarray,
    year_params: Mapping[int, GarchParams | None],
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter conditional variance and forecast using origin-date parameters.

    ``year_params[y]`` must contain the parameters estimated using information
    available before year ``y``. At each forecast origin, both the one-step
    variance and all subsequent horizon steps use that origin year's parameter
    set. This is important at calendar boundaries: a December forecast must not
    borrow the refitted parameters that first become active in January.

    Returns are expressed in percent units. Returned forecasts are average
    daily volatility in decimal-return units, matching the study target.
    """
    returns_pct = np.asarray(returns_pct, dtype="float64")
    years = np.asarray(years)

    if returns_pct.ndim != 1 or years.ndim != 1:
        raise ValueError("returns_pct and years must be one-dimensional arrays.")
    if len(returns_pct) != len(years):
        raise ValueError("returns_pct and years must have the same length.")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")

    sigma2 = np.full(len(returns_pct), np.nan)
    forecasts = np.full(len(returns_pct), np.nan)

    for i, year in enumerate(years):
        params = year_params.get(int(year))
        if params is None:
            continue

        omega, alpha, beta = params
        persistence = alpha + beta
        if omega <= 0 or alpha < 0 or beta < 0 or persistence >= 1:
            raise ValueError(f"Invalid stationary GARCH parameters for year {year}.")

        long_run_var = omega / (1 - persistence)

        if i == 0 or np.isnan(returns_pct[i - 1]) or np.isnan(sigma2[i - 1]):
            sigma2[i] = long_run_var
        else:
            sigma2[i] = (
                omega
                + alpha * returns_pct[i - 1] ** 2
                + beta * sigma2[i - 1]
            )

        if np.isnan(returns_pct[i]):
            continue

        # Compute h_{t+1|t} directly with the parameters available at t.
        # Looking up sigma2[i + 1] would use the next year's refitted parameters
        # when i is the final trading day of a calendar year.
        one_step_var = omega + alpha * returns_pct[i] ** 2 + beta * sigma2[i]

        if abs(persistence - 1) < 1e-8:
            geometric_sum = forecast_horizon
        else:
            geometric_sum = (1 - persistence**forecast_horizon) / (1 - persistence)

        average_var_pct = long_run_var + (
            (one_step_var - long_run_var) * geometric_sum / forecast_horizon
        )
        forecasts[i] = np.sqrt(max(average_var_pct, 1e-12)) / 100

    return sigma2, forecasts
