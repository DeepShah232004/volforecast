"""
Reusable evaluation metrics for volatility forecasting. Used by every model
-- classical and ML alike -- so the evaluation logic is written and tested.

QLIKE note: QLIKE is conventionally defined on 'variance' forecasts, not
volatility (std) forecasts, because it is derived from the Gaussian 
quasi-likelihood. Our target is volatility (std), so we square both
actual and forecast before computing QLIKE.
"""
import numpy as np

def rmse(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return np.sqrt(np.mean((actual-forecast)**2))

def mae(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return np.mean(np.abs(actual-forecast))

def qlike(actual_vol, forecast_vol):
    """
    QLIKE loss, computed on variance (volatility squared)

    Lower is better. Penalizes underprediction of variance more heavily
    than overprediction, which is generally considered more appropriate
    for volatility forecasting than RMSE/MSE.
    """
    actual_var = np.asarray(actual_vol) ** 2
    forecast_var = np.asarray(forecast_vol) ** 2

    if np.any(forecast_var <= 0):
        raise ValueError(
            "QLIKE requires strictly positive forecast_var values --" \
            "found zero or negative forecast(s)."
        )

    ratio = actual_var / forecast_var
    return np.mean(ratio - np.log(ratio) - 1)

def evaluate_all(actual, forecast, label=""):
    """
    Compute and print RMSE, MAE, QLIKE together.
    Drops any rows where actual or forecast is NaN before computing.
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    valid_mask = ~(np.isnan(actual) | np.isnan(forecast))
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f" (dropped {n_dropped} rows with NaN actual/forecast before evaluating)")

    actual = actual[valid_mask]
    forecast = forecast[valid_mask]

    results = {
        'rmse': rmse(actual, forecast),
        'mae': mae(actual, forecast),
        'qlike': qlike(actual, forecast),
        'n_obs': len(actual)
    }

    if label:
        print(f"Evaluation: {label}")
    print(f" N obs: {results['n_obs']}")
    print(f" RMSE: {results['rmse']:.6f}")
    print(f" MAE: {results['mae']:.6f}")
    print(f" QLIKE: {results['qlike']:.6f}")

    return results