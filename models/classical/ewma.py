"""
Second classical baseline: EWMA (exponentially weighted moving average)
volatility. Unlike the flat 21-day window used in historical_vol.py,
EWMA gives exponentially more weight to recent returns, so it should
react faster to genuine volatility regime changes.

Uses the RiskMetrics-standard decay parameter lambda = 0.94 for daily
data.

Forecast target: identical to historical_vol.py -- the non-overlapping
forward 21-day realized volatility window (t+1, t+21], forecast using
only information known as of t.
"""

import sys
import numpy as np
import pandas as pd

sys.path.append('evaluation')
from metrics import evaluate_all

TEST_START = pd.Timestamp('2015-01-01')
LAMBDA = 0.94
ALPHA = 1-LAMBDA
BURN_IN = 21

# load target variable
data = pd.read_parquet('data/target_variable_sp500_2005_2024.parquet')
data = data.sort_values(['permno', 'dlycaldt']).reset_index(drop=True)
print(f"Loaded {len(data)} rows, {data['permno'].nunique()} unique PERMNOs")
print()

# build the identical forward target
data['target_forward_21d'] = data.groupby('permno')['volatility_21d'].shift(-21)

# compute the EWMA volatility forecast
# EWMA variance: var_t = lambda * var_{t-1} + (1 - lambda)r_t^2
data['ewma_var'] = (
    data.groupby('permno')['dlyret']
    .transform(lambda x: (x**2).ewm(alpha=ALPHA, adjust=False).mean())
)
data['forecast_ewma'] = np.sqrt(data['ewma_var'])

data['obs_count'] = data.groupby('permno').cumcount() + 1
data.loc[data['obs_count'] < BURN_IN, 'forecast_ewma'] = np.nan

n_valid_forecast = data['forecast_ewma'].notna().sum()
print(f"Rows with a valid EWMA forecast (after {BURN_IN}-day burn-in): "
      f"{n_valid_forecast} out of {len(data)}")
print()

# split into train/test by date
train = data[data['dlycaldt'] < TEST_START]
test = data[(data['dlycaldt'] >= TEST_START) & (data['in_sp500_membership'])]

print(f"Train: {len(train)} rows ({train['dlycaldt'].min().date()} to "
      f"{train['dlycaldt'].max().date()})")
print(f"Test: {len(test)} rows ({test['dlycaldt'].min().date()} to "
      f"{test['dlycaldt'].max().date()})")
print()

# evaluate
result = evaluate_all(
    actual=test['target_forward_21d'],
    forecast=test['forecast_ewma'],
    label=f"EWMA (lambda={LAMBDA}), forward 21-day, test period 2015-2024"
)
print()

# quick comparison against the naive baseline
try:
    naive = pd.read_parquet('models/classical/forecasts_historical_vol.parquet')
    print("Reference: naive persistence baseline")
    evaluate_all(
        actual=naive['target_forward_21d'],
        forecast=naive['forecast_naive'],
        label="Historical Volatility (naive persistence), for comparision"
    )
    print()
    print("EWMA should outperform (lower RMSE/MAE/QLIKE than) naive persistence "
          "if it's adding genuine forecasting value beyond a flat-window average. "
          "If it is worse, investigate.")
except FileNotFoundError:
    print("(Naive baseline forecasts not found -- run historical_vol.py first " \
    "for a side-by-side comparision.)")
print()

# save
output = test[['permno', 'gvkey', 'dlycaldt', 'target_forward_21d', 'forecast_ewma']]
output.to_parquet('models/classical/forecasts_ewma.parquet', index=False)
print(f"Saved {len(output)} test-period forecasts to "
      f"models/classical/forecasts_ewma.parquet")