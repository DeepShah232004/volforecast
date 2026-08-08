"""
The simplest classical baseline: naive persistence at a 21-trading day
horizon. The forecast for the forward (non-overlapping) 21-day realized
volatility window (t+1, t+21] is simply the most recently observed
trailing 21-day volatility as of t.

This model has no parameters to fit, so there's no actual "training" step
-- but we still build the full rolling out-of-sample evaluation structure, 
since every later model will reuse this same train/test split and eval.

Forecast horizon: 21 trading days ahead, non-overlapping with the feature 
window.
"""

import sys
import pandas as pd

sys.path.append('evaluation')
from metrics import evaluate_all

TEST_START = pd.Timestamp('2015-01-01') # train: 2005-2014, test: 2015-2024

# load target variable
data = pd.read_parquet('data/target_variable_sp500_2005_2024.parquet')
data = data.sort_values(['permno', 'dlycaldt']).reset_index(drop=True)
print(f"Loaded {len(data)} rows, {data['permno'].nunique()} unique PERMNOs")
print()

# target_forward_21d[t] = volatility_21d[t+21]
# since the the trailing 21-day window ending at t+21 covers exactly
# (t+1, t+21] -- the forward window for t.
data['target_forward_21d'] = data.groupby('permno')['volatility_21d'].shift(-21)

# The naive/historical-volatility forecast for the forward window is
# simply the most recently observed training volatility -- no shift
# needed.
data['forecast_naive'] = data['volatility_21d']

n_valid_target = data['target_forward_21d'].notna().sum()
n_valid_forecast = data['forecast_naive'].notna().sum()
print(f"Rows with a valid forward target: {n_valid_target} out of {len(data)}")
print(f"Rows with a valid naive forecast: {n_valid_forecast} out of {len(data)}")
print("(Last 21 trading days per company will have no forward target yet -- expected.)")
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
    forecast=test['forecast_naive'],
    label="Historical Volatility (naive persistence), forward 21-day, full test period 2015-2024"
)
print()

# save forecasts for later cross-model comparison
output = test[['permno', 'gvkey', 'dlycaldt', 'target_forward_21d', 'forecast_naive']]
output.to_parquet('models/classical/forecasts_historical_vol.parquet', index=False)
print(f"Saved {len(output)} test-period forecasts to "
      f"models/classical/forecasts_historical_vol.parquet")