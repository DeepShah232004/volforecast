"""
Third and final classical baseline: GARCH(1, 1)

This model has real parameters (omega, alpha, beta) that must
be *estimated* via MLE.

1. Parameters are re-estimated ANNUALY per company, using an EXPANDING
window of all of that company's available history up to the end of
the prior calendar year -- not fit once on a fixed 2005-2014 window.

2. Each year's forecasts remain genuinely out-of-sample: forecasts made
during calendar year Y use only parameters estimated from data through
the end of year Y-1.

3. The conditional variance path (sigma^2) is recursed continuously
through a company's full history, switching to newly-estimated
parameters at each calendar year boundary, rather than resetting.

4. Forecast target: identical to historical_vol.py / ewma.py

5. Companies without at least MIN_TRAIN_OBS days of history before
their first possible fit year are excluded until they accumulate
enough -- logged explicitly
"""

import sys
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

sys.path.append('evaluation')
from metrics import evaluate_all
from garch_core import compute_garch_path_and_forecasts

warnings.filterwarnings('ignore')

TEST_START = pd.Timestamp('2015-01-01')
MIN_TRAIN_OBS = 252 # ~1 trading year
FORECAST_HORIZON = 21

# load target variable
data = pd.read_parquet('data/target_variable_sp500_2005_2024.parquet')
data = data.sort_values(['permno', 'dlycaldt']).reset_index(drop=True)
print(f"Loaded {len(data)} rows, {data['permno'].nunique()} unique PERMNOs")
print()

data['target_forward_21d'] = data.groupby('permno')['volatility_21d'].shift(-21)
data['year'] = data['dlycaldt'].dt.year

def fit_garch_params(return_pct):
    """Fit GARCH(1,1) on a percent-scaled return series. Returns
    (omega, alpha, beta, next_variance) or None if fitting failed / degenerate.

    next_variance is the fit's one-step conditional variance after the final
    training return. It initializes the newly refitted calendar year without
    borrowing the previous parameter regime's filtered variance state.
    """
    try:
        model = arch_model(return_pct, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
        fit_result = model.fit(disp='off', show_warning=False)
    except Exception:
        return None

    omega = fit_result.params.get('omega')
    alpha = fit_result.params.get('alpha[1]')
    beta = fit_result.params.get('beta[1]')

    if (
        omega is None
        or alpha is None
        or beta is None
        or omega <= 0
        or alpha < 0
        or beta < 0
        or (alpha + beta) >= 1
    ):
        return None

    next_variance = float(
        fit_result.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    )
    if not np.isfinite(next_variance) or next_variance <= 0:
        return None

    return omega, alpha, beta, next_variance

def fit_and_forecast_garch_expanding(group):
    """
    For a single company's data (sorted by date), fit GARCH(1,1) with
    ANNUAL expanding-window re-estimation, then compute the variance
    path and averaged 21-day-ahead forecast at every eligible date.
    """
    group = group.copy().reset_index(drop=True)
    group['forecast_garch'] = np.nan

    returns_pct = (group['dlyret'].astype('float64') * 100).to_numpy()
    dates = group['dlycaldt'].values
    years = group['year'].values

    # Determine, for each calendar year present, whether enough expanding
    # history exists by the START of that year to attempt a fit, and if
    # so, fit params using all data strictly before that year.
    year_params = {} # year -> (omega, alpha, beta, initial variance) or None
    fit_log = [] # (year, status) pairs, for reporting

    for y in sorted(set(years)):
        train_idx = np.where(years < y)[0]
        if len(train_idx) < MIN_TRAIN_OBS:
            year_params[y] = None
            fit_log.append((y, 'insufficient_history'))
            continue

        train_returns = returns_pct[train_idx]
        train_returns = train_returns[~np.isnan(train_returns)]
        if len(train_returns) < MIN_TRAIN_OBS:
            year_params[y] = None
            fit_log.append((y, 'insufficient_history'))
            continue

        params = fit_garch_params(pd.Series(train_returns))
        if params is None:
            year_params[y] = None
            fit_log.append((y, 'fit_failed_or_degenerate'))
        else:
            omega, alpha, beta, _ = params
            long_run_var = omega / (1 - alpha - beta)
            sample_var = np.var(train_returns)
            if sample_var <= 0 or not (0.05 <= long_run_var / sample_var <= 20):
                year_params[y] = None
                fit_log.append((y, 'implausible_unconditional_variance'))
            else:
                year_params[y] = params
                fit_log.append((y, 'ok'))

    # Recurse sigma^2 continuously while ensuring that every forecast uses
    # the parameter set available at its own origin date. In particular, a
    # December origin cannot use parameters that only become active in January.
    _, forecast_garch = compute_garch_path_and_forecasts(
        returns_pct=returns_pct,
        years=years,
        year_params=year_params,
        forecast_horizon=FORECAST_HORIZON,
    )
    group['forecast_garch'] = forecast_garch

    return group, fit_log

# fit + forecast per company, with annual expanding re-fits
print("Fitting GARCH(1,1) per company with annual expanding-window "
      f"re-estimation -- will take a while "
      f"({data['permno'].nunique()} companies, multiple fits each)...")

results = []
all_fit_logs = []
for permno, group in data.groupby('permno'):
    result_group, fit_log = fit_and_forecast_garch_expanding(group)
    results.append(result_group)
    all_fit_logs.extend(fit_log)

data = pd.concat(results, ignore_index=True)

fit_log_df = pd.DataFrame(all_fit_logs, columns=['year', 'status'])
print("Fit status counts, across all company-year attempts")
print(fit_log_df['status'].value_counts())
print()
print("Fit status by year (test period years only)")
print(fit_log_df[fit_log_df['year'] >= 2015].groupby(['year', 'status']).size())
print()

n_valid_forecast = data['forecast_garch'].notna().sum()
print(f"Rows with a valid GARCH forecast: {n_valid_forecast} out of {len(data)}")
print()

# split into train / test
test = data[(data['dlycaldt'] >= TEST_START) & (data['in_sp500_membership'])]

print(f"Test: {len(test)} rows ({test['dlycaldt'].min().date()} to "
      f"{test['dlycaldt'].max().date()})")
print(f"Companies with at least one valid GARCH forecast in test period: "
      f"{test.loc[test['forecast_garch'].notna(), 'permno'].nunique()}")
print()


# comparison against prior baselines 
# this comparison is only fully apples-to-apples on the SUBSET of
# rows where all three models have a valid forecast
# GARCH may still have fewer valid rows early in the test period for
# companies that entered close to 2015 without quite enough history yet.
print("Comparison against prior baselines (each on its own full valid set)")
for name, path, col in [
    ('Naive persistence', 'models/classical/forecasts_historical_vol.parquet', 'forecast_naive'),
    ('EWMA', 'models/classical/forecasts_ewma.parquet', 'forecast_ewma'),
]:
    try:
        prior = pd.read_parquet(path)
        evaluate_all(actual=prior['target_forward_21d'], forecast=prior[col], label=name)
        print()
    except FileNotFoundError:
        print(f"({name} forecasts not found, skipping comparison)")
        print()
 
# apples-to-apples comparison on the common valid subset
naive_df = pd.read_parquet('models/classical/forecasts_historical_vol.parquet')
ewma_df = pd.read_parquet('models/classical/forecasts_ewma.parquet')
 
common = test[['permno', 'dlycaldt', 'target_forward_21d', 'forecast_garch']].merge(
    naive_df[['permno', 'dlycaldt', 'forecast_naive']], on=['permno', 'dlycaldt'], how='inner'
).merge(
    ewma_df[['permno', 'dlycaldt', 'forecast_ewma']], on=['permno', 'dlycaldt'], how='inner'
)
common = common.dropna(subset=['target_forward_21d', 'forecast_garch', 'forecast_naive', 'forecast_ewma'])
 
print(f"Apples-to-apples comparison on common valid rows: {len(common)}")
for name, col in [('Naive', 'forecast_naive'), ('EWMA', 'forecast_ewma'), ('GARCH', 'forecast_garch')]:
    evaluate_all(actual=common['target_forward_21d'], forecast=common[col], label=name)
    print()


extreme = data[data['forecast_garch'] > 1.0]  # 1.0 = 100% daily-equivalent, clearly absurd
print(f"Rows with absurd GARCH forecasts (>1.0): {len(extreme)}")
print(extreme[['permno', 'dlycaldt', 'year', 'forecast_garch']].sort_values('forecast_garch', ascending=False).head(20))
print(f"Unique companies affected: {extreme['permno'].nunique()}")


# save
output = test[['permno', 'gvkey', 'dlycaldt', 'target_forward_21d', 'forecast_garch']]
output.to_parquet('models/classical/forecasts_garch.parquet', index=False)
print(f"Saved {len(output)} test-period forecasts to "
      f"models/classical/forecasts_garch.parquet")
