"""
construct the volatility target variable -- a 21-day rolling window standard deviation of daily total
returns, computed per company, from the validated CRSP daily price/return data.
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW = 21

# load the validated CRSP price data
prices = pd.read_parquet('data/crsp_daily_prices_sp500_2005_2024.parquet')
print(f"Loaded {len(prices)} rows, {prices['permno'].nunique()} unique PERMNOs")
print()

prices = prices.sort_values(['permno', 'dlycaldt']).reset_index(drop=True)

# drop rows with missing returns before computing volatility
n_missing = prices['dlyret'].isna().sum()
print(f"Rows with missing dlyret: {n_missing}")
print()

# compute the rolling volatility target, per company
# groupby + rolling ensures the window never crosses a PERMNO boundary
prices['volatility_21d'] = (
    prices.groupby('permno')['dlyret']
    .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std())
)

n_valid_target = prices['volatility_21d'].notna().sum()
print(f"Rows with a valid 21-day rolling volatility target: "
      f"{n_valid_target} out of {len(prices)}")
print("(First 20 trading days per company will have no target -- expected, "
      "not enough history yet for a full 21-day window.)")
print()

# summary statistics
print("Volatility target summary statistics (daily std of returns)")
print(prices['volatility_21d'].describe())
print()

# validate against the COVID crash (March 2020)
covid_window = prices[
    prices['dlycaldt'].between('2020-02-15', '2020-04-15')
]
covid_avg_by_date = covid_window.groupby('dlycaldt')['volatility_21d'].mean()
print("Average 21-day rolling volatility across universe, Feb 15 - Apr 15 2020")
print(covid_avg_by_date.to_string())
print()

pre_covid_avg = prices[
    prices['dlycaldt'].between('2019-10-01', '2019-12-31')
]['volatility_21d'].mean()
covid_peak = covid_avg_by_date.max()
print(f"Pre-COVID average (Q4 2019): {pre_covid_avg:.5f}")
print(f"COVID-window peak average: {covid_peak:.5f}")
print(f"Ratio (peak / pre-COVID): {covid_peak / pre_covid_avg:.2f}x")
print()

# single-company sanity check (AAPL)
aapl_covid = prices[
    (prices['permno'] == 14593) &
    (prices['dlycaldt'].between('2020-02-01', '2020-04-30'))
]
print("AAPL 21-day rolling volatility, Feb-Apr 2020 (spot check)")
print(aapl_covid[['dlycaldt', 'dlyret', 'volatility_21d']].to_string(index=False))
print()

# save
output = prices[['permno', 'gvkey', 'dlycaldt', 'dlyret', 'dlyprc', 'in_sp500_membership', 'volatility_21d']]
output.to_parquet('data/target_variable_sp500_2005_2024.parquet', index=False)
print(f"Saved {len(output)} rows to data/target_variable_sp500_2005_2024.parquet")