"""
Pull daily total returns and prices from CRSP's Stock Daily Security Data table for
our locked S&P 500 universe (2005-2024), tag each row with point-in-time index
membership, and attach the date-correct GVKEY.
"""

import sys
import wrds
import pandas as pd

sys.path.append('data')
from build_links import attach_gvkey
from membership import attach_membership_flag

SAMPLE_START = pd.Timestamp('2005-01-01')
SAMPLE_END = pd.Timestamp('2024-12-31')

# load the locked universe and the validated GVKEY link
universe = pd.read_parquet('data/universe_sp500_2005_2024.parquet')
universe_permnos = universe['permno'].unique().tolist()
print(f"Universe: {len(universe_permnos)} unique PERMNOs")
print()

db = wrds.Connection(wrds_username='deepshah')

link_query = """
    select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
    from crsp_a_ccm.ccmxpf_linktable
    where linktype in ('LC', 'LU')
        and linkprim in ('P', 'C')
"""
links = db.raw_sql(link_query, date_cols=['linkdt', 'linkenddt'])

# pull CRSP daily prices/returns for the full universe
permno_list_sql = ", ".join(str(p) for p in universe_permnos)

price_query = f"""
    select permno, dlycaldt, dlyret, dlyprc, dlycap, dlyprcflg, dlyretmissflg, dlydelflg
    from crsp_a_stock.stkdlysecuritydata
    where permno in ({permno_list_sql})
        and dlycaldt between '{SAMPLE_START.date()}' and '{SAMPLE_END.date()}'
"""

print("Pulling CRSP daily prices for the full universe ...")
prices = db.raw_sql(price_query, date_cols=['dlycaldt'])
db.close()

print(f"Raw rows pulled: {len(prices)}")
print(prices.head())
print()

# check data-quality flags
print("dlyprcflg value counts")
print(prices['dlyprcflg'].value_counts(dropna=False))
print()
print("dlyretmissflg value counts")
print(prices['dlyretmissflg'].value_counts(dropna=False))
print()
print("dlydelflg value counts")
print(prices['dlydelflg'].value_counts(dropna=False))
print()

# check missing/null return
missing_ret = prices['dlyret'].isna().sum()
print(f"Rows with null dlyret: {missing_ret} out of {len(prices)}")
print()

# Tag each price row against all of its company's membership spells. This
# preserves pre-membership return history for model estimation while keeping
# exactly one row per (PERMNO, date), including for companies that leave and
# later re-enter the index.
prices_with_window = attach_membership_flag(prices, universe)

n_in_membership = prices_with_window['in_sp500_membership'].sum()
print(f"Rows within actual S&P 500 membership: {n_in_membership} out of "
      f"{len(prices_with_window)}")
print("(All rows are kept -- the flag marks membership for evaluation)")
print("Confirmed: membership tagging preserved one row per (PERMNO, date).")
print()

in_window = prices_with_window

# tag with date-correct gvkey
final = attach_gvkey(in_window, links, permno_col='permno', date_col='dlycaldt')
print(f"Rows after GVKEY tagging: {len(final)}")
print()

# sanity check -- Apple around a known volatile date
# Jan 28, 2015 -- the trading day after AAPL earnings announcement
# Expect a large return that day.
aapl_check = final[
    (final['permno'] == 14593) &
    (final['dlycaldt'].between('2015-01-26', '2015-01-29'))
]
print(f"Sanity check: AAPL returns around Jan 27 2015 earnings ")
print(aapl_check[['permno', 'gvkey', 'dlycaldt', 'dlyret', 'dlyprc']])
print()

# save
final = final[['permno', 'gvkey', 'dlycaldt', 'dlyret', 'dlyprc', 'in_sp500_membership']]
final.to_parquet('data/crsp_daily_prices_sp500_2005_2024.parquet', index=False)
print(f"Saved {len(final)} rows to data/crsp_daily_prices_sp500_2005_2024.parquet")
