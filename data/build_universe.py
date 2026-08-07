"""
construct the study's equity universe as historical S&P 500 constituents,
avoiding survivorship bias by using membership start/end dates rather than current/fixed-date snapshot.

Sample period: 2005-01-01 to 2024-12-31.
A stock is included only for the portion of this window during which it was actually an S&P 500 member.
"""

import wrds
import pandas as pd

SAMPLE_START = pd.Timestamp('2005-01-01')
SAMPLE_END = pd.Timestamp('2024-12-31')

db = wrds.Connection(wrds_username='deepshah')

# pull S&P500 membership history
membership_query = """
    select permno, indno, mbrstartdt, mbrenddt, mbrflg, indfam
    from crsp_a_indexes.dsp500list_v2
    where mbrflg = 'NORM'
"""

membership = db.raw_sql(membership_query, date_cols=['mbrstartdt', 'mbrenddt'])
print(membership.head())
print()

# Check: is indno/indfam actually constant ?
print("Unique indno / indfam values (should be a single S&P 500 index)")
print(membership[['indno', 'indfam']].drop_duplicates())
print()

# filter rows overlapping the sample window
membership['mbrenddt_filled'] = membership['mbrenddt'].fillna(pd.Timestamp.today())

in_window = membership[
    (membership['mbrstartdt'] <= SAMPLE_END) &
    (membership['mbrenddt_filled'] >= SAMPLE_START)
].copy()

# clip each membersip window to the sample period
in_window['universe_start'] = in_window['mbrstartdt'].clip(lower=SAMPLE_START)  # type: ignore[call-overload]
in_window['universe_end'] = in_window['mbrenddt_filled'].clip(upper=SAMPLE_END)  # type: ignore[call-overload]

print(f"Membership rows overlapping {SAMPLE_START.date()} - {SAMPLE_END.date()}: "
      f"{len(in_window)}")
print(f"Unique PERMNOs in universe: {in_window['permno'].nunique()}")
print()

# sanity checks
stocknames_query = """
    select distinct permno, ticker, comnam
    from crsp.stocknames
"""

all_stocknames = db.raw_sql(stocknames_query)

db.close()

# Lehman Brothers and Bear Stearns should show a removal date in 2008
for name_fragment in ['LEHMAN', 'BEAR STEARNS']:
    matches = all_stocknames[all_stocknames['comnam'].str.contains(name_fragment, na=False)]
    permnos = matches['permno'].unique()
    check = membership[membership['permno'].isin(permnos)]
    print(f"Membership history check: {name_fragment}")
    print(check[['permno', 'mbrstartdt', 'mbrenddt']])
    print()

# Tesla should show an addition date around Dec 2020.
tesla_matches = all_stocknames[all_stocknames['comnam'].str.contains('TESLA', na=False)]
tesla_permnos = tesla_matches['permno'].unique()
tesla_check = membership[membership['permno'].isin(tesla_permnos)]
print("Membership history check: TESLA")
print(tesla_check[['permno', 'mbrstartdt', 'mbrenddt']])
print()

# save the universe
final_universe = in_window[['permno', 'universe_start', 'universe_end']].sort_values('permno')
print(f"Final universe: {final_universe['permno'].nunique()} unique PERMNOs, "
      f"{len(final_universe)} membership-window rows")
print(final_universe.head(10))

final_universe.to_parquet('data/universe_sp500_2005_2024.parquet', index=False)
print("\nSaved to data/universe_sp500_2005_2024.parquet")