"""
pull quarterly earnings announcement dates from IBES, join them to the S&P 500 universe via the validated
PERMO <-> IBES ticker link and produce a (permno, earnings_date) table for the 2005-2024 sample window.
"""

import wrds
import pandas as pd

SAMPLE_START = pd.Timestamp('2005-01-01')
SAMPLE_END = pd.Timestamp('2024-12-31')

db = wrds.Connection(wrds_username='deepshah')

# pull broad unfiltered-by-measure sample to check the
# actual pdicity / measure values
check_query = """
    select ticker, pdicity, measure, anndats, pends, usfirm
    from ibes.act_epsus
    where usfirm = 1
        and anndats between '2005-01-01' and '2024-12-31'
    limit 500000
"""
check_sample = db.raw_sql(check_query, date_cols=['anndats', 'pends'])

print("pdicity value counts (US firms, 2005-2024 sample)")
print(check_sample['pdicity'].value_counts())
print()

print("measure value counts (US firms, 2005-2024 sample)")
print(check_sample['measure'].value_counts())
print()

# pull the real earnings announcement data, filtered
earnings_query = """
    select ticker, cname, pends, anndats, measure, pdicity
    from ibes.act_epsus
    where usfirm = 1
        and pdicity = 'QTR'
        and measure = 'EPS'
        and anndats between '2005-01-01' and '2024-12-31'
"""
earnings = db.raw_sql(earnings_query, date_cols=['anndats', 'pends'])
print(f"Quaterly EPS earnings announcements pulled: {len(earnings)} rows")
print(earnings.head())
print()

# drop exact duplicate rows
before_dedup = len(earnings)
earnings = earnings.drop_duplicates()
after_dedup = len(earnings)
if before_dedup != after_dedup:
    print(f"Dropped {before_dedup - after_dedup} exact duplicate rows.")

print(f"Quaterly EPS earnings announcements pulled: {len(earnings)} rows.")
print(earnings.head())
print()

# Check: is it one row per ticker (ticker, pends) ?
dupe_check = earnings.groupby(['ticker', 'pends']).size()
dupes = dupe_check[dupe_check > 1]
if len(dupes) > 0:
    print(f"!! WARNING: {len(dupes)} (ticker, period-end) pairs have multiple "
          f"announcement rows. Inspect.")
    print(dupes.head())
 
    # Known case: CNHI 2013-06-30 has two distinct anndats, likely an
    # artifact of the 2013 Fiat Industrial / CNH Global merger. Keep the
    # earlier announcement date (first date the market actually had the
    # information) and drop the later duplicate. At 1 row out of ~470K,
    # this is handled explicitly rather than with a blanket rule, since
    # a blanket "keep earliest" rule could silently mask a different kind
    # of problem in a future data pull.
    earnings = earnings.sort_values('anndats').drop_duplicates(
        subset=['ticker', 'pends'], keep='first'
    )
    remaining = earnings.groupby(['ticker', 'pends']).size()
    remaining_dupes = remaining[remaining > 1]
    if len(remaining_dupes) > 0:
        print(f"!! STILL {len(remaining_dupes)} duplicate groups after "
              f"keep-first fix -- needs manual review.")
    else:
        print("Resolved: kept earliest anndats per (ticker, pends).")
else:
    print("OK: exactly one row per (ticker, period-end) -- no duplicate announcements.")
print()

# pull validated PERMNO <-> IBES ticker link
ibes_link_query = """
    select ticker as ibes_ticker, permno, sdate, edate, score
    from wrdsapps.ibcrsphist
    where score in (1, 2)
"""
ibes_links = db.raw_sql(ibes_link_query, date_cols=['sdate', 'edate'])

# load the locked universe
universe = pd.read_parquet('data/universe_sp500_2005_2024.parquet')
universe_permnos = universe['permno'].unique().tolist()

db.close()

def attach_permno_to_earnings(earnings_df, ibes_link_table):
    """
    Given earnings announcements (with 'ticker' and 'anndats'), attach the 
    PERMNO whose IBES ticker link was valid on announcement date. 
    """
    link_slim = ibes_link_table[['permno', 'ibes_ticker', 'sdate', 'edate']].copy()
    link_slim['edate_filled'] = link_slim['edate'].fillna(pd.Timestamp.today())

    merged = earnings_df.merge(
        link_slim,
        left_on='ticker',
        right_on='ibes_ticker',
        how='left'
    )

    valid_mask = (
        (merged['anndats'] >= merged['sdate']) &
        (merged['anndats'] <= merged['edate_filled'])
    )
    result = merged[valid_mask].copy()

    dupe_check = result.groupby(['ticker', 'pends']).size()
    dupes = dupe_check[dupe_check > 1]
    if len(dupes) > 0:
        print(f"!! WARNING: {len(dupes)} announcements matched multiple PERMNOs. "
              f"Inspect.")

    unmatched = len(earnings_df) - len(result)
    if unmatched > 0:
        print(f"!! NOTE: {unmatched} announcements has no valid PERMNO link "
              f"(IBES ticker not found in link table for that date, or no "
              f"score 1-2 match exists).")

    return result

earnings_with_permno = attach_permno_to_earnings(earnings, ibes_links)
print(f"Earnings announcements with PERMNO attached: {len(earnings_with_permno)} rows")
print()

# restrict to our locked universe
earnings_in_universe = earnings_with_permno[
    earnings_with_permno['permno'].isin(universe_permnos)
].copy()
print(f"Earnings announcements within our S&P 500 universe: "
      f"{len(earnings_in_universe)} rows, "
      f"{earnings_in_universe['permno'].nunique()} unique companies")
print()

# sanity check against real earnings data
# Apple's fiscal Q1 2025 earnings were announced 2015-01-27.
aapl_check = earnings_in_universe[
    (earnings_in_universe['permno'] == 14593) &
    (earnings_in_universe['anndats'].between('2015-01-01', '2015-02-15'))
]
print("Sanity check: AAPL earnings announcement, Jan 2015")
print(aapl_check[['permno', 'ticker', 'pends', 'anndats']])
print()

# save the output
final = earnings_in_universe[['permno', 'ticker', 'pends', 'anndats']].sort_values(
    ['permno', 'anndats']
)
final.to_parquet('data/earnings_dates_sp500_2005_2024.parquet', index=False)
print(f"Saved {len(final)} rows to data/earnings_dates_sp500_2005_2024.parquet")
