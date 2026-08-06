"""
build a PERMNO <-> IBES ticker link table, and provide a reusable function 
to attach the *date-correct* IBES ticker to any dataframe of (permno, date) rows
"""

import wrds
import pandas as pd

db = wrds.Connection(wrds_username='deepshah')

# get PERMNOs for test tickers from CRSP
test_tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']

stocknames_query = """
    select permno, ticker, comnam, namedt, nameenddt
    from crsp.stocknames
    where ticker in ({})
""".format(", ".join(f"'{t}'" for t in test_tickers))

stocknames = db.raw_sql(stocknames_query, date_cols=['namedt', 'nameenddt'])

print(stocknames)
print()

ibes_link_query = """
    select ticker as ibes_ticker, permno, ncusip, sdate, edate, score
    from wrdsapps.ibcrsphist
    where score in (1, 2)
"""

ibes_links = db.raw_sql(ibes_link_query, date_cols=['sdate', 'edate'])
print(ibes_links.head())
print()

test_permnos = stocknames['permno'].unique().tolist()
test_ibes_links = ibes_links[ibes_links['permno'].isin(test_permnos)]
print("IBES links for test tickers")
print(test_ibes_links[['permno', 'ibes_ticker', 'ncusip', 'sdate', 'edate', 'score']]
      .sort_values(['permno', 'sdate']))
print()

db.close()

def attach_ibes_ticker(df, ibes_link_table, permno_col='permno', date_col='date'):
    """
    Given a dataframe with a permno column and a date column, returh the same
    dataframe with a new 'ibes_ticker' column, correctly matched to the link that
    was valid on that specific date.

    Rows where no valid link exists for that permno/date will be dropped from the
    result (not silently).
    """
    link_slim = ibes_link_table[['permno', 'ibes_ticker', 'sdate', 'edate']].copy()
    link_slim['edate_filled'] = link_slim['edate'].fillna(pd.Timestamp.today())

    merged = df.merge(
        link_slim,
        left_on=permno_col,
        right_on='permno',
        how='left'
    )

    valid_mask = (
        (merged[date_col] >= merged['sdate']) &
        (merged[date_col] <= merged['edate_filled'])
    )
    result = merged[valid_mask].copy()

    # Check: did any (permno, date) pair match more than one link row?
    dupe_check = result.groupby([permno_col, date_col]).size()
    dupes = dupe_check[dupe_check > 1]
    if len(dupes) > 0:
        print(f"!! WARNING: {len(dupes)} (permno, date) pairs matched multiple "
              f"overlapping IBES links. Inspect before trusting.")

    # Check: did any (permno, date) pair fail to match any link at all?
    matched_keys = set(zip(result[permno_col], result[date_col]))
    all_keys = set(zip(df[permno_col], df[date_col]))
    unmatched = all_keys - matched_keys
    if len(unmatched) > 0:
        print(f"!! NOTE: {len(unmatched)} (permno, date) pairs had no valid "
              f"IBES link on that date.")

    keep_cols = list(df.columns) + ['ibes_ticker']
    return result[keep_cols].reset_index(drop=True)




if __name__ == '__main__':
    test_permnos = stocknames['permno'].unique().tolist()

    test_dates_multi = pd.DataFrame([
        {'permno': p, 'date': d}
        for p in test_permnos
        for d in pd.to_datetime(['1998-06-15', '2005-06-15', '2015-06-15'])
    ])

    result = attach_ibes_ticker(test_dates_multi, ibes_links)
    print(result.sort_values(['permno', 'date']))