"""
build a PERMNO <-> GVKEY link table (via WRDS's CRSP/Compustat Merged linking table), and provide a reusable function to attach *date-correct* GVKEY to any dataframe of (permno, date) rows
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

# pull the CCM linking table, filtered to good-quality links
link_query = """
    select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
    from crsp_a_ccm.ccmxpf_linktable
    where linktype in ('LC', 'LU')
        and linkprim in ('P', 'C')
"""

links = db.raw_sql(link_query, date_cols=['linkdt', 'linkenddt'])
print(links.head())
print()

db.close()

def attach_gvkey(df, link_table, permno_col='permno', date_col='date'):
    """
    Given a dataframe with a permno column and a date column, return the same dataframe
    with a new 'gvkey' column, correctly matched to the link that was valid on that
    specific date.

    Rows where no valid link exists for that permno/date (e.g. before the company's link
    starts, or after it ends and no new link took over) will have gvkey = NaN -- these
    should be inspected, not silently dropped, since they may indicate real data issues.
    """
    link_slim = link_table[['permno', 'gvkey', 'linkdt', 'linkenddt']].copy()
    link_slim['linkenddt_filled'] = link_slim['linkenddt'].fillna(pd.Timestamp.today())

    merged = df.merge(
        link_slim,
        left_on = permno_col,
        right_on = "permno",
        how="left"
    )

    valid_mask = (
        (merged[date_col] >= merged['linkdt']) &
        (merged[date_col] <= merged['linkenddt_filled'])
    )
    result = merged[valid_mask].copy()

    # Check: did any (permno, date) pair match more than one link row?
    dupe_check = result.groupby([permno_col, date_col]).size()
    dupes = dupe_check[dupe_check > 1]

    if len(dupes) > 0:
        print(f"!! WARNING: {len(dupes)} (permno, date) pairs matched multiple "
              f"overlapping links. Inspect before trusting results.")

    # Check: did any (permno, date) pair fail to match any link at all?
    matched_keys = set(zip(result[permno_col], result[date_col]))
    all_keys = set(zip(df[permno_col], df[date_col]))
    unmatched = all_keys - matched_keys
    if len(unmatched) > 0:
        print(f"!! NOTE: {len(unmatched)} (permno, date) pairs had no valid "
              f"link on that date.")

    keep_cols = list(df.columns) + ['gvkey']
    return result[keep_cols].reset_index(drop=True)



if __name__ == '__main__':
    test_permnos = stocknames['permno'].unique().tolist()

    test_dates_multi = pd.DataFrame([
        {'permno': p, 'date': d}
        for p in test_permnos
        for d in pd.to_datetime(['1998-06-15', '2005-06-15', '2015-06-15'])
    ])

    result = attach_gvkey(test_dates_multi, links)
    print(result.sort_values(['permno', 'date']))