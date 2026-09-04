"""Utilities for applying point-in-time index membership to security data."""

import pandas as pd


def attach_membership_flag(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    permno_col: str = "permno",
    date_col: str = "dlycaldt",
    start_col: str = "universe_start",
    end_col: str = "universe_end",
    flag_col: str = "in_sp500_membership",
) -> pd.DataFrame:
    """Tag each security-date row without expanding repeat membership spells.

    A PERMNO can leave and later re-enter the S&P 500. A direct merge from
    prices to membership intervals duplicates the security's complete price
    history once per spell. This function instead evaluates all intervals for
    each PERMNO and returns exactly one row for every input price row.
    """
    required_price_cols = {permno_col, date_col}
    required_membership_cols = {permno_col, start_col, end_col}

    missing_price_cols = required_price_cols.difference(prices.columns)
    missing_membership_cols = required_membership_cols.difference(membership.columns)
    if missing_price_cols:
        raise KeyError(f"Prices are missing required columns: {sorted(missing_price_cols)}")
    if missing_membership_cols:
        raise KeyError(
            "Membership data are missing required columns: "
            f"{sorted(missing_membership_cols)}"
        )

    duplicate_price_rows = prices.duplicated([permno_col, date_col])
    if duplicate_price_rows.any():
        raise ValueError(
            "Expected one price row per (PERMNO, date), but found "
            f"{int(duplicate_price_rows.sum())} duplicate rows."
        )

    invalid_intervals = membership[start_col].isna() | membership[end_col].isna()
    invalid_intervals |= membership[start_col] > membership[end_col]
    if invalid_intervals.any():
        raise ValueError(
            "Membership data contain "
            f"{int(invalid_intervals.sum())} missing or reversed intervals."
        )

    unknown_permnos = pd.Index(prices[permno_col].unique()).difference(
        membership[permno_col].unique()
    )
    if len(unknown_permnos) > 0:
        raise ValueError(
            f"Found {len(unknown_permnos)} price PERMNOs with no membership interval."
        )

    result = prices.copy()
    membership_flags = pd.Series(False, index=result.index, dtype=bool)
    intervals_by_permno = {
        permno: intervals[[start_col, end_col]]
        for permno, intervals in membership.groupby(permno_col, sort=False)
    }

    for permno, row_index in result.groupby(permno_col, sort=False).groups.items():
        dates = result.loc[row_index, date_col]
        in_any_interval = pd.Series(False, index=row_index, dtype=bool)

        for interval in intervals_by_permno[permno].itertuples(index=False, name=None):
            interval_start, interval_end = interval
            in_any_interval |= dates.between(interval_start, interval_end)

        membership_flags.loc[row_index] = in_any_interval

    result[flag_col] = membership_flags

    if len(result) != len(prices):
        raise AssertionError("Membership tagging changed the number of price rows.")
    if result.duplicated([permno_col, date_col]).any():
        raise AssertionError("Membership tagging created duplicate security-date rows.")

    return result
