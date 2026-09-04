"""Pull and validate daily VIX OHLC data from WRDS Cboe, 2005-2024."""

import os
from pathlib import Path

import pandas as pd
import psycopg2


SAMPLE_START = pd.Timestamp("2005-01-01")
SAMPLE_END = pd.Timestamp("2024-12-31")
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DATABASE = "wrds"
DEFAULT_WRDS_USERNAME = "deepshah"
OUTPUT_PATH = Path("data/vix_2005_2024.parquet")
CRSP_PATH = Path("data/crsp_daily_prices_sp500_2005_2024.parquet")

SOURCE_COLUMNS = ["date", "vixo", "vixh", "vixl", "vix"]
OUTPUT_COLUMNS = ["date", "vix_close"]


def clean_vix_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, int, list[pd.Timestamp]]:
    """Convert WRDS VIX fields to a validated, analysis-ready daily table.

    WRDS contains a small number of calendar placeholders where every VIX
    field is null. Those rows are removed. Partial OHLC missingness is treated
    as an error rather than being silently filled.
    """
    missing_columns = set(SOURCE_COLUMNS).difference(raw.columns)
    if missing_columns:
        raise KeyError(f"VIX data are missing required columns: {sorted(missing_columns)}")

    vix = raw[SOURCE_COLUMNS].copy()
    vix["date"] = pd.to_datetime(vix["date"])

    if vix["date"].isna().any():
        raise ValueError("VIX data contain a missing or invalid date.")
    if vix.duplicated("date").any():
        duplicate_count = int(vix.duplicated("date").sum())
        raise ValueError(f"VIX data contain {duplicate_count} duplicate dates.")

    source_value_columns = SOURCE_COLUMNS[1:]
    for column in source_value_columns:
        vix[column] = pd.to_numeric(vix[column], errors="coerce")

    all_ohlc_missing = vix[source_value_columns].isna().all(axis=1)
    partial_ohlc_missing = vix[source_value_columns].isna().any(axis=1) & ~all_ohlc_missing
    if partial_ohlc_missing.any():
        dates = vix.loc[partial_ohlc_missing, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"VIX data contain partial OHLC missingness on: {dates}")

    placeholder_count = int(all_ohlc_missing.sum())
    vix = vix.loc[~all_ohlc_missing].copy()
    vix = vix.rename(
        columns={
            "vixo": "vix_open",
            "vixh": "vix_high",
            "vixl": "vix_low",
            "vix": "vix_close",
        }
    )
    vix = vix.sort_values("date").reset_index(drop=True)

    value_columns = ["vix_open", "vix_high", "vix_low", "vix_close"]
    if (vix[value_columns] <= 0).any(axis=None):
        raise ValueError("VIX OHLC values must all be strictly positive.")

    invalid_trading_range = vix["vix_high"] < vix["vix_low"]
    invalid_trading_range |= ~vix["vix_close"].between(
        vix["vix_low"], vix["vix_high"]
    )
    if invalid_trading_range.any():
        invalid_dates = vix.loc[invalid_trading_range, "date"]
        dates = invalid_dates.dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"VIX close falls outside its high-low range on: {dates}")

    open_outside_range = ~vix["vix_open"].between(vix["vix_low"], vix["vix_high"])
    open_anomaly_dates = vix.loc[open_outside_range, "date"].tolist()

    if vix["date"].min() < SAMPLE_START or vix["date"].max() > SAMPLE_END:
        raise ValueError("VIX data fall outside the locked 2005-2024 sample window.")

    return vix[OUTPUT_COLUMNS], placeholder_count, open_anomaly_dates


def validate_crsp_date_coverage(
    vix: pd.DataFrame,
    crsp_dates: pd.Series,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Require a valid VIX close for every CRSP trading date."""
    vix_dates = pd.DatetimeIndex(vix["date"].drop_duplicates())
    equity_dates = pd.DatetimeIndex(pd.to_datetime(crsp_dates).drop_duplicates())

    missing_vix_dates = equity_dates.difference(vix_dates)
    extra_vix_dates = vix_dates.difference(equity_dates)
    if len(missing_vix_dates) > 0:
        dates = missing_vix_dates.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"VIX is missing on CRSP trading dates: {dates}")

    return missing_vix_dates, extra_vix_dates


def fetch_vix_from_wrds() -> pd.DataFrame:
    """Fetch the locked VIX sample using PostgreSQL and local pgpass auth."""
    username = os.environ.get("WRDS_USERNAME", DEFAULT_WRDS_USERNAME)
    query = """
        select date, vixo, vixh, vixl, vix
        from cboe.cboe
        where date between %s and %s
        order by date
    """

    connection = psycopg2.connect(
        host=WRDS_HOST,
        port=WRDS_PORT,
        dbname=WRDS_DATABASE,
        user=username,
        connect_timeout=30,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (SAMPLE_START.date(), SAMPLE_END.date()))
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
    finally:
        connection.close()

    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    print("Pulling daily VIX OHLC data from WRDS Cboe...")
    raw = fetch_vix_from_wrds()
    print(f"Raw rows: {len(raw):,}")

    vix, placeholder_count, open_anomaly_dates = clean_vix_data(raw)
    print(f"All-null non-trading placeholders removed: {placeholder_count:,}")
    print(
        "Open-only source anomalies excluded from the saved fields: "
        f"{len(open_anomaly_dates):,}"
    )
    if open_anomaly_dates:
        dates = [date.strftime("%Y-%m-%d") for date in open_anomaly_dates]
        print(f"Open-only anomaly dates: {dates}")
    print(f"Validated VIX rows: {len(vix):,}")
    print(f"Date range: {vix['date'].min().date()} to {vix['date'].max().date()}")
    print(f"VIX close range: {vix['vix_close'].min():.2f} to {vix['vix_close'].max():.2f}")

    if not CRSP_PATH.exists():
        raise FileNotFoundError(
            f"CRSP data not found at {CRSP_PATH}; run data/pull_crsp_prices.py first."
        )

    crsp_dates = pd.read_parquet(CRSP_PATH, columns=["dlycaldt"])["dlycaldt"]
    _, extra_vix_dates = validate_crsp_date_coverage(vix, crsp_dates)
    print(f"CRSP trading dates covered: {crsp_dates.nunique():,}")
    print(f"Additional valid Cboe dates retained: {len(extra_vix_dates):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    vix.to_parquet(OUTPUT_PATH, index=False)

    saved = pd.read_parquet(OUTPUT_PATH)
    if len(saved) != len(vix) or saved.duplicated("date").any():
        raise RuntimeError("Saved VIX parquet failed round-trip validation.")

    print(f"Saved {len(saved):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
