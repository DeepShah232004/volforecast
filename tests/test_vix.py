import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from pull_vix import clean_vix_data, validate_crsp_date_coverage


def valid_vix_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-02"],
            "vixo": [13.0, 14.0],
            "vixh": [15.0, 16.0],
            "vixl": [12.0, 13.0],
            "vix": [14.0, 15.0],
        }
    )


class CleanVixDataTests(unittest.TestCase):
    def test_drops_only_all_null_placeholders_and_sorts_dates(self):
        raw = valid_vix_rows()
        raw.loc[len(raw)] = ["2020-01-04", np.nan, np.nan, np.nan, np.nan]

        cleaned, placeholder_count, open_anomaly_dates = clean_vix_data(raw)

        self.assertEqual(placeholder_count, 1)
        self.assertEqual(open_anomaly_dates, [])
        self.assertListEqual(
            cleaned.columns.tolist(),
            ["date", "vix_close"],
        )
        self.assertListEqual(
            cleaned["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-01-02", "2020-01-03"],
        )

    def test_rejects_partial_ohlc_missingness(self):
        raw = valid_vix_rows()
        raw.loc[0, "vixh"] = None

        with self.assertRaisesRegex(ValueError, "partial OHLC"):
            clean_vix_data(raw)

    def test_rejects_duplicate_dates(self):
        raw = valid_vix_rows()
        raw.loc[1, "date"] = raw.loc[0, "date"]

        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            clean_vix_data(raw)

    def test_rejects_invalid_ohlc_ranges(self):
        raw = valid_vix_rows()
        raw.loc[0, "vixh"] = 13.5

        with self.assertRaisesRegex(ValueError, "outside its high-low range"):
            clean_vix_data(raw)

    def test_reports_open_anomaly_without_saving_open(self):
        raw = valid_vix_rows()
        raw.loc[0, "vixo"] = 41.6

        cleaned, _, open_anomaly_dates = clean_vix_data(raw)

        self.assertNotIn("vix_open", cleaned.columns)
        self.assertEqual(open_anomaly_dates, [pd.Timestamp("2020-01-03")])


class VixCoverageTests(unittest.TestCase):
    def test_requires_every_crsp_date(self):
        cleaned, _, _ = clean_vix_data(valid_vix_rows())
        crsp_dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-06"]))

        with self.assertRaisesRegex(ValueError, "2020-01-06"):
            validate_crsp_date_coverage(cleaned, crsp_dates)


if __name__ == "__main__":
    unittest.main()
