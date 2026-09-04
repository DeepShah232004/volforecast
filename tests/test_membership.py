import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from membership import attach_membership_flag


class AttachMembershipFlagTests(unittest.TestCase):
    def test_repeat_membership_spells_do_not_duplicate_prices(self):
        prices = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001, 10001, 10002],
                "dlycaldt": pd.to_datetime(
                    ["2010-01-04", "2011-01-03", "2012-01-03", "2013-01-02", "2010-01-04"]
                ),
                "dlyret": [0.01, 0.02, -0.01, 0.03, 0.005],
            }
        )
        membership = pd.DataFrame(
            {
                "permno": [10001, 10001, 10002],
                "universe_start": pd.to_datetime(
                    ["2010-01-01", "2012-01-01", "2009-01-01"]
                ),
                "universe_end": pd.to_datetime(
                    ["2010-12-31", "2012-12-31", "2014-12-31"]
                ),
            }
        )

        result = attach_membership_flag(prices, membership)

        self.assertEqual(len(result), len(prices))
        self.assertFalse(result.duplicated(["permno", "dlycaldt"]).any())
        self.assertListEqual(
            result["in_sp500_membership"].tolist(),
            [True, False, True, False, True],
        )

    def test_duplicate_price_keys_fail_fast(self):
        prices = pd.DataFrame(
            {
                "permno": [10001, 10001],
                "dlycaldt": pd.to_datetime(["2010-01-04", "2010-01-04"]),
            }
        )
        membership = pd.DataFrame(
            {
                "permno": [10001],
                "universe_start": pd.to_datetime(["2010-01-01"]),
                "universe_end": pd.to_datetime(["2010-12-31"]),
            }
        )

        with self.assertRaisesRegex(ValueError, "one price row"):
            attach_membership_flag(prices, membership)


if __name__ == "__main__":
    unittest.main()
