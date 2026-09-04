import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from build_classical_panel import (
    add_forward_earnings_flags,
    assign_vix_regime,
    compute_vix_regime_thresholds,
)


class VixRegimeTests(unittest.TestCase):
    def test_thresholds_use_only_pre_test_vix(self):
        vix = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2010-01-04", "2010-01-05", "2010-01-06", "2020-03-16"]
                ),
                "vix_close": [10.0, 20.0, 30.0, 1000.0],
            }
        )

        low, high = compute_vix_regime_thresholds(vix)

        self.assertAlmostEqual(low, 10.0 + (20.0 - 10.0) * 2 / 3)
        self.assertAlmostEqual(high, 20.0 + (30.0 - 20.0) * 1 / 3)

    def test_threshold_boundaries_are_inclusive(self):
        values = pd.Series([14.0, 14.01, 20.0, 24.99, 25.0, None])

        regimes = assign_vix_regime(values, low_threshold=14.0, high_threshold=25.0)

        self.assertListEqual(
            regimes.tolist(),
            ["low", "middle", "middle", "middle", "high", pd.NA],
        )


class EarningsWindowTests(unittest.TestCase):
    def test_uses_strict_origin_and_inclusive_end_boundaries(self):
        windows = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001, 10002],
                "dlycaldt": pd.to_datetime(
                    ["2020-01-01", "2020-01-31", "2020-02-28", "2020-01-01"]
                ),
                "target_end_date": pd.to_datetime(
                    ["2020-01-31", "2020-02-28", None, "2020-01-31"]
                ),
            }
        )
        earnings = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001, 10001, 10001],
                "anndats": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-15",
                        "2020-01-15",
                        "2020-01-31",
                        "2020-02-01",
                    ]
                ),
            }
        )

        result, duplicates_removed = add_forward_earnings_flags(windows, earnings)

        self.assertEqual(duplicates_removed, 1)
        self.assertListEqual(
            result["earnings_count_forward_21d"].tolist(),
            [2, 1, pd.NA, 0],
        )
        self.assertListEqual(
            result["has_earnings_forward_21d"].tolist(),
            [True, True, pd.NA, False],
        )


if __name__ == "__main__":
    unittest.main()
