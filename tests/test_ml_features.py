import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from build_ml_features import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    add_return_features,
    add_targets_and_splits,
    build_ml_feature_matrix,
    build_vix_features,
)


class ReturnFeatureTests(unittest.TestCase):
    def test_rolling_features_use_only_returns_through_origin(self):
        dates = pd.bdate_range("2020-01-01", periods=253)
        base = pd.DataFrame(
            {
                "permno": 10001,
                "dlycaldt": dates,
                "dlyret": np.linspace(-0.02, 0.02, len(dates)),
            }
        )
        changed_future = base.copy()
        changed_future.loc[252, "dlyret"] = 2.0

        original = add_return_features(base)
        changed = add_return_features(changed_future)

        for feature in FEATURE_COLUMNS:
            if feature.startswith("vix_"):
                continue
            self.assertAlmostEqual(
                original.loc[251, feature],
                changed.loc[251, feature],
                msg=f"{feature} changed before the altered future return",
            )

    def test_compounded_return_is_geometric_not_a_return_sum(self):
        frame = pd.DataFrame(
            {
                "permno": [10001] * 5,
                "dlycaldt": pd.bdate_range("2020-01-01", periods=5),
                "dlyret": [0.10, -0.10, 0.05, 0.00, 0.02],
            }
        )

        result = add_return_features(frame)
        expected = np.prod(1 + frame["dlyret"].to_numpy()) - 1

        self.assertAlmostEqual(result.loc[4, "compounded_return_5d"], expected)


class VixFeatureTests(unittest.TestCase):
    def test_lags_follow_crsp_calendar_not_extra_vix_dates(self):
        vix = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
                ),
                "vix_close": [10.0, 999.0, 20.0, 40.0],
            }
        )
        crsp_dates = pd.Series(
            pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-07"])
        )

        result = build_vix_features(vix, crsp_dates)

        self.assertAlmostEqual(
            result.loc[1, "vix_log_change_1d"],
            np.log(20.0) - np.log(10.0),
        )


class SplitBoundaryTests(unittest.TestCase):
    def test_forward_targets_crossing_split_boundaries_are_purged(self):
        frame = pd.DataFrame(
            {
                "permno": [10001] * 50 + [10002] * 50,
                "dlycaldt": list(pd.bdate_range("2012-11-20", periods=50))
                + list(pd.bdate_range("2014-11-20", periods=50)),
                "volatility_21d": np.linspace(0.01, 0.03, 100),
            }
        )

        result = add_targets_and_splits(frame)

        train_crossing = result[
            result["split"].eq("train")
            & result["target_end_date"].ge("2013-01-01")
        ]
        validation_crossing = result[
            result["split"].eq("validation")
            & result["target_end_date"].ge("2015-01-01")
        ]
        self.assertTrue(train_crossing["crosses_split_boundary"].all())
        self.assertTrue(validation_crossing["crosses_split_boundary"].all())
        self.assertFalse(
            result.loc[
                result["target_end_date"].notna()
                & ~result.index.isin(train_crossing.index)
                & ~result.index.isin(validation_crossing.index),
                "crosses_split_boundary",
            ].any()
        )


class MatrixSchemaTests(unittest.TestCase):
    def test_output_exposes_only_locked_features_and_audit_columns(self):
        dates = pd.bdate_range("2014-01-01", periods=300)
        target = pd.DataFrame(
            {
                "permno": [10001] * len(dates),
                "gvkey": ["001234"] * len(dates),
                "dlycaldt": dates,
                "dlyret": np.linspace(-0.02, 0.02, len(dates)),
                "dlyprc": np.linspace(10.0, 20.0, len(dates)),
                "in_sp500_membership": [True] * len(dates),
                "volatility_21d": [0.02] * len(dates),
            }
        )
        vix = pd.DataFrame(
            {
                "date": dates,
                "vix_close": np.linspace(12.0, 18.0, len(dates)),
            }
        )

        result = build_ml_feature_matrix(target, vix)

        self.assertListEqual(result.columns.tolist(), OUTPUT_COLUMNS)
        self.assertNotIn("dlyprc", result.columns)
        self.assertNotIn("dlyret", result.columns)


if __name__ == "__main__":
    unittest.main()
