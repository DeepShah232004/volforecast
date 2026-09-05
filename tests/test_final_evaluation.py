import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from build_final_panel import build_final_panel
from final_evaluation import (
    bootstrap_pooled_metric,
    classify_h1,
    classify_h2,
    classify_h3,
    h2_practically_comparable,
    moving_block_bootstrap_indices,
    positive_effect_supported,
    qlike_loss,
)


class FinalPanelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        self.classical = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001],
                "gvkey": ["1", "1", "1"],
                "dlycaldt": self.dates,
                "target_forward_21d": [0.01, 0.02, 0.03],
                "forecast_naive": [0.011, 0.021, 0.031],
                "forecast_ewma": [0.012, 0.022, 0.032],
                "forecast_garch": [0.013, 0.023, 0.033],
                "target_end_date": pd.to_datetime(
                    ["2020-02-03", "2020-02-04", "2020-02-05"]
                ),
                "vix_close": [12.0, 18.0, 30.0],
                "vix_regime": ["low", "middle", "high"],
                "earnings_count_forward_21d": [0, 1, 0],
                "has_earnings_forward_21d": [False, True, False],
                "common_valid": [True, False, True],
            }
        )
        self.rf = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001],
                "dlycaldt": self.dates,
                "target_forward_21d": [0.01, 0.02, 0.03],
                "forecast_random_forest": [0.014, 0.024, 0.034],
            }
        )
        self.xgb = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001],
                "dlycaldt": self.dates,
                "target_forward_21d": [0.01, 0.02, 0.03],
                "forecast_xgboost": [0.015, 0.025, 0.035],
            }
        )

    def test_builds_only_the_five_model_common_sample(self):
        panel = build_final_panel(self.classical, self.rf, self.xgb)

        self.assertEqual(len(panel), 2)
        self.assertListEqual(panel["dlycaldt"].tolist(), [self.dates[0], self.dates[2]])
        self.assertIn("forecast_random_forest", panel.columns)
        self.assertIn("forecast_xgboost", panel.columns)

    def test_rejects_target_mismatch(self):
        self.xgb.loc[1, "target_forward_21d"] = 0.99

        with self.assertRaisesRegex(ValueError, "targets differ"):
            build_final_panel(self.classical, self.rf, self.xgb)


class LossTests(unittest.TestCase):
    def test_qlike_is_calculated_on_variance(self):
        actual = np.array([2.0])
        forecast = np.array([1.0])

        loss = qlike_loss(actual, forecast)

        self.assertAlmostEqual(loss[0], 4.0 - np.log(4.0) - 1.0)


class BootstrapTests(unittest.TestCase):
    def test_moving_blocks_are_deterministic_and_consecutive(self):
        first = moving_block_bootstrap_indices(10, block_length=3, replicates=5, seed=42)
        second = moving_block_bootstrap_indices(10, block_length=3, replicates=5, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (5, 10))
        for row in first:
            for start in (0, 3, 6):
                np.testing.assert_array_equal(np.diff(row[start : start + 3]), [1, 1])

    def test_cluster_bootstrap_preserves_constant_rmse(self):
        values = np.full(8, 4.0)
        row_mask = np.ones(8, dtype=bool)
        date_codes = np.repeat(np.arange(4), 2)
        indices = moving_block_bootstrap_indices(
            4, block_length=2, replicates=20, seed=42
        )

        result = bootstrap_pooled_metric(
            values,
            row_mask,
            date_codes,
            n_dates=4,
            bootstrap_indices=indices,
            transform=np.sqrt,
        )

        self.assertEqual(result["estimate"], 2.0)
        self.assertEqual(result["ci_lower"], 2.0)
        self.assertEqual(result["ci_upper"], 2.0)


class HypothesisDecisionTests(unittest.TestCase):
    def test_positive_effect_requires_strictly_positive_lower_bound(self):
        self.assertTrue(positive_effect_supported(0.02, 0.001))
        self.assertFalse(positive_effect_supported(0.02, 0.0))
        self.assertFalse(positive_effect_supported(-0.01, -0.02))

    def test_h1_requires_both_ml_confidence_intervals_above_zero(self):
        full = {
            "rf": {"supports_h1": True},
            "xgb": {"supports_h1": True},
        }
        partial = {
            "rf": {"supports_h1": True},
            "xgb": {"supports_h1": False},
        }

        self.assertEqual(classify_h1(full), "full_support")
        self.assertEqual(classify_h1(partial), "partial_support")

    def test_h2_uses_both_practical_comparability_decisions(self):
        full = {
            "rf": {"practically_comparable": True},
            "xgb": {"practically_comparable": True},
        }
        rejected = {
            "rf": {"practically_comparable": True},
            "xgb": {"practically_comparable": False},
        }

        self.assertEqual(classify_h2(full), "full_support")
        self.assertEqual(classify_h2(rejected), "not_supported")

    def test_h2_bound_is_strict_and_garch_win_qualifies(self):
        self.assertTrue(h2_practically_comparable(0.02, 0.049))
        self.assertFalse(h2_practically_comparable(0.02, 0.05))
        self.assertTrue(h2_practically_comparable(-0.001, 0.20))

    def test_h3_requires_every_model(self):
        all_supported = {
            model: {"supports_h3": True}
            for model in ("persistence", "ewma", "garch", "rf", "xgb")
        }
        one_failure = {key: value.copy() for key, value in all_supported.items()}
        one_failure["garch"]["supports_h3"] = False

        self.assertEqual(classify_h3(all_supported), "full_support")
        self.assertEqual(classify_h3(one_failure), "not_supported")


if __name__ == "__main__":
    unittest.main()
