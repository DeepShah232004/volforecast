import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "interpretability"))

from shap_analysis import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    allocate_stratum_counts,
    build_analysis_population,
    compute_tree_shap,
    draw_stratified_sample,
    ranked_group_importance,
    sample_key_sha256,
    validate_feature_groups,
)


class StratifiedSamplingTests(unittest.TestCase):
    def setUp(self):
        records = []
        counts = {
            (2015, "low"): 40,
            (2015, "high"): 20,
            (2020, "low"): 30,
            (2020, "high"): 10,
        }
        permno = 10000
        for (year, regime), count in counts.items():
            for day in range(count):
                records.append(
                    {
                        "permno": permno,
                        "dlycaldt": pd.Timestamp(year=year, month=1, day=1)
                        + pd.Timedelta(day, unit="D"),
                        "year": year,
                        "vix_regime": regime,
                    }
                )
                permno += 1
        self.population = pd.DataFrame.from_records(records)

    def test_allocation_is_complete_bounded_and_represents_every_stratum(self):
        sizes = self.population.groupby(["year", "vix_regime"]).size()
        allocation = allocate_stratum_counts(sizes, 25)

        self.assertEqual(int(allocation.sum()), 25)
        self.assertTrue((allocation >= 1).all())
        self.assertTrue((allocation <= sizes).all())

    def test_sample_and_hash_are_deterministic(self):
        first, first_strata = draw_stratified_sample(
            self.population, sample_size=25, seed=42
        )
        second, second_strata = draw_stratified_sample(
            self.population.sample(frac=1, random_state=7), sample_size=25, seed=42
        )

        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_strata, second_strata)
        self.assertEqual(sample_key_sha256(first), sample_key_sha256(second))
        observed = first.groupby(["year", "vix_regime"], observed=True).size()
        expected = first_strata.set_index(["year", "vix_regime"])["sample_rows"]
        pd.testing.assert_series_equal(observed, expected, check_names=False)


class PopulationTests(unittest.TestCase):
    def make_inputs(self):
        matrix = pd.DataFrame(
            {
                "permno": [10001, 10002, 10003],
                "dlycaldt": pd.to_datetime(
                    ["2015-01-02", "2015-01-02", "2015-01-05"]
                ),
                **{feature: [1.0, 2.0, 3.0] for feature in FEATURE_COLUMNS},
            }
        )
        panel = pd.DataFrame(
            {
                "permno": [10001, 10003],
                "dlycaldt": pd.to_datetime(["2015-01-02", "2015-01-05"]),
                "vix_regime": ["low", "high"],
                "forecast_random_forest": [0.02, 0.03],
                "forecast_xgboost": [0.021, 0.031],
            }
        )
        return matrix, panel

    def test_population_uses_only_final_common_panel_keys(self):
        matrix, panel = self.make_inputs()
        population = build_analysis_population(
            matrix, np.array([True, True, True]), panel
        )

        self.assertListEqual(population["permno"].tolist(), [10001, 10003])
        self.assertListEqual(population["year"].tolist(), [2015, 2015])

    def test_population_rejects_panel_key_missing_from_matrix(self):
        matrix, panel = self.make_inputs()
        panel.loc[1, "permno"] = 99999

        with self.assertRaisesRegex(ValueError, "absent from the ML test matrix"):
            build_analysis_population(matrix, np.array([True, True, True]), panel)


class ImportanceTests(unittest.TestCase):
    def test_locked_groups_partition_all_features(self):
        validate_feature_groups()
        grouped = [feature for group in FEATURE_GROUPS.values() for feature in group]
        self.assertCountEqual(grouped, FEATURE_COLUMNS)

    def test_group_shares_sum_to_one(self):
        values = np.arange(1, 1 + 20 * len(FEATURE_COLUMNS), dtype="float64").reshape(
            20, len(FEATURE_COLUMNS)
        )
        groups = ranked_group_importance(values)

        self.assertAlmostEqual(sum(row["normalized_share"] for row in groups), 1.0)
        self.assertListEqual([row["rank"] for row in groups], [1, 2, 3, 4])

    def test_random_forest_tree_shap_reconstructs_log_prediction(self):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, len(FEATURE_COLUMNS))).astype("float32")
        target = features[:, 0] - 0.5 * features[:, 1]
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(features, target)

        values, bases, predictions = compute_tree_shap(
            "random_forest", model, features[:10]
        )

        self.assertEqual(values.shape, (10, len(FEATURE_COLUMNS)))
        np.testing.assert_allclose(
            bases + values.sum(axis=1), predictions, rtol=0, atol=1e-8
        )

    def test_xgboost_tree_shap_reconstructs_log_prediction(self):
        import xgboost as xgb

        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, len(FEATURE_COLUMNS))).astype("float32")
        target = features[:, 0] - 0.5 * features[:, 1]
        training = xgb.DMatrix(
            features, label=target, feature_names=FEATURE_COLUMNS
        )
        model = xgb.train(
            {"objective": "reg:squarederror", "max_depth": 2, "seed": 42},
            training,
            num_boost_round=5,
        )

        values, bases, predictions = compute_tree_shap(
            "xgboost", model, features[:10]
        )

        self.assertEqual(values.shape, (10, len(FEATURE_COLUMNS)))
        np.testing.assert_allclose(
            bases + values.sum(axis=1), predictions, rtol=0, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
