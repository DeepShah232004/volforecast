import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models/ml"))

from train_models import RF_GRID, XGB_GRID, finish_selection
from training_common import FEATURE_COLUMNS, load_and_validate_matrix


class LockedGridTests(unittest.TestCase):
    def test_random_forest_grid_has_all_twelve_locked_combinations(self):
        self.assertEqual(len(RF_GRID), 12)
        self.assertEqual({row["max_depth"] for row in RF_GRID}, {8, 16, None})
        self.assertEqual({row["min_samples_leaf"] for row in RF_GRID}, {20, 100})
        self.assertEqual({row["max_features"] for row in RF_GRID}, {"sqrt", 0.5})

    def test_xgboost_grid_has_all_eight_locked_combinations(self):
        self.assertEqual(len(XGB_GRID), 8)
        self.assertEqual({row["learning_rate"] for row in XGB_GRID}, {0.03, 0.08})
        self.assertEqual({row["max_depth"] for row in XGB_GRID}, {3, 6})
        self.assertEqual({row["min_child_weight"] for row in XGB_GRID}, {5, 20})


class SampleConstructionTests(unittest.TestCase):
    def test_final_refit_readmits_train_boundary_but_not_test_boundary(self):
        rows = [
            ("2012-12-20", "2012-12-31", "train", False),
            ("2012-12-21", "2013-01-15", "train", True),
            ("2014-12-01", "2014-12-31", "validation", False),
            ("2014-12-02", "2015-01-05", "validation", True),
            ("2015-01-02", "2015-02-02", "test", False),
        ]
        matrix = pd.DataFrame(
            {
                "permno": np.arange(10001, 10006),
                "gvkey": ["1", "2", "3", "4", "5"],
                "dlycaldt": pd.to_datetime([row[0] for row in rows]),
                "in_sp500_membership": True,
                **{feature: 1.0 for feature in FEATURE_COLUMNS},
                "target_forward_21d": 0.02,
                "log_target_forward_21d": np.log(0.02),
                "target_end_date": pd.to_datetime([row[1] for row in rows]),
                "split": [row[2] for row in rows],
                "crosses_split_boundary": [row[3] for row in rows],
                "eligible_ml": [not row[3] for row in rows],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / "matrix.parquet"
            matrix.to_parquet(temporary_path, index=False)
            _, masks = load_and_validate_matrix(temporary_path)

        self.assertListEqual(
            masks.tuning_train.tolist(), [True, False, False, False, False]
        )
        self.assertListEqual(
            masks.validation.tolist(), [False, False, True, False, False]
        )
        self.assertListEqual(
            masks.final_refit.tolist(), [True, True, True, False, False]
        )
        self.assertListEqual(masks.test.tolist(), [False, False, False, False, True])


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation_qlike(self):
        run = {
            "grid": [
                {"candidate_id": "model_01", "params": {"depth": 1}},
                {"candidate_id": "model_02", "params": {"depth": 2}},
            ],
            "candidates": [
                {
                    "candidate_id": "model_01",
                    "params": {"depth": 1},
                    "validation_metrics": {"qlike": 0.4, "rmse": 0.01, "mae": 0.01},
                },
                {
                    "candidate_id": "model_02",
                    "params": {"depth": 2},
                    "validation_metrics": {"qlike": 0.3, "rmse": 0.02, "mae": 0.02},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "selection.json"
            selected = finish_selection(run, result_path)

        self.assertEqual(selected["selected"]["candidate_id"], "model_02")


if __name__ == "__main__":
    unittest.main()

