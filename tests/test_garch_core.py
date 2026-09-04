import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models" / "classical"))

from garch_core import compute_garch_path_and_forecasts


class GarchBoundaryTests(unittest.TestCase):
    def test_december_forecast_uses_december_parameters(self):
        returns_pct = np.array([1.0, 2.0, 3.0])
        years = np.array([2020, 2020, 2021])
        year_params = {
            2020: (1.0, 0.1, 0.8),
            2021: (4.0, 0.2, 0.5),
        }

        sigma2, forecasts = compute_garch_path_and_forecasts(
            returns_pct,
            years,
            year_params,
            forecast_horizon=2,
        )

        december_sigma2 = 1.0 + 0.1 * 1.0**2 + 0.8 * 10.0
        december_one_step = 1.0 + 0.1 * 2.0**2 + 0.8 * december_sigma2
        december_two_step = 1.0 + (0.1 + 0.8) * december_one_step
        expected_december_forecast = np.sqrt(
            (december_one_step + december_two_step) / 2
        ) / 100

        january_sigma2 = 4.0 + 0.2 * 2.0**2 + 0.5 * december_sigma2

        self.assertAlmostEqual(sigma2[1], december_sigma2)
        self.assertAlmostEqual(sigma2[2], january_sigma2)
        self.assertAlmostEqual(forecasts[1], expected_december_forecast)

        # This is the contaminated one-step value used by the previous
        # np.roll implementation. It must not determine December's forecast.
        self.assertNotAlmostEqual(december_one_step, january_sigma2)

    def test_last_observation_can_produce_a_forecast(self):
        _, forecasts = compute_garch_path_and_forecasts(
            returns_pct=np.array([1.0, 2.0]),
            years=np.array([2020, 2020]),
            year_params={2020: (1.0, 0.1, 0.8)},
            forecast_horizon=21,
        )

        self.assertTrue(np.isfinite(forecasts[-1]))


if __name__ == "__main__":
    unittest.main()
