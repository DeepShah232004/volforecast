import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "reporting"))

import build_report_assets as reporting


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation, cls.shap = reporting.load_locked_results()

    def test_locked_sources_and_headline_content(self):
        markdown = reporting.build_markdown(self.evaluation, self.shap)

        self.assertIn("0.379788", markdown)
        self.assertIn("H1 decision: **not supported**", markdown)
        self.assertIn("H2 decision: **full support under the locked rule**", markdown)
        self.assertIn("H3 decision: **full support**", markdown)
        self.assertIn("83.27%", markdown)
        self.assertIn("74.31%", markdown)
        self.assertIn("Mean \\|return\\| (63d)", markdown)

    def test_all_figures_render_as_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            original_directory = reporting.FIGURE_DIR
            reporting.FIGURE_DIR = Path(temporary_directory)
            try:
                plt = reporting.configure_plotting()
                reporting.figure_overall_metrics(self.evaluation, plt)
                reporting.figure_regime_tests(self.evaluation, plt)
                reporting.figure_earnings_penalty(self.evaluation, plt)
                reporting.figure_shap_groups(self.shap, plt)
                reporting.figure_shap_stability(self.shap, plt)
            finally:
                reporting.FIGURE_DIR = original_directory

            outputs = list(Path(temporary_directory).iterdir())
            self.assertEqual(len(outputs), 10)
            self.assertEqual({path.suffix for path in outputs}, {".png", ".pdf"})
            self.assertTrue(all(path.stat().st_size > 1_000 for path in outputs))


if __name__ == "__main__":
    unittest.main()
