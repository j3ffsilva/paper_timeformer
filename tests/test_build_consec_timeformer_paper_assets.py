import unittest

from scripts.build_consec_timeformer_paper_assets import latex_escape


class BuildPaperAssetsTest(unittest.TestCase):
    def test_latex_escape(self):
        self.assertEqual(
            latex_escape("plane_nn & 95%"),
            r"plane\_nn \& 95\%",
        )


if __name__ == "__main__":
    unittest.main()
