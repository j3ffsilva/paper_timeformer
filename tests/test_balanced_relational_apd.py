import unittest

import torch

from scripts.diagnose_balanced_relational_apd import balanced_apd


class BalancedRelationalAPDTest(unittest.TestCase):
    def test_balanced_apd_is_reproducible(self) -> None:
        before = torch.nn.functional.normalize(torch.randn(12, 8, generator=torch.Generator().manual_seed(1)), dim=1)
        after = torch.nn.functional.normalize(torch.randn(20, 8, generator=torch.Generator().manual_seed(2)), dim=1)

        first = balanced_apd(before, after, sample_size=10, seeds=5, seed=42)
        second = balanced_apd(before, after, sample_size=10, seeds=5, seed=42)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
