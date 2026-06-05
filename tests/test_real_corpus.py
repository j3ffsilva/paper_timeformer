import tempfile
import unittest
from pathlib import Path

from timeformers.real_corpus import (
    RealMLMDataset,
    RealWordProbeDataset,
    build_vocabulary,
    read_period_corpora,
    tokenize,
)


class RealCorpusTest(unittest.TestCase):
    def test_tokenize_keeps_lowercase_words(self) -> None:
        self.assertEqual(tokenize("The broadcast's meaning changed."), ["the", "broadcast's", "meaning", "changed"])

    def test_read_period_corpora_accepts_period_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1950.txt").write_text("gay meant happy\n", encoding="utf-8")
            (root / "1980.txt").write_text("gay rights movement\n", encoding="utf-8")

            corpora = read_period_corpora(root)

        self.assertEqual([corpus.period for corpus in corpora], ["1950", "1980"])
        self.assertEqual(corpora[0].documents[0], ["gay", "meant", "happy"])

    def test_required_tokens_are_kept_in_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1950.txt").write_text("common common common rare", encoding="utf-8")
            corpora = read_period_corpora(root)

        vocab, token_to_id = build_vocabulary(
            corpora,
            min_count=3,
            required_tokens=["rare", "missing"],
        )

        self.assertIn("common", token_to_id)
        self.assertIn("rare", token_to_id)
        self.assertIn("missing", token_to_id)
        self.assertEqual(vocab[token_to_id["[MASK]"]], "[MASK]")

    def test_real_mlm_dataset_and_probe_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1950.txt").write_text("gay meant happy and cheerful", encoding="utf-8")
            corpora = read_period_corpora(root)
            _, token_to_id = build_vocabulary(corpora, min_count=1)

        dataset = RealMLMDataset(corpora[0], token_to_id, period_idx=0, seq_len=8, stride=4)
        item = dataset[0]
        self.assertEqual(tuple(item["input_ids"].shape), (8,))
        self.assertEqual(tuple(item["labels"].shape), (8,))
        self.assertGreaterEqual(int((item["labels"] != -100).sum()), 1)

        probe = RealWordProbeDataset(["gay"], token_to_id, period_idx=0, seq_len=8)
        probe_item = probe[0]
        self.assertEqual(tuple(probe_item["input_ids"].shape), (8,))
        self.assertEqual(int(probe_item["word_idx"]), 0)


if __name__ == "__main__":
    unittest.main()
