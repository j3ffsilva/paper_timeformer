import tempfile
import unittest
from pathlib import Path

from timeformers.real_corpus import (
    RealMLMDataset,
    RealTargetOccurrenceDataset,
    RealWordProbeDataset,
    build_vocabulary,
    make_windows,
    read_period_corpora,
    tokenize,
)


class RealCorpusTest(unittest.TestCase):
    def test_tokenize_keeps_lowercase_words(self) -> None:
        self.assertEqual(tokenize("The broadcast's meaning changed."), ["the", "broadcast's", "meaning", "changed"])

    def test_tokenize_keeps_semeval_target_pos_tokens(self) -> None:
        self.assertEqual(tokenize("attack_nn circle_vb changed."), ["attack_nn", "circle_vb", "changed"])

    def test_read_period_corpora_accepts_period_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1950.txt").write_text(
                "gay meant happy\n"
                "separate document remains separate\n",
                encoding="utf-8",
            )
            (root / "1980.txt").write_text("gay rights movement\n", encoding="utf-8")

            corpora = read_period_corpora(root)

        self.assertEqual([corpus.period for corpus in corpora], ["1950", "1980"])
        self.assertEqual(len(corpora[0].documents), 2)
        self.assertEqual(corpora[0].documents[0], ["gay", "meant", "happy"])
        self.assertEqual(
            corpora[0].documents[1],
            ["separate", "document", "remains", "separate"],
        )

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

    def test_make_windows_includes_document_tail(self) -> None:
        windows = make_windows(list(range(13)), seq_len=8, stride=4)

        self.assertEqual(windows, [list(range(6)), list(range(4, 10)), list(range(7, 13))])

    def test_real_mlm_masking_is_reproducible_and_changes_by_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = " ".join(f"word{chr(97 + index)}" for index in range(20))
            (root / "1950.txt").write_text(text, encoding="utf-8")
            corpora = read_period_corpora(root)
            _, token_to_id = build_vocabulary(corpora, min_count=1)

        dataset = RealMLMDataset(
            corpora[0],
            token_to_id,
            period_idx=0,
            seq_len=24,
            stride=12,
            mask_probability=0.5,
            seed=123,
        )
        first = dataset[0]
        repeated = dataset[0]
        self.assertTrue(first["input_ids"].equal(repeated["input_ids"]))
        self.assertTrue(first["labels"].equal(repeated["labels"]))

        dataset.set_epoch(1)
        next_epoch = dataset[0]
        self.assertFalse(first["labels"].equal(next_epoch["labels"]))

    def test_real_mlm_masks_multiple_lexical_tokens_but_not_special_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1950.txt").write_text(
                "alpha beta gamma delta epsilon zeta",
                encoding="utf-8",
            )
            corpora = read_period_corpora(root)
            _, token_to_id = build_vocabulary(corpora, min_count=1)

        dataset = RealMLMDataset(
            corpora[0],
            token_to_id,
            period_idx=0,
            seq_len=10,
            mask_probability=1.0,
            mask_replace_probability=1.0,
            random_replace_probability=0.0,
        )
        item = dataset[0]

        self.assertEqual(int((item["labels"] != -100).sum()), 6)
        self.assertEqual(int(item["labels"][0]), -100)
        self.assertEqual(int(item["labels"][7]), -100)
        self.assertEqual(int((item["input_ids"] == token_to_id["[MASK]"]).sum()), 6)

    def test_target_occurrence_probe_masks_real_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1810-1860.txt").write_text(
                "the attack_nn on the town was sudden\n"
                "another attack_nn happened later\n",
                encoding="utf-8",
            )
            corpora = read_period_corpora(root)
            _, token_to_id = build_vocabulary(corpora, min_count=1, required_tokens=["attack_nn"])

        dataset = RealTargetOccurrenceDataset(
            corpora[0],
            ["attack_nn"],
            token_to_id,
            period_idx=0,
            seq_len=8,
        )

        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertEqual(tuple(item["input_ids"].shape), (8,))
        self.assertEqual(int(item["input_ids"][int(item["mask_pos"])]), token_to_id["[MASK]"])
        self.assertEqual(int(item["word_idx"]), 0)


if __name__ == "__main__":
    unittest.main()
