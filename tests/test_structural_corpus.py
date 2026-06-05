import unittest

from timeformers.corpus import STRUCTURAL_CONDITIONS, SUBJECTS
from timeformers.dataset import CLASS2ID, ContextPairMLMDataset
from timeformers.structural_corpus import (
    N_STRUCTURAL_QUARTETS,
    STRUCTURAL_CLASS_NAMES,
    STRUCTURAL_TARGETS,
    generate_structural_examples,
    generate_structural_null_examples,
    generate_structural_trajectories,
    structural_condition,
    structural_quartet,
)


class StructuralCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectories, self.metadata = generate_structural_trajectories(seed=7)

    def test_each_quartet_pairs_all_conditions_with_shared_endpoints(self) -> None:
        for quartet in range(N_STRUCTURAL_QUARTETS):
            subjects = STRUCTURAL_TARGETS[quartet * 4 : quartet * 4 + 4]
            self.assertEqual(
                {self.metadata[subject]["condition"] for subject in subjects},
                set(STRUCTURAL_CONDITIONS),
            )
            self.assertEqual(len({self.metadata[subject]["start"] for subject in subjects}), 1)
            self.assertEqual(len({self.metadata[subject]["alternate"] for subject in subjects}), 1)
            self.assertEqual(len({self.metadata[subject]["direction"] for subject in subjects}), 1)

    def test_directions_are_balanced_across_quartets(self) -> None:
        directions = {
            self.metadata[STRUCTURAL_TARGETS[quartet * 4]]["direction"]
            for quartet in range(N_STRUCTURAL_QUARTETS)
        }
        self.assertEqual(directions, {"n1_to_n2", "n2_to_n1"})
        for direction in directions:
            count = sum(
                self.metadata[STRUCTURAL_TARGETS[quartet * 4]]["direction"] == direction
                for quartet in range(N_STRUCTURAL_QUARTETS)
            )
            self.assertEqual(count, N_STRUCTURAL_QUARTETS // 2)

    def test_temporal_forms_match_preregistered_behavior(self) -> None:
        for quartet in range(N_STRUCTURAL_QUARTETS):
            base = quartet * 4
            gradual = self.trajectories[STRUCTURAL_TARGETS[base]]
            abrupt = self.trajectories[STRUCTURAL_TARGETS[base + 1]]
            transient = self.trajectories[STRUCTURAL_TARGETS[base + 2]]
            oscillating = self.trajectories[STRUCTURAL_TARGETS[base + 3]]
            start = gradual[0]
            alternate = gradual[-1]
            increasing = alternate > start

            self.assertEqual(abrupt[0], start)
            self.assertEqual(abrupt[-1], alternate)
            self.assertEqual(transient[0], start)
            self.assertEqual(transient[-1], start)
            self.assertIn(alternate, transient)
            self.assertEqual(oscillating[0], start)
            self.assertEqual(oscillating[-1], start)
            self.assertIn(alternate, oscillating)

            steps = [after - before for before, after in zip(gradual, gradual[1:])]
            self.assertTrue(all(step > 0 for step in steps) if increasing else all(step < 0 for step in steps))

    def test_subject_metadata_helpers_follow_quartets(self) -> None:
        for index in range(len(SUBJECTS)):
            if index < len(SUBJECTS) - len(STRUCTURAL_TARGETS):
                self.assertEqual(structural_condition(index), "structural_anchor")
                self.assertEqual(structural_quartet(index), -1)
            else:
                target_index = index - (len(SUBJECTS) - len(STRUCTURAL_TARGETS))
                self.assertEqual(structural_condition(index), STRUCTURAL_CONDITIONS[target_index % 4])
                self.assertEqual(structural_quartet(index), target_index // 4)

    def test_structural_examples_are_dataset_compatible(self) -> None:
        rows, trajectories, metadata = generate_structural_examples(
            seed=11,
            examples_per_subject_period=5,
        )

        self.assertEqual(len(rows), len(SUBJECTS) * 10 * 5)
        self.assertEqual(set(trajectories), set(SUBJECTS))
        self.assertEqual(set(metadata), set(SUBJECTS))
        self.assertTrue({row.subject_class for row in rows}.issubset(CLASS2ID))

        dataset = ContextPairMLMDataset(rows)
        self.assertEqual(len(dataset), len(rows))
        self.assertEqual(
            {int(item["class_id"]) for item in dataset},
            {CLASS2ID[condition] for condition in STRUCTURAL_CONDITIONS} | {CLASS2ID["structural_anchor"]},
        )
        self.assertEqual(
            STRUCTURAL_CLASS_NAMES,
            {
                CLASS2ID[condition]: condition
                for condition in STRUCTURAL_CONDITIONS + ("structural_anchor",)
            },
        )

    def test_invalid_number_of_periods_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_structural_trajectories(seed=7, n_periods=5)

    def test_abrupt_switch_period_can_be_configured(self) -> None:
        trajectories, _ = generate_structural_trajectories(
            seed=7,
            abrupt_switch_period=3,
        )

        for quartet in range(N_STRUCTURAL_QUARTETS):
            abrupt = trajectories[STRUCTURAL_TARGETS[quartet * 4 + 1]]
            self.assertEqual(abrupt[:3], [abrupt[0]] * 3)
            self.assertEqual(abrupt[3:], [abrupt[-1]] * 7)

    def test_transient_window_can_be_configured(self) -> None:
        trajectories, _ = generate_structural_trajectories(
            seed=7,
            transient_onset_period=2,
            transient_width=3,
        )

        for quartet in range(N_STRUCTURAL_QUARTETS):
            transient = trajectories[STRUCTURAL_TARGETS[quartet * 4 + 2]]
            start = transient[0]
            alternate = transient[2]
            self.assertEqual(transient[:2], [start] * 2)
            self.assertEqual(transient[2:5], [alternate] * 3)
            self.assertEqual(transient[5:], [start] * 5)

    def test_invalid_temporal_form_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_structural_trajectories(seed=7, abrupt_switch_period=0)
        with self.assertRaises(ValueError):
            generate_structural_trajectories(seed=7, transient_onset_period=9, transient_width=2)
        with self.assertRaises(ValueError):
            generate_structural_trajectories(seed=7, transient_width=0)

    def test_resampled_null_keeps_semantics_constant_but_draws_new_texts(self) -> None:
        real_rows, real_trajectories, _ = generate_structural_examples(
            seed=13,
            examples_per_subject_period=20,
        )
        null_rows, null_trajectories, _ = generate_structural_null_examples(
            seed=13,
            examples_per_subject_period=20,
        )

        for subject in SUBJECTS:
            self.assertEqual(len(set(null_trajectories[subject])), 1)
            self.assertEqual(null_trajectories[subject][0], real_trajectories[subject][0])
        real_sentences = [(row.epoch, row.subject, row.verb, row.obj) for row in real_rows]
        null_sentences = [(row.epoch, row.subject, row.verb, row.obj) for row in null_rows]
        self.assertNotEqual(real_sentences, null_sentences)


if __name__ == "__main__":
    unittest.main()
