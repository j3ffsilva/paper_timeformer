import unittest

import torch

from scripts.evaluate_unsupervised_community_profiles import (
    knn_graph,
    labels_from_communities,
    mutual_knn_graph,
)


class UnsupervisedCommunityProfilesTest(unittest.TestCase):
    def test_mutual_knn_keeps_only_reciprocal_edges(self) -> None:
        points = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            dim=1,
        )

        graph = mutual_knn_graph(points, k=1)

        self.assertEqual(set(graph.edges()), {(0, 1)})
        self.assertIn(2, graph.nodes())

    def test_union_knn_keeps_nonreciprocal_edges(self) -> None:
        points = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            dim=1,
        )

        graph = knn_graph(points, k=1, mode="union")

        self.assertEqual(set(graph.edges()), {(0, 1), (1, 2)})

    def test_partition_labels_cover_every_node(self) -> None:
        labels = labels_from_communities([{0, 2}, {1}], 3)
        self.assertEqual(labels.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
