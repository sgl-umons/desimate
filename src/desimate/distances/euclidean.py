import numpy as np
from scipy.spatial.distance import cdist, pdist

from desimate.distances import DistanceMetric


class EuclideanDistance(DistanceMetric):
    def __init__(self, vectors: np.ndarray):
        super().__init__(vectors)

    def compute_distance(self, i1, i2):
        return np.linalg.norm(i1 - i2)

    def _compute_distances(self, i1) -> np.ndarray:
        return np.linalg.norm(self.items - i1, axis=1)

    def _compute_distance_matrix(self) -> np.ndarray:
        # while scikit-learn euclidean_distances is slightly
        # faster, it is also on multiple threads.
        # Scipy version, while single threaded, is only
        # slightly longer, but allow to multi process
        # anything using this distance.
        return cdist(self.items, self.items, metric="euclidean")

    def _compute_compressed_distance_matrix(self) -> np.ndarray:
        return pdist(self.items, metric="euclidean")


class SquaredEuclideanDistance(DistanceMetric):
    def __init__(self, vectors: np.ndarray):
        super().__init__(vectors)

    def compute_distance(self, i1, i2):
        return np.sum((i1 - i2) ** 2)

    def _compute_distances(self, i1) -> np.ndarray:
        return np.sum((self.items - i1) ** 2, axis=1)

    def _compute_distance_matrix(self) -> np.ndarray:
        return cdist(self.items, self.items, metric="sqeuclidean")

    def _compute_compressed_distance_matrix(self) -> np.ndarray:
        return pdist(self.items, metric="sqeuclidean")


class NormalizedEuclideanDistance(DistanceMetric):
    def __init__(self, texts: list[str], vectors: np.ndarray):
        if len(texts) != len(vectors):
            raise ValueError("The length of texts and vectors should be equal.")
        super().__init__((t, v) for t, v in zip(texts, vectors))
        self.lens = np.array([len(_t) for _t in texts])

    def compute_distance(self, i1, i2) -> float:
        _t1, _v1 = i1
        _t2, _v2 = i2
        return np.linalg.norm(_v1 - _v2) / max(len(_t1), len(_t2))

    def _compute_distances(self, i1) -> np.ndarray:
        _t, _v = i1
        _d = np.linalg.norm(self.items - _v, axis=1)
        return _d / np.maximum(self.lens, len(_t))

    # TODO: use cdist for matrix
