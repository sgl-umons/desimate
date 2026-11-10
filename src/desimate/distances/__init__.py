from abc import abstractmethod

import numpy as np
from joblib import Parallel, delayed

from desimate import Step, ProgressStep


class ComputeAllPairDistance(ProgressStep):
    # TODO: handle number of CPUs
    def __init__(self, distance_metric: DistanceMetric, n_jobs=-1):
        super().__init__()
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs

    def run(self) -> np.ndarray:
        task = self.progress.add_task(
            "Computing pairwise distance...", total=len(self.distance_metric)
        )
        matrix = np.empty(
            (len(self.distance_metric), len(self.distance_metric)), dtype=int
        )
        parallel = Parallel(return_as="generator", n_jobs=self.n_jobs)
        output = parallel(
            delayed(self.distance_metric.compute_distance_between)(i)
            for i in range(len(self.distance_metric))
        )
        for _i, _arr in enumerate(output):
            matrix[_i] = _arr
            self.progress.update(task, advance=1)
        return matrix


class DistanceMetric(Step):
    def __init__(self, items):
        super().__init__()
        self.items = items

    @abstractmethod
    def compute_distance(self, i1, i2) -> float:
        pass

    def _compute_distances(self, i1) -> np.ndarray:
        return np.vectorize(self.compute_distance)(self.items, i1)

    def _compute_distance_matrix(self) -> np.ndarray:
        matrix = np.empty((len(self.items), len(self.items)), dtype=np.float32)
        for _i in range(len(self.items)):
            matrix[_i] = self._compute_distances(_i)
        return matrix

    def _compute_compressed_distance_matrix(self) -> np.ndarray:
        return np.array(
            [
                self.compute_distance(self.items[i], self.items[j])
                for i in range(len(self.items))
                for j in range(i + 1, len(self.items))
            ]
        )

    def compute_distance_between(
        self, idx1: int | None = None, idx2: int | None = None, compressed: bool = False
    ) -> float | np.ndarray:
        """
        :param idx1:
        :param idx2:
        :param compressed: Only applies when idx1 and idx2 are both None.
        If true, compute a 1D compressed matrix distance instead of a full 2D distance.
        :return:
        """
        if idx1 is not None and idx2 is not None:
            return self.compute_distance(self.items[idx1], self.items[idx2])
        elif idx1 is not None:
            return self._compute_distances(self.items[idx1])
        elif compressed:
            return self._compute_compressed_distance_matrix()
        else:
            return self._compute_distance_matrix()

    def get_data_length(self):
        return len(self.items)

    def get_vector_length(self):
        if len(self.items) > 0:
            return len(self.items[0])
        return 0

    def run(
        self, idx1: int | None = None, idx2: int | None = None
    ) -> float | np.ndarray:
        return self.compute_distance_between(idx1, idx2)

    def __len__(self):
        return self.get_data_length()
