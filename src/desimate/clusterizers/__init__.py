from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import joblib
import numpy as np
from joblib import Memory

from desimate import StatusProgressStep
from desimate.distances import DistanceMetric


class VectorClustering(StatusProgressStep, ABC):
    def __init__(
        self,
        db_path: Path,
        memory: joblib.Memory = Memory(),
        show_progress: bool = True,
    ):
        super().__init__(show_progress=show_progress)
        self.db_path = db_path
        self.memory = memory

    @abstractmethod
    def cluster(
        self,
        distance: Type[DistanceMetric],
        max_distance: int,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Given a 2D numpy arrays, cluster each rows
        and returns a single 1D numpy array giving
        the cluster ID of each row (in the same
        order).
        :param memory:
        :param distance:
        :return: A single 1D numpy array.
        """
        pass

    def run(self, *args, **kwargs) -> np.ndarray:
        return self.cluster(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__
