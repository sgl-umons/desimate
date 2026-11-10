from pathlib import Path
from typing import Type

import duckdb
import joblib
import numpy as np
import hdbscan
from joblib import Memory

from desimate.clusterizers import VectorClustering
from desimate.distances import DistanceMetric
from desimate.distances.euclidean import EuclideanDistance


class HDBSCAN(VectorClustering):
    def __init__(
        self,
        db_path: Path,
        memory: joblib.Memory = Memory(),
        show_progress: bool = True,
    ):
        super().__init__(db_path=db_path, show_progress=show_progress, memory=memory)

    def cluster(
        self,
        distance: Type[DistanceMetric],
        max_distance: int,
        min_cluster_size: int = 2,
        *args,
        **kwargs,
    ) -> np.ndarray:
        with duckdb.connect(self.db_path) as db:
            items = db.execute(
                """
                SET memory_limit = '1GB';
                SELECT value
                FROM paths e
                ORDER BY rowid
                """,
            ).fetchnumpy()["value"]

        d: DistanceMetric = distance(items)
        matrix: np.ndarray = d.compute_distance_between(compressed=True)
        if distance == EuclideanDistance:
            _cluster = hdbscan.HDBSCAN(
                metric="euclidean",
                memory=self.memory,
                leaf_size=40,
                min_cluster_size=min_cluster_size,
                cluster_selection_method="leaf",
                cluster_selection_epsilon=max_distance,
            )
            _cluster.fit(items)
        else:
            _cluster = hdbscan.HDBSCAN(
                metric="precomputed",
                memory=self.memory,
                leaf_size=40,
                min_cluster_size=min_cluster_size,
                cluster_selection_method="leaf",
                cluster_selection_epsilon=max_distance,
            )
            _cluster.fit(matrix)
        return _cluster.labels_
