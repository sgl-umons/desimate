import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Type

import duckdb
import joblib
import numpy as np
from joblib import Memory
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, Birch

from desimate.clusterizers import VectorClustering
from desimate.distances import DistanceMetric
from desimate.distances.euclidean import EuclideanDistance

logger = logging.getLogger(__name__)


class Agglomerative(VectorClustering):
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
    ) -> np.ndarray:
        """
        Cluster a group of vectors.

        :param distance:
        :param memory:
        :return:
        """
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

        _linkage = linkage(matrix, method="complete")
        labels = fcluster(_linkage, t=max_distance, criterion="distance")

        # process noise
        ulabels, ulabels_count = np.unique(labels, return_counts=True, sorted=False)
        noise_labels = ulabels[(ulabels_count <= 1)]
        labels[np.isin(labels, noise_labels)] = -1

        return labels


class GridAgglomerative(VectorClustering):
    def __init__(
        self,
        db_path: Path,
        memory: joblib.Memory = Memory(),
        show_progress: bool = True,
    ):
        super().__init__(db_path=db_path, show_progress=show_progress, memory=memory)

    def _create_grid(
        self,
        db,
        vector_length: int,
        min_length: int,
        max_embedding_size: int | None = None,
    ):
        columns = ",".join(
            f"list_extract(xs, {i + 1})::INTEGER AS x{i}" for i in range(vector_length)
        )
        db.execute(
            f"""
            CREATE OR REPLACE TABLE grid_positions AS
            SELECT p.rowid AS id, p.value AS value, e.embedding, 
                list_transform(e.embedding, lambda x: ((x - x % 1))::INTEGER) AS xs,
                {columns},
                len(value{("[:" + str(max_embedding_size) + "]") if max_embedding_size else ""}) AS vlength,
                list_append(xs, vlength) AS cell_position
            FROM embeds e
            JOIN paths p
            ON p.file_id = e.file_id
            AND p.ind = e.ind
            AND len(p.value) > ?
            """,
            parameters=[min_length],
        )
        for i in range(vector_length):
            db.execute(f"CREATE INDEX x{i}_positions_idx ON grid_positions(x{i})")

    def _get_vector_length(self, db):
        return db.execute(
            """
            SELECT len(embedding) AS length FROM embeds;
            """
        ).fetchone()[0]

    def _get_max_id(self, db):
        return db.execute(
            """
            SELECT MAX(id) FROM grid_positions;
            """
        ).fetchone()[0]

    def _get_neighborhood_cells(
        self,
        db,
        cell: np.ndarray,
        vector_length: int,
        max_distance: int | float,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        _m = int(math.ceil(max_distance))
        condition = " AND ".join(
            f"x{i} IN ({','.join([str(cell[i] + j) for j in range(-_m, _m + 1, 1)])})"
            for i in range(vector_length)
        )
        # cell[-1] is the length
        condition += f" AND vlength IN ({','.join([str(cell[-1] + j) for j in range(-_m, _m + 1, 1)])})"
        neighborhood = db.execute(
            f"""
            SELECT id, embedding, value, xs
            FROM grid_positions gp
            WHERE {condition}
            AND sqrt(
                list_reduce(
                    list_transform(
                        list_zip(xs, ?),
                        lambda p: greatest(0, abs(p[1] - p[2]) - 1) ^ 2
                    ),
                    lambda acc, x: acc + x,
                    0
                )
            ) <= {max_distance}
            """,
            [cell[:vector_length]],
        ).fetchnumpy()
        # make sure euclidean distance is lower than
        # sqrt(sum(max(diff_x_i - 1, 0)**2 for each i))

        ids = neighborhood["id"]
        embedding = np.stack(neighborhood["embedding"], dtype=np.float64)
        value = np.array(neighborhood["value"])
        xs = np.stack(neighborhood["xs"], dtype=np.int32)

        return ids, embedding, value, xs

    def _get_grid_cells(self, db):
        return db.sql(
            """
            SELECT cell_position, COUNT(*) AS count
            FROM grid_positions
            GROUP BY cell_position
            ORDER BY count, MIN(rowid) DESC
            """
        ).fetchnumpy()["cell_position"]

    def _apply_agglomerative_clustering(self, distance: DistanceMetric, max_distance):
        def _aggl(_items):
            agg = AgglomerativeClustering(
                n_clusters=None,
                metric="euclidean",
                distance_threshold=max_distance,
                linkage="complete",
                compute_full_tree=True,
            )
            return agg.fit_predict(_items)

        if isinstance(distance, EuclideanDistance):
            should_reduce = len(distance) > 50_000
            # should_reduce = False
            embedding = distance.items
            if should_reduce:
                birch = Birch(threshold=1, branching_factor=10, n_clusters=None)
                birch.fit(embedding)
                centers = birch.subcluster_centers_
                if len(centers) == 1:
                    # Only one BIRCH subcluster
                    labels = np.zeros(len(embedding), dtype=int)
                else:
                    center_labels = _aggl(centers)
                    labels = center_labels[birch.predict(embedding)]
            else:
                labels = _aggl(embedding)

            return labels
        else:
            matrix: np.ndarray = distance.compute_distance_between(compressed=True)
            _linkage = linkage(matrix, method="complete")
            return fcluster(_linkage, t=max_distance, criterion="distance")

    def _compute_subres(self, labels, in_current_cell):
        labels_in_cell = np.unique(labels[in_current_cell])
        if len(labels_in_cell) == 0:
            return

        ulabels, ulabels_count = np.unique(labels, return_counts=True, sorted=False)
        subres = -np.ones(len(labels))
        fltr_labels = ulabels[
            (ulabels_count > 1) & (np.isin(ulabels, labels_in_cell, assume_unique=True))
        ]
        noise_labels = ulabels[
            (ulabels_count <= 1)
            & (np.isin(ulabels, labels_in_cell, assume_unique=True))
        ]
        for _j in range(len(fltr_labels)):
            subres[labels == fltr_labels[_j]] = _j
        subres[np.isin(labels, noise_labels)] = -2
        return subres

    def _process_cell(
        self,
        db,
        cell,
        vector_length,
        max_distance,
        error_margin,
        distance,
        real_distance,
        mask,
    ):
        ids, vectors, values, xs = self._get_neighborhood_cells(
            db,
            cell,
            vector_length,
            max_distance=max_distance,
        )
        if len(ids) <= 1:
            return None

        mask = mask[ids]
        if mask.sum() <= 1:
            return None

        in_current_cell = np.array(
            [np.array_equal(xs[_i], cell[:vector_length]) for _i in range(len(xs))],
            dtype=bool,
        )
        ids = ids[mask]
        vectors = vectors[mask]
        values = values[mask]
        in_current_cell = in_current_cell[mask]

        labels = self._apply_agglomerative_clustering(
            distance(vectors),
            max_distance=max_distance + error_margin,
        )

        subres = self._compute_subres(labels, in_current_cell)
        if subres is None:
            return None

        if real_distance:
            ulabels = np.unique(subres, sorted=False)
            for _j in ulabels:
                if _j is None or _j < 0:
                    continue
                submask = subres == _j
                sublabels = self._apply_agglomerative_clustering(
                    real_distance(values[submask]),
                    max_distance=max_distance,
                )
                subres = self._compute_subres(sublabels, in_current_cell[submask])
                # res[ids[submask]] = subres
                return len(labels), ids[submask], subres
        else:
            return len(labels), ids, subres
            # res[ids] = subres

        # res[ids] = subres

    def _process_cell_db(
        self,
        cell,
        vector_length,
        max_distance,
        error_margin,
        distance,
        real_distance,
        mask,
    ):
        with duckdb.connect(self.db_path, read_only=True) as db:
            db.execute(
                """
                SET memory_limit = '1GB'; SET preserve_insertion_order = false;
                SET threads=3;
                """
            )
            return self._process_cell(
                db,
                cell,
                vector_length,
                max_distance,
                error_margin,
                distance,
                real_distance,
                mask,
            )

    def multi_process_grid(
        self,
        grid_cells,
        vector_length,
        max_distance,
        error_margin,
        distance,
        real_distance,
        mask,
        max_workers: int | None = None,
        mem_ideal_usage=10_000_000_000,
        increase_thread_after=10,
    ):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = []
            threads = 1
            current_cell = 0
            last_count = []
            could_increase = 0
            while len(results) > 0 or current_cell < len(grid_cells):
                if (
                    len(last_count) > 0
                    and sum([(i * (i - 1) * 4) for i in last_count[-threads - 1 :]])
                    < mem_ideal_usage
                ):
                    could_increase += 1
                    if could_increase > increase_thread_after:
                        threads = min(max_workers, threads + 1)
                        could_increase = 0
                else:
                    could_increase = 0
                while current_cell < len(grid_cells) and len(results) < threads:
                    args = (
                        grid_cells[current_cell],
                        vector_length,
                        max_distance,
                        error_margin,
                        distance,
                        real_distance,
                        mask,
                    )
                    results.append(
                        (current_cell, executor.submit(self._process_cell_db, *args))
                    )
                    current_cell += 1
                if len(results) > 0:
                    _cc, _f = results.pop(0)
                    _r = _f.result()
                    if _r is not None:
                        _l, _ids, _subres = _r
                        last_count.append(_l)
                        last_count = last_count[-max_workers:]
                        if not np.all(mask[_ids]):
                            _a = (
                                grid_cells[_cc],
                                vector_length,
                                max_distance,
                                error_margin,
                                distance,
                                real_distance,
                                mask,
                            )
                            _f2 = executor.submit(self._process_cell_db, *_a)
                            results.insert(0, (_cc, _f2))
                            continue
                    yield _r

    def cluster(
        self,
        distance: Type[DistanceMetric],
        max_distance: int,
        real_distance: Type[DistanceMetric] = None,
        error_margin: float = 0.0,
        min_length: float = 0.0,
    ) -> np.ndarray:
        # This version for now uses a temporary res array, which
        # remember the results, as the DuckDB database is opened
        # in read-only by multiple workers.
        # Note that this does not impact scalability, as, considering
        # a 8-byte representation, a 1M items clustering would
        # result in a RAM usage of 8Mb, which is fairly low.
        # This will be updated in future versions of the tool.

        error_margin = max_distance * error_margin
        with duckdb.connect(self.db_path) as db:
            db.execute(
                """
                SET memory_limit = '1GB'; SET preserve_insertion_order = false;
                SET threads=3;
                """
            )
            vector_length = self._get_vector_length(db)
            self._create_grid(
                db, vector_length, min_length=min_length, max_embedding_size=2048
            )
            _t = self.progress.add_task("Creating grid cells...")
            grid_cells = self._get_grid_cells(db)
            self.progress.remove_task(_t)
            logger.debug("Found %d grid cells", len(grid_cells))

            _max_id = self._get_max_id(db)

        res = -np.ones(_max_id + 1, dtype=int)

        cells_task = self.progress.add_task(
            description="Clustering grid cells", total=len(grid_cells)
        )
        ids_task = self.progress.add_task(
            description="Clustering vectors", total=_max_id
        )
        mask = np.full(len(res), True, dtype=bool)  # where it is == -1
        max_res_id = 0

        iter_ = self.multi_process_grid(
            grid_cells,
            vector_length,
            max_distance,
            error_margin,
            distance,
            real_distance,
            mask,
        )

        # Not exactly clean, but no choice due to multiprocessing...
        _progress = self.progress
        _live = self.live

        self.progress = None
        self.live = None

        for _r in iter_:
            if _r is not None:
                _, _ids, _subres = _r
                _subres[_subres >= 0] += max_res_id
                res[_ids] = _subres
                mask[_ids] = _subres == -1
                max_res_id = max(max_res_id, _subres.max()) + 1
                _progress.update(ids_task, advance=(_subres != -1).sum())
            _progress.update(cells_task, advance=1)

        self.progress = _progress
        self.live = _live

        with duckdb.connect(self.db_path) as db:
            db.execute(
                """
                SET memory_limit = '1GB'; SET preserve_insertion_order = false;
                SET threads=3;
                CREATE OR REPLACE TABLE clusters (
                    item_id INTEGER,
                    cluster_id INTEGER
                );
                DROP TABLE grid_positions;
                """
            )
            BATCH_SIZE = 1_000_000
            for start in range(0, len(res), BATCH_SIZE):
                batch = res[start : start + BATCH_SIZE]
                item_ids = np.arange(start, start + len(batch))

                db.execute(
                    """
                    INSERT INTO clusters
                    SELECT unnest(?) AS item_id,
                           unnest(?) AS cluster_id
                """,
                    [item_ids, batch],
                )

        # return res
