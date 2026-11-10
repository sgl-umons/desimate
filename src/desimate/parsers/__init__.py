import os
import time
from abc import ABC
from collections import namedtuple
from itertools import chain

import duckdb
import pandas as pd
from joblib import Parallel, delayed

from desimate import Step, ProgressStep
from pathlib import Path
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Parser(Step, ABC):
    def get_config(self):
        return super().get_config().get("parsing", {})


class FileValue(NamedTuple):
    ind: int
    path: list[str | int]
    abstract_path: list[str]
    value: str
    offset_start: int
    offset_end: int
    # extra: dict[str, str] = {}


NamedFileValue: type[NamedTuple] = namedtuple(
    "NamedFileValue", ["file", *FileValue._fields]
)


class ReadFile(Parser):
    def run(self, file_path: Path) -> str:
        try:
            with open(file_path, "r") as f:
                data = f.read()
        except ValueError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    data = f.read()
            except ValueError:
                logger.warning(
                    "Errors occurred while reading '%s'! Ignoring encoding errors as a fallback...",
                    file_path,
                )
                with open(file_path, "r", errors="ignore"):
                    data = f.read()
        return data


class MultiParserDB(Parser):
    def __init__(self, db: Path, parser: Parser):
        super().__init__()
        self.ps = parser
        self.db = db

    def _save(self, values):
        df = pd.DataFrame(values, columns=NamedFileValue._fields)
        r = False
        # basically try until it works (per suggestion of DuckDB docs)
        # they are surely better ways but it works good enough for our
        # purpose and multiple workers.
        # Note that you should reduce memory limit such that
        # your number of workers * your memory limit < total available memory of system
        while not r:
            try:
                with duckdb.connect(self.db) as db:
                    db.execute(
                        """
                        SET memory_limit = '1GB'; SET preserve_insertion_order = false;
                        CREATE TABLE IF NOT EXISTS paths (file_id STRING,
                        ind INTEGER, path STRING[], abstract_path STRING[],
                        value STRING, offset_start INTEGER, offset_end INTEGER, 
                        PRIMARY KEY (file_id, ind));
                        INSERT INTO paths SELECT * FROM df
                        """
                    )
                    r = True
            except duckdb.IOException as e:
                if "lock" in str(e.with_traceback(None)):
                    logger.debug("Database seems locked... Waiting for unlock...")
                    time.sleep(0.1)
                else:
                    raise e

    def run(self, batch: list[Path], *args, **kwargs) -> int:
        res = []
        for _f in batch:
            # try:
            _name = os.path.basename(_f)
            res.append(
                [NamedFileValue(_name, *_t) for _t in self.ps(_f, *args, **kwargs)]
            )
            # except Exception as e:
            #     logger.error(e)
        self._save(chain(*res))
        return len(batch)


class MultiParser(Parser, ProgressStep):
    def __init__(self, db: Path, parser: Parser):
        super().__init__()
        self.db = db
        self.ps = MultiParserDB(self.db, parser)

    def run(self, files: list[Path], granularity: str, batch_size=1000) -> None:
        batches = [
            (i, min(len(files), i + batch_size))
            for i in range(0, len(files), batch_size)
        ]

        task = self.progress.add_task("Reading and storing...", total=len(files))
        parallel = Parallel(return_as="generator_unordered", n_jobs=-1)
        output = parallel(
            delayed(self.ps)(files[b[0] : b[1]], granularity) for b in batches
        )
        for _n in output:
            self.progress.update(task, advance=_n)
