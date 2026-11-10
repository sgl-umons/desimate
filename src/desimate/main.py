import json
import logging
import os
import tomllib
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
from rich.logging import RichHandler
from rich import print

from desimate import get_progress
from desimate.charencoders.hashing import HashingCE
from desimate.charencoders.onehot import OneHotCE, AsciiOneHotCE
from desimate.charencoders.ord import OrdCV, OrdCV2
from desimate.clusterizers.agglomerative import Agglomerative, GridAgglomerative
from desimate.distances.euclidean import EuclideanDistance
from desimate.distances.levenshtein import (
    LevenshteinDistance,
    NormalizedLevenshteinDistance,
)
from desimate.parsers.markdown import MultiAST
from desimate.parsers.text import MultiTextParser
from desimate.utils import read_input
from desimate.config import config as toolconfig
from desimate.encoders import BaseVectorizer, ValueEmbedding
from desimate.encoders.transformers import TransformersVectorizers

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)

logger = logging.getLogger(__name__)

np.random.seed(42)


CHAR_VECTORIZERS = {  # TODO: Auto-detect or auto-register
    "hash": HashingCE,
    "onehot": OneHotCE,
    "onehot-ascii": AsciiOneHotCE,
    "utf8": OrdCV,
    "utf8-large": OrdCV2,
}
DISTANCE_METRIC = {
    "levenshtein": LevenshteinDistance,
    "norm_levenshtein": NormalizedLevenshteinDistance,
}
PARSERS = {"markdown": MultiAST, "text": MultiTextParser}


@click.group(context_settings={"auto_envvar_prefix": "DESIMATE"})
@click.option(
    "--config",
    default="./config.toml",
    type=click.Path(readable=True),
    help="The path to a TOML configuration file.",
)
@click.pass_context
def cli(ctx, config):
    print("Welcome to DeSiMate! Feel free to report any bugs you might encounter!")
    config_path = Path(config)
    if config_path.exists():
        with open(config_path) as f:
            config_map = tomllib.loads(f.read())
        ctx.default_map = config_map
        toolconfig.update(config_map)
        logger.debug("Read a configuration file at '%s'", config)


## DEFAULT PIPELINE


@cli.command("process")
@click.argument(
    "inputs",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
    nargs=-1,
    help="The files or directories (given one or multiple times; directories "
    "are traversed recursively) to parse.",
)
@click.option(
    "-i",
    "--input-list",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="A file containing a list of additional inputs (one path per line) to "
    "add to the inputs.",
)
@click.option(
    "-p",
    "--parser",
    type=click.Choice(list(PARSERS.keys())),
    default="markdown",
    help="The parser to use at which the fragments are extracted.",
)
@click.option(
    "-g",
    "--granularity",
    type=str,
    default="paragraph",
    help="The granularity at which the fragments are extracted.",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(readable=True, file_okay=True, dir_okay=False, exists=True),
    help="The trained model used to embed the values.",
)
@click.option(
    "-f",
    "--focus-on",
    type=str,
    multiple=True,
    help="Only keep the fragments of the given type (can be given multiple "
    "times); every other fragment is removed from the database. (Everything "
    "is kept is not given.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True, file_okay=True, dir_okay=False, exists=False),
    help="The path where the cluster labels will be saved as a NumPy array.",
)
@click.option(
    "-d",
    "--distance",
    type=int,
    help="The maximum distance for the clustering. When given, the "
    "clustering step is run.",
)
@click.option(
    "--syntactic",
    "syntactic",
    flag_value=True,
    default=True,
    help="Use the syntactic embedding, i.e., a trained model (default).",
)
@click.option(
    "--semantic",
    "syntactic",
    flag_value=False,
    help="Use the semantic embedding, i.e., a sentence transformer.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1024,
    help="The batch size used for the inference (default: 1024).",
)
@click.option(
    "--db",
    "--database",
    required=True,
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    help="The DuckDB database used throughout the pipeline.",
)
def run(
    inputs,
    input_list,
    parser,
    granularity,
    focus_on,
    model,
    output,
    distance,
    syntactic,
    batch_size,
    db,
):
    """
    Run the full pipeline in a single command: parse the given inputs (if any is given),
    embed the values with the given model, and cluster the values of the
    database. The embedding step is only run if `--model` is given, and
    the clustering step is only run if `--distance` is given.
    """

    if db is None:
        db = toolconfig["database"]
    if inputs or input_list:
        parse_inputs_list(inputs, input_list, parser, granularity, focus_on, db=db)
    if model:
        infer(model, None, None, batch_size, db_path=db, syntactic=syntactic)
    if distance:
        distance = int(distance)
        cluster_grid_agglomerative(distance, output, db)


## READ PIPELINE


@cli.command("parse")
@click.argument(
    "inputs",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
    nargs=-1,
    help="The files or directories (given one or multiple times; directories "
    "are traversed recursively) to parse.",
)
@click.option(
    "-i",
    "--input-list",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="A file containing a list of additional inputs (one path per line) to "
    "add to the inputs.",
)
@click.option(
    "-p",
    "--parser",
    type=str,
    default="markdown",
    help="The granularity at which the fragments are extracted.",
)
@click.option(
    "-g",
    "--granularity",
    type=str,
    default="paragraph",
    help="The granularity at which the fragments are extracted.",
)
@click.option(
    "-f",
    "--focus-on",
    type=str,
    multiple=True,
    help="Only keep the fragments of the given type (can be given multiple "
    "times); every other fragment is removed from the database. Everything is "
    "kept if not given.",
)
@click.option(
    "--db",
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The DuckDB database file in which the fragments are stored. "
    "Defaults to the `database` key of the configuration file.",
)
def parse_cli(
    inputs: tuple[str, ...],
    input_list: str,
    parser: str,
    granularity: str,
    focus_on,
    db: str | None,
):
    """
    Parse the given inputs (a file or a directory, given one or multiple
    times; directories are traversed recursively), and store the extracted
    fragments in the DuckDB database given to `--db`.
    """

    if db is None:
        db = toolconfig["database"]
    parse_inputs_list(inputs, input_list, parser, granularity, focus_on, db)


def parse_inputs_list(
    inputs: tuple[str, ...],
    input_list: str,
    parser: str,
    granularity: str,
    focus_on: list[str],
    db: str | None,
):
    if input_list:
        inputs = list(inputs)
        inputs.extend(read_input(Path(input_list)))
    parse([Path(i) for i in inputs], parser, granularity, focus_on, Path(db))


def parse(
    data: Path | list[Path],
    parser: str,
    granularity: str,
    focus_on: list[str],
    database: Path,
):
    if database.exists():
        raise ValueError(
            "Database already exists! Refusing to overwrite! Please first delete the database."
        )

    if not (isinstance(data, list) or isinstance(data, tuple)):
        data = [data]

    inputs = []
    for d in data:
        if d.is_file():
            inputs.append(d)
        elif d.is_dir():
            for root, dirs, files in os.walk(d):
                inputs.extend([Path(os.path.join(root, f)) for f in files])
        else:
            logger.error(
                "Unsupported input format... The input should be a directory or a file! Stopping now!"
            )
            exit(1)

    logger.info("Begin the parsing of %d files", len(inputs))
    parser = PARSERS[parser](database)
    parser(inputs, granularity)
    if len(focus_on) > 0:
        logger.debug("Deleting everything not in %s", str(focus_on))
        with duckdb.connect(database) as db:
            db.execute(
                "DELETE FROM paths WHERE abstract_path[-1] NOT IN ?;",
                parameters=[focus_on],
            )


## TRAINING PIPELINE ##


@cli.group()
def train():
    """Train a string-embedding model."""


@train.command("nscnn")
@click.option(
    "--input",
    type=click.Path(exists=True, readable=True),
    default=None,
    help="The path to a file (JSON, JSON Lines or plain text) from which "
    "the training texts are sampled (one per line if plain text; list of string "
    "if JSON or one string per line if JSONL).",
)
@click.option(
    "--output",
    type=click.Path(writable=True, dir_okay=False, exists=False),
    required=True,
    help="The path where the trained model will be saved.",
)
@click.option(
    "--char-encoder",
    type=click.Choice(list(CHAR_VECTORIZERS.keys())),
    default="utf8",
    help="The character vectorization used to feed the model (default: utf8).",
)
@click.option(
    "--n-training",
    type=int,
    default=1000,
    help="The number of texts used for training (default: 1000).",
)
@click.option(
    "--n-query",
    type=int,
    default=1000,
    help="The number of texts used for the validation query (default: 1000).",
)
@click.option(
    "--maximum-length",
    type=int,
    default=None,
    help="The maximum length of the texts.",
)
@click.option(
    "--epochs",
    type=int,
    default=20,
    help="The number of training epochs (default: 20).",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="The batch size (default: 64).",
)
@click.option(
    "--embedding-size",
    type=int,
    default=128,
    help="The size of the output embedding (default: 128).",
)
@click.option(
    "--channel",
    type=int,
    default=8,
    help="The number of convolution channels (default: 8).",
)
@click.option(
    "--layers",
    type=int,
    default=4,
    help="The number of convolutional layers (default: 4).",
)
@click.option(
    "--lr",
    type=float,
    default=1e-3,
    help="The learning rate (default: 1e-3).",
)
@click.option(
    "--weight-decay",
    type=float,
    default=0,
    help="The weight decay (default: 0).",
)
@click.option(
    "--save-every",
    type=int,
    default=5,
    help="Save the model every N epochs (default: 5).",
)
@click.option(
    "--patience",
    type=int,
    default=-1,
    help="The number of epochs without improvement to wait before stopping "
    "the training (default: -1, i.e., no early stopping).",
)
@click.option(
    "--distance-metric",
    type=click.Choice(list(DISTANCE_METRIC.keys())),
    default="levenshtein",
    help="The distance used as the training target (default: levenshtein).",
)
@click.option(
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="Sample the training texts from this DuckDB database instead of `--input`.",
)
def nscnn_train_cli(
    input: str,
    output: str,
    char_encoder: str,
    n_training: int,
    n_query: int,
    maximum_length: int | None,
    epochs: int,
    batch_size: int,
    embedding_size: int,
    channel: int,
    layers: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    patience: int,
    distance_metric: str,
    database: str | None,
):
    """
    Train a string-embedding model such that the Euclidean distance between
    the embeddings of two strings approximates their (normalized)
    Levenshtein distance.
    """

    nscnn_train(
        input,
        output,
        char_encoder,
        n_training,
        n_query,
        maximum_length,
        epochs,
        batch_size,
        embedding_size,
        channel,
        layers,
        lr,
        weight_decay,
        save_every,
        patience,
        distance_metric,
        database,
    )


def nscnn_train(
    input: str | None,
    output: str,
    char_vectorizer: str,
    n_training: int,
    n_query: int,
    maximum_length: int | None,
    epochs: int,
    batch_size: int,
    embedding_size: int,
    channel: int,
    layers: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    patience: int,
    distance_metric: str,
    database: str | None,
):
    output = Path(output)
    cv = CHAR_VECTORIZERS.get(char_vectorizer, None)
    if not cv:
        logger.error(
            "Unknown char encoders '%s'. Must be one of '%s'. Stopping now...",
            char_vectorizer,
            ",".join(CHAR_VECTORIZERS.keys()),
        )
        exit(1)
    if database is None:
        database = toolconfig.get("database", None)

    if input:
        input_file = Path(input)
        texts = read_input(input_file)
    elif database:
        import duckdb

        n = int(n_training) + int(n_query)
        texts = []
        sql = f"""
        SELECT value
        FROM (SELECT * FROM paths) 
        USING SAMPLE {n};
        """
        with duckdb.connect(database) as db:
            result = db.execute(sql)
            while rows := result.fetchmany(1_000):
                texts.extend((_t[0] for _t in rows))
    else:
        logger.error("No data source was given! Stopping now...")
        exit(1)

    from desimate.encoders.neuroseed import NeuroSeedCNNTrainer

    cv = cv(max_length=maximum_length)

    trainer = NeuroSeedCNNTrainer(cv)
    trainer.execute(
        output_model=output,
        texts=texts,
        n_training=n_training,
        n_query=n_query,
        maximum_length=maximum_length,
        epochs=epochs,
        batch_size=batch_size,
        embedding_size=embedding_size,
        channel=channel,
        layers=layers,
        lr=lr,
        weight_decay=weight_decay,
        save_every=save_every,
        patience=patience,
        distance_metric=DISTANCE_METRIC[distance_metric],
    )


## INFERENCE PIPELINE ##


@cli.command("embed")
@click.option(
    "--model",
    type=str,
    required=True,
    help="Either the path to a trained model (syntactic embedding), or the "
    "name of a Hugging Face sentence transformer (semantic embedding).",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, readable=True),
    default=None,
    help="A file (JSON, JSON Lines or plain text) containing the strings to embed "
    "(one per line if plain text; list of string "
    "if JSON or one string per line if JSONL).",
)
@click.option(
    "--output-file",
    type=click.Path(writable=True),
    default=None,
    help="The path where the embeddings will be saved as a NumPy array.",
)
@click.option(
    "--batch-size",
    type=int,
    default=2048,
    help="The batch size used for the inference (default: 2048).",
)
@click.option(
    "--syntactic",
    "syntactic",
    flag_value=True,
    default=True,
    help="Use the syntactic embedding, i.e., a trained model (default).",
)
@click.option(
    "--semantic",
    "syntactic",
    flag_value=False,
    help="Use the semantic embedding, i.e., a sentence transformer.",
)
@click.option(
    "--db",
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The DuckDB database whose values will be embedded (the embeddings "
    "are stored in its `embeds` table).",
)
def infer_cli(model: str, input_file, output_file, batch_size, db, syntactic):
    """
    Embed strings with a model. Either embed the strings of a given input
    file into a NumPy file (via `--input-file`/`--output-file`), or embed
    all the values stored in the database and store the results in its
    `embeds` table (via `--db`).
    """

    infer(model, input_file, output_file, batch_size, db, syntactic)


def infer(model: str, input_file, output_file, batch_size, db_path, syntactic):
    if syntactic:
        model = Path(model)
        if not model.exists():
            raise ValueError(f"The given model {model} does not exist!")
        vec = BaseVectorizer(model, OrdCV(), show_progress=False)
    else:
        vec = TransformersVectorizers(model, show_progress=False)
    if input_file:
        input_file = Path(input_file)
        output_file = Path(output_file)
        texts = read_input(input_file)
        logger.debug("Read %d texts to vectorize.", len(texts))
        embeds: np.ndarray = vec(texts, batch_size)
        embeds = np.concatenate(list(embeds))
        np.save(output_file, embeds)
    else:
        import duckdb

        with duckdb.connect(db_path) as db, get_progress() as progress:
            n = db.execute("""
            SET memory_limit = '1GB'; SET preserve_insertion_order = false;
            CREATE OR REPLACE TABLE embeds (file_id STRING,
            ind INTEGER, embedding FLOAT[], PRIMARY KEY (file_id, ind));
            SELECT COUNT(*) FROM paths;
            """).fetchone()[0]
            sql = """
            SELECT file_id, ind, value
            FROM paths;
            """
            reader = db.cursor()
            writer = db.cursor()
            result = reader.execute(sql)
            t = progress.add_task("Inferring from the DB", total=n)
            while rows := result.fetchmany(batch_size):
                texts = [_t[-1] for _t in rows]
                batch_embeds_iter = vec(texts, batch_size)
                current = 0
                for batch_embeds in batch_embeds_iter:
                    res = [
                        ValueEmbedding(
                            rows[current + _i][0],
                            rows[current + _i][1],
                            batch_embeds[_i],
                        )
                        for _i in range(len(batch_embeds))
                    ]

                    df = pd.DataFrame(res, columns=ValueEmbedding._fields)
                    writer.execute(
                        """
                        INSERT INTO embeds 
                        SELECT * FROM df
                        """
                    )
                    progress.advance(t, advance=len(df))
                    current += len(batch_embeds)


## CLUSTERING PIPELINE ##


@cli.group()
def cluster():
    """Cluster the values stored in the database."""


@cluster.command("levenshtein")
@click.option(
    "-d",
    "--max-distance",
    type=int,
    default=10,
    help="The maximum distance between two strings for them to belong to "
    "the same cluster (default: 10).",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The path where the resulting cluster labels will be saved as a NumPy array.",
)
@click.option(
    "--db",
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The DuckDB database to cluster. Defaults to the `database` key of "
    "the configuration file.",
)
def cluster_levenshtein(max_distance: int, output_file: str, db: str | None):
    """
    Cluster the values stored in the database using the Levenshtein
    distance directly. Scales quadratically with the number of values.
    """

    if output_file:
        output_file: Path = Path(output_file)
    if db is None:
        db = toolconfig.get("database", None)
    clustering = Agglomerative(Path(db))
    res = clustering.execute(max_distance=max_distance, distance=LevenshteinDistance)
    if output_file:
        np.save(output_file, res)


@cluster.command("grid")
@click.option(
    "-d",
    "--max-distance",
    type=int,
    default=10,
    help="The maximum distance between two strings for them to belong to "
    "the same cluster (default: 10).",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The path where the resulting cluster labels will be saved as a NumPy array.",
)
@click.option(
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The DuckDB database to cluster. Defaults to the `database` key of "
    "the configuration file.",
)
def cluster_grid_agglomerative_cli(
    max_distance: int, output_file: str, database: str | None
):
    """
    Cluster the values stored in the database with a grid-based
    agglomerative clustering: the Euclidean distance between the embeddings
    is used as a cheap first filter, and the Levenshtein distance is used
    to refine the clusters. This is the scalable version.

    Be cautious when clustering a lot of small fragments with a large
    threshold, as such fragments will have a tendency for form a dense cluster,
    which might take time to be clustered.
    """

    cluster_grid_agglomerative(max_distance, output_file, database)


def cluster_grid_agglomerative(
    max_distance: int, output_file: str, database: str | None
):
    if output_file:
        output_file: Path = Path(output_file)
    if database is None:
        database = toolconfig.get("database", None)
    clustering = GridAgglomerative(Path(database))
    clustering.execute(
        max_distance=max_distance,
        distance=EuclideanDistance,
        real_distance=LevenshteinDistance,
        error_margin=0.3,
    )
    if output_file:
        with duckdb.connect(database) as db:
            db.execute(
                f"""
                COPY (
                SELECT p.file_id AS file_id, p.ind AS fragment_id, p.value AS value, c.cluster_id AS cluster_id 
                FROM paths p 
                JOIN clusters c 
                ON p.rowid = c.item_id )
                TO '{output_file}' (FORMAT parquet)
                """
            )


@cli.command()
@click.argument(
    "output-json",
    help="The path to the JSON Lines file where the sampled values will be written.",
)
@click.option(
    "--count",
    type=int,
    help="The number of values to sample.",
)
@click.option(
    "--path-ends-with",
    help="Only sample values of the given fragment type "
    "(e.g., paragraph, table, code).",
)
@click.option(
    "--min-length",
    type=int,
    default=None,
    help="The minimum length of the sampled values.",
)
@click.option(
    "--max-length",
    type=int,
    default=None,
    help="The maximum length of the sampled values.",
)
@click.option(
    "--db",
    "--database",
    type=click.Path(readable=True, writable=True, dir_okay=False, file_okay=True),
    default=None,
    help="The DuckDB database to sample from.",
)
def create_training_dataset(
    output_json, count, path_ends_with, min_length, max_length, db
):
    """
    Sample values from the database, and write them in a JSON Lines file
    (one JSON-encoded string per line) that can be given to `train` via its
    `--input` option.
    """

    with duckdb.connect(db) as _db:
        sql = """
        SELECT value FROM (SELECT * FROM paths
        """
        conditions = []
        if path_ends_with:
            conditions.append("abstract_path[-1] = '{path_ends_with}'")
        if min_length:
            conditions.append(f"len(value) >= {min_length}")
        if max_length:
            conditions.append(f"len(value) < {max_length}")
        if len(conditions) > 0:
            sql += " WHERE " + " AND ".join(conditions)
        sql += ")"
        if count is not None and count > 0:
            sql += f"USING SAMPLE {count}"
        result = _db.execute(sql)
        with open(output_json, "w") as f:
            while rows := result.fetchmany(10_000):
                for _s in rows:
                    f.write(json.dumps(_s[0], indent=None))
                    f.write("\n")


if __name__ == "__main__":
    cli()
