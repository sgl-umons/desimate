# DeSiMaTe: Detecting Similar Markdown Text

[![License](https://badgen.net/pypi/license/desimate)](https://github.com/sgl-umons/desimate/blob/master/LICENSE.txt)
[![Last release](https://badgen.net/github/release/sgl-umons/desimate/stable)](https://badgen.net/github/release/sgl-umons/desimate/stable)
[![PyPI](https://badgen.net/pypi/v/desimate)](https://pypi.org/project/desimate)
[![Last commit](https://badgen.net/github/last-commit/sgl-umons/desimate)](https://github.com/sgl-umons/desimate/commits/)

An automated tool for detecting similar text in large collections of documents, written in Python.
`desimate` is designed to be used as a CLI tool.
Given a set of Markdown documents, it splits each of them into fragments (paragraphs, lists, tables, code blocks, etc.), stores them in a DuckDB database, embeds them with a learned string-embedding model, and clusters the fragments according to their similarity.

The pipeline is composed of the following steps, each available as a dedicated subcommand:

1. `run`: run the complete pipeline on a set of documents. It is the easiest way for quick results. Alternatively, you can use subcommands to control each step of the pipeline:
2. `parse`: parse the given documents into fragments at a configurable granularity, and store them in a DuckDB database.
3. `create-training-dataset`: sample values from the database into a JSON Lines file, to be used for training.
4. `train`: train a string-embedding model (a character-level convolutional network such as NeuroSEED or CNN-ED) such that the Euclidean distance between two embeddings approximates the Levenshtein distance between the embedded strings.
5. `embed`: embed the fragments with the trained model (syntactic embedding), or with a given sentence transformer (semantic embedding), and store the results in the database.
6. `cluster`: group the fragments into clusters of near-duplicates, either directly with the Levenshtein distance (small datasets) or with a grid-based agglomerative clustering that uses the embeddings as a cheap first filter and the Levenshtein distance to refine the results (large datasets).

This project is developed by Guillaume Cardoen at the Software Engineering Lab of the University of Mons (Belgium).

## Installation

`desimate` requires a Python 3.14 or newer interpreter. The easiest way to install `desimate` is to install it from PyPI
```
pip install desimate
```

Another easy way to install `desimate` is via `pip` from this GitHub repository
```
pip install git+https://github.com/sgl-umons/desimate
```

Alternatively, you can clone this repository and install it locally
```
git clone https://github.com/sgl-umons/desimate
cd desimate
pip install . # or uv sync and uv run
```

You may wish to use `desimate` in a virtual environment
```
python -m venv venv
source venv/bin/activate
pip install desimate
```

## Usage

After installation, the `desimate` command-line tool should be available in your shell. All commands accept a `--config` option (defaulting to `./config.toml`) pointing to a TOML file whose content is used as default for the command options:

```
Usage: desimate [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH    The path to a TOML configuration file.
  --help           Show this message and exit.

Commands:
  cluster                      Cluster the values stored in the database.
  create-training-dataset      Sample values from the database, and write
                               them in a JSON Lines file.
  embed                        Embed the values stored in the database (or
                               a given input file) with a model.
  parse                        Parse the given inputs, and store the
                               extracted fragments in the database.
  process                      Run the full pipeline (parse, embed and
                               cluster) in a single command.
  train                        Train a string-embedding model.
```

The easiest way to use desimate is through the `process` subcommand. 
```
Usage: desimate process [OPTIONS] [INPUTS]...

  Run the full pipeline in a single command: parse the given inputs (if any is
  given), embed the values with the given model, and cluster the values of the
  database. The embedding step is only run if `--model` is given, and the
  clustering step is only run if `--distance` is given.

Positional arguments:
  [INPUTS]...  The files or directories (given one or multiple times;
               directories are traversed recursively) to parse.

Options:
  -i, --input-list FILE         A file containing a list of additional inputs
                                (one path per line) to add to the inputs.
  -p, --parser [markdown|text]  The parser to use at which the fragments are
                                extracted.
  -g, --granularity TEXT        The granularity at which the fragments are
                                extracted.
  -m, --model FILE              The trained model used to embed the values.
  -f, --focus-on TEXT           Only keep the fragments of the given type (can
                                be given multiple times); every other fragment
                                is removed from the database. (Everything is
                                kept is not given.
  -o, --output FILE             The path where the cluster labels will be
                                saved as a NumPy array.
  -d, --distance INTEGER        The maximum distance for the clustering. When
                                given, the clustering step is run.
  --syntactic                   Use the syntactic embedding, i.e., a trained
                                model (default).
  --semantic                    Use the semantic embedding, i.e., a sentence
                                transformer.
  --batch-size INTEGER          The batch size used for the inference
                                (default: 1024).
  --db, --database FILE         The DuckDB database used throughout the
                                pipeline.  [required]
  --help                        Show this message and exit.
```

For Markdown, each granularity is a list of Markdown AST node types at which the traversal stops, i.e., the fragment types that will be extracted. The default configuration defines two granularities:
- `paragraph`: fragments are paragraphs, lists and tables (default);
- `md-line`: fragments are the smallest line-level elements (text, list items, table cells and table rows).

Other granularities (`line` and `file`) are available through the `text` parser, which read the file as pure text. 



### Examples

As an example, the following commands will parse every Markdown file of the `all_content` directory into the `my_dataset.db` database, keeping only the `paragraph`, `table`, `list` and `code` fragments.

The fragments can be parsed, embedded and clustered, with a maximum distance of 15, in a single command:

```bash
desimate process all_content --db my_dataset.db --model model.torch -d 15 -o clusters.parquet  -f paragraph -f table -f list -f code
```

Alternatively, you can use subcommands to do each step at a time.

```bash
desimate parse all_content --db my_dataset.db -f paragraph -f table -f list -f code
```

A training set can then be sampled from the database:

```bash
desimate create-training-dataset training.jsonl --count 25000 --db my_dataset.db
```

An embedding model can be trained on this dataset:

```bash
desimate train nscnn --input training.jsonl --output model.torch --n-training 20000 --n-query 5000
```

## License

Distributed under [GNU Lesser General Public License v3](https://github.com/sgl-umons/desimate/blob/master/LICENSE.txt).
