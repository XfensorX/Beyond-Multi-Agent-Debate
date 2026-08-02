# `analyzer`

`analyzer` converts generated experiment runs into normalized Parquet tables. It reads compressed track files, loads Hydra and metadata files, retrieves Phoenix span attributes through GraphQL, and writes model-shaped tables for downstream Dagster assets.

## Entry Point

```shell
uv run ana --help
```

Console scripts:

```text
ana = "social_groups.analyzer.__main__:main"
analyzer = "social_groups.analyzer.__main__:main"
```

## Commands

| Command | Purpose |
| --- | --- |
| `uv run ana parse` | Build Parquet files from final experiment result directories. |
| `uv run ana sync-tex` | Sync Dagster report outputs into the adjacent thesis project. |

## Parse Inputs

The parser scans:

```text
results/runs/final
results/multirun/final
```

Each run directory is expected to contain compressed track data and Hydra metadata written by `trialrunner`.

## Parse Outputs

`ana parse` writes:

```text
results/analysis/parquet/Answer.parquet
results/analysis/parquet/Experiment.parquet
results/analysis/parquet/Question.parquet
results/analysis/parquet/Run.parquet
```

These are the raw analysis inputs consumed by Dagster assets in `analysis/defs/raw.py`.

## Important Files

| Path | Purpose |
| --- | --- |
| `__main__.py` | Typer CLI for `parse` and `sync-tex`. |
| `algorithm/experiment_reader.py` | Main async parser and Parquet writer. |
| `algorithm/phoenix_connector.py` | Batched Phoenix GraphQL lookup loop. |
| `algorithm/utils/` | Queue packages, cache helpers, IO operations, and lazy dictionaries. |
| `models/answer.py` | Parquet model for answer-level rows. |
| `models/question.py` | Parquet model for question-level rows and dataset-specific question extraction. |
| `models/run.py` | Parquet model for run-level config and metadata. |
| `models/experiment.py` | Parquet model for experiment names. |
| `sync_tex.py` | Exact report-directory sync and git helper functions for the thesis project. |

## Data Model

| Table | Main content |
| --- | --- |
| `Answer` | Run id, question id, Phoenix span id/url, model answers, final answer, token usage. |
| `Question` | Dataset connector, source/category, question text, answer options, target answer. |
| `Run` | Experiment config JSON, execution config JSON, metadata JSON. |
| `Experiment` | Experiment id and experiment name. |

## Phoenix Enrichment

Each track entry stores a Phoenix span id. The analyzer batches these ids, queries Phoenix GraphQL, and joins returned span attributes back into the normalized rows. The code uses queues and bounded async requests to keep network usage controlled.

This documentation was generated using an LLM
