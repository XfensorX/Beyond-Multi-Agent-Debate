# `analysis`

`analysis` is the Dagster project layer. It loads normalized Parquet files, builds cleaned experiment-specific assets, runs asset checks, executes notebooks through Dagstermill, and materializes thesis report artifacts.

## Dagster Entry

```shell
uv run task dagster
```

The task sets `DAGSTER_HOME=.tmp_dagster` and runs `uv run dg dev`. Dagster discovers this project through `pyproject.toml`:

```text
root_module = "social_groups.analysis"
```

## Important Files

| Path | Purpose |
| --- | --- |
| `definitions.py` | Dagster `@definitions` entry point using `load_from_defs_folder`. |
| `defs/raw.py` | Loads raw Parquet tables and builds `combined_data`. |
| `defs/experiments/` | Earlier experiment-family assets. |
| `defs/final_experiments/` | Final experiment-family assets used by thesis reporting. |
| `defs/external/` | External comparison/reference assets. |
| `defs/notebooks/definitions.py` | Notebook asset registry and extra materialized report assets. |
| `asset_checks.py` | Reusable Dagster asset checks. |
| `notebook_assets.py` | Helpers for Dagstermill notebooks and extra asset materialization. |
| `polars_transformations/` | Shared Polars transformations for parsing, voting, constellations, Pareto fronts, and distances. |

## Inputs

Raw assets read Parquet files from:

```text
results/analysis/parquet
```

The expected files are:

- `Answer.parquet`
- `Experiment.parquet`
- `Question.parquet`
- `Run.parquet`

## Outputs

Dagster IO managers write below:

```text
results/analysis/dagster
```

Report-facing notebook assets are materialized below:

```text
results/analysis/dagster/report
```

## Asset Layers

1. Raw assets scan the Parquet model tables.
2. `combined_data` joins answers, runs, experiments, and questions.
3. Experiment-family assets filter and enrich `combined_data`.
4. Asset checks validate assumptions such as dataset uniqueness, expected length, or group constellations.
5. Notebook assets create thesis tables, plots, and intermediate report Parquet files.

## Notebook Materializations

`notebook_assets.py` defines `ExtraNotebookAsset`, which supports:

- `csv`
- `tex`
- `svg`
- `pdf`
- `png`
- `parquet`

Notebooks register outputs through `register_materialization(...)`; paths and metadata are exposed through Dagster.

## Common Transformations

| Module | Purpose |
| --- | --- |
| `apply_parsing_and_group_decision.py` | Parse individual answers, aggregate group replies, and compare against target answers. |
| `deserialize_experiment_configuration.py` | Convert stored experiment config JSON into Polars struct data. |
| `make_group_constellation.py` | Build model-family and group-constellation columns. |
| `pareto_front.py` | Compute Pareto-front style comparisons. |
| `wasserstein_distance.py` | Distance calculation used in analysis. |

This documentation was generated using an LLM
