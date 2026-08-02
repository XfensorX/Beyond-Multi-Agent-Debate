# `social_groups` Package

`social_groups` is the project package for running group-based language-model experiments and processing their outputs. It contains three major layers:

1. Execution: `trialrunner` runs configured experiments.
2. Infrastructure: `orchestrator` starts SLURM services and cluster jobs.
3. Evaluation: `analyzer`, `analysis`, and `reporting` transform raw runs into thesis figures and tables.

## Package Map

| Path | Role |
| --- | --- |
| `trialrunner/` | Hydra entry point, config models, data connectors, decision schemes, experiment loop. |
| `orchestrator/` | Typer CLI for SLURM, model-serving, tracing, SSH piping, and result sync. |
| `analyzer/` | Track-file and Phoenix parser that builds normalized Parquet tables. |
| `analysis/` | Dagster project definitions, assets, checks, notebook materialization utilities. |
| `reporting/` | Parsing, voting, metric, and plotting helpers used by assets and notebooks. |
| `tools/` | Streamlit inspection utilities. |
| `general/` | Shared helpers for tracking, command execution, registries, URLs, and utility types. |

## Shared Files

| File | Purpose |
| --- | --- |
| `directories.py` | Central filesystem paths for `results`, analysis output, Dagster report output, and thesis report sync. |
| `config.py` | Shared constants for group sizes, model letters, and tool-call tags. |
| `polars_columns.py` | Canonical string constants for frequently used Polars columns. |
| `polars_values.py` | Shared value mappings, including model-name to size-letter mappings. |

## Generated Output Locations

```text
results/runs
results/multirun
results/analysis/parquet
results/analysis/dagster
results/analysis/dagster/report
```

These directories are written by experiment execution, parsing, and report generation.

## Design Conventions

- Runtime extension points use small registries and decorators.
- Configs are resolved by Hydra and then validated by Pydantic models.
- Experiment outputs are stored as compressed JSONL first and converted to Parquet later.
- Analysis code uses Polars for tabular transformations.

This documentation was generated using an LLM
