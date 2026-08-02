# Social Groups Experiment Framework

This repository contains the experiment, orchestration, tracing, analysis, and reporting code for evaluating group-based language-model decision strategies. The code is organized around a reproducible pipeline: define experiments with Hydra, run them locally or on SLURM-managed cluster nodes, record model interactions and traces, parse completed runs into analysis tables, and materialize report artifacts for the thesis.

## System Overview

```text
configs/trials
  -> trialrunner
  -> model servers and Phoenix tracing
  -> compressed experiment tracks
  -> analyzer parquet build
  -> Dagster assets and notebooks
  -> report tables and figures
```

The architecture diagram source is stored at `assets/information/system_architecture.mmd`. The intended rendered thesis figure is `diagrams/architecture.svg`.

## Main Source Packages

| Package | Purpose |
| --- | --- |
| `src/social_groups/trialrunner` | Runs Hydra-configured experiments over datasets and decision schemes. |
| `src/social_groups/orchestrator` | Starts, inspects, pipes, and syncs SLURM jobs for model servers, tracing, and experiments. |
| `src/social_groups/analyzer` | Converts compressed experiment tracks plus Phoenix span data into normalized Parquet tables. |
| `src/social_groups/analysis` | Defines Dagster assets, notebook assets, checks, and Polars transformations. |
| `src/social_groups/reporting` | Provides answer parsing, voting aggregation, comparison helpers, and plotting functions. |
| `src/social_groups/tools` | Contains Streamlit utilities for manually inspecting generated artifacts. |
| `src/social_groups/general` | Holds shared command, tracking, registry, URL, and standard-library helpers. |

Each package has a local `README.md` with module-level details.

## Repository Layout

```text
assets/
  information/        Generated or thesis-facing architecture/info artifacts
  orchestration/      NGINX config and Apptainer image locations used by orchestration
  scripts/            Convenience scripts for starting groups of cluster jobs
configs/
  orchestration/      Per-location service configs for pascal and neumann
  trials/             Hydra configs for experiment runs and sweeps
notebooks/            Exploratory notebooks and scratch analysis
src/social_groups/    Python package
results/              Generated experiment, Phoenix, analysis, and report outputs
```

`results/` can contain many large generated files. It is an output area, not a source-of-truth code area.

## Installation

The project is a Python 3.11 package managed with `uv`.

```shell
uv sync
```

The package exposes console scripts through `pyproject.toml`. Most project commands should be run through `uv run`.

## Command Reference

| Command | Defined by | Purpose |
| --- | --- | --- |
| `uv run trial ...` | `social_groups.trialrunner.__main__` | Run a Hydra experiment locally or inside a cluster experiment job. |
| `uv run orch start <location> <service>` | `social_groups.orchestrator.__main__` | Submit a SLURM job for `experiment`, `phoenix`, `phoenix-pg`, `tgi`, or `vllm`. |
| `uv run orch show <location>` | `orchestrator` | Show current SLURM jobs for the configured remote user. |
| `uv run orch stop <location>` | `orchestrator` | Interactively select running jobs to cancel. |
| `uv run orch log <location> [service]` | `orchestrator` | Fetch and print a selected SLURM log. |
| `uv run orch pipe <location> <service>` | `orchestrator` | Open an SSH local port forward to a running service. |
| `uv run orch sync <location>` | `orchestrator` | Interactively sync selected remote multirun result directories. |
| `uv run ana parse` | `social_groups.analyzer.__main__` | Build normalized Parquet files from final run outputs and Phoenix spans. |
| `uv run ana sync-tex` | `analyzer` | Sync generated Dagster report assets into the adjacent thesis project. |
| `uv run task dagster` | `taskipy` | Start Dagster with `DAGSTER_HOME=.tmp_dagster`. |
| `uv run task tools` | `taskipy` | Start the Streamlit tools app. |
| `uv run task pull_tgi_apptainer` | `taskipy` | Pull the TGI Apptainer image into `assets/orchestration/tgi.sif`. |
| `uv run task pull_pg_apptainer` | `taskipy` | Pull the PostgreSQL Apptainer image. |
| `uv run task pull_nginx_apptainer` | `taskipy` | Pull the NGINX Apptainer image. |

## Typical Workflows

### Run an experiment locally

Local execution requires reachable model endpoints matching `configs/trials/local.yaml` and a Phoenix endpoint if tracing is enabled by the selected config.

```shell
uv run trial --config-dir configs/trials --config-name local +experiment=standard
```

Hydra writes single runs under `results/runs/${hostname:}/...` and multiruns under `results/multirun/${hostname:}/...`. Each completed run writes metadata, Hydra config, logs, and a compressed `experiment_result.jsonl.zst`.

### Start cluster infrastructure

Cluster execution is driven by typed service configs in `configs/orchestration/<location>`.

```shell
uv run orch start neumann phoenix-pg
uv run orch start neumann vllm --llm Qwen/Qwen3-4B
uv run orch start neumann experiment -m -e final/baseline
```

The experiment service discovers running Phoenix and model-server jobs, injects their endpoints into the Hydra command, and submits a new SLURM job.

### Parse final runs

The analyzer reads final run locations from:

```text
results/runs/final
results/multirun/final
```

Run:

```shell
uv run ana parse
```

The output is written to:

```text
results/analysis/parquet/Answer.parquet
results/analysis/parquet/Question.parquet
results/analysis/parquet/Run.parquet
results/analysis/parquet/Experiment.parquet
```

### Build report assets

Dagster loads definitions from `src/social_groups/analysis`.

```shell
uv run task dagster
```

Raw Parquet assets, combined data assets, final experiment assets, checks, notebooks, and extra report materializations are stored below `results/analysis/dagster`.

### Sync report assets to the thesis project

```shell
uv run ana sync-tex
```

This command copies `results/analysis/dagster/report` to the adjacent `TexProject/data/report` directory and runs git operations in the target directory.

## Data and Artifact Flow

1. `configs/trials` define experiment names, datasets, strategies, backends, parameters, and sweeps.
2. `trialrunner` validates the resolved Hydra config with Pydantic.
3. Runtime registries instantiate a data connector and a decision scheme.
4. Examples are processed concurrently with retries.
5. LLM calls go through LangChain adapters to OpenAI-compatible, Hugging Face, Qwen, Mistral, TGI, vLLM, LM Studio, or related endpoints configured in code.
6. Phoenix captures per-example traces and span attributes.
7. `ExperimentTracker` writes `experiment_result.jsonl`, then compresses it to `experiment_result.jsonl.zst`.
8. `analyzer` reads run folders, enriches entries with Phoenix GraphQL span attributes, and writes normalized Parquet tables.
9. `analysis` and `reporting` transform the normalized tables into evaluation assets, plots, and thesis-ready report files.

## Experiment Configuration

The final experiment configurations are primarily under:

```text
configs/trials/experiment/final
configs/trials/experiment/final_gemma
```

Config families include single-agent baselines, no-discussion voting, standard and changed-prompt multi-agent debate, thinking visibility experiments, diversity parameter sweeps, different-family experiments, and Gemma-specific GPQA runs.

## Development Notes

- Prefer adding new decision schemes through `trialrunner/decision_schemes` and the `@register_decision_scheme(...)` decorator.
- Prefer adding new datasets through `trialrunner/data_connectors` and the `@register_data_connector(...)` decorator.
- New cluster services should subclass `SlurmService` and register through `@register_slurm_service(...)`.
- Analysis code should keep generated artifacts below `results/analysis`, not inside source directories.
- The project uses shared Polars column constants in `src/social_groups/polars_columns.py`.

This documentation was generated using an LLM
