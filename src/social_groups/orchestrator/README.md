# `orchestrator`

`orchestrator` manages remote SLURM services needed by experiments. It turns typed YAML service configs into SLURM batch files, submits them over SSH, inspects running jobs, forwards service ports, and syncs completed result directories.

## Entry Point

```shell
uv run orch --help
```

Console scripts:

```text
orch = "social_groups.orchestrator.__main__:main"
orchestrator = "social_groups.orchestrator.__main__:main"
```

## Configuration Inputs

```text
configs/orchestration/pascal
configs/orchestration/neumann
```

Each location has a `general.yaml` with SSH login, remote project directory, log directory, and home directory. Service YAML files define SLURM resources and service-specific settings.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run orch start <location> <service>` | Start a SLURM service. |
| `uv run orch show <location>` | Print current SLURM jobs for the configured user. |
| `uv run orch stop <location>` | Interactively cancel selected jobs. |
| `uv run orch log <location> [service]` | Select and print a SLURM log. |
| `uv run orch pipe <location> <service>` | Open SSH local port forwarding to a service. |
| `uv run orch sync <location>` | Sync selected remote multirun directories to local `results/multirun/<target>`. |

## Services

| Service | Class | Role |
| --- | --- | --- |
| `experiment` | `ExperimentConfiguration` | Starts `uv run trial`, injects model endpoints and Phoenix endpoints, and optionally enables Hydra multirun. |
| `vllm` | `VLLMConfiguration` | Starts OpenAI-compatible vLLM model servers. |
| `tgi` | `TgiConfiguration` | Starts Hugging Face TGI through Apptainer. |
| `phoenix` | `PhoenixConfiguration` | Starts Phoenix with a SQLite backend. |
| `phoenix-pg` | `PhoenixWithPostgresConfiguration` | Starts Phoenix instances, PostgreSQL, and NGINX reverse proxy. |

Services subclass `SlurmService`, define environment variables and run commands, and register with `@register_slurm_service(...)`.

## Cluster Experiment Flow

```text
orch start neumann phoenix-pg
orch start neumann vllm --llm Qwen/Qwen3-4B
orch start neumann experiment -m -e final/baseline
```

For experiment jobs, the CLI:

1. Reads the execution-location config.
2. Reads the requested service config.
3. Queries running SLURM jobs.
4. Finds the Phoenix service.
5. Parses model backend endpoints from running model-server jobs.
6. Builds an `uv run trial` command with Hydra overrides.
7. Submits the generated batch file via `sbatch --parsable --export=NONE`.

## Outputs

- Remote SLURM logs are written below the configured `slurm_log_dir_inside_project`.
- Experiment outputs are written by Hydra under the remote project's `results/runs` or `results/multirun`.
- `orch sync` copies selected remote multirun directories into local `results/multirun/<target>`.

## Implementation Notes

- `models/execution_environment.py` handles SSH execution and `sbatch` submission.
- `models/slurm_config.py` creates the shared SLURM header.
- `services/base.py` contains the service registry and batch-file assembly.
- `utils/running_backends.py` and `utils/slurm_jobs.py` connect running SLURM jobs back to Phoenix and model endpoint metadata.

This documentation was generated using an LLM
