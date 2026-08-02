# `general`

`general` contains shared helpers used across execution, orchestration, analysis, and reporting.

## Important Files

| Path | Purpose |
| --- | --- |
| `tracking.py` | `TrackEntry`, `ExperimentTracker`, zstandard compression, and `.jsonl.zst` iteration. |
| `run_commands.py` | Local subprocess and SSH command helpers. |
| `utils/registry.py` | Simple named registry used by data connectors, decision schemes, and services. |
| `utils/standard_library.py` | Shared utility functions, including submodule import helpers and Pydantic helper types. |
| `utils/urls.py` | URL helper functions. |
| `debug.py` | Debug helpers. |
| `types.py` | Shared type helpers. |

## Tracking Format

`ExperimentTracker` writes one JSON object per benchmark example. Each line validates as `TrackEntry` and contains:

- prepared input
- output or `null`
- Phoenix span handle

On close, the tracker compresses `experiment_result.jsonl` into `experiment_result.jsonl.zst` and removes the uncompressed file.

## Command Helpers

`run_local(...)` wraps local subprocess calls. `run_ssh(...)` runs commands through:

```text
ssh <login> bash -lc "<remote_cmd>"
```

It can optionally create and enter a remote working directory before running the command, and it can pass text to stdin.

## Registry Pattern

The registry helper supports the main plugin points in the repository:

- `trialrunner` data connectors
- `trialrunner` decision schemes
- `orchestrator` SLURM services

The pattern keeps configuration values decoupled from concrete classes while still validating registered class types.

This documentation was generated using an LLM
