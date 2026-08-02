# `trialrunner`

`trialrunner` executes a single Hydra-resolved experiment or Hydra multirun. It builds the configured dataset connector and decision scheme, runs benchmark examples concurrently, records Phoenix spans, and writes compressed track files.

## Entry Point

```shell
uv run trial --config-dir configs/trials --config-name local +experiment=standard
```

The console script is defined in `pyproject.toml` as:

```text
trial = "social_groups.trialrunner.__main__:main"
trialrunner = "social_groups.trialrunner.__main__:main"
```

## Runtime Flow

```text
Hydra config
  -> setup_config
  -> MainConfig Pydantic validation
  -> data connector registry
  -> decision scheme registry
  -> concurrent example execution
  -> Phoenix spans and compressed track file
```

## Important Files

| Path | Purpose |
| --- | --- |
| `__main__.py` | Hydra entry point; initializes config, logging, Phoenix, metadata, and experiment execution. |
| `config.py` | Backend, LLM, debate-agent, and LangChain model-construction logic. |
| `utils/hydra_config.py` | Pydantic models for `MainConfig`, `ExperimentConfig`, and `ExecutionConfig`. |
| `experiment/main_registry.py` | Registries for data connectors and decision schemes. |
| `experiment/run_experiment.py` | Main experiment loop, retries, concurrent workers, tracking writes. |
| `data_connectors/` | Dataset adapters that produce `ExampleInput` values. |
| `decision_schemes/` | Single-agent, voting, and multi-agent debate strategies. |
| `utils/phoenix.py` | Phoenix setup, span helpers, and deep-link helpers. |
| `utils/meta_info.py` | Experiment metadata generation. |

## Inputs

- Hydra configs under `configs/trials`.
- Dataset names registered in `data_connectors`.
- Strategy names registered in `decision_schemes`.
- Model endpoints from `execution.model_backends`.
- Phoenix URLs from `execution.phoenix_server_url` and `execution.phoenix_graphql_url`.

## Outputs

Hydra controls the output directory:

```text
results/runs/${hostname:}/${experiment.name}/${now:%Y-%m-%d-%H-%M-%S}
results/multirun/${hostname:}/${experiment.name}/${now:%Y-%m-%d-%H-%M-%S}
```

Each run writes:

| File | Meaning |
| --- | --- |
| `experiment_result.jsonl.zst` | Compressed line-oriented track entries with input, output, and Phoenix span handles. |
| Hydra files | Resolved configuration and runtime metadata. |
| logs | Experiment stdout/logging output. |

## Registered Data Connectors

The current source registers:

- `mmlu-pro`
- `mmlu-pro-subset`
- `mmlu-pro-medium-subset`
- `mmlu-pro-big-subset`
- `mmlu-pro-single`
- `gpqa-diamond`

Connectors subclass `DataConnector` and implement:

- `prepare_example`
- `iterate_data`
- `data_length`

## Registered Decision Schemes

The current source registers:

- `single-agent`
- `single-agent-tool`
- `single-agent-structured-output`
- `multi-agent-debate`
- `custom-multi-agent-debate`
- `changed_prompt_mad`
- `changed_prompt_thinking_mad`
- `thinking-mad`
- `tribal-council`
- `encouraging-divergent-thinking`

Decision schemes subclass `DecisionScheme[ConfigurationModel]` and implement `run_example`.

## Concurrency and Failure Handling

`run_experiment.py` uses a `ThreadPoolExecutor` with `execution.num_workers`. New examples are started gradually using `execution.seconds_between_start_of_new_experiment` to avoid bursting the model backend. Transient API and timeout errors are retried up to `MAXIMUM_RETRIES_PER_EXPERIMENT`.

## Adding a Strategy

1. Add a module under `decision_schemes/`.
2. Define a Pydantic-compatible configuration model.
3. Subclass `DecisionScheme[YourConfig]`.
4. Register it with `@register_decision_scheme("your-name")`.
5. Reference it from a Hydra config through `experiment.strategy.name`.

This documentation was generated using an LLM
