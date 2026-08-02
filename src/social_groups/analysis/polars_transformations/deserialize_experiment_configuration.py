import json

import polars as pl

from social_groups.trialrunner.utils.hydra_config import ExperimentConfig


def deserialize_experiment_configuration(expr: pl.Expr) -> pl.Expr:
    try:
        elements = expr.map_elements(
            lambda x: ExperimentConfig.model_validate(json.loads(x)).model_dump_json(),
            return_dtype=pl.Utf8,
        ).str.json_decode(
            dtype=pl.Struct(
                {
                    "strategy": pl.Struct(
                        {
                            "name": pl.String,
                            "configuration": pl.Struct(
                                {
                                    "openly_thinking_models": pl.List(
                                        pl.Struct(
                                            {
                                                "model_name": pl.String,
                                            }
                                        ),
                                    ),
                                    "number_of_rounds": pl.Int64,
                                    "debate_agents": pl.List(
                                        pl.Struct(
                                            {
                                                "backend": pl.Struct(
                                                    {"model_name": pl.String}
                                                ),
                                                "params": pl.Struct(
                                                    {
                                                        "max_new_tokens": pl.Int64,
                                                        "temperature": pl.Float32,
                                                        "top_p": pl.Float32,
                                                        "seed": pl.Int64,
                                                    }
                                                ),
                                            }
                                        )
                                    ),
                                    "llm": pl.Struct(
                                        {
                                            "max_new_tokens": pl.Int64,
                                            "temperature": pl.Float32,
                                            "top_p": pl.Float32,
                                            "seed": pl.Int64,
                                        }
                                    ),
                                    "use_few_shot_prompting": pl.Boolean,
                                    "use_thinking": pl.Boolean,
                                    "backend": pl.Struct(
                                        {
                                            "model_name": pl.String,
                                        }
                                    ),
                                }
                            ),
                        }
                    )
                }
            )
        )

        return elements
    except pl.exceptions.StructFieldNotFoundError as e:
        raise NotImplementedError(
            "The Field is not extracted from the json decoding module. Please add it."
        ) from e
