from __future__ import annotations

import logging

from config import CLI_SUBTITLE, CLI_TITLE
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from openai import APIConnectionError
from pydantic import ValidationError
from pyfiglet import figlet_format
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.traceback import Traceback
from utils.hydra_config import ExecutionConfig, MainConfig
from utils.meta_info import generate_meta_information
from utils.phoenix import phoenix_server_is_up, setup_phoenix

log = logging.getLogger(__name__)


def print_title(console: Console):
    console.print(
        Panel.fit(
            figlet_format(CLI_TITLE, font="ansi_shadow", width=200),
            subtitle=CLI_SUBTITLE,
        ),
        justify="center",
    )


def initialize_phoenix(console: Console, conf: ExecutionConfig):
    log.info("Checking Phoenix Server connection...")

    if phoenix_server_is_up(url=f"{conf.phoenix_server_url}/healthz"):
        log.info("Writing LLM Interactions into Phoenix. It is up and running.")
        setup_phoenix(endpoint=conf.phoenix_graphql_url, project_name="experiment")

        return

    console.print(
        Panel(
            Group(
                Text("❌ Abort", style="bold red"),
                Text("\n\n"),
                Text(
                    "You have to start the phoenix server first! (See Readme.md for help)."
                ),
            ),
            title="Failed",
        )
    )

    exit(1)


def print_config_overview(console: Console, config: MainConfig):
    console.print(
        Panel(
            Syntax(
                OmegaConf.to_yaml(
                    OmegaConf.create(
                        config.model_dump(mode="json", exclude={"meta_info"})
                    ),
                    resolve=True,
                ),
                "yaml",
                word_wrap=True,
            ),
            title=f"Resolved config - {HydraConfig.get().runtime.choices['experiment']}",
            title_align="left",
        ),
    )


def print_final_message(console: Console, config: MainConfig):
    console.print(
        Panel(
            Group(
                Text("✅ Done", style="bold green"),
                Text("\n"),
                Text(f"Results: {config.meta_info.output_directory}"),
            ),
            title="Success",
        )
    )


def handle_failure_exception(console: Console, exception: Exception):
    console.print(Panel(str(exception), title="Run failed", style="red"))
    if isinstance(exception, APIConnectionError):
        console.print(
            Panel(repr(exception.request), title="Failed Request", style="yellow")
        )
    elif isinstance(exception, ValidationError):
        pass
    else:
        console.print(
            Traceback.from_exception(
                type(exception), exception, exception.__traceback__
            )
        )

    log.info("Error raised: %s", repr(exception))
    exit(1)


def setup_config(console: Console, cfg: DictConfig) -> MainConfig:
    try:
        config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
        config.meta_info = generate_meta_information()
        return config

    except ValidationError:
        console.print(
            "\n[bold yellow] Did you select an experiment config? (experiment=<exp name>)[/bold yellow]\n"
        )
        raise
