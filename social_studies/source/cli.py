from __future__ import annotations

import logging

import dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from openai import APIConnectionError
from pydantic import ValidationError
from rich.console import Console, Group
from typing import Literal
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import Traceback
from rich.tree import Tree
from rich.text import Text

import hydra
from pyfiglet import figlet_format

from config import config_dir, LOG_FILE_NAME
from experiment.run_experiment import (
    run_experiment,
)
from utils.hydra_config import MainConfig
from utils.meta_info import generate_meta_information
from experiment.main_registry import DECISION_SCHEMES, DATA_CONNECTORS
from utils.logging import setup_logging
from utils.phoenix import phoenix_server_is_up, setup_phoenix

console = Console()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    try:
        config = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
        config.meta_info = generate_meta_information()

    except ValidationError as e:
        console.print(
            "\n[bold yellow] Did you select an experiment config? (experiment=<exp name>)[/bold yellow]\n"
        )
        raise e

    console.print(
        Panel.fit(
            figlet_format("Agent Swarm", font="ansi_shadow", width=200),
            subtitle="Experiments",
        ),
        justify="center",
    )

    log_file_path = config.meta_info.output_directory / LOG_FILE_NAME
    log_level = "INFO"

    setup_logging(level=log_level, log_file=log_file_path)
    log.info("Logging file: %s", log_file_path)
    log.info("Log Level: %s", log_level)

    log.info("Checking Phoenix Server connection...")

    if phoenix_server_is_up(url=f"{config.execution.phoenix_server_url}/healthz"):
        log.info("Writing LLM Interactions into Phoenix. It is up and running.")
        setup_phoenix(
            endpoint=config.execution.phoenix_graphql_url, project_name="experiment"
        )
    else:
        console.print(
            Panel(
                "[bold red]❌ Abort[/bold red]\n\nYou have to start the phoenix server first! (See Readme.md for help).",
                title="Failed",
            )
        )
        exit(1)

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

    try:
        run_experiment(config)

    except Exception as e:
        console.print(Panel(str(e), title="Run failed", style="red"))
        if isinstance(e, APIConnectionError):
            console.print(
                Panel(repr(e.request), title="Failed Request", style="yellow")
            )
        elif isinstance(e, ValidationError):
            pass
        else:
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))

        log.info("Error raised: %s", repr(e))
        exit(1)

    console.print(
        Panel(
            Group(
                Text("✅ Done", style="bold green"),
                Text("\n\n"),
                Text(f"Results: {config.meta_info.output_directory}"),
            ),
            title="Success",
        )
    )


def list_(what: Literal["schemes", "data", "configs"]):
    match what:
        case "schemes":
            console.print(DECISION_SCHEMES)
        case "data":
            console.print(DATA_CONNECTORS)
        case "configs":
            root = config_dir()
            tree = Tree("📂 config", guide_style="bold bright_blue")

            for p in sorted(root.rglob("*.y*ml")):
                tree.add(f"📃 {p.relative_to(root).with_suffix('').as_posix()}")

            console.print(tree)
        case _:
            raise NotImplementedError(f"{what} not implemented")


if __name__ == "__main__":
    dotenv.load_dotenv("../.env")
    main()
