from __future__ import annotations

import logging
import socket

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from openai import APIConnectionError
from pydantic import ValidationError
from pyfiglet import figlet_format
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.traceback import Traceback

from social_groups.trialrunner.config import CLI_SUBTITLE, CLI_TITLE
from social_groups.trialrunner.utils import global_config_holder
from social_groups.trialrunner.utils.hydra_config import MainConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo
from social_groups.trialrunner.utils.phoenix import phoenix_server_is_up, setup_phoenix

log = logging.getLogger("startup")


def print_title():
    banner = figlet_format(CLI_TITLE, font="ansi_shadow", width=200)
    log.info(
        Panel(
            banner, subtitle=CLI_SUBTITLE, title_align="center", subtitle_align="center"
        )
    )


def initialize_phoenix(conf: MainConfig, meta_info: ExperimentMetaInfo):
    log.info("Checking Phoenix Server connection...")

    if phoenix_server_is_up(url=f"{conf.execution.phoenix_server_url}/healthz"):
        log.info("Writing LLM Interactions into Phoenix. It is up and running.")
        setup_phoenix(
            endpoint=f"{conf.execution.phoenix_server_url}/v1/traces",
            project_name=meta_info.phoenix_project_name,
        )

        return

    log.warning(
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


def print_config_overview(config: MainConfig):
    log.info(
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
        )
    )


def print_final_message(meta_info: ExperimentMetaInfo):
    log.info(
        Panel(
            Group(
                Text("✅ Done\n", style="bold green"),
                Text(f"Results: {meta_info.output_directory}"),
                Text(f"Phoenix Project: {meta_info.phoenix_project_name}"),
            ),
            title="Success",
        )
    )


def handle_failure_exception(exception: Exception):
    log.info(Panel(str(exception), title="Run failed", style="red"))
    if isinstance(exception, APIConnectionError):
        log.exception(
            Panel(repr(exception.request), title="Failed Request", style="yellow")
        )
    elif isinstance(exception, ValidationError):
        pass
    else:
        log.exception(
            Traceback.from_exception(
                type(exception), exception, exception.__traceback__, show_locals=True
            )
        )

    log.info("Error raised: %s", repr(exception))
    exit(1)


def setup_config(cfg: DictConfig) -> MainConfig:
    try:
        if not OmegaConf.has_resolver("hostname"):
            OmegaConf.register_new_resolver("hostname", lambda: socket.gethostname())

        cfg = MainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

        global_config_holder.global_hydra_config = cfg
        return cfg

    except ValidationError:
        log.warning(
            "\n[bold yellow] Did you select an experiment config? (experiment=<exp name>)[/bold yellow]\n"
        )
        raise
