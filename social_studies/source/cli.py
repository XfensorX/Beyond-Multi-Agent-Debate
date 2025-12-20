from __future__ import annotations

import dotenv
from openai import APIConnectionError
from pydantic import ValidationError
from rich.console import Console
from typing import List, Literal
import typer
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import Traceback
from rich.tree import Tree

import hydra
from omegaconf import OmegaConf

from config import config_dir
from experiment.run_experiment import ExperimentConfig, run_experiment
from main_registry import DECISION_SCHEMES, DATA_CONNECTORS

app = typer.Typer(add_completion=False, help="Experiment runner CLI")
console = Console()  # Todo: Move into dependencies


@app.command("run")
def run(
    overrides: List[str] = typer.Argument(
        None,
        help="Hydra-style overrides like source=csv experiment=wordcount experiment.field=body",
    ),
    config_name: str = typer.Option(
        "config", "--config", "-c", help="Base config name (without .yaml)"
    ),
    show_config: bool = typer.Option(True, help="Print resolved config before running"),
):
    console.print(
        Panel.fit(
            " Agent Group Experiments ",
            subtitle="CLI App",
        ),
        justify="center",
    )

    overrides = overrides or []
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = hydra.compose(config_name, overrides)

    if show_config:
        resolved = OmegaConf.to_yaml(cfg, resolve=True)
        console.print(
            Panel(
                Syntax(resolved, "yaml", word_wrap=True),
                title="Resolved config",
                title_align="left",
            ),
        )

    try:
        info = run_experiment(ExperimentConfig(**cfg))

    except Exception as e:
        console.print(Panel(str(e), title="Run failed", style="red"))
        if isinstance(e, APIConnectionError):
            console.print(
                Panel(repr(e.request), title="Failed Request", style="yellow")
            )

        elif not isinstance(e, ValidationError):
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold green]✅ Done[/bold green]\n\nResults: {info.results_dir}",
            title="Success",
        )
    )


@app.command("list")
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
    app()
