from __future__ import annotations

from rich.console import Console
from typing import List
import typer
from rich.panel import Panel
from rich.syntax import Syntax

import hydra
from omegaconf import OmegaConf

from config import config_dir
from experiment.run_experiment import ExperimentConfig, run_experiment

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
        console.print(Panel(str(e), title="Run failed", style="bold red"))
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold green]✅ Done[/bold green]\n\nResults: {info.results_dir}",
            title="Success",
        )
    )


if __name__ == "__main__":
    app()
