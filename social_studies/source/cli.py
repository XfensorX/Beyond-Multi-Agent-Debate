import hydra
from experiment.run_experiment import run_experiment
from omegaconf import DictConfig
from rich.console import Console
from utils import global_config_holder
from utils.cli_utils import (
    handle_failure_exception,
    initialize_phoenix,
    print_config_overview,
    print_final_message,
    print_title,
    setup_config,
)
from utils.logging import setup_logging

console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    config = setup_config(console, cfg)
    global_config_holder.global_hydra_config = config

    try:
        print_title(console)
        setup_logging(output_directory=config.meta_info.output_directory)
        print_config_overview(console, config)

        initialize_phoenix(console, config)

        run_experiment(config)

    except Exception as e:
        handle_failure_exception(console, e)

    finally:
        print_final_message(console, config)


if __name__ == "__main__":
    main()
