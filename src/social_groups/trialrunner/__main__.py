import hydra
from omegaconf import DictConfig
from rich.console import Console

from social_groups.trialrunner.experiment.run_experiment import run_experiment
from social_groups.trialrunner.utils import global_config_holder
from social_groups.trialrunner.utils.cli_utils import (
    handle_failure_exception,
    initialize_phoenix,
    print_config_overview,
    print_final_message,
    print_title,
    setup_config,
)
from social_groups.trialrunner.utils.logging import setup_logging
from social_groups.trialrunner.utils.meta_info import generate_meta_information

console = Console()


@hydra.main(version_base=None, config_name="base")
def main(cfg: DictConfig):
    config = setup_config(console, cfg)
    meta_info = generate_meta_information()

    global_config_holder.global_hydra_config = config

    try:
        print_title(console)
        setup_logging(output_directory=meta_info.output_directory)
        print_config_overview(console, config)

        initialize_phoenix(console, config, meta_info)

        run_experiment(config, meta_info.output_directory)

    except Exception as e:
        handle_failure_exception(console, e)

    finally:
        print_final_message(console, config, meta_info)


if __name__ == "__main__":
    main()
