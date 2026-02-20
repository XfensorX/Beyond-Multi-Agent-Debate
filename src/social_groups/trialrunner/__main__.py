import socket

import hydra
from omegaconf import DictConfig, OmegaConf

from social_groups.trialrunner.experiment.run_experiment import run_experiment
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


@hydra.main(version_base=None, config_name="base")
def main(cfg: DictConfig):
    config = setup_config(cfg)
    meta_info = generate_meta_information()

    try:
        setup_logging(output_directory=meta_info.output_directory)

        print_title()
        print_config_overview(config)

        initialize_phoenix(config, meta_info)

        run_experiment(config, meta_info.output_directory)

    except Exception as e:
        handle_failure_exception(e)

    finally:
        print_final_message(meta_info)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("hostname", lambda: socket.gethostname())
    main()
