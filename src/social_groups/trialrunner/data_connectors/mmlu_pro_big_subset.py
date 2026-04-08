from social_groups.trialrunner.data_connectors.mmlu_pro import (
    MMLUProConnector,
)
from social_groups.trialrunner.experiment.main_registry import register_data_connector

SEED = 128
TOTAL_N = 1000


@register_data_connector("mmlu-pro-big-subset")
class MMLUProBigSubsetConnector(MMLUProConnector):
    prompts = None

    def __init__(self):
        super().__init__()

        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=TOTAL_N,
            seed=SEED,
        )["test"]
