from social_groups.trialrunner.data_connectors.mmlu_pro import (
    MMLUProConnector,
)
from social_groups.trialrunner.experiment.main_registry import register_data_connector

SEED = 64
TOTAL_N = 100


@register_data_connector("mmlu-pro-subset")
class MMLUProSubsetConnector(MMLUProConnector):
    prompts = None

    def __init__(self):
        super().__init__()

        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=TOTAL_N,
            seed=SEED,
        )["test"]
