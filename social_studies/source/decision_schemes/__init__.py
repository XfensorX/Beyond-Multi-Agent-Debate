from enum import Enum
from typing import Callable

from data_connectors.mmlu_pro import ExperimentInput, ExperimentOutput
from decision_schemes.single_agent import single_model_baseline


class GroupDecisionScheme(Enum):
    SingleAgent = "SingleAgent"


DecisionSchemeFunction = Callable[[ExperimentInput], ExperimentOutput]
DECISION_SCHEMES: dict[GroupDecisionScheme, DecisionSchemeFunction] = {
    GroupDecisionScheme.SingleAgent: single_model_baseline
}
