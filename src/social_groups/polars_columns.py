"""
Use this as
import social_groups.polars_columns as plc
"""

from social_groups.reporting.analysis_columns import AnalysisColumn

# The constellation of the group, e.g. LLM, LHM, etc.
group_constellation = "group_constellation"

final_answer = "final_answer"

# the model names of the models involved as a list
model_names = "model_names"

model_family = "model_family"


# The models that are allowed to openly think
thinking_models = "thinking_models"


experiment_configuration_json = "experiment_configuration_json"
answers_at_beginning = "answers_at_beginning"
answers_at_end = "answers_at_end"

# the correct answer as string form
answer_string = "answer_string"

# If the answer of this row is correct:
is_correct = "is_correct"

accuracy = "accuracy"

# TODO: refactor
parsed_individual_answers_after: str = (
    AnalysisColumn.parsed_individual_answers_after.value
)
