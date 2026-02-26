import enum


class AnalysisColumn(enum.Enum):
    # TODO: integrate in polars columns
    parsed_individual_answers_before = "___parsed_individual_answers_before___"
    parsed_individual_answers_after = "___parsed_individual_answers_after___"
    parsed_combined_answers_before = "___parsed_combined_answers_before___"
    parsed_combined_answers_after = "___parsed_combined_answers_after___"

    parsed_answer = "___parsed_answer___"
