from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Any
from langchain_core.messages import BaseMessage


class HistoryMessage(BaseModel):
    input_context: list[BaseMessage]
    answer: str
    agent_id: int
    model_name: str
    options: dict[str, Any]


class ExampleInput(BaseModel):
    example_questions: str
    question: str


class ExampleOutput(BaseModel):
    number_of_agents: int
    used_input_tokens: int
    used_output_tokens: int
    used_rounds: int
    answers_at_beginning: list[str] | None = None
    answers_at_end: list[str] | None = None
    final_answer: str
    history: list[HistoryMessage]


class DecisionScheme(ABC):
    @abstractmethod
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        """
        !!! This method has to be thread safe !!!

        :param example_input: the experiment_input
        :return:
        """
        raise NotImplementedError()
