from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, get_args, get_origin

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from social_groups.general.utils.standard_library import BaseModelWithExtraFields


class HistoryMessage(BaseModel):
    input_context: list[BaseMessage]
    answer: str
    agent_id: int


class ExampleInput(BaseModel):
    presolved_questions: str
    question: str


class ExampleOutput(BaseModel):
    used_input_tokens: int
    used_output_tokens: int
    answers_at_beginning: list[str] | None = None
    answers_at_end: list[str] | None = None
    final_answer: str | None = None
    history: list[HistoryMessage]


ConfigurationOptions = TypeVar("ConfigurationOptions", bound=BaseModelWithExtraFields)


class DecisionScheme(ABC, Generic[ConfigurationOptions]):
    # this will be filled automatically on creation
    config: ConfigurationOptions

    # >>>>>>>>>>>>>>>>>>>>>> has to be overridden in subclass >>>>>>>>>>>>>>>>>>>>>>>>>>
    @abstractmethod
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        """
        !!! This method has to be thread safe !!!

        You have access to the configuration parameters on self.config


        :param example_input: the experiment_input for one example
        :return: the experiment output of that example
        """
        raise NotImplementedError()

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def __init__(self, config: ConfigurationOptions) -> None:
        self.config: ConfigurationOptions = self.validate_config(config)

    _config_type_arg: ConfigurationOptions

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for b in getattr(cls, "__orig_bases__", ()):
            if get_origin(b) is DecisionScheme:
                (arg,) = get_args(b)
                cls._config_type_arg = arg
                return

        raise TypeError(
            f"{cls.__name__} must subclass DecisionScheme[SomeType] with the config parameters type"
        )

    def validate_config(self, config: Any) -> ConfigurationOptions:
        return self._config_type_arg.model_validate(config)
