from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Any, Generic, TypeVar
from langchain_core.messages import BaseMessage

from utils.general import BaseModelWithExtraFields


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
    used_input_tokens: int
    used_output_tokens: int
    answers_at_beginning: list[str] | None = None
    answers_at_end: list[str] | None = None
    final_answer: str
    history: list[HistoryMessage]


ConfigurationOptions = TypeVar("ConfigurationOptions", bound=BaseModelWithExtraFields)


class DecisionScheme(ABC, Generic[ConfigurationOptions]):
    # >>>>>>>>>>>>>>>>>>>>>> has to be overridden in subclass >>>>>>>>>>>>>>>>>>>>>>>>>>

    # the pydantic model for the input parameters
    configuration_parameters: ConfigurationOptions

    @abstractmethod
    def run_example(  # TODO: make the handling of config_params more intuitive
        self, example_input: ExampleInput, config_params: ConfigurationOptions
    ) -> ExampleOutput:
        """
        !!! This method has to be thread safe !!!


        :param config_params: the configuration parameters given from the experiment configuration
        :param example_input: the experiment_input
        :return: the experiment output
        """
        raise NotImplementedError()

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def __init__(self, options: ConfigurationOptions) -> None:
        self.config_params: ConfigurationOptions = (
            self.configuration_parameters.model_validate(options)
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls is DecisionScheme:
            return

        model = getattr(cls, "configuration_parameters", None)
        if model is None:
            raise TypeError(
                f"{cls.__name__} must define class attribute "
                f"`configuration_parameters = <pydantic.BaseModel subclass>`"
            )
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.configuration_parameters must be a subclass of pydantic.BaseModel"
            )
