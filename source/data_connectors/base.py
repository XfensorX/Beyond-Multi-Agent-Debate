from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from decision_schemes.base import ExampleInput
from pydantic import BaseModel


class ExampleBase(BaseModel):
    question_id: int


Example = TypeVar("Example", bound=ExampleBase)


class DataConnector(ABC, Generic[Example]):
    @abstractmethod
    def prepare_example(self, example: Example) -> ExampleInput:
        """
        !!! This has to be thread safe !!!
        :param example:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def iterate_data(self) -> Iterable[Example]:
        raise NotImplementedError()

    @abstractmethod
    def data_length(self) -> int:
        raise NotImplementedError()
