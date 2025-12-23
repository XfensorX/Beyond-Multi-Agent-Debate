from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable

from pydantic import BaseModel

from decision_schemes.base import ExampleInput

Example = TypeVar("Example", bound=BaseModel)


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
