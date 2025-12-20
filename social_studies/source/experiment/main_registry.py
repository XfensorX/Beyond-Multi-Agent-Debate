from __future__ import annotations

from enum import Enum

import data_connectors
import decision_schemes
from data_connectors.base import DataConnector
from decision_schemes.base import DecisionScheme
from utils.general import import_all_submodules, make_enum
from utils.registry import Registry


def _validate_decision_scheme(cls: type[DecisionScheme]):
    if not isinstance(cls, type) or not issubclass(cls, DecisionScheme):
        raise TypeError("Only DecisionScheme subclasses can be registered")


def _validate_data_connector(cls: type[DataConnector]):
    if not isinstance(cls, type) or not issubclass(cls, DataConnector):
        raise TypeError("Only DataConnector subclasses can be registered")


DECISION_SCHEMES: Registry[type[DecisionScheme]] = Registry("decision scheme")
DATA_CONNECTORS: Registry[type[DataConnector]] = Registry("data connector")


def register_data_connector(name):
    return DATA_CONNECTORS.register(name, validate=_validate_data_connector)


def register_decision_scheme(name):
    return DECISION_SCHEMES.register(name, validate=_validate_decision_scheme)


import_all_submodules(decision_schemes)
import_all_submodules(data_connectors)


DecisionSchemeName: type[Enum] = make_enum(
    "GroupDecisionScheme", set(DECISION_SCHEMES.names())
)
DataConnectorName: type[Enum] = make_enum("DataConnector", set(DATA_CONNECTORS.names()))
