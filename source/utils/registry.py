from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Generic, Iterable, Optional, TypeVar

from source.utils.general import contains_whitespace

T = TypeVar("T", bound=type)


class DuplicateRegistrationError(KeyError):
    pass


class UnknownRegistrationError(KeyError):
    pass


class InvalidRegistryNameError(ValueError):
    messages = ["Registration name must not contain whitespace"]

    def __str__(self):
        return "\n - ".join([super().__str__()] + self.messages)


@dataclass(frozen=True)
class Registry(Generic[T]):
    """
    Registry of named classes (or callables) with a decorator-based API.

    - Use .register("name") as a decorator
    - Use .get("name") to retrieve
    - Enforces uniqueness by default
    """

    kind: str
    _items: Dict[str, T]

    def __init__(self, kind: str):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_items", {})

    def register(
        self,
        name: str,
        *,
        override: bool = False,
        validate: Optional[Callable[[T], None]] = None,
    ) -> Callable[[T], T]:
        """
        Register an item under `name`.

        - override=False -> raises on duplicates
        - validate(item) can enforce base-class constraints, etc.
        """

        if contains_whitespace(name):
            raise InvalidRegistryNameError(name)

        def decorator(item: T) -> T:
            if validate is not None:
                validate(item)

            if not override and name in self._items and self._items[name] is not item:
                existing = self._items[name]
                raise DuplicateRegistrationError(
                    f"Duplicate {self.kind} registration for '{name}': "
                    f"{existing.__module__}.{existing.__qualname__} already registered, "
                    f"cannot register {item.__module__}.{item.__qualname__}"
                )

            self._items[name] = item
            return item

        return decorator

    def get(self, name: str | Enum) -> T:
        try:
            if isinstance(name, Enum):
                name = name.value

            return self._items[name]
        except KeyError as e:
            raise UnknownRegistrationError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            ) from e

    def names(self) -> Iterable[str]:
        return iter(sorted(self._items.keys()))

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def __contains__(self, name: str) -> bool:
        return name in self._items
