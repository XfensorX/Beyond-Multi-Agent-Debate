from functools import lru_cache
from typing import Callable, Generic, Hashable, TypeVar

TKey = TypeVar("TKey", bound=Hashable)
TParam = TypeVar("TParam")
TVal = TypeVar("TVal")


class LazyDict(Generic[TKey, TParam, TVal]):
    """
    A dictionary-like object that computes values lazily using a heavy function
    and caches them with LRU (Least Recently Used) eviction.
    """

    def __init__(self, compute_func: Callable[[TParam], TVal], max_cached: int):
        """
        Args:
            compute_func: Function that takes a key (id) and returns the computed value.
                          Example: lambda i: heavy(load(i))
            max_cached: Maximum number of items to keep in cache (None = unlimited)
        """
        super().__init__()
        self.content: dict[TKey, TParam] = {}
        self.compute_func = compute_func
        self.maxsize = max_cached

        @lru_cache(maxsize=max_cached)
        def cached_compute(key: TParam) -> TVal:
            return self.compute_func(key)

        self._cached_compute = cached_compute

    def __getitem__(self, key: TKey) -> TVal:
        if key not in self.content:
            raise KeyError(key)

        return self._cached_compute(self.content[key])

    def register(self, key: TKey, value: TParam) -> None:
        self.content[key] = value

    def __contains__(self, key: TKey) -> bool:
        return key in self.content

    def __len__(self) -> int:
        """Return current number of cached items."""
        return len(self.content)
