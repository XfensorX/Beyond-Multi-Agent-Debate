from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Package:
    id: int
    run_id: int


@dataclass(slots=True)
class SendPackage(Package):
    span_id: str
    phoenix_graphql_endpoint: str


@dataclass(slots=True)
class ReceivePackage(Package):
    span_info: dict[str, Any] | None


class PendingCollection:
    def __init__(self) -> None:
        self._pending: dict[str, list[SendPackage]] = defaultdict(list)
        self._ids_pending = 0  # indicating how many are inside totally

        self._ids_largest_endpoint = 0  # len of the largest queue
        self._largest_endpoint: str | None = None

    def _update_largest_endpoint(self) -> None:
        self._ids_largest_endpoint = 0
        self._largest_endpoint = None

        for endpoint, pending in self._pending.items():
            if (new_len := len(pending)) > self._ids_largest_endpoint:
                self._ids_largest_endpoint = new_len
                self._largest_endpoint = endpoint

    def add(self, job: SendPackage) -> None:
        pending_list = self._pending[job.phoenix_graphql_endpoint]

        pending_list.append(job)
        self._ids_pending += 1

        if (new_len := len(pending_list)) > self._ids_largest_endpoint:
            self._ids_largest_endpoint = new_len
            self._largest_endpoint = job.phoenix_graphql_endpoint

    def is_empty(self) -> bool:
        return self._ids_pending == 0

    def current_largest_batch_size(self) -> int:
        return self._ids_largest_endpoint

    def get_batch(self, maximum_length: int) -> list[SendPackage]:
        length = min(maximum_length, self._ids_largest_endpoint)
        endpoint = self._largest_endpoint

        if endpoint is None:
            return []

        batch = self._pending[endpoint][:length]
        self._pending[endpoint] = self._pending[endpoint][length:]

        self._ids_pending -= length
        self._update_largest_endpoint()

        return batch

    @property
    def total_pending(self):
        return self._ids_pending
