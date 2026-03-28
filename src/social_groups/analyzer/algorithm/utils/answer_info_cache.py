import threading
from itertools import count

from social_groups.general.tracking import TrackEntry


class AnswerInfoCache:
    """Thread Safe"""

    def __init__(self, ids_starting_at: int = 0):
        self._cache: dict[tuple[int, int], TrackEntry] = {}
        self._id_generator = count(start=ids_starting_at)
        self._lock = threading.RLock()

    def register_answer_info(self, entry: TrackEntry, run_id: int) -> int:
        with self._lock:
            _id = next(self._id_generator)

            self._cache[(_id, run_id)] = entry
            return _id

    def retrieve_answer_info(self, _id: int, run_id: int) -> TrackEntry:
        with self._lock:
            return self._cache.pop((_id, run_id))

    def update(self, other: "AnswerInfoCache") -> None:
        """This can cause deadlocks. Use carefully."""
        with other._lock:
            with self._lock:
                self._cache.update(other._cache)
