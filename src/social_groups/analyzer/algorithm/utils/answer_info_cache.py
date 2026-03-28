import threading
from itertools import count

from social_groups.general.tracking import TrackEntry


class AnswerInfoCache:
    """Thread Safe"""

    def __init__(self):
        self._cache = {}
        self._id_generator = count()
        self._lock = threading.RLock()

    def register_answer_info(self, entry: TrackEntry, run_id: int) -> int:
        with self._lock:
            _id = next(self._id_generator)

            self._cache[_id] = (entry, run_id)
            return _id

    def retrieve_answer_info(self, _id: int) -> tuple[TrackEntry, int]:
        with self._lock:
            return self._cache.pop(_id)
