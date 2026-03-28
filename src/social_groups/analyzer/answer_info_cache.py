from itertools import count

from social_groups.general.tracking import TrackEntry


class AnswerInfoCache:
    def __init__(self):
        self._cache = {}
        self._id_generator = count()

    def register_answer_info(self, entry: TrackEntry, run_id: int) -> int:
        _id = next(self._id_generator)

        self._cache[_id] = (entry, run_id)
        return _id

    def retrieve_answer_info(self, _id: int) -> tuple[TrackEntry, int]:
        return self._cache.pop(_id)
