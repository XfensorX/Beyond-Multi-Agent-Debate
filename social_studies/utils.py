class AccuracyTracker:
    success = 0
    failed = 0

    def register_failed(self):
        self.failed += 1

    def register_success(self):
        self.success += 1

    @property
    def total(self):
        return self.success + self.failed

    @property
    def accuracy(self):
        return self.success / self.total

    def __repr__(self):
        return f"Accuracy {self.success} / {self.total}"
