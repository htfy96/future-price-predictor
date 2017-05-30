

class SequentialIndexList(object):
    def __init__(self, from_idx, to_idx):
        """
        Returns an object which behaves like [from, from+1, ..., to]
        """
        self._from = from_idx
        self._to = to_idx
    def __len__(self):
        return self._to - self._from + 1
    def __getitem__(self, item):
        return self._from + item
