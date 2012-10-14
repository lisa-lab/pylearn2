"""A key-aware analog to defaultdict."""


class KeyAwareDefaultDict(dict):
    """
    Like a standard library defaultdict, but pass the key
    to the default factory.
    """
    def __init__(self, default_factory=None):
        self.default_factory = default_factory

    def __getitem__(self, key):
        if key not in self and self.default_factory is not None:
            self[key] = val = self.default_factory(key)
            return val
        else:
            raise KeyError(str(key))
