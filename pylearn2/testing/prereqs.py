""" Objects to be used as Monitor prereqs during testing. """


class ReadVerifyPrereq(object):
    """
    Part of tests/test_monitor.py. Just put here so it be serialized.

    Parameters
    ----------
    counter_idx : WRITEME
    counter : WRITEME
    """
    def __init__(self, counter_idx, counter):
        self.counter_idx = counter_idx
        self.counter = counter

    def __call__(self, *data):
        # We set up each dataset with a different batch size
        # check here that we're getting the right one
        X, = data
        assert X.shape[0] == self.counter_idx + 1
        # Each dataset has different content, make sure we
        # get the right one
        assert X[0,0] == self.counter_idx
        prereq_counter = self.counter
        prereq_counter.set_value(prereq_counter.get_value() + 1)
