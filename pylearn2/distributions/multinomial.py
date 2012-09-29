"""TODO: document me."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import numpy as N


class Multinomial(object):
    """TODO: document me."""
    def __init__(self, rng, pi, renormalize=False):
        """TODO: document me."""
        self.pi = pi
        assert self.pi.min() >= 0.0
        self.rng = rng
        if renormalize:
            self.pi = self.pi / self.pi.sum()
        else:
            assert abs(1.0 - self.pi.sum()) < 1e-10

    def sample_integer(self, m):
        return N.nonzero(
            self.rng.multinomial(pvals=self.pi, n=1, size=(m,))
        )[1]
