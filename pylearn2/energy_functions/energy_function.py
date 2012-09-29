"""TODO: module level docstring."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import theano.tensor as T


class EnergyFunction(object):
    """TODO: class level docstring."""
    def __init__(self):
        pass

    def score(self, X):
        assert X.dtype.find('int') == -1

        X_name = 'X' if X.name is None else X.name

        E = self.free_energy(X)

        #There should be one energy value for each example in the batch
        assert len(E.type.broadcastable) == 1

        dummy = T.sum(E)
        rval = T.grad(dummy, X)
        rval.name = 'score(' + X_name + ')'
        return rval

    def free_energy(self, X):
        raise NotImplementedError(str(type(self)) +
                                  ' has not implemented free_energy(self,X)')

    def energy(self, varlist):
        raise NotImplementedError(str(type(self)) +
                                  ' has not implemented energy(self,varlist)')

    def __call__(self, varlist):
        return self.energy(varlist)
